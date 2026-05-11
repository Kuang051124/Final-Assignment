"""
抓取规划与路径生成模块。
规划门型轨迹（pick-and-place），用逆运动学求解关节角，驱动机器人。
"""
import time
import pybullet as p
import numpy as np


# KUKA iiwa 参数
END_EFFECTOR_INDEX = 6       # 末端关节索引
NUM_JOINTS = 7               # 关节数量
TOOL_TIP_OFFSET = np.array([0.0, 0.0, 0.02])  # link6 原点到工具尖端的 Z 偏移
GRIP_OFFSET = np.array([0.0, 0.0, 0.03])       # 夹爪到物体中心的偏置
SAFE_Z_OFFSET = 0.15         # 安全高度（物体上方）
PLACE_POS = np.array([0.3, 0.25, 0.02])    # 默认放置位置


def plan_pick_and_place(obj_world_pos, place_pos=None, safe_z_offset=None):
    """
    生成门型抓取搬运路径点序列。

    参数:
        obj_world_pos: (3,) 物体世界坐标
        place_pos:     (3,) 放置位置，None 则用默认
        safe_z_offset: 安全高度偏移

    返回:
        waypoints: [(3,) * N] 路径点列表（世界坐标）
    """
    if place_pos is None:
        place_pos = PLACE_POS.copy()
    if safe_z_offset is None:
        safe_z_offset = SAFE_Z_OFFSET

    grasp_pt = obj_world_pos + GRIP_OFFSET
    approach  = obj_world_pos + np.array([0, 0, safe_z_offset])
    retreat   = approach.copy()
    above_place = place_pos + np.array([0, 0, safe_z_offset])
    place_pt    = place_pos + GRIP_OFFSET

    return [approach, grasp_pt, retreat, above_place, place_pt]


def solve_ik(robot_id, target_pos, end_effector_idx=None,
             target_orn=None, max_iters=200, threshold=1e-4):
    """
    逆运动学求解：给定末端目标位姿，返回关节角。

    参数:
        robot_id:         机器人 ID
        target_pos:       (3,) 末端目标世界坐标
        end_effector_idx: 末端关节索引
        target_orn:       (4,) 末端目标四元数 [x,y,z,w]，默认垂直向下

    返回:
        joint_angles: (N,) 关节弧度，失败返回 None
    """
    if end_effector_idx is None:
        end_effector_idx = END_EFFECTOR_INDEX

    # 默认：末端垂直向下（Z 轴指向 -Z 世界方向）
    if target_orn is None:
        # 绕 Y 轴旋转 180 度使 Z 朝下
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

    # 补偿 link 原点到工具尖端的偏移（末端朝下时，local +Z=world -Z）
    adjusted_pos = np.array(target_pos, dtype=np.float64) + TOOL_TIP_OFFSET

    joint_angles = p.calculateInverseKinematics(
        robot_id,
        end_effector_idx,
        adjusted_pos,
        targetOrientation=target_orn,
        maxNumIterations=max_iters,
        residualThreshold=threshold)

    if joint_angles is None:
        return None
    return np.array(joint_angles[:NUM_JOINTS], dtype=np.float64)


def move_to_joints(robot_id, target_angles, steps=200, speed=1.0):
    """
    平滑移动关节到目标位置（位置控制模式）。

    参数:
        robot_id:       机器人 ID
        target_angles:  (7,) 目标关节角
        steps:          仿真步数
        speed:          运动速度系数（越大越快）
    """
    # 获取当前关节角
    current = np.array([p.getJointState(robot_id, i)[0]
                         for i in range(NUM_JOINTS)])

    for t in range(steps):
        alpha = (t + 1) / steps
        # 梯形速度：缓起缓停
        alpha_smooth = 0.5 - 0.5 * np.cos(alpha * np.pi)
        interp = current + (target_angles - current) * alpha_smooth * speed

        p.setJointMotorControlArray(
            robot_id,
            list(range(NUM_JOINTS)),
            p.POSITION_CONTROL,
            targetPositions=interp)

        p.stepSimulation()
        # GUI 模式下需要给渲染线程留时间，否则动画会「瞬移」
        time.sleep(1.0 / 240.0)


def execute_waypoints(robot_id, waypoints, end_effector_idx=None,
                      steps_per_point=200):
    """
    依次执行路径点：对每个点做 IK 求解，驱动关节到达。

    参数:
        robot_id:          机器人 ID
        waypoints:         路径点列表 [(3,), ...]
        end_effector_idx:  末端关节索引
        steps_per_point:   每个路径点的仿真步数

    返回:
        success: 是否所有路径点都成功
    """
    if end_effector_idx is None:
        end_effector_idx = END_EFFECTOR_INDEX

    for i, wp in enumerate(waypoints):
        joints = solve_ik(robot_id, wp, end_effector_idx)
        if joints is None:
            print(f"  [WARN] IK 求解失败，路径点 {i}: {wp}")
            return False

        # 检查关节角是否在合理范围
        if np.any(np.abs(joints) > 6.28):  # 关节角异常大
            print(f"  [WARN] 关节角异常，路径点 {i}: {wp}")
            return False

        move_to_joints(robot_id, joints, steps=steps_per_point)

    return True


def simulate_grasp(robot_id, obj_id):
    """
    模拟抓取：将物体固定到机器人末端（创建约束）。

    返回:
        constraint_id: 约束 ID，用于后续释放
    """
    constraint = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=END_EFFECTOR_INDEX,
        childBodyUniqueId=obj_id,
        childLinkIndex=-1,          # -1 = 物体 base
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0])
    return constraint


def simulate_release(constraint_id):
    """释放抓取：删除约束。"""
    p.removeConstraint(constraint_id)


def get_end_effector_pose(robot_id, end_effector_idx=None):
    """
    获取机器人末端执行器当前世界坐标。

    返回:
        pos: (3,) 世界坐标
        orn: (4,) 四元数
    """
    if end_effector_idx is None:
        end_effector_idx = END_EFFECTOR_INDEX
    state = p.getLinkState(robot_id, end_effector_idx)
    return np.array(state[0]), np.array(state[1])


def compute_path_pixels(waypoints, K, cam_pos, look_at):
    """
    将世界坐标路径点投影到像素坐标（用于在图像上叠加可视化）。

    参数:
        waypoints: [(3,), ...] 世界坐标路径点
        K:         (3,3) 内参矩阵
        cam_pos, look_at: 相机位姿

    返回:
        pixels: [(u, v), ...] 像素坐标列表
    """
    from depth_pose import compute_cam_rotation
    R_cw = compute_cam_rotation(cam_pos, look_at)
    R_wc = R_cw.T
    t = np.array(cam_pos)

    pixels = []
    for wp in waypoints:
        pt_cam = R_wc @ (wp - t)
        if pt_cam[2] < 1e-6:
            pixels.append(None)
            continue
        u = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
        v = K[1, 2] - K[1, 1] * pt_cam[1] / pt_cam[2]  # v 轴朝下，需反转
        pixels.append(np.array([u, v]))

    return pixels


def plan_path_with_obstacle(obj_pos, place_pos, obstacle_pos, safe_z, grid_size=0.02):
    """
    A* 避障路径规划（2D 俯视图，在 safe_z 高度平面）。

    参数:
        obj_pos:      (3,) 物体位置
        place_pos:    (3,) 放置位置
        obstacle_pos: (3,) 障碍物位置
        safe_z:       运动安全高度
        grid_size:    网格大小（米）

    返回:
        path_2d: [(x, y), ...] 2D 路径点（在 safe_z 高度）
    """
    start = (int(obj_pos[0] / grid_size), int(obj_pos[1] / grid_size))
    goal = (int(place_pos[0] / grid_size), int(place_pos[1] / grid_size))
    obs = (int(obstacle_pos[0] / grid_size), int(obstacle_pos[1] / grid_size))

    # 障碍物膨胀
    obstacles = set()
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            obstacles.add((obs[0] + dx, obs[1] + dy))

    # A* 搜索
    import heapq

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            # 转为世界坐标
            return [np.array([p[0] * grid_size, p[1] * grid_size, safe_z])
                    for p in path]

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in obstacles:
                continue

            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    # A* 失败，返回直线
    return [np.array([obj_pos[0], obj_pos[1], safe_z]),
            np.array([place_pos[0], place_pos[1], safe_z])]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from scene import build_scene
    from camera import SimCamera
    from detection import detect_objects
    from depth_pose import restore_all_objects
    from utils import draw_detection_overlay, draw_path_overlay
    import cv2
    import time

    print("构建场景...")
    robot, obj_infos, table, z = build_scene(gui=True)
    cam = SimCamera(640, 480, fov=70)
    cam_pos = [0.5, -0.3, 0.8]
    look_at = [0.6, 0.0, z]

    # 让场景稳定
    for _ in range(50):
        p.stepSimulation()

    # 捕获 + 检测 + 3D 定位
    rgb, depth = cam.capture(cam_pos, look_at)
    objects = detect_objects(rgb)
    restore_all_objects(objects, depth, cam.K, cam_pos, look_at)

    if not objects:
        print("未检测到物体！")
        p.disconnect()
        exit()

    # 取第一个物体作为抓取目标
    target = objects[0]
    print(f"\n抓取目标: {target.class_name} at {target.world_pos}")

    # 规划路径
    waypoints = plan_pick_and_place(target.world_pos)
    print("路径点:")
    for i, wp in enumerate(waypoints):
        print(f"  {i}: ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})")

    # IK + 执行
    print("\n执行抓取...")
    success = execute_waypoints(robot, waypoints[:2])  # approach -> grasp
    if success:
        print("到达抓取点！")

        # 模拟抓取
        constraint = simulate_grasp(robot, obj_infos[0]["id"])

        # 搬运到放置点
        print("搬运中...")
        time.sleep(0.5)
        execute_waypoints(robot, waypoints[2:])  # retreat -> above_place -> place

        # 释放
        simulate_release(constraint)
        print("放置完成！")

    # 保持窗口
    print("\n按 Ctrl+C 退出...")
    try:
        while p.isConnected():
            p.stepSimulation()
            rgb, _ = cam.capture(cam_pos, look_at)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Grasp Demo", bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    p.disconnect()
