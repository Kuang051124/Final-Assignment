"""
PyBullet 场景搭建：地面、桌面、工件、机器人、相机。
"""
import pybullet as p
import pybullet_data
import numpy as np


def build_scene(gui=True):
    """
    构建完整的仿真场景。

    返回:
        robot_id:   机器人 ID
        obj_infos:  工件信息列表 [{"id": int, "name": str, "color": str}, ...]
        table_id:   桌面 ID
    """
    if gui:
        client = p.connect(p.GUI)
    else:
        client = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # ---- 地面 ----
    plane_id = p.loadURDF("plane.urdf")

    # ---- 桌面（用方块模拟） ----
    table_pos = [0.6, 0.0, 0.0]
    table_half = [0.4, 0.3, 0.02]
    table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half)
    table_visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=table_half,
        rgbaColor=[0.7, 0.7, 0.7, 1])
    table_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_shape,
        baseVisualShapeIndex=table_visual,
        basePosition=table_pos)

    # 四条桌腿
    leg_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.3])
    leg_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.3],
                                     rgbaColor=[0.5, 0.5, 0.5, 1])
    for dx, dy in [(-0.35, -0.25), (-0.35, 0.25), (0.35, -0.25), (0.35, 0.25)]:
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=leg_shape,
            baseVisualShapeIndex=leg_visual,
            basePosition=[table_pos[0] + dx, table_pos[1] + dy, table_pos[2] - 0.32])

    table_surface_z = table_pos[2] + table_half[2]  # 桌面顶部高度

    # ---- 工件 ----
    obj_infos = []
    obj_configs = [
        {"name": "gear",    "color": [1, 0, 0, 1], "pos": [0.65, -0.08, table_surface_z],
         "shape": "cylinder", "radius": 0.025, "height": 0.02, "mass": 0.5},
        {"name": "bearing", "color": [0, 0, 1, 1], "pos": [0.55,  0.10, table_surface_z],
         "shape": "box",      "half": [0.025, 0.025, 0.01],     "mass": 0.3},
        {"name": "washer",  "color": [0, 1, 0, 1], "pos": [0.60, -0.02, table_surface_z],
         "shape": "cylinder", "radius": 0.018, "height": 0.01, "mass": 0.2},
    ]

    for cfg in obj_configs:
        if cfg["shape"] == "cylinder":
            col_shape = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=cfg["radius"], height=cfg["height"])
            vis_shape = p.createVisualShape(
                p.GEOM_CYLINDER, radius=cfg["radius"], length=cfg["height"],
                rgbaColor=cfg["color"])
        else:
            col_shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=cfg["half"])
            vis_shape = p.createVisualShape(
                p.GEOM_BOX, halfExtents=cfg["half"],
                rgbaColor=cfg["color"])

        obj_id = p.createMultiBody(
            baseMass=cfg["mass"],
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=cfg["pos"])
        obj_infos.append({
            "id": obj_id, "name": cfg["name"], "color": cfg["color"],
            "initial_pos": np.array(cfg["pos"]),
        })

    # ---- 机器人 KUKA iiwa ----
    robot_start_pos = [0.0, 0.0, table_surface_z]
    try:
        robot_id = p.loadURDF("kuka_iiwa/model.urdf", robot_start_pos)
        # 根据URDF结构调整关节数量
        num_joints = p.getNumJoints(robot_id)
    except Exception:
        # Windows 上 URDF 路径可能有问题，换用 Franka
        print("[WARN] KUKA 模型加载失败，尝试 Franka Panda...")
        try:
            robot_id = p.loadURDF("franka_panda/panda.urdf", robot_start_pos)
            num_joints = p.getNumJoints(robot_id)
        except Exception:
            # 最终降级：自己搭一个简易臂
            print("[WARN] Franka 模型也加载失败，创建简易机械臂...")
            robot_id = _build_simple_arm(table_surface_z)
            num_joints = 4

    print(f"[OK] 场景构建完成。桌面高度: {table_surface_z:.2f}m, "
          f"工件数量: {len(obj_infos)}, 机器人关节数: {num_joints}")

    return robot_id, obj_infos, table_id, table_surface_z


def _build_simple_arm(base_z):
    """如果真的加载不了 URDF，用基本几何体搭一个简易臂。"""
    arm_parts = []
    pos = [0.0, 0.0, base_z]
    arm_parts.append(p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.02]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.02],
                                                  rgbaColor=[0.5, 0.5, 0.5, 1]),
        basePosition=pos))
    return arm_parts[0]  # 返回底座作为"机器人"


def get_ground_truth(obj_infos):
    """获取所有工件的真实世界坐标（用于定量评估）。"""
    gt = {}
    for info in obj_infos:
        pos, orn = p.getBasePositionAndOrientation(info["id"])
        gt[info["id"]] = {
            "name": info["name"],
            "position": np.array(pos),
            "orientation": np.array(orn),
        }
    return gt


if __name__ == "__main__":
    robot, objs, table, z = build_scene(gui=True)
    print("\n工件 ground truth 坐标：")
    for info in objs:
        pos, _ = p.getBasePositionAndOrientation(info["id"])
        print(f"  {info['name']}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    print("\n按 Ctrl+C 或关闭 GUI 窗口退出...")
    try:
        while p.isConnected():
            p.stepSimulation()
    except KeyboardInterrupt:
        p.disconnect()
