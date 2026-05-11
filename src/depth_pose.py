"""
深度获取与 3D 位姿还原模块。
利用 PyBullet 深度缓冲 + 内参矩阵，将检测到的 2D 像素坐标反投影为 3D 世界坐标。
"""
import numpy as np
import pybullet as p


def compute_cam_rotation(cam_pos, look_at, up=None):
    """
    从相机位置和朝向计算相机→世界的旋转矩阵。

    返回:
        R_cw: (3,3) 相机坐标系 → 世界坐标系的旋转矩阵
    """
    if up is None:
        up = np.array([0, 0, 1], dtype=np.float64)

    forward = np.array(look_at, dtype=np.float64) - np.array(cam_pos, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    cam_up = np.cross(right, forward)

    # 旋转矩阵：列向量为相机坐标系的基在世界中的表示
    # 相机坐标系: X→right, Y→cam_up, Z→forward（指向场景）
    R_cw = np.column_stack([right, cam_up, forward])
    return R_cw


def pixel_to_camera(u, v, depth, K):
    """
    像素坐标 + 深度 → 相机坐标系 3D 坐标。

    相机坐标系约定：X→右, Y→上, Z→光轴指向场景（与 OpenCV/OpenGL 一致）。

    参数:
        u, v:  像素坐标（OpenCV 约定，原点左上，v 向下）
        depth: 深度值（米），沿光轴方向的距离
        K:     (3,3) 内参矩阵

    返回:
        (3,) 相机坐标系下的点 (Xc, Yc, Zc)
    """
    Xc = (u - K[0, 2]) * depth / K[0, 0]
    # v 轴向下，相机 Y 轴向上，所以需取反
    Yc = (K[1, 2] - v) * depth / K[1, 1]
    Zc = depth
    return np.array([Xc, Yc, Zc], dtype=np.float64)


def camera_to_world(pt_cam, R_cw, cam_pos):
    """
    相机坐标系 → 世界坐标系。

    参数:
        pt_cam:  (3,) 相机坐标系下的点
        R_cw:    (3,3) 相机→世界的旋转矩阵
        cam_pos: (3,) 相机在世界中的位置

    返回:
        (3,) 世界坐标系下的点
    """
    return R_cw @ pt_cam + np.array(cam_pos, dtype=np.float64)


def get_depth_at_pixel(depth_img, u, v, patch_radius=2):
    """
    取像素点周围小区域的深度中值（鲁棒于边缘和噪声）。

    参数:
        depth_img:    (H, W) 深度图，单位：米
        u, v:         像素坐标
        patch_radius: 采样半径

    返回:
        depth: 深度值（米），如果区域无有效深度返回 -1
    """
    h, w = depth_img.shape
    u_int, v_int = int(round(u)), int(round(v))
    r = patch_radius

    y1, y2 = max(0, v_int - r), min(h, v_int + r + 1)
    x1, x2 = max(0, u_int - r), min(w, u_int + r + 1)

    patch = depth_img[y1:y2, x1:x2]
    valid = patch[patch > 0]

    if len(valid) == 0:
        return -1.0

    return float(np.median(valid))


def restore_3d_position(obj, depth_img, K, cam_pos, look_at):
    """
    从检测结果 + 深度图恢复物体的 3D 世界坐标。

    参数:
        obj:       DetectedObject（含 cx, cy 像素坐标）
        depth_img: (H, W) 深度图（米）
        K:         (3,3) 内参矩阵
        cam_pos:   相机世界位置
        look_at:   相机朝向目标

    返回:
        world_pos: (3,) 世界坐标 (Xw, Yw, Zw)，失败返回 None
        depth_val: 使用的深度值
    """
    depth_val = get_depth_at_pixel(depth_img, obj.cx, obj.cy)
    if depth_val <= 0:
        return None, depth_val

    R_cw = compute_cam_rotation(cam_pos, look_at)
    pt_cam = pixel_to_camera(obj.cx, obj.cy, depth_val, K)
    world_pos = camera_to_world(pt_cam, R_cw, cam_pos)

    return world_pos, depth_val


def restore_all_objects(objects, depth_img, K, cam_pos, look_at):
    """
    批量还原所有检测物体的 3D 坐标，结果写入 obj.world_pos。
    返回成功还原的数量。
    """
    R_cw = compute_cam_rotation(cam_pos, look_at)
    success_count = 0

    for obj in objects:
        depth_val = get_depth_at_pixel(depth_img, obj.cx, obj.cy)
        if depth_val <= 0:
            obj.world_pos = None
            obj.depth_val = -1
            continue

        pt_cam = pixel_to_camera(obj.cx, obj.cy, depth_val, K)
        obj.world_pos = camera_to_world(pt_cam, R_cw, cam_pos)
        obj.depth_val = depth_val
        success_count += 1

    return success_count


def evaluate_accuracy(objects, obj_infos):
    """
    定量评估：比对算法还原的坐标与 PyBullet 的 ground truth。

    参数:
        objects:   DetectedObject 列表（含 world_pos）
        obj_infos: scene 中的工件信息列表（含 id, name, initial_pos）

    返回:
        errors: [{"name": str, "calc": (3,), "gt": (3,), "error_mm": float}, ...]
    """
    errors = []

    for obj in objects:
        if obj.world_pos is None:
            continue

        # 按颜色/类别匹配 ground truth
        matched_gt = None
        for info in obj_infos:
            if info["name"] == obj.class_name:
                gt_pos, _ = p.getBasePositionAndOrientation(info["id"])
                matched_gt = np.array(gt_pos)
                break

        if matched_gt is None:
            continue

        error_m = np.linalg.norm(obj.world_pos - matched_gt)
        errors.append({
            "name": obj.class_name,
            "calculated": obj.world_pos.copy(),
            "ground_truth": matched_gt.copy(),
            "error_mm": error_m * 1000,
        })

    return errors


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from scene import build_scene
    from camera import SimCamera
    from detection import detect_objects

    robot, obj_infos, table, z = build_scene(gui=False)
    cam = SimCamera(640, 480, fov=70)
    cam_pos = [0.5, -0.3, 0.8]
    look_at = [0.6, 0.0, z]

    # 捕获 + 检测
    rgb, depth = cam.capture(cam_pos, look_at)
    objects = detect_objects(rgb)

    # 还原 3D 坐标
    n = restore_all_objects(objects, depth, cam.K, cam_pos, look_at)
    print(f"成功还原 {n}/{len(objects)} 个物体的 3D 坐标\n")

    # 定量评估
    errors = evaluate_accuracy(objects, obj_infos)
    total_err = 0
    for e in errors:
        print(f"  {e['name']}:")
        print(f"    计算值: ({e['calculated'][0]:.4f}, {e['calculated'][1]:.4f}, {e['calculated'][2]:.4f}) m")
        print(f"    真值:   ({e['ground_truth'][0]:.4f}, {e['ground_truth'][1]:.4f}, {e['ground_truth'][2]:.4f}) m")
        print(f"    误差:   {e['error_mm']:.2f} mm")
        total_err += e['error_mm']

    if errors:
        print(f"\n  平均 3D 定位误差: {total_err/len(errors):.2f} mm")

    p.disconnect()
