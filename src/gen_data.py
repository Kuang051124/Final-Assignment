"""
自动生成 YOLO 训练数据。
PyBullet 渲染 RGB 图，利用已知 3D 位姿自动推算 2D 包围盒标签，零人工标注。
"""
import os
import sys
import random
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from pathlib import Path

sys.path.insert(0, '.')
from camera import SimCamera
from depth_pose import compute_cam_rotation


# 工件定义：(名称, 类别ID, 颜色RGBA, 几何参数)
OBJECT_DEFS = [
    {
        "name": "gear",
        "class_id": 0,
        "color": [0.9, 0.1, 0.1, 1.0],
        "shape": "cylinder",
        "radius": 0.025,
        "half_height": 0.015,
    },
    {
        "name": "bearing",
        "class_id": 1,
        "color": [0.1, 0.1, 0.9, 1.0],
        "shape": "box",
        "half_extents": [0.025, 0.025, 0.012],
    },
    {
        "name": "washer",
        "class_id": 2,
        "color": [0.1, 0.85, 0.1, 1.0],
        "shape": "cylinder",
        "radius": 0.020,
        "half_height": 0.008,
    },
    {
        "name": "bolt",
        "class_id": 3,
        "color": [0.6, 0.6, 0.6, 1.0],
        "shape": "cylinder",
        "radius": 0.012,
        "half_height": 0.035,
    },
    {
        "name": "block",
        "class_id": 4,
        "color": [0.9, 0.7, 0.1, 1.0],
        "shape": "box",
        "half_extents": [0.020, 0.030, 0.018],
    },
]

TABLE_CENTER = np.array([0.6, 0.0, 0.02])
TABLE_HALF = np.array([0.35, 0.25, 0.001])


def build_data_scene():
    """构建用于数据生成的场景（无GUI）。"""
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf")

    # 桌面
    ts = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.3, 0.02])
    tv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.3, 0.02],
                              rgbaColor=[0.6, 0.6, 0.6, 1])
    table = p.createMultiBody(0, ts, tv, TABLE_CENTER)

    # 随机背景色（模拟不同光照）
    bg_r = random.uniform(0.3, 0.7)
    bg_g = random.uniform(0.3, 0.7)
    bg_b = random.uniform(0.5, 0.9)

    return table, bg_r, bg_g, bg_b


def create_object(obj_def, position, orientation=None):
    """根据定义创建工件，返回 body id。"""
    if orientation is None:
        orientation = p.getQuaternionFromEuler([
            random.uniform(0, np.pi * 2),
            random.uniform(0, np.pi * 2),
            random.uniform(0, np.pi * 2)])

    if obj_def["shape"] == "cylinder":
        cs = p.createCollisionShape(p.GEOM_CYLINDER,
                                     radius=obj_def["radius"],
                                     height=obj_def["half_height"] * 2)
        vs = p.createVisualShape(p.GEOM_CYLINDER,
                                  radius=obj_def["radius"],
                                  length=obj_def["half_height"] * 2,
                                  rgbaColor=obj_def["color"])
    else:  # box
        cs = p.createCollisionShape(p.GEOM_BOX,
                                     halfExtents=obj_def["half_extents"])
        vs = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=obj_def["half_extents"],
                                  rgbaColor=obj_def["color"])

    body = p.createMultiBody(
        baseMass=random.uniform(0.1, 0.5),
        baseCollisionShapeIndex=cs,
        baseVisualShapeIndex=vs,
        basePosition=position,
        baseOrientation=orientation)
    return body


def compute_local_corners(obj_def):
    """计算工件在局部坐标系下的 8 个包围盒角点。"""
    if obj_def["shape"] == "cylinder":
        r = obj_def["radius"]
        h = obj_def["half_height"]
        half = np.array([r, r, h])
    else:
        half = np.array(obj_def["half_extents"])

    corners = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                corners.append(np.array([dx * half[0], dy * half[1], dz * half[2]]))
    return np.array(corners)


def compute_2d_bbox(body_id, obj_def, cam, cam_pos, look_at, img_w, img_h):
    """
    根据物体 3D 位姿自动计算 2D 包围盒（YOLO 归一化格式）。

    返回: (class_id, x_center_norm, y_center_norm, w_norm, h_norm)
    """
    pos, orn = p.getBasePositionAndOrientation(body_id)
    pos = np.array(pos)
    rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    # 局部 → 世界
    local_corners = compute_local_corners(obj_def)
    world_corners = (rot_mat @ local_corners.T).T + pos

    # 世界 → 相机
    R_cw = compute_cam_rotation(cam_pos, look_at)
    R_wc = R_cw.T
    t = np.array(cam_pos)
    cam_corners = (R_wc @ (world_corners - t).T).T

    # 只保留在相机前方的角点
    valid = cam_corners[:, 2] > 1e-4
    if not valid.any():
        return None

    # 相机 → 像素
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    u = fx * cam_corners[valid, 0] / cam_corners[valid, 2] + cx
    v = fy * cam_corners[valid, 1] / cam_corners[valid, 2] + cy

    # 边界裁剪
    u = np.clip(u, 0, img_w - 1)
    v = np.clip(v, 0, img_h - 1)

    if u.max() - u.min() < 2 or v.max() - v.min() < 2:
        return None

    # YOLO 格式：归一化 [0, 1]
    x_c = (u.min() + u.max()) / 2 / img_w
    y_c = (v.min() + v.max()) / 2 / img_h
    width = (u.max() - u.min()) / img_w
    height = (v.max() - v.min()) / img_h

    return (obj_def["class_id"], float(x_c), float(y_c), float(width), float(height))


def random_position_on_table():
    """在桌面范围内随机生成位置。"""
    x = TABLE_CENTER[0] + random.uniform(-TABLE_HALF[0] + 0.04, TABLE_HALF[0] - 0.04)
    y = TABLE_CENTER[1] + random.uniform(-TABLE_HALF[1] + 0.04, TABLE_HALF[1] - 0.04)
    z = TABLE_CENTER[2] + TABLE_HALF[2] + 0.005
    return [x, y, z]


def generate_dataset(output_dir, num_images=500, val_split=0.15):
    """
    生成训练+验证数据集。

    参数:
        output_dir: 输出根目录
        num_images: 总图像数
        val_split:  验证集比例
    """
    output_dir = Path(output_dir)
    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"
    val_img_dir = output_dir / "images" / "val"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cam = SimCamera(640, 480, fov=65)

    print(f"生成 {num_images} 张训练数据...")
    for i in range(num_images):
        # 每张图重建场景（随机背景色）
        table, bg_r, bg_g, bg_b = build_data_scene()

        # 随机相机位置
        cam_x = TABLE_CENTER[0] + random.uniform(-0.1, 0.1)
        cam_y = TABLE_CENTER[1] + random.uniform(-0.3, -0.1)
        cam_z = random.uniform(0.7, 1.1)
        cam_pos = [cam_x, cam_y, cam_z]
        look_at = TABLE_CENTER + np.array([random.uniform(-0.05, 0.05),
                                            random.uniform(-0.05, 0.05),
                                            random.uniform(0.0, 0.03)])

        # 随机放置 2-5 个工件
        num_objs = random.randint(2, 5)
        bodies = []
        obj_defs_used = []

        for _ in range(num_objs):
            obj_def = random.choice(OBJECT_DEFS)
            pos = random_position_on_table()
            # 避免重叠：简单重试
            for _ in range(10):
                too_close = False
                for existing in bodies:
                    ep, _ = p.getBasePositionAndOrientation(existing)
                    if np.linalg.norm(np.array(ep) - np.array(pos)) < 0.04:
                        too_close = True
                        break
                if not too_close:
                    break
                pos = random_position_on_table()

            body = create_object(obj_def, pos)
            bodies.append(body)
            obj_defs_used.append(obj_def)

        # 模拟几步让物体落稳定
        for _ in range(50):
            p.stepSimulation()

        # 捕获图像
        rgb, _ = cam.capture(cam_pos, look_at)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 计算标签
        labels = []
        for body, obj_def in zip(bodies, obj_defs_used):
            bbox = compute_2d_bbox(body, obj_def, cam, cam_pos, look_at,
                                    cam.width, cam.height)
            if bbox is not None:
                labels.append(bbox)

        if len(labels) == 0:
            p.disconnect()
            continue  # 跳过此帧

        # 训练/验证分离
        is_val = random.random() < val_split
        img_dir = val_img_dir if is_val else train_img_dir
        lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        fname = f"frame_{i:05d}"
        # cv2.imwrite 在中文路径下会失败，用 imencode + tofile 绕过
        enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
        enc.tofile(str(img_dir / f"{fname}.jpg"))

        with open(lbl_dir / f"{fname}.txt", "w") as f:
            for lbl in labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

        p.disconnect()

        if (i + 1) % 50 == 0:
            print(f"  已生成 {i + 1}/{num_images} ...")

    # 写 data.yaml
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.resolve().as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\nnc: 5\n")
        names = [d["name"] for d in OBJECT_DEFS]
        f.write(f"names: {names}\n")

    print(f"\n数据集生成完成:")
    print(f"  {output_dir}")
    print(f"  data.yaml -> {yaml_path}")
    print(f"  类别: {[d['name'] for d in OBJECT_DEFS]}")


if __name__ == "__main__":
    from pathlib import Path
    output = str(Path(__file__).resolve().parent.parent / "data_yolo")
    generate_dataset(output, num_images=500, val_split=0.15)
