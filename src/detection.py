"""
目标检测模块 —— HSV 颜色分割 + 轮廓检测。
在 PyBullet 仿真环境中，工件颜色纯净，传统方法即可可靠工作。
"""
import cv2
import numpy as np


class DetectedObject:
    """检测到的工件信息。"""
    def __init__(self, class_name, cx, cy, angle, w, h, box_points, contour):
        self.class_name = class_name
        self.cx = float(cx)        # 像素中心 x
        self.cy = float(cy)        # 像素中心 y
        self.angle = float(angle)  # minAreaRect 旋转角（度）
        self.width_px = float(w)   # 像素宽度
        self.height_px = float(h)  # 像素高度
        self.box_points = box_points  # (4,2) 外接矩形四角点
        self.contour = contour        # 原始轮廓
        self.world_pos = None         # 3D 世界坐标（后续填充）


# HSV 颜色范围（根据 PyBullet 渲染实测数据定义）
HSV_RANGES = {
    "gear": {
        # 红色在 HSV 中分两段（hue 绕 0/180）
        "lower1": np.array([0, 80, 80]),
        "upper1": np.array([12, 255, 255]),
        "lower2": np.array([170, 80, 80]),
        "upper2": np.array([180, 255, 255]),
    },
    "bearing": {
        "lower1": np.array([105, 80, 80]),
        "upper1": np.array([135, 255, 255]),
    },
    "washer": {
        "lower1": np.array([45, 80, 80]),
        "upper1": np.array([80, 255, 255]),
    },
}

MIN_CONTOUR_AREA = 80    # 最小轮廓面积（像素）
MAX_CONTOUR_AREA = 2000  # 最大轮廓面积（像素），排除大块背景误检


def detect_objects(rgb_img, debug=False):
    """
    从 RGB 图像中检测工件。

    参数:
        rgb_img: (H, W, 3) uint8 RGB 图像
        debug:   是否返回调试用的 mask 图

    返回:
        objects:  DetectedObject 列表
        debug_masks: (仅 debug=True) {class_name: mask} 字典
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    results = []
    debug_masks = {} if debug else None

    for class_name, ranges in HSV_RANGES.items():
        # 构建该类颜色的 mask
        mask = cv2.inRange(hsv, ranges["lower1"], ranges["upper1"])
        if "lower2" in ranges:
            mask2 = cv2.inRange(hsv, ranges["lower2"], ranges["upper2"])
            mask = cv2.bitwise_or(mask, mask2)

        # 形态学去噪：开运算去除小噪声，闭运算填充小孔洞
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)

        if debug:
            debug_masks[class_name] = mask

        # 轮廓提取
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue

            # 最小外接矩形
            rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect)    # (4, 2) 四角点

            obj = DetectedObject(
                class_name=class_name,
                cx=rect[0][0],
                cy=rect[0][1],
                angle=rect[2],
                w=rect[1][0],
                h=rect[1][1],
                box_points=box,
                contour=cnt)

            results.append(obj)

    return (results, debug_masks) if debug else results


def detect_from_bgr(bgr_img, debug=False):
    """从 BGR 图像检测（会自动转 RGB）。"""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return detect_objects(rgb, debug)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, '.')
    from scene import build_scene
    from camera import SimCamera
    from utils import draw_detection_overlay
    import pybullet as p

    # 构建场景
    robot, objs, table, z = build_scene(gui=False)
    cam = SimCamera(640, 480, fov=70)

    # 捕获一帧
    rgb, depth = cam.capture([0.5, -0.3, 0.8], [0.6, 0.0, z])

    # 检测（debug 模式返回 mask）
    objects, masks = detect_objects(rgb, debug=True)

    print(f"检测到 {len(objects)} 个工件:")
    for obj in objects:
        print(f"  {obj.class_name}: center=({obj.cx:.1f},{obj.cy:.1f}), "
              f"angle={obj.angle:.1f}deg, size=({obj.width_px:.1f},{obj.height_px:.1f})")

    # 可视化
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    vis = draw_detection_overlay(bgr, objects)

    # 拼接：原图 | masks | 检测结果
    mask_display = np.zeros_like(bgr)
    colors = {"gear": (0, 0, 255), "bearing": (255, 0, 0), "washer": (0, 255, 0)}
    for name, mask in masks.items():
        mask_display[mask > 0] = colors.get(name, (255, 255, 255))

    top_row = np.hstack([bgr, mask_display])
    h1, w1 = top_row.shape[:2]
    vis_resized = cv2.resize(vis, (w1, h1))
    combined = np.vstack([top_row, vis_resized])

    out_path = Path(__file__).resolve().parent.parent / "output" / "detection_test.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.png', combined)[1].tofile(str(out_path))
    print(f"Result saved to {out_path}")

    p.disconnect()
