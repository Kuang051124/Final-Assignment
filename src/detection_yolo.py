"""
目标检测模块 —— YOLOv8 粗定位 + HSV 精修。
YOLO 给出类别和大致区域，HSV 在区域内精修轮廓和旋转角。
"""
from pathlib import Path
import cv2
import numpy as np


class DetectedObject:
    """检测到的工件信息（与 HSV 版本接口兼容）。"""
    def __init__(self, class_name, cx, cy, angle, w, h, box_points, contour):
        self.class_name = class_name
        self.cx = float(cx)
        self.cy = float(cy)
        self.angle = float(angle)
        self.width_px = float(w)
        self.height_px = float(h)
        self.box_points = box_points
        self.contour = contour
        self.world_pos = None
        self.depth_val = -1


MODEL_PATH = Path(__file__).resolve().parent.parent / "runs" / "detect_workpieces" / "weights" / "best.pt"
CLASS_NAMES = ["gear", "bearing", "washer", "bolt", "block"]

HSV_RANGES = {
    "gear":    (np.array([0, 80, 80]),   np.array([12, 255, 255]),
                np.array([170, 80, 80]), np.array([180, 255, 255])),
    "bearing": (np.array([105, 80, 80]), np.array([135, 255, 255]),
                None, None),
    "washer":  (np.array([45, 80, 80]),  np.array([80, 255, 255]),
                None, None),
    "bolt":    (np.array([0, 0, 60]),    np.array([180, 30, 200]),
                None, None),
    "block":   (np.array([15, 80, 80]),  np.array([35, 255, 255]),
                None, None),
}

MIN_CONTOUR_AREA = 30
MAX_CONTOUR_AREA = 5000

_model = None


def _load_model():
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _refine_with_hsv(rgb_img, bbox_xyxy, class_name):
    """在 YOLO 包围盒内用 HSV 精修轮廓，获取精确的 minAreaRect。"""
    h, w = rgb_img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    margin = 8
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    roi = rgb_img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    r = HSV_RANGES.get(class_name)
    if r is None:
        return None
    lo1, hi1, lo2, hi2 = r

    mask = cv2.inRange(hsv_roi, lo1, hi1)
    if lo2 is not None:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_roi, lo2, hi2))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
        return None

    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)

    cx = rect[0][0] + x1
    cy = rect[0][1] + y1
    box[:, 0] += x1
    box[:, 1] += y1

    return DetectedObject(
        class_name=class_name,
        cx=cx, cy=cy,
        angle=rect[2],
        w=rect[1][0], h=rect[1][1],
        box_points=box,
        contour=best)


def detect_objects(rgb_img, debug=False, conf_threshold=0.05):
    """
    YOLOv8 粗定位 + HSV 精修轮廓。

    参数:
        rgb_img:       (H, W, 3) uint8 RGB
        debug:         是否返回调试 mask
        conf_threshold: YOLO 置信度阈值

    返回:
        objects:       DetectedObject 列表
        debug_masks:   (仅 debug=True) {class_name: mask}
    """
    model = _get_model()
    results = model(rgb_img, conf=conf_threshold, verbose=False)

    objects = []
    debug_masks = {} if debug else None

    if not results or len(results) == 0:
        return (objects, debug_masks) if debug else objects

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return (objects, debug_masks) if debug else objects

    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id]
        bbox = box.xyxy[0].cpu().numpy()

        obj = _refine_with_hsv(rgb_img, bbox, class_name)
        if obj is not None:
            objects.append(obj)

        if debug and obj is not None:
            mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            debug_masks[class_name] = mask if class_name not in debug_masks else \
                cv2.bitwise_or(debug_masks[class_name], mask)

    return (objects, debug_masks) if debug else objects


def detect_from_bgr(bgr_img, debug=False):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return detect_objects(rgb, debug)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from scene import build_scene
    from camera import SimCamera
    from utils import draw_detection_overlay
    import pybullet as p

    print("Loading YOLO model...")
    robot, objs, table, z = build_scene(gui=False)
    cam = SimCamera(640, 480, fov=70)

    rgb, depth = cam.capture([0.5, -0.3, 0.8], [0.6, 0.0, z])

    objects, masks = detect_objects(rgb, debug=True)
    print(f"Detected {len(objects)} objects:")
    for obj in objects:
        print(f"  {obj.class_name}: ({obj.cx:.1f},{obj.cy:.1f}) angle={obj.angle:.1f}")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    vis = draw_detection_overlay(bgr, objects)
    combined = np.vstack([bgr, vis])

    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "yolo_detection_test.jpg"
    cv2.imencode('.jpg', combined)[1].tofile(str(out_path))
    print(f"Saved to {out_path}")

    p.disconnect()
