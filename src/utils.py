"""
工具函数：可视化叠加、坐标转换、数据保存。
"""
import cv2
import numpy as np


def draw_detection_overlay(bgr_img, objects):
    """
    在图像上叠加检测结果可视化。

    参数:
        bgr_img:  (H, W, 3) uint8 BGR 图像
        objects:  检测结果列表，每个元素需有:
                  .class_name, .cx, .cy, .angle, .contour

    返回:
        vis: 叠加了标注的图像
    """
    vis = bgr_img.copy()
    colors = {
        "gear":    (0, 0, 255),   # BGR 红色
        "bearing": (255, 0, 0),   # BGR 蓝色
        "washer":  (0, 255, 0),   # BGR 绿色
    }

    for obj in objects:
        color = colors.get(obj.class_name, (255, 255, 255))

        # 画轮廓
        cv2.drawContours(vis, [obj.contour.astype(np.int32)], -1, color, 2)

        # 画中心十字
        cx, cy = int(obj.cx), int(obj.cy)
        cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)

        # 画最小外接矩形
        if hasattr(obj, 'box_points') and obj.box_points is not None:
            cv2.drawContours(vis, [obj.box_points.astype(np.int32)], -1, color, 1)

        # 标签文字
        label = f"{obj.class_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (cx - tw//2 - 3, cy - 22), (cx + tw//2 + 3, cy - 6), color, -1)
        cv2.putText(vis, label, (cx - tw//2, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def draw_path_overlay(bgr_img, path_pixels, color=(255, 0, 0)):
    """
    在图像上叠加路径箭头。

    参数:
        bgr_img:     BGR 图像
        path_pixels: 路径点列表 [(u, v), ...]，像素坐标
        color:       BGR 颜色
    """
    vis = bgr_img.copy()
    for i in range(len(path_pixels) - 1):
        p1 = (int(path_pixels[i][0]), int(path_pixels[i][1]))
        p2 = (int(path_pixels[i+1][0]), int(path_pixels[i+1][1]))
        cv2.arrowedLine(vis, p1, p2, color, 2, tipLength=0.1)
    return vis


def world_to_pixel(world_pt, K, cam_pos, cam_orn):
    """
    将世界坐标点投影到像素坐标（用于验证检测结果）。

    参数:
        world_pt: (3,) 世界坐标
        K:         (3,3) 内参矩阵
        cam_pos:   (3,) 相机在世界中的位置
        cam_orn:   (4,) 相机在世界中的朝向四元数

    返回:
        (u, v): 像素坐标
    """
    from pybullet import getMatrixFromQuaternion

    R = np.array(getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
    t = np.array(cam_pos)

    # 世界 → 相机
    pt_cam = R.T @ (np.array(world_pt) - t)

    # 相机 → 像素
    if pt_cam[2] <= 1e-6:
        return None
    u = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
    v = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]
    return np.array([u, v])


def add_status_bar(bgr_img, fps, obj_count, mode=""):
    """在图像底部叠加状态栏。"""
    h, w = bgr_img.shape[:2]
    bar = bgr_img.copy()
    cv2.rectangle(bar, (0, h - 24), (w, h), (0, 0, 0), -1)
    text = f"FPS: {fps:.1f} | Objects: {obj_count}"
    if mode:
        text += f" | {mode}"
    cv2.putText(bar, text, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return bar


def add_3d_info(bgr_img, objects):
    """
    在图像上标注物体的 3D 坐标信息（如果有 world_pos 属性）。
    """
    vis = bgr_img.copy()
    for obj in objects:
        if not hasattr(obj, 'world_pos') or obj.world_pos is None:
            continue
        cx, cy = int(obj.cx), int(obj.cy)
        wp = obj.world_pos
        text = f"({wp[0]*1000:.0f},{wp[1]*1000:.0f},{wp[2]*1000:.0f})mm"
        cv2.putText(vis, text, (cx + 28, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return vis


def save_results(save_path, rgb_img, depth_img, objects, frame_id=0):
    """保存中间结果到 output 目录。"""
    import os
    prefix = os.path.join(save_path, f"frame_{frame_id:04d}")
    cv2.imwrite(f"{prefix}_rgb.png", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    # 深度图保存为 16-bit PNG（毫米单位）
    depth_mm = (depth_img * 1000).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(f"{prefix}_depth.png", depth_mm)
    print(f"[SAVE] {prefix}_*.png")
