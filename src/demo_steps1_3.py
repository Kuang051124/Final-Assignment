"""
前三步可视化演示 —— 场景 + 相机画面 + 检测结果。
在 PyBullet GUI 窗口中运行，按 ESC 退出。
"""
import sys
sys.path.insert(0, '.')
from scene import build_scene
from camera import SimCamera
from detection import detect_objects
from utils import draw_detection_overlay
import cv2
import pybullet as p
import numpy as np
import time


def main():
    print("构建场景...")
    robot, objs, table, z = build_scene(gui=True)

    cam = SimCamera(640, 480, fov=70)
    cam_pos = [0.5, -0.3, 0.8]
    look_at = [0.6, 0.0, z]

    fps_counter = []
    print("\n运行中... 请查看两个窗口：")
    print("  1) PyBullet 3D 场景窗口")
    print("  2) OpenCV 检测结果窗口（原图 + mask + 检测框）")
    print("\n按 ESC 键退出\n")

    while p.isConnected():
        t0 = time.time()

        # 步骤2: 捕获相机图像
        rgb, depth = cam.capture(cam_pos, look_at)

        # 步骤3: 目标检测
        objects, masks = detect_objects(rgb, debug=True)

        # ---- 可视化 ----
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 左上：原图
        # 右上：mask（三色叠加）
        mask_display = np.zeros_like(bgr)
        cls_colors = {"gear": (0, 0, 255), "bearing": (255, 0, 0), "washer": (0, 255, 0)}
        for name, mask in masks.items():
            mask_display[mask > 0] = cls_colors.get(name, (255, 255, 255))

        # 左下：检测结果（框+标签+中心十字）
        vis = draw_detection_overlay(bgr, objects)

        # 右下：深度图（归一化显示）
        depth_valid = depth.copy()
        valid = depth_valid > 0
        if valid.any():
            dmin, dmax = depth_valid[valid].min(), depth_valid[valid].max()
            depth_valid[valid] = (depth_valid[valid] - dmin) / (dmax - dmin + 1e-8)
        depth_color = cv2.applyColorMap((depth_valid * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 拼接四宫格
        top = np.hstack([bgr, mask_display])
        bot = np.hstack([vis, depth_color])
        combined = np.vstack([top, bot])

        # FPS
        fps_counter.append(1.0 / (time.time() - t0 + 1e-8))
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        fps = np.mean(fps_counter)
        cv2.putText(combined, f"FPS: {fps:.1f} | Objects: {len(objects)}",
                    (8, combined.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Step 1-3 Demo | RGB | Masks | Detection | Depth", combined)

        # 步进仿真
        p.stepSimulation()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    p.disconnect()
    print("退出。")


if __name__ == "__main__":
    main()
