"""
虚拟相机模块 —— 从 PyBullet 场景获取 RGB 图像和深度图。
"""
import pybullet as p
import numpy as np
import cv2


class SimCamera:
    """PyBullet 虚拟相机，输出 OpenCV 兼容的 RGB 图像和深度图。"""

    def __init__(self, width=640, height=480, fov=60, near=0.01, far=5.0):
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self._compute_intrinsics()

    def _compute_intrinsics(self):
        """从 FOV 和图像尺寸计算内参矩阵（方形像素，无畸变）。"""
        aspect = self.width / self.height
        fy = self.height / (2 * np.tan(np.radians(self.fov) / 2))
        fx = fy  # 方形像素
        cx = self.width / 2
        cy = self.height / 2
        self.K = np.array([[fx, 0,  cx],
                           [0,  fy, cy],
                           [0,  0,   1]], dtype=np.float64)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def get_view_matrix(self, cam_pos, look_at, up=None):
        """计算 viewMatrix。"""
        if up is None:
            up = [0, 0, 1]
        return p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=look_at,
            cameraUpVector=up)

    def get_proj_matrix(self):
        """计算 projectionMatrix（透视投影）。"""
        aspect = self.width / self.height
        return p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=aspect,
            nearVal=self.near, farVal=self.far)

    def capture(self, cam_pos, look_at, up=None):
        """
        捕获一帧。

        参数:
            cam_pos:  相机位置 (x, y, z)
            look_at:  相机看向的目标点 (x, y, z)

        返回:
            rgb:    (H, W, 3) uint8  RGB 图像
            depth:  (H, W) float64  深度图，单位：米（PyBullet 深度缓冲）
        """
        vm = self.get_view_matrix(cam_pos, look_at, up)
        pm = self.get_proj_matrix()

        # PyBullet 渲染
        img = p.getCameraImage(
            width=self.width, height=self.height,
            viewMatrix=vm, projectionMatrix=pm,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # 提取 RGB：img[2] 是 RGBA (H*W*4) uint8
        rgba = np.reshape(img[2], (self.height, self.width, 4))
        rgb = rgba[:, :, :3].astype(np.uint8)

        # 提取深度缓冲：img[3] 是 (H*W) float32，OpenGL 非线性 0~1 格式
        depth_buf = np.reshape(img[3], (self.height, self.width)).astype(np.float64)
        # 转换为线性深度（米）：z_eye = far*near / (far - depth*(far-near))
        depth = (self.far * self.near) / (
            self.far - depth_buf * (self.far - self.near) + 1e-10)

        # 获取相机在渲染时的实际位姿（用于后续坐标变换）
        self._last_cam_pos = np.array(cam_pos, dtype=np.float64)
        self._last_look_at = np.array(look_at, dtype=np.float64)

        return rgb, depth

    def get_bgr(self, cam_pos, look_at, up=None):
        """同 capture()，但返回 BGR 格式（OpenCV 常用）。"""
        rgb, depth = self.capture(cam_pos, look_at, up)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr, depth

    def get_cam_pose(self):
        """返回最近一次 capture 时的相机位置和朝向point。"""
        return self._last_cam_pos.copy(), self._last_look_at.copy()


if __name__ == "__main__":
    # 简单测试：需要先运行 scene.py 构建场景
    from scene import build_scene
    import time

    robot, objs, table, z = build_scene(gui=True)
    time.sleep(0.5)

    cam = SimCamera(640, 480, fov=70)
    rgb, depth = cam.capture(
        cam_pos=[0.5, -0.3, 0.8],
        look_at=[0.6, 0.0, z])

    print(f"RGB shape: {rgb.shape}, dtype={rgb.dtype}")
    print(f"Depth shape: {depth.shape}, dtype={depth.dtype}")
    print(f"Depth range: [{depth[depth > 0].min():.3f}, {depth.max():.3f}] m")
    print(f"内参矩阵 K:\n{cam.K}")

    # 显示
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # 归一化深度用于显示
    depth_vis = depth.copy()
    valid = depth_vis > 0
    if valid.any():
        depth_vis[valid] = (depth_vis[valid] - depth_vis[valid].min()) / \
                           (depth_vis[valid].max() - depth_vis[valid].min() + 1e-8)
    cv2.imshow("RGB", bgr)
    cv2.imshow("Depth", depth_vis)
    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    p.disconnect()
