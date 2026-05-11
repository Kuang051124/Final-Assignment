"""
自动运动测试 —— 机械臂自动画方形轨迹，你在 PyBullet GUI 里观察。
不依赖任何按键，直接运行就能看结果。
"""
import sys
sys.path.insert(0, '.')
from scene import build_scene
from grasp_plan import solve_ik, move_to_joints, get_end_effector_pose, NUM_JOINTS
import pybullet as p
import numpy as np
import time

print("=" * 50)
print("  机械臂自动正方形轨迹测试")
print("=" * 50)

# 构建场景（GUI 模式——你会看到 3D 窗口）
robot, obj_infos, table, z = build_scene(gui=True)

# 等 GUI 初始化
print("等待 GUI 就绪...")
for _ in range(500):
    p.stepSimulation()

# 设置机器人到弯曲的 Home 位姿
home_joints = np.array([0, np.pi/4, 0, -np.pi/2, 0, np.pi/4, 0])
for i, angle in enumerate(home_joints):
    p.resetJointState(robot, i, angle)
for _ in range(300):
    p.stepSimulation()

ee, _ = get_end_effector_pose(robot)
print(f"\nHome 位姿 末端: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m")
print(f"Home 关节角: {np.round(np.degrees(home_joints), 1)} deg\n")

# 定义正方形的五个顶点（最后一个=第一个，形成闭环）
targets = [
    ("点0-前方近",  np.array([0.55, -0.12, 0.12])),
    ("点1-右方",    np.array([0.55,  0.12, 0.12])),
    ("点2-远方右",  np.array([0.70,  0.12, 0.12])),
    ("点3-远方左",  np.array([0.70, -0.12, 0.12])),
    ("点4-回到起点", np.array([0.55, -0.12, 0.12])),
]

for label, target in targets:
    print(f"-> {label} : ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")

    # 逆运动学求解
    joints = solve_ik(robot, target)
    if joints is None:
        print(f"   [FAIL] IK 无解!")
        continue

    print(f"   关节角(deg): {np.round(np.degrees(joints), 1)}")

    # 执行移动（300步，让你看清运动过程）
    move_to_joints(robot, joints, steps=300)

    # 检查精度
    ee_actual, _ = get_end_effector_pose(robot)
    err = np.linalg.norm(ee_actual - target) * 1000
    print(f"   末端到达: ({ee_actual[0]:.3f}, {ee_actual[1]:.3f}, {ee_actual[2]:.3f})")
    print(f"   误差: {err:.1f} mm {'OK' if err < 25 else 'BIG'}")
    print()

    time.sleep(0.5)  # 停半秒让你看清

print("=" * 50)
print("  测试完成！请在 PyBullet 窗口中确认")
print("  机器人末端沿正方形轨迹运动了 5 个点。")
print("  关闭 PyBullet 窗口退出。")
print("=" * 50)

# 保持GUI
while p.isConnected():
    p.stepSimulation()
