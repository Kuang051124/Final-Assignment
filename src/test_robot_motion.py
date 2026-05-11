"""
机器人运动检验脚本 —— 键盘实时控制机械臂末端，验证运动是否正常。
"""
import sys
sys.path.insert(0, '.')
import pybullet as p
import pybullet_data
import numpy as np
import cv2
from scene import build_scene
from grasp_plan import (
    solve_ik, move_to_joints,
    get_end_effector_pose, END_EFFECTOR_INDEX, NUM_JOINTS,
)


def get_joint_angles(robot_id):
    """获取当前关节角（度）。"""
    return np.degrees([p.getJointState(robot_id, i)[0]
                        for i in range(NUM_JOINTS)])


def draw_status_panel(robot_id, ee_pos, step_size, joint_mode):
    """绘制状态信息面板（640x200 的纯文字画面）。"""
    h, w = 300, 640
    panel = np.zeros((h, w, 3), dtype=np.uint8)

    angles = get_joint_angles(robot_id)
    y = 25
    cv2.putText(panel, f"End-Effector: X={ee_pos[0]:.4f} Y={ee_pos[1]:.4f} Z={ee_pos[2]:.4f} m",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    y += 28
    if joint_mode:
        cv2.putText(panel, f">> JOINT MODE (current joint: {joint_mode}) <<",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    else:
        cv2.putText(panel, f">> END-EFFECTOR MODE  step={step_size*1000:.0f}mm <<",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    y += 30
    for i in range(NUM_JOINTS):
        color = (0, 255, 255) if (joint_mode == i) else (200, 200, 200)
        cv2.putText(panel, f"Joint {i}: {angles[i]:+7.1f} deg",
                    (10 + (i % 4) * 160, y + (i // 4) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    y += 60
    lines = [
        "Keys:",
        "  W/S       - EE +/- X    (前后)",
        "  A/D       - EE +/- Y    (左右)",
        "  Q/E       - EE +/- Z    (上下)",
        "  [/]       - Step +/-    (步长)",
        "  0-6       - Select joint for joint control",
        "  UP/DOWN   - Move selected joint +/-",
        "  R         - Reset robot to home",
        "  ESC       - Quit",
    ]
    for line in lines:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        y += 17

    return panel


def main():
    print("=" * 50)
    print("  机械臂键盘控制测试")
    print("=" * 50)

    # 构建场景
    robot, obj_infos, table, table_z = build_scene(gui=True)
    for _ in range(50):
        p.stepSimulation()

    # 初始状态
    home_joints = [0, np.pi/4, 0, -np.pi/2, 0, np.pi/4, 0]
    for i, angle in enumerate(home_joints):
        p.resetJointState(robot, i, angle)
    for _ in range(100):
        p.stepSimulation()

    ee_pos, _ = get_end_effector_pose(robot)
    print(f"末端初始位置: {np.round(ee_pos, 4)} m")
    print(f"关节初始角度: {np.round(get_joint_angles(robot), 1)} deg")

    step_size = 0.02       # 末端步长（米），[ ] 调整
    joint_mode = None      # None=末端模式, 0-6=关节模式
    ee_target = ee_pos.copy()

    print("\n操作说明见弹出窗口。按 ESC 退出。")

    while p.isConnected():
        # 获取当前末端位置
        ee_pos, _ = get_end_effector_pose(robot)

        # 绘制状态面板
        panel = draw_status_panel(robot, ee_pos, step_size, joint_mode)
        cv2.imshow("Robot Control Panel", panel)

        key = cv2.waitKey(50) & 0xFF

        if key == 27:  # ESC
            break

        # ---- 模式切换 ----
        if ord('0') <= key <= ord('6'):
            j = key - ord('0')
            joint_mode = j if joint_mode != j else None
            print(f"切换到 {'关节'+str(joint_mode)+' 控制' if joint_mode is not None else '末端控制模式'}")
            continue

        # ---- 步长 ----
        if key == ord('['):
            step_size = max(0.005, step_size - 0.005)
            print(f"步长: {step_size*1000:.0f}mm")
            continue
        if key == ord(']'):
            step_size = min(0.1, step_size + 0.005)
            print(f"步长: {step_size*1000:.0f}mm")
            continue

        # ---- 归位 ----
        if key == ord('r') or key == ord('R'):
            print("归位...")
            move_to_joints(robot, np.array(home_joints), steps=150)
            ee_target = np.array(p.getLinkState(robot, END_EFFECTOR_INDEX)[0])
            print(f"末端位置: {np.round(ee_target, 4)}")
            continue

        # ---- 移动 ----
        delta = np.zeros(3)

        if joint_mode is not None:
            # 关节模式：直接控制单个关节
            current_angles = np.array([p.getJointState(robot, i)[0]
                                        for i in range(NUM_JOINTS)])
            if key == 82:       # UP arrow
                current_angles[joint_mode] += np.radians(2)
            elif key == 84:     # DOWN arrow
                current_angles[joint_mode] -= np.radians(2)
            else:
                continue

            move_to_joints(robot, current_angles, steps=50)
            ee_pos_new, _ = get_end_effector_pose(robot)
            print(f"Joint {joint_mode}: {np.degrees(current_angles[joint_mode]):+.1f}deg  "
                  f"-> EE=({ee_pos_new[0]:.3f},{ee_pos_new[1]:.3f},{ee_pos_new[2]:.3f})")
            continue

        # 末端模式：XYZ 方向移动
        if key == ord('w') or key == ord('W'):
            delta[0] += step_size  # +X
        if key == ord('s') or key == ord('S'):
            delta[0] -= step_size  # -X
        if key == ord('a') or key == ord('A'):
            delta[1] += step_size  # +Y
        if key == ord('d') or key == ord('D'):
            delta[1] -= step_size  # -Y
        if key == ord('q') or key == ord('Q'):
            delta[2] += step_size  # +Z
        if key == ord('e') or key == ord('E'):
            delta[2] -= step_size  # -Z

        if np.all(delta == 0):
            p.stepSimulation()
            continue

        ee_target += delta
        joints = solve_ik(robot, ee_target)

        if joints is None:
            ee_target -= delta  # 回退
            print(f"IK 失败! 目标={np.round(ee_target, 3)}, delta={np.round(delta, 3)}")
            continue

        move_to_joints(robot, joints, steps=80)
        ee_actual, _ = get_end_effector_pose(robot)
        err = np.linalg.norm(ee_actual - ee_target) * 1000
        print(f"Target: ({ee_target[0]:.3f},{ee_target[1]:.3f},{ee_target[2]:.3f})  "
              f"Actual: ({ee_actual[0]:.3f},{ee_actual[1]:.3f},{ee_actual[2]:.3f})  "
              f"err={err:.0f}mm")

        p.stepSimulation()

    cv2.destroyAllWindows()
    p.disconnect()
    print("退出。")


if __name__ == "__main__":
    main()
