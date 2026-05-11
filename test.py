import pybullet as p

try:
    client = p.connect(p.GUI)  # 会弹出可视化窗口
    print("✅ PyBullet 工作正常！")
    input("按 Enter 键关闭窗口...")  # 等待用户按键
    p.disconnect()
except Exception as e:
    print("❌ 错误:", e)