"""
训练 YOLOv8 目标检测模型。
"""
import os
from pathlib import Path
from ultralytics import YOLO

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

data_yaml = str(ROOT / "data_yolo" / "data.yaml")

model = YOLO("yolov8n.pt")

results = model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    device="cuda",
    workers=0,
    project=str(ROOT / "runs"),
    name="detect_workpieces",
    exist_ok=True,
    verbose=True,
)
