# 🚀 train_model_90plus_final.py
# Goal: ≥ 90% mAP, Precision, Recall


from ultralytics import YOLO
import torch

# 🔥 Detect device
if torch.cuda.is_available():
    device = 0
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("⚙️ Running on CPU mode (slow)")

# 🔥 Load most accurate YOLO model
model = YOLO("yolov8x.pt")

# 🏋️‍♂️ Train
model.train(
    data="/home/user/Desktop/Soumith/project/data.yaml",
    epochs=300,
    imgsz=960,
    batch=2,
    device=device,
    workers=0,
    name="beach_safety_90plus_final",
    cache=False,

    # Optimizer
    optimizer="AdamW",
    lr0=0.0006,
    lrf=0.005,
    momentum=0.937,
    weight_decay=0.0004,
    warmup_epochs=5,
    warmup_bias_lr=0.1,

    # Augmentations
    augment=True,
    degrees=15,
    translate=0.08,
    scale=0.9,
    shear=0.15,
    mosaic=1.0,
    mixup=0.3,
    flipud=0.5,
    fliplr=0.5,
    perspective=0.0005,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.45,
    close_mosaic=20,

    # Training boost
    label_smoothing=0.05,
    dropout=0.05,
    freeze=5,
    plots=True,
)

print("\n🎯 Training completed successfully!")
print("📦 Best model saved at: runs/detect/beach_safety_90plus_final/weights/best.pt")
