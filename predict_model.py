# 🎯 prediction.py
# Run inference using YOLOv8 trained model


from ultralytics import YOLO
import torch

def main():
    # Detect GPU or CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🔍 Using device: {'GPU' if device == 0 else 'CPU'}")

    # Path to your best model
    model_path = "D:\RESEARCH\soumith\project\runs\detect\val4"

    # Load model
    model = YOLO(model_path)

    # Input source (change this as needed)
    source = "D:\RESEARCH\soumith\project\test"

    # Run prediction
    results = model.predict(
        source=source,
        imgsz=960,
        conf=0.25,
        device=device,
        save=True,           # Saves output to runs/
        save_txt=False,      # Set True if you need YOLO txt output
        save_conf=True,
        vid_stride=1,
        show=False           # Make True to see images on screen
    )

    print("\n🎉 Prediction Completed!")
    print("📁 Output saved in: runs/detect/predict/")

if __name__ == "__main__":
    main()
