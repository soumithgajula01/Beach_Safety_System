# 🧪 validate_model.py
# Validate YOLOv8 model performance + accuracy


from ultralytics import YOLO
import torch

def main():
    # ✅ Device detect
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running on: {'GPU' if device == 0 else 'CPU'}")

    # ✅ Model path (update for Ubuntu)
    model_path = "/home/user/Desktop/Soumith/project/runs/detect/beach_safety_90plus_final/weights/best.pt"
    model = YOLO(model_path)

    # ✅ Run validation
    metrics = model.val(
        data="/home/user/Desktop/Soumith/project/data.yaml",
        imgsz=960,
        batch=2,
        device=device,
        split="val",
        conf=0.25,
        iou=0.6,
        save_json=True,
        plots=True,
        workers=0
    )

    # ✅ Extract metrics
    try:
        mAP50 = metrics.box.map50
        precision = metrics.box.mp
        recall = metrics.box.mr
    except AttributeError:
        mAP50 = metrics.results_dict.get("metrics/mAP50(B)", 0)
        precision = metrics.results_dict.get("metrics/precision(B)", 0)
        recall = metrics.results_dict.get("metrics/recall(B)", 0)

    # 🎯 Overall Accuracy
    overall = (mAP50 + precision + recall) / 3

    # ✅ Print results
    print("\n Validation Summary:")
    print(f"mAP@0.5:    {mAP50:.3f}")
    print(f"Precision:  {precision:.3f}")
    print(f"Recall:     {recall:.3f}")
    print(f" Overall Accuracy: {overall * 100:.2f}%")

    # ✅ Save validation summary
    summary_path = "/home/user/Desktop/Soumith/project/validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"mAP@0.5: {mAP50:.3f}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall: {recall:.3f}\n")
        f.write(f"Overall Accuracy: {overall * 100:.2f}%\n")

    print(f"\n Summary saved at: {summary_path}")


if __name__ == "__main__":
    main()
