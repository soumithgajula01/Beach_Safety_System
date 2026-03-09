# 🧪 advanced_validate.py
# Validates YOLO model + computes false detection
# metrics and inference speed.


from ultralytics import YOLO
import torch
import numpy as np

def main():

    # Device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running on: {'GPU' if device == 0 else 'CPU'}")

    # Load trained model
    model = YOLO("/home/user/Desktop/Soumith/project/runs/detect/beach_safety_90plus_final/weights/best.pt")

    # Run validation
    results = model.val(
        data="/home/user/Desktop/Soumith/project/data.yaml",
        imgsz=960,
        batch=2,
        device=device,
        conf=0.25,
        iou=0.6,
        split="val",
        plots=True,
        save_json=False,
        workers=0
    )

    # Extract default metrics
    mAP50 = results.box.map50
    precision = results.box.mp
    recall = results.box.mr
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    # ========================
    # EXTRACT CONFUSION MATRIX
    # ========================
    cm = results.confusion_matrix.matrix.astype(int)

    # YOLO confusion matrix layout:
    # [ [TP, FN],
    #   [FP, TN] ]
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

    # Compute false detection metrics
    FPR = FP / (FP + TN + 1e-6)
    FNR = FN / (FN + TP + 1e-6)

    # ======================
    #   INFERENCE SPEED
    # ======================
    inf_speed = results.speed['inference']   # milliseconds per image
    FPS = 1000 / inf_speed

    # Print summary
    print("\n========== VALIDATION SUMMARY ==========")
    print(f"mAP@0.5:      {mAP50:.3f}")
    print(f"Precision:    {precision:.3f}")
    print(f"Recall:       {recall:.3f}")
    print(f"F1 Score:     {f1:.3f}")
    print("----------------------------------------")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"False Positive Rate (FPR): {FPR:.3f}")
    print(f"False Negative Rate (FNR): {FNR:.3f}")
    print("----------------------------------------")
    print(f"Inference Time: {inf_speed:.2f} ms/image")
    print(f"FPS (speed):    {FPS:.2f} frames/sec")
    print("========================================")

    # Save summary to a text file
    with open("/home/user/Desktop/Soumith/project/validation_summaryfinal.txt", "w") as f:
        f.write("===== VALIDATION SUMMARY =====\n")
        f.write(f"mAP@0.5: {mAP50:.3f}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall: {recall:.3f}\n")
        f.write(f"F1 Score: {f1:.3f}\n\n")
        f.write(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}\n")
        f.write(f"False Positive Rate: {FPR:.3f}\n")
        f.write(f"False Negative Rate: {FNR:.3f}\n\n")
        f.write(f"Inference Time: {inf_speed:.2f} ms per image\n")
        f.write(f"FPS (speed): {FPS:.2f} FPS\n")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
