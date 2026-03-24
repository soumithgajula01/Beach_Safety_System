# 🌊 AI-Powered Beach Safety System  
### (Drowning Detection using Deep Learning - YOLOv8)

An AI-based beach safety system designed to detect drowning persons in real-time using **YOLOv8 (a CNN-based object detection model)**.  
This project focuses on improving beach safety by assisting lifeguards with automated monitoring and alert systems.

---

## 🧠 Project Overview

Drowning is one of the leading causes of accidental death worldwide, with over 300,000 deaths annually. Traditional monitoring systems rely heavily on lifeguards, which becomes ineffective due to:

- Large crowd sizes  
- Limited human attention  
- Night-time visibility issues  

This project proposes an **AI-based automated surveillance system** that detects drowning behavior using computer vision techniques.

---

## 🤖 Model & Neural Network Used

- Model: **YOLOv8x (Ultralytics)**
- Type: **Convolutional Neural Network (CNN)**
- Framework: **PyTorch**

### 🔍 How the Model Works

- Input image/video is processed by CNN layers
- Features like shape, motion, and posture are extracted
- Model predicts:
  - Bounding boxes
  - Class labels (`drowning_person`, `person`)
  - Confidence scores

✔ Real-time detection  
✔ High accuracy  
✔ Suitable for safety-critical applications  

---

## ⚠️ Important Clarification

> 🔴 **Note:**  
> This implementation focuses on **YOLO-based drowning detection only**.

The following components are part of the **research concept** but **NOT implemented in this project**:

- ❌ AELIS system (Aquatic Environment Lifesaving Integrated System)  
- ❌ Thermal camera integration  

👉 These are included as **proposed enhancements based on research study** :contentReference[oaicite:0]{index=0}  

---

## 📚 Conceptual Extensions (From Research)

### 🌙 Thermal Cameras (Concept Only)

Thermal cameras can:
- Detect human body heat
- Work in night / fog / low visibility

👉 In research, system switches:
- Day → RGB camera  
- Night → Thermal camera  

✔ Enables 24/7 monitoring  
❌ Not implemented in this project  

---

### 🧬 AELIS Technology (Concept Only)

AELIS is a conceptual integrated system that:

- Combines cameras, AI models, and alert systems  
- Detects boundary crossing and drowning  
- Sends alerts (SMS, alarms) to lifeguards  
- Uses GIS for multi-location tracking  

👉 In this project:
- Only the **AI detection (YOLO)** part is implemented  
- AELIS is discussed as a **future integration framework**

---

## ⚙️ Methodology (Implemented)

1. Input images/videos are provided to the system  
2. YOLO model processes frames  
3. Detects:
   - Drowning person
   - Normal person  
4. Bounding boxes are generated  
5. Detection results are saved/output  

---

## 🚀 Features

- Deep learning-based detection (CNN)
- Real-time object detection using YOLOv8
- Detects drowning behavior from images/videos
- High-performance training (target ≥ 90% accuracy)
- Advanced validation metrics:
  - mAP@0.5
  - Precision
  - Recall
  - F1 Score
  - FPR / FNR
  - FPS

---

## 📊 Performance

- Precision: ~0.84  
- Recall: ~0.85  
- F1 Score: ~0.85  
- False Positive Rate: 0.027  
- False Negative Rate: 0.014  
- FPS: ~11.9  

---

## 📁 Project Structure

```
project/
│
├── train/ (not included)
├── valid/ (not included)
├── test/  (not included)
│
├── data.yaml
├── clean_label.py
├── train_model.py
├── validate_model.py
├── validation_final.py
├── predict_model.py
│
└── runs/
```

---

## ⚠️ Dataset Note

Dataset is not included due to size limitations.

Format required:

```
<class_id> <x_center> <y_center> <width> <height>
```

---

## 💡 Future Scope

- Integration with AELIS system  
- Thermal camera-based night detection  
- Real-time alert system (SMS / alarm)  
- CCTV live monitoring  
- Edge deployment (Jetson / Raspberry Pi)  

---

## 🎯 Applications

- Beach safety monitoring  
- Swimming pools  
- Water parks  
- Rescue systems  

---

## 👨‍💻 Author

Soumith Gajula  
B.Tech CSE, NIIT University  

---

## 🎤 Interview Tip (VERY IMPORTANT)

If asked:

👉 *“Did you implement AELIS and thermal cameras?”*

You should say:

> “No, I implemented the YOLO-based drowning detection system. AELIS and thermal cameras are part of my research-based system design and future work.”

---

## 📜 License

MIT License
