# 🚗 Innovation in Vehicle Tracking: YOLOv8 + Deep Learning for Automatic Number Plate Recognition (ANPR)

This project presents a cutting-edge Automatic Number Plate Recognition (ANPR) system designed for real-time vehicle tracking. It integrates YOLOv8, YOLOv5, EasyOCR, DeepSORT, and a custom License Plate Detector into an ensemble-based architecture that enhances detection accuracy and tracking reliability. Built with a Flask-based web interface, the system processes both live and pre-recorded video feeds to detect, track, and recognize license plates in real-world traffic conditions.

---

## 🎯 Objective

To improve the reliability and accuracy of vehicle tracking and number plate recognition by integrating multiple deep learning models and OCR technologies into a unified, real-time system.

---

## 🛠️ Technologies Used

- **YOLOv8 & YOLOv5** – For real-time detection of vehicles and license plates.
- **EasyOCR** – For reading alphanumeric license plates via optical character recognition.
- **DeepSORT** – For identity-preserving multi-object tracking across video frames.
- **Custom License Plate Detector** – Enhances plate localization and complements YOLO models.
- **Flask** – For the web-based frontend and video stream handling.

---

## 🧪 Methodology

### 1. Ensemble Model Architecture
- Combines predictions from YOLOv8, YOLOv5, and a custom license plate model.
- DeepSORT links frame-to-frame detections with unique IDs for continuous tracking.
- EasyOCR extracts plate text from detected regions, linking OCR to tracking output.

### 2. Dataset
- **Roboflow Number Plate ANPR Dataset**
- **Total Images**: 8,999
  - Training: 7,186
  - Validation: 899
  - Test: 914
- **Preprocessing**: Standardized to 640×640 pixels. No data augmentation applied to maintain label accuracy.

### 3. System Workflow

1. **Input**: Live stream or uploaded video
2. **Preprocessing**: Scaling and frame conversion
3. **Detection**: YOLOv8 and YOLOv5 detect vehicles and plates
4. **Tracking**: DeepSORT assigns consistent IDs across frames
5. **OCR**: EasyOCR extracts license plate text
6. **Output**: Annotated real-time video with vehicle IDs and plate numbers

---

## 📈 Results

### Performance Metrics

| Metric            | Description                                                     |
|-------------------|-----------------------------------------------------------------|
| **Precision**     | Accuracy of vehicle and license plate detection                 |
| **Recall**        | Completeness in detecting all relevant vehicles and plates      |
| **Tracking Accuracy** | Stability of DeepSORT in preserving unique identities over time |

The ensemble architecture significantly reduces false positives and improves OCR-linking accuracy, especially in cluttered or low-light scenes.

---

## 🌱 Future Work

- **Real-Time Enhancements**: Optimize latency and tracking in dense traffic.
- **Multilingual OCR**: Support for regional and international license plate formats.
- **Edge Deployment**: Lightweight versions for Jetson Nano, Raspberry Pi, etc.
- **Security Improvements**: Add encryption for video and plate data to ensure privacy.

---

## 🚀 How to Run

### Requirements

- Python 3.x
- YOLOv5 + YOLOv8
- EasyOCR
- DeepSORT
- Flask

### Install Dependencies

```bash
pip install -r requirements.txt

python app.py

```

## 📺 Output

-**Bounding boxes around vehicles**

-**Tracked vehicle IDs across video frames**

-**Extracted license plate text overlaid on the video**

-**Web interface for input selection and output display**
