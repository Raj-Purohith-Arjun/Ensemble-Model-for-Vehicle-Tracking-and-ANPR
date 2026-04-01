# 🚗 Innovation in Vehicle Tracking: YOLOv8 + Deep Learning for Automatic Number Plate Recognition (ANPR)

This project presents a cutting-edge Automatic Number Plate Recognition (ANPR) system designed for real-time vehicle tracking. It integrates YOLOv8, YOLOv5, EasyOCR, DeepSORT, and a custom License Plate Detector into an ensemble-based architecture that enhances detection accuracy and tracking reliability. Built with a Flask-based web interface, the system processes both live and pre-recorded video feeds to detect, track, and recognize license plates in real-world traffic conditions.

LINK: [https://link.springer.com/chapter/10.1007/978-3-031-82383-1_6]

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

## 📁 Project Structure

```
.
├── app.py                      # Flask web application entry point
├── requirements.txt            # Python dependencies
├── config/
│   └── config.yaml             # YOLOv5 training dataset configuration
├── data/                       # CSV output files (detection results)
├── docs/
│   └── research_paper.pdf      # Published research paper
├── models/
│   ├── README.md               # Instructions for placing model weights
│   ├── best.pt                 # YOLOv5 license plate detector
│   ├── yolov8s.pt              # YOLOv8s COCO vehicle detector
│   └── yolov8n.pt              # YOLOv8n lightweight variant
├── notebooks/
│   └── detector.ipynb          # Exploratory Jupyter notebook
├── src/
│   ├── __init__.py
│   ├── utils.py                # Shared utilities (OCR, CSV, bbox helpers)
│   ├── main.py                 # Core detection + tracking pipeline
│   ├── add_missing_data.py     # Bounding box interpolation for missing frames
│   ├── visualize.py            # Annotated video generation
│   └── detect_plates.py        # YOLOv5-based plate detection script
├── templates/
│   └── index.html              # Flask web UI template
├── outputs/                    # Processed video output (runtime, gitignored)
└── uploads/                    # Uploaded videos (runtime, gitignored)
```

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
```

### Run the Web Application

```bash
python app.py
```

Then open your browser at `http://localhost:5000` and upload a video to process.

### Run Pipeline Scripts Directly

```bash
# Step 1 – Detect vehicles and license plates, write results to data/test.csv
python src/main.py

# Step 2 – Interpolate missing bounding boxes across frames
python src/add_missing_data.py

# Step 3 – Generate annotated output video (saved to outputs/out.mp4)
python src/visualize.py

# Alternative – YOLOv5-based detection with target plate search
python src/detect_plates.py <TARGET_PLATE>
```

---

## 📺 Output

- **Bounding boxes around vehicles**
- **Tracked vehicle IDs across video frames**
- **Extracted license plate text overlaid on the video**
- **Web interface for input selection and output display**
