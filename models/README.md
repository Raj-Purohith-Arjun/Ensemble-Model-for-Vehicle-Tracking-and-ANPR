# Model Weights

Place the following pre-trained model weight files in this directory before running the system:

| File | Description | Source |
|---|---|---|
| `license_plate_detector.pt` | Custom YOLOv5/YOLOv8 license plate detector | Trained on Roboflow ANPR dataset |
| `yolov8s.pt` | YOLOv8s general object detection (COCO) | `ultralytics` auto-download |
| `yolov8n.pt` | YOLOv8n lightweight variant | `ultralytics` auto-download |
| `best.pt` | YOLOv5 license plate detector | Trained on Roboflow ANPR dataset |

> **Note:** The `.pt` files are excluded from version control via `.gitignore` due to their size.  
> Download them from the project release page or train your own using `config/config.yaml`.
