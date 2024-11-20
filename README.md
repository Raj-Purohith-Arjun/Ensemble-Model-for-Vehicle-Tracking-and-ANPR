Innovation in Vehicle Tracking: Harnessing YOLOv8 and Deep Learning Tools for Automatic Number Plate Detection



This research project explores the development of an advanced Automatic Number Plate Recognition (ANPR) system that integrates several cutting-edge deep learning technologies for enhanced vehicle tracking. By using YOLOv8, YOLOv5, EasyOCR, DeepSORT, and a specially designed License Plate Detector, this project aims to improve the precision and reliability of vehicle identification, even in complex traffic conditions. The system features a robust ensemble model that combines multiple deep learning models to achieve accurate detection and tracking of vehicles in real-time.


The main goal of this research is to improve the reliability and accuracy of vehicle tracking using Automatic Number Plate Recognition (ANPR). By combining advanced models like YOLOv8, YOLOv5, EasyOCR, and DeepSORT, this system provides improved vehicle detection, number plate recognition, and tracking capabilities. The system's strength lies in data augmentation, preprocessing techniques, and advanced OCR, all of which ensure high performance even in challenging real-world environments. Future improvements include enhancing real-time tracking, multilingual OCR support, and hardware optimization.

Technologies Used:
YOLOv8 and YOLOv5: For real-time object detection, including vehicle and license plate detection.
EasyOCR: For optical character recognition (OCR) to accurately read alphanumeric license plates.
DeepSORT: For vehicle tracking across frames, assigning unique identities to vehicles.
Ensemble Model: Combines YOLOv8, YOLOv5, and License Plate Detector outputs to maximize detection accuracy.
Web Application: Provides a user-friendly interface for real-time video input or pre-recorded video processing.


Methodology:
1. Ensemble Model:
The project leverages an ensemble model that integrates outputs from YOLOv8, YOLOv5, and a specialized License Plate Detector model. The combination of these models enables the system to detect vehicles and license plates simultaneously, improving the accuracy of detection and reducing false positives. After detecting objects, the system uses the DeepSORT algorithm to track vehicles and assign unique identities, ensuring continuous monitoring across video frames.

The model further enhances accuracy by using EasyOCR for optical character recognition, extracting text from license plates. This process links the OCR results with the tracked vehicles, providing a clear relationship between the vehicle's movement and its license plate details.

2. Dataset:
The dataset used for training and testing is the ROBOFLOW Number Plate ANPR dataset, which includes 8,999 images. The dataset is split into:

Training set: 7,186 images
Validation set: 899 images
Test set: 914 images
The images are preprocessed to standardize orientation and resize them to 640x640 pixels. No augmentations were applied during preprocessing, preserving the dataset’s integrity.

3. Workflow:
Live Video or Pre-recorded Footage: The system supports both live streaming and pre-recorded video uploads.
Preprocessing: The input video frames are standardized, with dynamic scaling and cropping applied to meet the requirements of YOLO models.
Real-time Detection and Tracking: The YOLO models detect vehicles and license plates in each frame, producing bounding boxes, class labels, and confidence scores. These outputs are fed into DeepSORT for continuous vehicle tracking.
OCR Processing: The regions of interest (ROIs) detected as license plates undergo OCR processing using EasyOCR to extract the alphanumeric characters.
Final Output: The system presents real-time video output, displaying tracked vehicles with corresponding license plate details.
4. Integration of Technologies:
The system's core lies in the integration of YOLOv8, YOLOv5, and EasyOCR, which together enhance the accuracy and speed of license plate detection and vehicle tracking. The combination of these models allows the system to handle real-time traffic data and provide insights into vehicle movements with minimal latency.


​

Results:

The system is evaluated on key performance metrics such as precision, recall, and tracking accuracy. The ensemble model enhances both vehicle tracking and license plate recognition, offering a highly reliable system for real-time traffic monitoring. The integration of YOLO models and DeepSORT ensures that vehicles are tracked continuously across frames, while EasyOCR provides accurate license plate recognition.

Performance Metrics:

Precision: Measures the accuracy of vehicle and license plate detection.
Recall: Evaluates the system’s ability to detect all vehicles and plates within the video.
Tracking Accuracy: Assesses the reliability of DeepSORT in maintaining unique vehicle identities across frames.

Future Work:
Future developments will focus on:

Real-Time Tracking: Improving tracking accuracy under dynamic traffic conditions.
Multilingual OCR: Expanding the OCR capabilities to recognize plates in different languages.
Hardware Optimization: Ensuring that the system can be deployed on edge devices for real-time applications.
Security: Implementing more stringent data security measures for privacy and safety in transportation systems.
How to Run the Project:
1. Requirements:
Python 3.x
YOLOv8 and YOLOv5 models
EasyOCR
DeepSORT
Flask (for web application)
2. Install Dependencies:
bash
Copy code
pip install -r requirements.txt
3. Run the Web Application:
bash
Copy code
python app.py
Access the system via a web browser and choose to either stream live video or upload pre-recorded footage for processing.

4. Output:
The application will display the processed video with real-time vehicle tracking and license plate recognition, offering a comprehensive view of the traffic situation.

For any questions or contributions:
Raj Purohith Arjun
Email: raj2001@tamu.edu
LinkedIn: Raj Purohith Arjun
