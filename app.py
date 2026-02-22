"""
Flask web application entry point for the Vehicle Tracking & ANPR system.

Usage:
    pip install -r requirements.txt
    python app.py
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from ultralytics import YOLO
from src.utils import get_car, read_license_plate, write_csv

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))  # Set SECRET_KEY env var in production

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB upload limit

# Load models at startup (only once)
coco_model = None
license_plate_detector = None


def load_models():
    """Load YOLO models. Call once before processing."""
    global coco_model, license_plate_detector
    if coco_model is None:
        coco_model = YOLO('yolov8s.pt')
    if license_plate_detector is None:
        model_path = os.path.join('models', 'license_plate_detector.pt')
        if os.path.exists(model_path):
            license_plate_detector = YOLO(model_path)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video(video_path, output_csv, output_video_path):
    """
    Run vehicle detection, tracking and license plate recognition on a video.

    Args:
        video_path (str): Path to the input video file.
        output_csv (str): Path to save the detection results CSV.
        output_video_path (str): Path to save the annotated output video.
    """
    load_models()

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    vehicles = [2, 3, 5, 7]  # COCO class IDs: car, motorcycle, bus, truck
    results = {}
    frame_nmr = -1
    ret = True

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break

        results[frame_nmr] = {}

        # Detect and track vehicles with YOLOv8
        detections = coco_model.track(frame, persist=True)[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection
            if int(class_id) in vehicles:
                if license_plate_detector is not None:
                    # Detect license plates within vehicle ROI
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    license_plates = license_plate_detector(roi)[0]
                    for lp in license_plates.boxes.data.tolist():
                        px1, py1, px2, py2, lp_score, _ = lp
                        plate = roi[int(py1):int(py2), int(px1):int(px2)]
                        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                        _, plate_thresh = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
                        lp_text, lp_score_ocr = read_license_plate(plate_thresh)
                        if lp_text is not None:
                            results[frame_nmr][track_id] = {
                                'car': {'bbox': [x1, y1, x2, y2]},
                                'license_plate': {
                                    'bbox': [px1, py1, px2, py2],
                                    'text': lp_text,
                                    'bbox_score': lp_score,
                                    'text_score': lp_score_ocr
                                }
                            }

                # Annotate frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    write_csv(results, output_csv)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'video' not in request.files:
        flash('No video file provided.', 'error')
        return redirect(url_for('index'))

    video_file = request.files['video']
    if video_file.filename == '' or not allowed_file(video_file.filename):
        flash('Invalid file. Please upload a video (mp4, avi, mov, mkv).', 'error')
        return redirect(url_for('index'))

    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    output_csv = os.path.join(DATA_FOLDER, 'results.csv')
    output_video = os.path.join(OUTPUT_FOLDER, 'out.mp4')

    try:
        process_video(video_path, output_csv, output_video)
        flash('Video processed successfully!', 'success')
    except Exception as e:
        flash(f'Processing error: {e}', 'error')
        return redirect(url_for('index'))

    return render_template('index.html', output_video=url_for('output_file', filename='out.mp4'))


@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
