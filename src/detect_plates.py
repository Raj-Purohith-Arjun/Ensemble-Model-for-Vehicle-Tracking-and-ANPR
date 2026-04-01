import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2
import pandas as pd
import argparse
from timeit import default_timer as timer
from csv import writer
pd.options.mode.chained_assignment = None

import easyocr
import torch
from difflib import SequenceMatcher

ALPR_RECORD_PATH = os.path.join('data', 'ALPR_Record.csv')
ALLOWED_CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def make_record(elapsed_time):
    timestamp = convert(elapsed_time)
    with open(ALPR_RECORD_PATH, 'a') as f:
        wobj = writer(f)
        wobj.writerow(["Match Found At: ", timestamp])


def get_plates_xy(frame: np.ndarray, labels: list, row: list, width: int, height: int, reader: easyocr.Reader) -> tuple:
    """Get the results from easyOCR for each frame and return them with bounding box coordinates.

    Returns:
        tuple: (ocr_result, x1, y1) where ocr_result is the list of OCR detections and
               x1, y1 are the top-left coordinates of the bounding box.
    """

    x1, y1, x2, y2 = int(row[0]*width), int(row[1]*height), int(row[2]*width), int(row[3]*height)
    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ocr_result = reader.readtext(np.asarray(plate_crop), allowlist=ALLOWED_CHARACTERS)

    return ocr_result, x1, y1


def detect_text(i: int, row: list, x1: int, y1: int, ocr_result: list, detections: list, yolo_detection_prob: float = 0.3) -> list:
    """Check the detection's probability, discard those with low prob and rewrite output from ocr_reader to detections list."""

    if row[4] >= yolo_detection_prob:
        if len(ocr_result) > 0:
            for item in ocr_result:
                detections[i][0] = item[1]
                detections[i][1] = [x1, y1]
                detections[i][2] = item[2]

    return detections


def is_adjacent(coord1: list, coord2: list) -> bool:
    """Check if [x, y] from list coord1 is similar to coord2."""

    MAX_PIXELS_DIFF = 50

    if (abs(coord1[0] - coord2[0]) <= MAX_PIXELS_DIFF) and (abs(coord1[1] - coord2[1]) <= MAX_PIXELS_DIFF):
        return True
    else:
        return False


def sort_detections(detections: list, plates_data: list) -> list:
    """Look at detections from last frame and rewrite indexes for similar coordinates."""

    for m in range(0, len(detections)):
        for n in range(0, len(plates_data)):
            if not detections[m][1] == [0, 0] and not plates_data[n][1] == [0, 0]:
                if is_adjacent(detections[m][1], plates_data[n][1]):
                    if m != n:
                        temp = detections[m]
                        detections[m] = detections[n]
                        detections[n] = temp

    return detections


def delete_old_labels(detections: list, count_empty_labels: list, plates_data: list, frames_to_reset: int = 3) -> tuple:
    """If earlier detected plate isn't spotted for the next frames_to_reset frames, delete it from plates_data."""

    for m in range(0, len(detections)):
        if detections[m][0] == 'None' and not count_empty_labels[m] == frames_to_reset:
            count_empty_labels[m] += 1
        elif count_empty_labels[m] == frames_to_reset:
            count_empty_labels[m] = 0
            plates_data[m] = ['None', [0, 0], 0]
        else:
            count_empty_labels[m] = 0

    return plates_data, count_empty_labels


def overwrite_plates_data(i: int, detections: list, plates_data: list, plate_length=None) -> list:
    """Check coordinates from detections, if there is similar record in plates_data try to overwrite it (only if probability is higher)."""

    if (detections[i][2] > plates_data[i][2] or detections[i][2] == 0):
        if plate_length:
            if len(detections[i][0]) == plate_length:
                plates_data[i][0] = detections[i][0]
                plates_data[i][2] = detections[i][2]
        else:
            plates_data[i][0] = detections[i][0]
            plates_data[i][2] = detections[i][2]
    plates_data[i][1] = detections[i][1]

    return plates_data


if __name__ == '__main__':
    # Initialise ALPR record file
    with open(ALPR_RECORD_PATH, 'w+') as f:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('target', metavar='target', type=str, help='Enter the license plate number to search for')
    args = parser.parse_args()
    target = args.target

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best.pt', force_reload=True)

    reader = easyocr.Reader(['en'])
    video_path = "./sample.mp4"
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)
    result = cv2.VideoWriter('./outputs/cv2_out.mp4',
                             cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                             10, size)

    plates_data = [['None', [0, 0], 0] for n in range(5)]
    count_empty_labels = [0] * 5

    assert cap.isOpened()
    start = timer()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        width, height = frame.shape[1], frame.shape[0]

        detections = [['None', [0, 0], 0] for n in range(5)]
        i = 0

        while i < len(labels):
            row = coordinates[i]
            ocr_result, x1, y1 = get_plates_xy(frame, labels, row, width, height, reader)
            detections = detect_text(i, row, x1, y1, ocr_result, detections, 0.5)
            i += 1
        i = 0
        detections = sort_detections(detections, plates_data)
        plates_data, count_empty_labels = delete_old_labels(detections, count_empty_labels, plates_data, 3)

        while i < len(labels):
            plates_data = overwrite_plates_data(i, detections, plates_data, 7)
            print(f"{plates_data[i][0]}")
            match_percent = SequenceMatcher(None, target, f"{plates_data[i][0]}").ratio()
            if match_percent > 0.7:
                end = timer()
                make_record(end - start)
            cv2.putText(frame, f"{plates_data[i][0]}", (plates_data[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            i += 1

        result.write(frame)

    cap.release()
    cv2.destroyAllWindows()
