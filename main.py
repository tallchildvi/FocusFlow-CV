import cv2
import time
import os
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.core.base_options import (
    BaseOptions
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions
)
from utils.draw_landmarks import draw_landmarks
from utils.draw_eye_boxes import draw_eye_boxes
from utils.draw_gaze_vectors import draw_gaze_vectors
from utils.get_features import extract_features
from data.data_collector import FocusDataCollector
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
model_path = str(BASE_DIR / "models" / "face_landmarker_v2_with_blendshapes.task")
svm_model_path = str(BASE_DIR / "models" / "focus_svm_model.pkl")
scaler_path = str(BASE_DIR / "models" / "scaler.pkl")
try:
    clf = joblib.load(svm_model_path)
except:
    print(f'Cannot load model from file: {svm_model_path}')
    exit()

try:
    scaler = joblib.load(scaler_path)
except:
    print(f'Cannot load scaler from: {scaler_path}')
    exit()

latest_result = None
last_timestamp_ms = 0

def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


base_options = BaseOptions(model_asset_path=model_path)
options = FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       running_mode=VisionTaskRunningMode.LIVE_STREAM,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1,
                                       result_callback=print_result)
detector = FaceLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)
away_counter = 0 
AWAY_THRESHOLD = 50
print("Press 'q' to quit")

while True:

    ret, frame = cap.read()

    if not ret: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms
    
    detector.detect_async(mp_image, timestamp_ms)

    current_result = latest_result 
    color = (255, 255, 255)

    if current_result and current_result.face_landmarks and current_result.facial_transformation_matrixes:
        frame = draw_landmarks(frame, current_result)
        
        landmarks = current_result.face_landmarks[0]
        matrix = current_result.facial_transformation_matrixes[0]

        features = extract_features(landmarks, matrix) 

        feature_names = [str(i) for i in range(len(features))]
        features_df = pd.DataFrame([features], columns=feature_names)
        features_scaled = scaler.transform(features_df)
        prediction = clf.predict(features_scaled)[0]

        if prediction == 0: 
            away_counter += 1
        else: 
            away_counter = max(0, away_counter - 3) 

        if away_counter > AWAY_THRESHOLD:
            status_text = "AWAY"
            color = (0, 0, 255)
        else:
            status_text = "FOCUSING"
            color = (0, 255, 0)
        
        cv2.putText(frame, f"STATUS: {status_text}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        away_counter = min(AWAY_THRESHOLD + 1, away_counter + 1)

    cv2.imshow("AI Monitor", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()