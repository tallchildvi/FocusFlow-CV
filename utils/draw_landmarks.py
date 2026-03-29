import numpy as np
import cv2

def draw_landmarks(image, detection_result):
    if not detection_result or not detection_result.face_landmarks:
        return image

    h, w, _ = image.shape
    annotated_image = image.copy()

    for face_landmarks in detection_result.face_landmarks:
        for landmark in face_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(annotated_image, (cx, cy), 1, (0, 255, 0), -1)
            
    return annotated_image