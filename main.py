import cv2
import time
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
model_path=r"C:\Users\Andrew\Documents\projects\FocusFlow-CV\face_landmarker_v2_with_blendshapes.task"

latest_result = None
def print_result(result, output_image, timestanp_ms):
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

print("Press 'q' to quit")

while True:

    ret, frame = cap.read()

    if not ret: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    detector.detect_async(mp_image, timestamp_ms)


    if latest_result:
        # frame = draw_landmarks(frame, latest_result)
        frame = draw_eye_boxes(frame, latest_result)
        cv2.putText(frame, f"Faces: {len(latest_result.face_landmarks)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("AI Monitor", frame)

detector.close()
cap.release()
cv2.destroyAllWindows()