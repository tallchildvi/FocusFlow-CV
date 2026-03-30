import cv2

def draw_gaze_vectors(image, detection_result, length_scale=100):
    if not detection_result or not detection_result.face_landmarks:
        return image

    h, w, _ = image.shape
    annotated_image = image.copy()

    LEFT_EYE = [468, 133, 33]
    RIGHT_EYE = [473, 362, 263]

    for face_landmarks in detection_result.face_landmarks:
        for eye in [LEFT_EYE, RIGHT_EYE]:

            iris = face_landmarks[eye[0]]
            p_inner = face_landmarks[eye[1]]
            p_outer = face_landmarks[eye[2]]

            iris_px = (int(iris.x * w), int(iris.y * h))
            
            center_x = (p_inner.x + p_outer.x) / 2
            center_y = (p_inner.y + p_outer.y) / 2
            center_px = (int(center_x * w), int(center_y * h))

            dx = iris.x - center_x
            dy = iris.y - center_y

            end_point = (
                int(iris_px[0] + dx * length_scale * w),
                int(iris_px[1] + dy * length_scale * h)
            )

            cv2.arrowedLine(annotated_image, iris_px, end_point, (255, 0, 0), 2, tipLength=0.3)
            
            cv2.circle(annotated_image, center_px, 2, (0, 0, 255), -1)

    return annotated_image