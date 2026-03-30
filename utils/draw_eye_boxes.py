import cv2

def draw_eye_boxes(image, detection_result, padding=10):
    if not detection_result or not detection_result.face_landmarks:
        return image

    h, w, _ = image.shape
    annotated_image = image.copy()

    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    for face_landmarks in detection_result.face_landmarks:
        for eye_indeces in [LEFT_EYE_INDICES, RIGHT_EYE_INDICES]:
            x_coords = [int(face_landmarks[i].x * w) for i in eye_indeces]
            y_coords = [int(face_landmarks[i].y * h) for i in eye_indeces]

            min_x, max_x = min(x_coords) - padding, max(x_coords) + padding
            min_y, max_y = min(y_coords) - padding, max(y_coords) + padding

            cv2.rectangle(annotated_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            
            label = "l_eye" if eye_indeces == LEFT_EYE_INDICES else "r_eye"
            cv2.putText(annotated_image, label, (min_x, min_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return annotated_image