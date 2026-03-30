def get_head_orientation(detection_result):
    if not detection_result or not detection_result.facial_transformation_matrixes:
        return None

    matrix = detection_result.facial_transformation_matrixes[0]
    
    return matrix

def calculate_eye_deviation(face_landmarks, eye_center_idx, iris_idx):

    iris = face_landmarks[iris_idx]
    eye_center = face_landmarks[eye_center_idx]
    
    dx = iris.x - eye_center.x
    dy = iris.y - eye_center.y
    return dx, dy

