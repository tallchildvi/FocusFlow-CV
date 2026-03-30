from utils.get_ear_index import get_ear
def extract_features(landmarks, matrix):
    # Відносні координати зіниць
    def get_rel_eye(iris_idx, inner_idx, outer_idx):
        iris, inner, outer = landmarks[iris_idx], landmarks[inner_idx], landmarks[outer_idx]
        return [(iris.x - inner.x) / (outer.x - inner.x), 
                (iris.y - inner.y) / (outer.y - inner.y)]

    eye_rel = get_rel_eye(468, 133, 33) + get_rel_eye(473, 362, 263)
    ear_l = get_ear(landmarks, [33, 160, 158, 133, 153, 144])
    ear_r = get_ear(landmarks, [362, 385, 387, 263, 373, 380])
    head_pose = matrix.flatten().tolist()
    
    return eye_rel + [ear_l, ear_r] + head_pose