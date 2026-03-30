import numpy as np
def get_ear(landmarks, p_indices):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in p_indices]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return (v1 + v2) / (2.0 * h)