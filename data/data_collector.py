import pandas as pd
import numpy as np


class FocusDataCollector:
    def __init__(self):
        self.data = []
        self.labels = []

    def get_ear(self, landmarks, p_indices):
        p = [np.array([landmarks[i].x, landmarks[i].y]) for i in p_indices]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * h)
    
    def collect(self, detection_result, label):
        if not detection_result or not detection_result.face_landmarks or \
           not detection_result.facial_transformation_matrixes:
            return False

        landmarks = detection_result.face_landmarks[0]
        matrix = detection_result.facial_transformation_matrixes[0] 

        def get_rel_eye(iris_idx, inner_idx, outer_idx):
            iris = landmarks[iris_idx]
            inner = landmarks[inner_idx]
            outer = landmarks[outer_idx]
            
            x_rel = (iris.x - inner.x) / (outer.x - inner.x)
            y_rel = (iris.y - inner.y) / (outer.y - inner.y)
            return [x_rel, y_rel]

        eye_features = get_rel_eye(468, 133, 33) + get_rel_eye(473, 362, 263)

        matrix_features = matrix.flatten().tolist()

        final_vector = eye_features + matrix_features
        
        self.data.append(final_vector)
        self.labels.append(label)
        return True

    def save_to_csv(self, filename="focus_dataset.csv"):
        df = pd.DataFrame(self.data)
        df['label'] = self.labels
        df.to_csv(filename, index=False)
        print(f"Dataset saved: {len(self.data)} samples")