# FocusFlow-CV: Real-Time Human Focus Detection

## Project Objective

FocusFlow-CV is an real-time concentration monitoring system. This project implements a comprehensive **3D analysis of head orientation and ocular movements** to classify user states as **"Focus"** or **"Away."**

The system is designed to minimize distractions during deep work or study sessions using a standard webcam and a lightweight SVM model, making it highly efficient for standard laptops without dedicated GPUs.

---

![Python](https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-%235C3EE8.svg?style=flat&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-%230097A7.svg?style=flat&logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)

---

## Technical Approach: 3D Feature Engineering

The engine extracts **23 high-dimensional features** using the MediaPipe Face Landmarker API.

### Feature Vector Composition

1. **Head Pose Matrix (16 features):** A full 4×4 transformation matrix capturing Pitch, Yaw, and Roll with high spatial resolution.
2. **Eye Aspect Ratio (EAR) (2 features):** Numerical coefficient representing eye openness to detect blinking and fatigue patterns.
3. **Relative Pupil Coordinates (4 features):** Pupil positioning relative to eye corners to track gaze directionality.

```
Feature Vector = [ Head_Matrix(1..16), EAR_L, EAR_R, Pupil_X_L, Pupil_Y_L, Pupil_X_R, Pupil_Y_R ]
```

---

## Dataset: The Challenge of Temporal Correlation

The training data was manually collected **(491 samples)** under authentic workspace conditions.

### Combating "Falsely Perfect" Accuracy

During development, a classic video-data challenge emerged: **Temporal Correlation**. Since consecutive video frames are nearly identical, the model initially showed a "fake" **98.5% accuracy** by simply memorizing adjacent frames.

To obtain a robust performance metric, **Dataset Decimation** was applied:

- Discarded **80% of the raw data**, retaining only every 5th frame to break the temporal link between samples.
- This forced the SVM to learn **generalized focus patterns** rather than memorizing a specific time sequence.

---

## Results Summary (SVM RBF Kernel)

| Dataset Strategy     | Train Accuracy | Test Accuracy | CV Score (Mean) |
|----------------------|----------------|---------------|-----------------|
| Full Dataset         | 97.03%         | 94.06%        | 98.52%          |
| Decimated (1/5)      | 95.00%         | 85.71%        | 90.00%          |

> **Note:** While the "on-paper" accuracy decreased after decimation, the model's stability in real-world scenarios improved significantly. The parameters **C=10.0** and **γ=0.1** proved to be the most optimal across both testing phases.

---

## Project Structure

```
FocusFlow-CV/
├── models/                 # Pre-trained SVM weights and MediaPipe .task files
├── utils/                  # Utilities scripts
├── data/                   # Data collection and CSV processing scripts
├── main.py                 # Live inference script with Confidence Buffer logic
└── requirements.txt        # Project dependencies
```

---

## Future Work

1. **Personalized Calibration:** Implementing a "calibration phase" at startup to adapt the model to a user's unique facial anatomy.
2. **Dataset Expansion:** Diversifying data with varying lighting conditions and eyewear.


---

## Installation & Usage

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/FocusFlow-CV.git
cd FocusFlow-CV
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Launch monitoring:**
```bash
python main.py
```

---

*Developed by Andrii (tallchildvi), 2nd year Software Engineering student at Taras Shevchenko National University of Kyiv (KNU).*