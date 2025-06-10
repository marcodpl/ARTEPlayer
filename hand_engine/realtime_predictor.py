import cv2
import mediapipe as mp
import numpy as np
import torch
import pickle
from collections import deque
from train_model import GestureLSTM  # Import your model class
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# === Load model and label encoder ===
with open("data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
model = GestureLSTM(output_size=len(label_encoder.classes_))
model.load_state_dict(torch.load("data/gesture_lstm.pt"))
model.eval()

# === Mediapipe hand tracking ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# === Parameters ===
SEQ_LEN = 45
INPUT_SIZE = 42
MIN_MOV_MAG = 1.0
sequence = deque(maxlen=SEQ_LEN)
current_prediction = "Waiting..."


# === Helper functions ===
def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    scale = np.linalg.norm(np.array(wrist) - np.array(landmarks[9])) + 1e-6
    return [((x - wrist[0]) / scale, (y - wrist[1]) / scale) for (x, y) in landmarks]


def flatten_landmarks(landmarks):
    return [coord for point in landmarks for coord in point]


def motion_magnitude(seq):
    diffs = np.diff(seq, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))


# === Webcam loop ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            norm = normalize_landmarks(landmarks)
            flattened = flatten_landmarks(norm)
            sequence.append(flattened)

        if len(sequence) == SEQ_LEN:
            mag = motion_magnitude(np.array(sequence))
            if mag > MIN_MOV_MAG:  # sort out minimal movement as idle
                input_tensor = torch.tensor([sequence], dtype=torch.float32)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_index = torch.argmax(output, dim=1).item()
                    current_prediction = l abel_encoder.inverse_transform([pred_index])[0]
            else:
                current_prediction = "idle"
    else:
        current_prediction = "idle"

    # === Display ===
    cv2.putText(image, f"Prediction: {current_prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Live Gesture Recognition", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
