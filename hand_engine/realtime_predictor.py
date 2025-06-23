import cv2
import mediapipe as mp
import numpy as np
import torch
import pickle
from collections import deque
from train_model import GestureLSTM
from train_model import GestureTransformer # Import your model class
import warnings
from sys import platform
from helpers.prediction_handler import PredictionHandler
from helpers.conflict_resolver import ConflictResolver
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_TYPE = "transformer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === Load model and label encoder ===
with open("data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
model = GestureLSTM(output_size=len(label_encoder.classes_)) if MODEL_TYPE == "lstm" else GestureTransformer(output_size=len(label_encoder.classes_))
model_path = "data/gesture_lstm.pt" if MODEL_TYPE == "lstm" else "data/gesture_model.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Mediapipe hand tracking ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


# === Prediction handler ===
handle_prediction = PredictionHandler()


# === Parameters ===
SEQ_LEN = 45
INPUT_SIZE = 42
MIN_MOV_MAG = 5
MIN_CONFIDENCE = 0.9
sequence = deque(maxlen=SEQ_LEN)
current_prediction = "Waiting..."
last_prediction = "Waiting..."
handler = PredictionHandler()
resolver = ConflictResolver()


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
            handedness = "right"

        if len(sequence) == SEQ_LEN:
            mag = motion_magnitude(np.array(sequence))
            if mag > MIN_MOV_MAG:  # sort out minimal movement as idle
                input_tensor = torch.tensor([sequence], dtype=torch.float32).to(device)  # define input tensor and move to GPU if available
                with torch.inference_mode():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, pred_class = torch.max(probs, dim=1)
                    if confidence >= MIN_CONFIDENCE:
                        pred_index = torch.argmax(output, dim=1).item()  # get maximum tensor item
                        current_prediction = label_encoder.inverse_transform([pred_index])[0]  # translate value to label via encoder
                        if resolver.should_be_resolved(current_prediction):  # check if prediction is in the "troublesome" list
                            current_prediction = resolver.generic_resolve(current_prediction, flattened, handedness)  # resolve potential mismatches
                        handler.pred = current_prediction  # set current prediction to PredictionHandler, triggers PredictionHandler.on_pred_update()
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
