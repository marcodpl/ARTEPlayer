import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from collections import deque

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Storage
DATA_FILE = "data/gesture_sequences.pkl"
gesture_data = []
# assert directory exists
os.makedirs("data", exist_ok=True)
# Load existing data if available
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        gesture_data = pickle.load(f)
    print(f"Loaded {len(gesture_data)} sequences from existing data.")
else:
    print("Starting new gesture recording session.")
gesture_name = ""
sequence = deque(maxlen=45)  # 1.5 seconds of gesture


# Normalize landmarks relative to wrist and scale
def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    normalized = []
    scale = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[9])) + 1e-6
    for x, y in landmarks:
        nx = (x - wrist[0]) / scale
        ny = (y - wrist[1]) / scale
        normalized.append((nx, ny))
    return normalized


# Flatten landmarks
def flatten_landmarks(landmarks):
    return [coord for point in landmarks for coord in point]


# Main capture loop
cap = cv2.VideoCapture(0)
print("Enter gesture name:")
gesture_name = input(">> ").strip()
print(f"Recording gesture: {gesture_name}. Press 'r' to record, 'n' for next gesture, 'q' to quit.")

i = 1
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
            norm_landmarks = normalize_landmarks(landmarks)
            flattened = flatten_landmarks(norm_landmarks)
            sequence.append(flattened)

    cv2.putText(image, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Recording Gestures", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        if len(sequence) == 45:
            gesture_data.append((gesture_name, list(sequence)))
            print(f"{i}: Recorded 1 sequence for '{gesture_name}'")
            i += 1
        else:
            print("Sequence too short, wait for 1.5 seconds of hand motion.")
    elif key == ord('n'):
        gesture_name = input("Next gesture name >> ").strip()
        print(f"Switched to gesture: {gesture_name}")
        i = 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save data
with open(DATA_FILE, "wb") as f:
    pickle.dump(gesture_data, f)

print(f"Saved {len(gesture_data)} sequences to 'data/gesture_sequences.pkl'")
