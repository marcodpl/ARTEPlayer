import cv2
import torch
import pickle
import numpy as np
import mediapipe as mp
from collections import deque
from sklearn.preprocessing import LabelEncoder
import os
import time

MODEL_TYPE = "transformer"  # or "lstm"
MAX_SEQ_LEN = 45
INPUT_SIZE = 42

# === Load label encoder ===
with open("data/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL_TYPE == "transformer":
    from train_model import GestureTransformer as Net
else:
    from train_model import GestureLSTM as Net

model = Net(input_size=INPUT_SIZE, hidden_size=64, output_size=len(le.classes_))
model.load_state_dict(torch.load("data/gesture_model.pt", map_location=device))
model.to(device).eval()

# === MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Capture ===
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=MAX_SEQ_LEN)
data_file = "data/gesture_sequences.pkl"

# Load existing data
if os.path.exists(data_file):
    with open(data_file, "rb") as f:
        data = pickle.load(f)
else:
    data = []

print("Press 'y' to confirm prediction, or type the correct label. Press 'q' to quit.")

recording = False
start_time = 0
COLLECT_DURATION = 1.5  # seconds to collect full gesture before predicting

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            coords = []
            for i in range(21):
                coords.extend([lm[i].x, lm[i].y])
            sequence.append(coords)

            if not recording:
                print("🟡 Gesture detected. Recording started...")
                recording = True
                start_time = time.time()

            elif recording and time.time() - start_time >= COLLECT_DURATION:
                if len(sequence) == MAX_SEQ_LEN:
                    input_tensor = torch.tensor([sequence], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        confidence, pred_class = torch.max(probs, dim=1)
                        pred_label = le.inverse_transform([pred_class.item()])[0]
                        cv2.putText(image, f"{pred_label} ({confidence.item()*100:.1f}%)", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow("Verify Gesture", image)
                    key = cv2.waitKey(0) & 0xFF

                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        with open(data_file, "wb") as f:
                            pickle.dump(data, f)
                        print("✅ Data saved and exiting.")
                        exit()
                    elif key == ord('y'):
                        print(f"✔️ Confirmed: {pred_label}")
                        data.append((list(sequence), pred_label))
                    else:
                        cv2.destroyWindow("Verify Gesture")
                        true_label = input(f"❌ Enter correct label (or empty to skip): ").strip()
                        if true_label in le.classes_:
                            print(f"✔️ Corrected to: {true_label}")
                            data.append((list(sequence), true_label))
                        elif true_label == "":
                            print("⏭️ Skipped.")
                        else:
                            print("⚠️ Unknown label — not saved.")

                recording = False
                sequence.clear()
    else:
        recording = False
        sequence.clear()
        cv2.putText(image, "Show your hand...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Verify Gesture", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

with open(data_file, "wb") as f:
    pickle.dump(data, f)

print("📝 Session complete. Data updated.")
