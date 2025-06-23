import pickle
import os
import numpy as np

DATA_FILE = "../data/gesture_sequences.pkl"
MAX_LANDMARKS = 42

def is_valid_frame(frame):
    return (
        isinstance(frame, (list, np.ndarray)) and
        len(frame) == MAX_LANDMARKS and
        all(isinstance(x, (int, float)) for x in frame)
    )

def is_valid_sequence(seq):
    return (
        isinstance(seq, list) and
        len(seq) > 0 and
        all(is_valid_frame(f) for f in seq)
    )

if not os.path.exists(DATA_FILE):
    print("❌ gesture_sequences.pkl not found.")
    exit()

with open(DATA_FILE, "rb") as f:
    data = pickle.load(f)

print(f"📂 Loaded {len(data)} gestures.")
clean_data = []
errors = []

for i, (seq, label) in enumerate(data):
    if not is_valid_sequence(seq):
        errors.append((i, label))
    else:
        clean_data.append((seq, label))

# Summary
print(f"\n✅ Valid sequences: {len(clean_data)}")
print(f"⚠️ Invalid sequences: {len(errors)}")

if errors:
    print("\n🔍 Listing invalid entries:")
    for idx, label in errors:
        print(f"  - Index {idx}: Label = {label}")

    choice = input("\n❓ Remove invalid entries and overwrite file? (y/n): ").strip().lower()
    if choice == "y":
        with open(DATA_FILE, "wb") as f:
            pickle.dump(clean_data, f)
        print("✅ Saved cleaned dataset.")
    else:
        print("❌ Aborted cleanup. File unchanged.")
else:
    print("🎉 All sequences are valid!")
