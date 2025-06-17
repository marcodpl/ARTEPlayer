import pickle
import tkinter as tk
from tkinter import Listbox, messagebox
import cv2
import numpy as np
import os

# === Load Data ===
DATA_FILE = "data/gesture_sequences.pkl"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"File not found: {DATA_FILE}")

with open(DATA_FILE, "rb") as f:
    raw_data = pickle.load(f)

# Group by gesture
gesture_dict = {}
for label, seq in raw_data:
    gesture_dict.setdefault(label, []).append(seq)

# === GUI ===
root = tk.Tk()
root.title("🖐️ Gesture Sample Replayer")

gesture_listbox = Listbox(root, width=30)
gesture_listbox.pack(pady=5)

sequence_listbox = Listbox(root, width=30)
sequence_listbox.pack(pady=5)

selected_gesture = None
selected_index = None

def refresh_gestures():
    gesture_listbox.delete(0, tk.END)
    for label in sorted(gesture_dict):
        count = len(gesture_dict[label])
        gesture_listbox.insert(tk.END, f"{label} ({count})")

def on_gesture_select(event=None):
    global selected_gesture
    sel = gesture_listbox.curselection()
    if not sel:
        return
    item = gesture_listbox.get(sel[0])
    selected_gesture = item.split("(")[0].strip()
    update_sequence_list()

def update_sequence_list():
    sequence_listbox.delete(0, tk.END)
    if not selected_gesture:
        return
    for i in range(len(gesture_dict[selected_gesture])):
        sequence_listbox.insert(tk.END, f"Sample #{i+1}")

def on_sequence_select(event=None):
    global selected_index
    sel = sequence_listbox.curselection()
    if not sel:
        return
    selected_index = sel[0]

def play_sequence():
    if selected_gesture is None or selected_index is None:
        messagebox.showinfo("Select something!", "Pick a gesture and a sample first.")
        return
    seq = gesture_dict[selected_gesture][selected_index]

    for i, frame in enumerate(seq):
        landmarks = np.array(frame).reshape(-1, 2)
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255

        for (x, y) in landmarks:
            cx, cy = int(x * 640), int(y * 480)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(img, f"{selected_gesture} - Frame {i + 1}/{len(seq)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2)

        cv2.imshow("Replay", img)
        if cv2.waitKey(33) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

# === GUI Layout ===
gesture_listbox.bind("<<ListboxSelect>>", on_gesture_select)
sequence_listbox.bind("<<ListboxSelect>>", on_sequence_select)

tk.Button(root, text="▶ Play", command=play_sequence).pack(pady=5)
tk.Button(root, text="Quit", command=root.quit).pack(pady=2)

refresh_gestures()
root.mainloop()
