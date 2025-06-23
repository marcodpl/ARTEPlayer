import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import simpledialog, messagebox, Listbox
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# === Settings ===
DATA_FILE = "data/gesture_sequences.pkl"
MAX_SEQ_LEN = 45

# === Data setup ===
gesture_data = defaultdict(list)
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        for seq, label in pickle.load(f):
            gesture_data[label].append(seq)


i_frame = 1
# === Mediapipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# === GUI state ===
current_label = None
sequence = []
recording = False
running = True

# === Create GUI ===
root = tk.Tk()
root.title("Gesture Recorder")


def flatten_landmarks(landmarks):
    return [coord for point in landmarks for coord in point]


def save_data():
    all_data = [(seq, label) for sequences, label in gesture_data.items() for seq in sequences]
    with open(DATA_FILE, "wb") as f:
        pickle.dump(all_data, f)


def show_stats():
    if not gesture_data:
        messagebox.showinfo("Stats", "No gesture data recorded.")
        return

    labels = []
    counts = []
    lengths = []

    for sequences, label in gesture_data.items():
        labels.append(label)
        counts.append(len(sequences))
        seq_lengths = [len(seq) for seq in sequences]
        lengths.append((
            min(seq_lengths),
            max(seq_lengths),
            round(sum(seq_lengths) / len(seq_lengths), 2)
        ))

    # Text stats
    stats_text = "\n".join(
        f"{label}: {count} recordings (min {min_l}, max {max_l}, avg {avg_l})"
        for (label, count, (min_l, max_l, avg_l)) in zip(labels, counts, lengths)
    )
    messagebox.showinfo("Gesture Stats", stats_text)

    # Bar plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, counts, color='skyblue')
    ax.set_title("Gesture Count")
    ax.set_ylabel("Recordings")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Embed chart in Tkinter window
    chart_window = tk.Toplevel(root)
    chart_window.title("Gesture Chart")

    canvas = FigureCanvasTkAgg(fig, master=chart_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


def refresh_listbox():
    listbox.delete(0, tk.END)
    for label in sorted(gesture_data.keys()):
        count = len(gesture_data[label])
        listbox.insert(tk.END, f"{label}: {count}")


def set_label(event=None):
    global current_label
    sel = listbox.curselection()
    if sel:
        item_text = listbox.get(sel[0])
        label = item_text.split(":")[0].strip()
        current_label = label
        print(f"🎯 Selected: {current_label}")


def add_label():
    global current_label
    name = simpledialog.askstring("New Gesture", "Enter gesture name:")
    if name:
        gesture_data[name] = gesture_data.get(name, [])
        current_label = name
        refresh_listbox()
        print(f"➕ Added: {name}")


def delete_label():
    sel = listbox.curselection()
    if sel:
        label1 = listbox.get(sel[0])
        if messagebox.askyesno("Confirm", f"Delete '{label}'?"):
            del gesture_data[label1.split(":")[0].strip()]
            refresh_listbox()
            save_data()
            print(f"🗑️ Deleted: {label}")


def quit_app():
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()
    root.quit()


listbox = Listbox(root, width=30)
listbox.pack(pady=5)
listbox.bind("<<ListboxSelect>>", set_label)

tk.Button(root, text="➕ New Gesture", command=add_label).pack(pady=2)
tk.Button(root, text="❌ Delete Gesture", command=delete_label).pack(pady=2)
tk.Button(root, text="📊 Show Stats", command=show_stats).pack(pady=2)
tk.Label(root, text="Press & hold 'R' to record").pack()
tk.Button(root, text="🚪 Quit", command=quit_app).pack(pady=5)

refresh_listbox()

# === Webcam + Keyboard Handling ===
cap = cv2.VideoCapture(0)


def process_frame():
    global sequence, recording, i_frame

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if recording and current_label:
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                flat = flatten_landmarks(landmarks)
                sequence.append(np.array(flat))
                if len(sequence) > MAX_SEQ_LEN:
                    sequence = sequence[:MAX_SEQ_LEN]

    if recording:
        cv2.putText(image, f"Recording: {i_frame}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        i_frame += 1
    else:
        cv2.putText(image, f"Gesture: {current_label or 'None'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Gesture Recorder", image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC for emergency quit
        quit_app()

    root.after(10, process_frame)


# === Keyboard events (Tkinter handles keydown/up)
def on_key(event):
    global recording, sequence, i_frame
    if event.keysym.lower() == 'r' and not recording and current_label:
        print("🔴 Recording started")
        recording = True
        i_frame = 1
        sequence = []


def on_key_release(event):
    global recording
    if event.keysym.lower() == 'r' and recording:
        if sequence and current_label:
            gesture_data[current_label].append(sequence)
            save_data()
            print(f"✅ Recorded for '{current_label}'")
        recording = False
        refresh_listbox()

root.bind("<KeyPress>", on_key)
root.bind("<KeyRelease>", on_key_release)

# === Start everything
root.after(0, process_frame)
root.mainloop()
