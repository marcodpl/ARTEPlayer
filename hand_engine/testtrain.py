import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# === Load data ===
DATA_FILE = "data/gesture_sequences.pkl"
with open(DATA_FILE, "rb") as f:
    data = pickle.load(f)

sequences = [np.array(seq) for _, seq in data]
labels = [label for label, _ in data]

# === Sequence handling ===
MAX_SEQ_LEN = 30
INPUT_SIZE = 42  # 21 landmarks x 2 (x, y)

def pad_sequence(seq):
    if len(seq) < MAX_SEQ_LEN:
        pad = np.zeros((MAX_SEQ_LEN - len(seq), INPUT_SIZE))
        return np.vstack((seq, pad))
    return np.array(seq[:MAX_SEQ_LEN])

X = np.array([pad_sequence(seq) for seq in sequences])
le = LabelEncoder()
y = le.fit_transform(labels)
y = torch.tensor(y, dtype=torch.long)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# === Move to tensors and device ===
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# === LSTM Model ===
class GestureLSTM(nn.Module):
    def __init__(self, input_size=42, hidden_size=64, output_size=len(le.classes_)):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# === Training ===
model = GestureLSTM(output_size=len(le.classes_)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 30
print("🧠 Training model...")

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        preds = torch.argmax(val_outputs, dim=1)
        acc = (preds == y_test).float().mean().item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} | Val Acc: {acc:.2f}")

# === Save model and label encoder
torch.save(model.state_dict(), "data/gesture_lstm.pt")
with open("data/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model trained and saved.")
