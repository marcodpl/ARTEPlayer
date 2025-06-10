import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# LSTM model
class GestureLSTM(nn.Module):
    def __init__(self, input_size=42, hidden_size=64, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


if __name__ == "__main__":
    # Load gesture data
    with open("data/gesture_sequences.pkl", "rb") as f:
        data = pickle.load(f)

    sequences = [np.array(seq) for _, seq in data]
    labels = [label for label, _ in data]

    # Pad/truncate to 45 frames
    MAX_SEQ_LEN = 45
    INPUT_SIZE = 42  # 21 landmarks x 2 (x, y)


    def pad_sequence(seq):
        if len(seq) < MAX_SEQ_LEN:
            padding = np.zeros((MAX_SEQ_LEN - len(seq), INPUT_SIZE))
            return np.vstack((seq, padding))
        else:
            return np.array(seq[:MAX_SEQ_LEN])


    X = np.array([pad_sequence(seq) for seq in sequences])
    le = LabelEncoder()
    y = le.fit_transform(labels)
    y = torch.tensor(y, dtype=torch.long)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    model = GestureLSTM(output_size=len(le.classes_))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    EPOCHS = 40
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            pred = torch.argmax(test_outputs, dim=1)
            acc = (pred == y_test).float().mean()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} | Val Acc: {acc:.2f}")

    # Save model and label encoder
    torch.save(model.state_dict(), "data/gesture_lstm.pt")
    with open("data/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("✅ Model trained and saved!")
