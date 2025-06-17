import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# helper function for EPOCH control
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, save_path="../data/gesture_lstm.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, lmodel):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            torch.save(lmodel.state_dict(), self.save_path)
            print("💾 Model improved — saved checkpoint.")
        else:
            self.counter += 1
            print(f"📉 No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


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
    with open("../data/gesture_sequences.pkl", "rb") as f:
        data = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sequences = [np.array(seq) for _, seq in data]
    labels = [label for label, _ in data]
    early_stopping = EarlyStopping(patience=2, min_delta=0.001)

    # Pad/truncate to 45 frames
    MAX_SEQ_LEN = 45
    INPUT_SIZE = 42  # 21 landmarks x 2 (x, y)
    MAX_EPOCHS = 100


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
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    model = GestureLSTM(output_size=len(le.classes_))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(MAX_EPOCHS):
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
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Loss: {loss.item():.4f} | Val Acc: {acc:.2f}")
        early_stopping(loss.item(), model)
        if early_stopping.early_stop:
            print(f"⏹️ Early stopping at epoch {epoch}")
            break

    # Save model and label encoder - deprecated
    with open("../data/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("✅ Model trained and saved!")
    y_true = y_test.cpu().numpy()
    y_pred = pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot()
    print("Showing confusion matrix plot...")
    plt.show()
