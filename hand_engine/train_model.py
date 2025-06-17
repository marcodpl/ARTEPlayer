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

# === TOGGLE MODEL TYPE ===
MODEL_TYPE = "transformer"  # "lstm" or "transformer"

# === Hyperparameters ===
MAX_SEQ_LEN = 45
INPUT_SIZE = 42
HIDDEN_SIZE = 64
MAX_EPOCHS = 100


# === Early stopping class ===
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, save_path="data/gesture_model.pt"):
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
            torch.save(lmodel.state_dict(), self.save_path)
            print("💾 Model improved — saved checkpoint.")
        else:
            self.counter += 1
            print(f"📉 No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


# === LSTM model ===
class GestureLSTM(nn.Module):
    def __init__(self, input_size=42, hidden_size=64, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


# === Transformer Model and Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class GestureTransformer(nn.Module):
    def __init__(self, input_size=42, hidden_size=64, output_size=4, num_layers=2, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.fc(x[:, 0])


# === MAIN LOOP - TRAIN_MODEL.PY ===
if __name__ == "__main__":
    with open("data/gesture_sequences.pkl", "rb") as f:
        data = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sequences = [np.array(seq) for _, seq in data]
    labels = [label if isinstance(label, str) else label[0] for label, _ in data]
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    if MODEL_TYPE == "transformer":
        model = GestureTransformer(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=len(le.classes_)).to(device)
    else:
        model = GestureLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=len(le.classes_)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            pred = torch.argmax(test_outputs, dim=1)
            acc = (pred == y_test).float().mean()

        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Loss: {loss.item():.4f} | Val Acc: {acc:.2f}")
        early_stopping(loss.item(), model)
        if early_stopping.early_stop:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break

    with open("data/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("✅ Model trained and saved!")
    y_true = y_test.cpu().numpy()
    y_pred = pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot()
    print("Showing confusion matrix plot...")
    plt.title(f"Confusion Matrix ({MODEL_TYPE.upper()})")
    plt.show()