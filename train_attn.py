# train_bilstm_attention.py
import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import wandb

# --- config ---
DATA_ROOT = '/data/minkyu/P_project/keypoint_npy_augmented'
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-3
PATIENCE = 20

wandb.init(project="sign_cnn_bilstm_attention", config={
    "model": "CNN + BiLSTM + Attention",
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "epochs": EPOCHS
})

# --- Dataset ---
class SignDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        x = np.load(self.file_paths[idx])  # (125, 126)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def load_data(data_root):
    file_paths, labels = [], []
    for label_name in sorted(os.listdir(data_root)):
        label_path = os.path.join(data_root, label_name)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if fname.endswith(".npy"):
                file_paths.append(os.path.join(label_path, fname))
                labels.append(label_name)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return list(file_paths), list(labels), le

# --- Model ---
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):  # (B, T, 2H)
        weights = torch.softmax(self.attn(lstm_output), dim=1)  # (B, T, 1)
        context = torch.sum(weights * lstm_output, dim=1)       # (B, 2H)
        return context, weights

class SignCNNBiLSTMAttn(nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True, bidirectional=True)
        self.attn = AttentionLayer(hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):  # x: (B, T, C)
        x = x.transpose(1, 2)             # (B, C, T)
        x = self.relu(self.conv1(x))      # (B, 64, T)
        x = x.transpose(1, 2)             # (B, T, 64)
        lstm_out, _ = self.lstm(x)        # (B, T, 2H)
        context, weights = self.attn(lstm_out)  # (B, 2H), (B, T, 1)
        return self.fc(self.dropout(context)), weights

# --- Training ---
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for x, y in loader:
        print("x.shape:", x.shape)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# --- Main ---
def main():
    file_paths, labels, le = load_data(DATA_ROOT)
    combined = list(zip(file_paths, labels))
    random.shuffle(combined)
    file_paths, labels = zip(*combined)

    print("샘플 확인:")
    for i in range(100):
        print(f"{i+1}. {file_paths[i]} --> label: {labels[i]} ({le.inverse_transform([labels[i]])[0]})")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42)

    train_loader = DataLoader(SignDataset(train_paths, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SignDataset(val_paths, val_labels), batch_size=BATCH_SIZE)

    num_classes = len(le.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SignCNNBiLSTMAttn(input_size=126, hidden_size=128, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(f"[{epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "CNN_LSTM_Attn_best_model.pth")
            np.save("label_classes.npy", le.classes_)
            print("Best model saved.")
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()
