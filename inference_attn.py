# infer_bilstm_attention.py
import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- 모델 정의 (CNN + BiLSTM + Attention) ---
class AttentionLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):  # (B, T, 2H)
        weights = torch.softmax(self.attn(lstm_output), dim=1)  # (B, T, 1)
        context = torch.sum(weights * lstm_output, dim=1)       # (B, 2H)
        return context, weights

class SignCNNBiLSTMAttn(torch.nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_classes=10):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.lstm = torch.nn.LSTM(64, hidden_size, batch_first=True, bidirectional=True)
        self.attn = AttentionLayer(hidden_size)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)             # (B, C, T)
        x = self.relu(self.conv1(x))      # (B, 64, T)
        x = x.transpose(1, 2)             # (B, T, 64)
        lstm_out, _ = self.lstm(x)        # (B, T, 2H)
        context, weights = self.attn(lstm_out)
        return self.fc(self.dropout(context)), weights  # (B, num_classes), (B, T, 1)

# --- 설정 ---
SEQ_LEN = 125
LABEL_PATH = "label_classes.npy"
MODEL_PATH = "/Users/songmingyu/Desktop/CNN_LSTM_Attn_best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_classes = np.load(LABEL_PATH)
num_classes = len(label_classes)

model = SignCNNBiLSTMAttn(input_size=126, hidden_size=128, num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Mediapipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
hand_connections = mp_hands.HAND_CONNECTIONS

def draw_hand(image, landmarks, color):
    h, w, _ = image.shape
    for x, y, z in landmarks:
        cx, cy = int(x * w), int(y * h)
        cv2.circle(image, (cx, cy), 3, color, -1)
    for i, j in hand_connections:
        x0, y0, _ = landmarks[i]
        x1, y1, _ = landmarks[j]
        p0 = int(x0 * w), int(y0 * h)
        p1 = int(x1 * w), int(y1 * h)
        cv2.line(image, p0, p1, color, 2)

def extract_keypoints_and_draw(frame, canvas):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    hand0 = np.zeros((21, 3))
    hand1 = np.zeros((21, 3))
    if results.multi_hand_landmarks and results.multi_handedness:
        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            label = handedness.classification[0].label
            if label == 'Right':
                hand0 = coords
                draw_hand(canvas, coords, color=(0, 255, 0))
            elif label == 'Left':
                hand1 = coords
                draw_hand(canvas, coords, color=(255, 0, 0))
    return np.concatenate([hand0, hand1], axis=0).flatten()

def predict_sequence(seq, top_k=3):
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out, attn_weights = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        topk_indices = probs.argsort()[-top_k:][::-1]

    predictions = [(label_classes[i], probs[i]) for i in topk_indices]
    attn_weights = attn_weights.squeeze(0).cpu().numpy()  # (T, 1)
    return predictions, attn_weights

def plot_attention(attn_weights):
    plt.figure(figsize=(10, 2))
    plt.plot(attn_weights, label="Attention")
    plt.title("Attention Weights per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 추론 루프 ---
cap = cv2.VideoCapture(1)
collecting = False
sequence = []
print("스페이스바 누르면 수어 추론 시작")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()
    key = cv2.waitKey(1)

    if key == ord(' '):
        print("수어 인식 시작")
        collecting = True
        sequence = []

    if collecting:
        keypoints = extract_keypoints_and_draw(frame, display)
        sequence.append(keypoints)
        cv2.putText(display, f"Collecting: {len(sequence)}/{SEQ_LEN}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if len(sequence) == SEQ_LEN:
            print("🧠 추론 중...")
            top_preds, attn_weights = predict_sequence(sequence, top_k=3)
            for i, (label, prob) in enumerate(top_preds, 1):
                print(f"{i}. {label} ({prob * 100:.1f}%)")
            cv2.putText(display, f"Prediction: {top_preds[0][0]}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)
            plot_attention(attn_weights)
            collecting = False

    cv2.imshow("Sign Inference with Attention", display)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
