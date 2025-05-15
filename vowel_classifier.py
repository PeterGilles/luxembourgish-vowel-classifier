
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Settings ---
AUDIO_DIR = "exported_vowels"
SAMPLE_RATE = 16000
MAX_FRAMES = 160
N_MELS = 40  # number of Mel bands

# --- Feature Extraction ---
features, labels, filenames = [], [], []
for fname in os.listdir(AUDIO_DIR):
    if not fname.endswith(".wav"):
        continue
    label = fname.rsplit("_", 1)[-1].replace(".wav", "")
    filepath = os.path.join(AUDIO_DIR, fname)
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    
    # Compute Mel spectrogram (power)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=512, hop_length=160, win_length=400
    )
    
    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or crop to MAX_FRAMES
    if mel_spec_db.shape[1] < MAX_FRAMES:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, MAX_FRAMES - mel_spec_db.shape[1])), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :MAX_FRAMES]

    features.append(mel_spec_db)
    labels.append(label)
    filenames.append(fname)

# Convert to arrays
X = np.stack(features)  # shape: (n_samples, n_mels, max_frames)
le = LabelEncoder()
y = le.fit_transform(labels)
labels = le.classes_

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Class Balancing ---
class_sample_counts = np.bincount(y_train)
class_weights = 1.0 / class_sample_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# --- Dataset Wrapper ---
class VowelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X[:, np.newaxis, :, :], dtype=torch.float32)  # Add channel dim
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(VowelDataset(X_train, y_train), batch_size=32, sampler=sampler)
test_loader = DataLoader(VowelDataset(X_test, y_test), batch_size=32)

# --- CNN Model ---
class VowelCNNImproved(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)

        # Use a dummy input to compute the correct fc1 input size
        dummy_input = torch.zeros(1, 1, 39, 100)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        flatten_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        
# --- Training ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VowelCNNImproved(n_classes=len(labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss = {total_loss / len(train_loader):.4f}")
    
# --- Evaluate ---
model.eval()
correct = 0
total = 0
all_preds = []
all_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch).argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(y_batch.cpu().numpy())

print(f"Test accuracy: {correct / total:.2%}")

# --- Save ---
torch.save(model.state_dict(), "vowel_cnn_model_augmented.pt")
print("✅ Model saved to vowel_cnn_model_augmented.pt")
