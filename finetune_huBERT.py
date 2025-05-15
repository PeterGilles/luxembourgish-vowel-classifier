import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments
)

# SETTINGS
AUDIO_DIR = "exported_vowels"
SAMPLE_RATE = 16000
BATCH_SIZE = 8
EPOCHS = 5
MODEL_NAME = "facebook/hubert-base-ls960"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data and count labels
all_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
all_labels = [f.rsplit("_", 1)[-1].replace(".wav", "") for f in all_files]
unique_labels = set(all_labels)
num_labels = len(unique_labels)
print(f"✅ Detected {num_labels} unique labels in the data")

# Load pretrained model and feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(DEVICE)
model.config.mask_time_prob = 0.0  # turn off masking explicitly

# Prepare label encoder
le = LabelEncoder()
le.fit(all_labels)
print(f"✅ Labels: {le.classes_}")

# Custom dataset
class VowelDataset(Dataset):
    def __init__(self, file_list, labels, feature_extractor):
        self.file_list = file_list
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        label = self.labels[idx]
        path = os.path.join(AUDIO_DIR, fname)
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        return {
            'input_values': inputs.input_values.squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Encode labels
encoded_labels = le.transform(all_labels)
dataset = VowelDataset(all_files, encoded_labels, feature_extractor)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Custom collate function to pad input lengths
def audio_data_collator(features):
    input_values = [f["input_values"] for f in features]
    labels = torch.tensor([f["labels"] for f in features])

    max_length = max(x.shape[-1] for x in input_values)
    padded_inputs = torch.stack([torch.nn.functional.pad(x, (0, max_length - x.shape[-1])) for x in input_values])

    return {"input_values": padded_inputs, "labels": labels}

# Trainer setup
training_args = TrainingArguments(
    output_dir="./results2",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = np.mean(preds == labels)
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=audio_data_collator,
    compute_metrics=compute_metrics,
)

# Start fine-tuning
trainer.train()

# Save model and label encoder
model.save_pretrained("./fine_tuned_vowel_model2")
feature_extractor.save_pretrained("./fine_tuned_vowel_model2")
import joblib
joblib.dump(le, "./fine_tuned_vowel_model2/label_encoder.pkl")
print("✅ Fine-tuned model and label encoder saved!")
