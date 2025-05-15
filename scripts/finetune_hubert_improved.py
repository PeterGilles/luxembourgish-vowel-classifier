import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fine-tune HuBERT model for vowel classification')
parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--use_class_weights', action='store_true', help='Use class weighting to handle imbalanced classes')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor (0-1)')
parser.add_argument('--output_dir', type=str, default='./fine_tuned_vowel_model', help='Directory to save the fine-tuned model')
parser.add_argument('--audio_dir', type=str, default='exported_vowels', help='Directory containing audio files')
args = parser.parse_args()

# SETTINGS
AUDIO_DIR = args.audio_dir
SAMPLE_RATE = 16000
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
OUTPUT_DIR = args.output_dir
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/hubert-base-ls960"

# Settings for short vowel processing
MAX_AUDIO_LENGTH = int(SAMPLE_RATE * 0.25)  # 250ms in samples
ENABLE_TIME_STRETCH = True
TIME_STRETCH_RANGE = (1.0, 1.2)
FEATURE_N_FFT = 1024
FEATURE_HOP_LENGTH = 160
FEATURE_WIN_LENGTH = 400

# Prepare data and count labels
print(f"Loading audio files from {AUDIO_DIR}...")
all_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
all_labels = [f.rsplit("_", 1)[-1].replace(".wav", "") for f in all_files]
unique_labels = set(all_labels)
num_labels = len(unique_labels)
print(f"✅ Detected {num_labels} unique labels in the data: {unique_labels}")

# Load pretrained model and feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

# Update feature extractor config for short vowel segments
feature_params = feature_extractor.to_dict()
if 'feature_extractor' in feature_params:
    orig_params = feature_params['feature_extractor']
    print(f"Original feature params: n_fft={orig_params.get('n_fft', 'N/A')}, "
          f"hop_length={orig_params.get('hop_length', 'N/A')}, "
          f"win_length={orig_params.get('win_length', 'N/A')}")
    
    # Update with optimized values for short segments
    if 'n_fft' in orig_params:
        feature_params['feature_extractor']['n_fft'] = FEATURE_N_FFT
    if 'hop_length' in orig_params:
        feature_params['feature_extractor']['hop_length'] = FEATURE_HOP_LENGTH
    if 'win_length' in orig_params:
        feature_params['feature_extractor']['win_length'] = FEATURE_WIN_LENGTH
    
    # Recreate feature extractor with updated params
    feature_extractor = Wav2Vec2FeatureExtractor.from_dict(feature_params)
    print(f"Updated feature params: n_fft={FEATURE_N_FFT}, "
          f"hop_length={FEATURE_HOP_LENGTH}, win_length={FEATURE_WIN_LENGTH}")

# Load the model
model = HubertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(DEVICE)

# Special configuration
model.config.mask_time_prob = 0.0  # turn off masking explicitly
print(f"Using learning rate: {args.learning_rate}")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save model type info for later
with open(f"{OUTPUT_DIR}/model_info.txt", "w") as f:
    f.write(f"model_type: hubert\n")
    f.write(f"model_name: {MODEL_NAME}\n")
    f.write(f"num_labels: {num_labels}\n")
    f.write(f"labels: {', '.join(unique_labels)}\n")

# Prepare label encoder
le = LabelEncoder()
le.fit(all_labels)
print(f"✅ Labels: {le.classes_}")

# Custom dataset
class VowelDataset(Dataset):
    def __init__(self, file_list, labels, feature_extractor, is_train=True):
        self.file_list = file_list
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.is_train = is_train  # Training mode enables augmentations

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        label = self.labels[idx]
        path = os.path.join(AUDIO_DIR, fname)
        
        # Set explicit window parameters to avoid warnings
        import warnings
        warnings.filterwarnings("ignore", message="PySoundFile failed.*", category=UserWarning)
        
        # Load audio with custom parameters
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        
        # Check if we need to pad very short segments before processing
        if len(audio) < 1024:  # If smaller than our FFT size
            # Pad to at least 1024 samples to avoid warnings
            padding = np.zeros(1024 - len(audio))
            audio = np.concatenate([audio, padding])
        
        # Apply time stretching augmentation during training
        if self.is_train and ENABLE_TIME_STRETCH:
            # Randomly stretch between 1.0 and 1.2 (0-20% longer)
            stretch_factor = np.random.uniform(TIME_STRETCH_RANGE[0], TIME_STRETCH_RANGE[1])
            # Only apply stretching 50% of the time to maintain variety
            if np.random.rand() > 0.5:
                # Use a smaller n_fft for time_stretch to avoid warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Pass complete set of parameters to ensure consistency
                    audio = librosa.effects.time_stretch(audio, 
                                                        rate=stretch_factor, 
                                                        n_fft=FEATURE_N_FFT,
                                                        hop_length=FEATURE_HOP_LENGTH,
                                                        win_length=FEATURE_WIN_LENGTH)
        
        # Ensure we don't exceed our maximum length
        if len(audio) > MAX_AUDIO_LENGTH:
            audio = audio[:MAX_AUDIO_LENGTH]
        
        # Extract features with consistent padding strategy
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=SAMPLE_RATE, 
            padding="max_length",
            max_length=MAX_AUDIO_LENGTH,
            return_tensors="pt"
        )
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Encode labels
encoded_labels = le.transform(all_labels)

# Create training and validation datasets separately with appropriate flags
# Use indices for splitting to maintain data distribution
indices = np.arange(len(all_files))
np.random.shuffle(indices)
train_size = int(0.8 * len(indices))

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create train dataset with augmentation enabled
train_files = [all_files[i] for i in train_indices]
train_labels = [encoded_labels[i] for i in train_indices]
train_dataset = VowelDataset(train_files, train_labels, feature_extractor, is_train=True)

# Create validation dataset with augmentation disabled
val_files = [all_files[i] for i in val_indices]
val_labels = [encoded_labels[i] for i in val_indices]
val_dataset = VowelDataset(val_files, val_labels, feature_extractor, is_train=False)

# Custom collate function to pad input lengths
def audio_data_collator(features):
    input_values = [f["input_values"] for f in features]
    labels = torch.tensor([f["labels"] for f in features])

    max_length = max(x.shape[-1] for x in input_values)
    padded_inputs = torch.stack([torch.nn.functional.pad(x, (0, max_length - x.shape[-1])) for x in input_values])

    return {"input_values": padded_inputs, "labels": labels}

# Compute class weights for balanced loss if requested
if args.use_class_weights:
    class_sample_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_sample_counts + 1e-6)
    # Normalize weights to prevent extremely high weights
    class_weights = class_weights / np.sum(class_weights) * len(class_weights)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Using class weights: {class_weights}")
else:
    class_weights_tensor = None

# Trainer setup
training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/checkpoints",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    label_smoothing_factor=args.label_smoothing if args.label_smoothing > 0 else None,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = np.mean(preds == labels)
    return {"accuracy": accuracy}

# Custom weighted loss function for class imbalance if requested
if args.use_class_weights:
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            loss = F.cross_entropy(
                logits.view(-1, self.model.config.num_labels),
                labels.view(-1),
                weight=class_weights_tensor,
                label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0.0
            )
            
            return (loss, outputs) if return_outputs else loss
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=audio_data_collator,
        compute_metrics=compute_metrics,
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=audio_data_collator,
        compute_metrics=compute_metrics,
    )

# Start fine-tuning
print(f"Starting training HuBERT model...")
trainer.train()

# Evaluate on validation set and generate confusion matrix
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Validation results: {eval_results}")

# Generate predictions for confusion matrix
predictions = trainer.predict(val_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

# Create and save confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
class_names = le.classes_

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - HuBERT')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
print(f"Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")

# Generate classification report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f"{OUTPUT_DIR}/classification_report.csv")
print(f"Classification report saved to {OUTPUT_DIR}/classification_report.csv")

# Print classification report to console
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# Save model and label encoder
model.save_pretrained(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)
joblib.dump(le, f"{OUTPUT_DIR}/label_encoder.pkl")
print(f"✅ Fine-tuned HuBERT model and label encoder saved to {OUTPUT_DIR}!")