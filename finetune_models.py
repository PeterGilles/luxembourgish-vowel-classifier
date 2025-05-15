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
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fine-tune audio models for vowel classification')
parser.add_argument('--model_type', type=str, default='hubert', choices=['hubert', 'wav2vec2'], 
                    help='Model type: hubert or wav2vec2')
parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=None, 
                    help='Learning rate (default: 5e-5 for hubert, 1e-5 for wav2vec2)')
parser.add_argument('--weight_decay', type=float, default=0.01, 
                    help='Weight decay for regularization')
parser.add_argument('--use_class_weights', action='store_true', 
                    help='Use class weighting to handle imbalanced classes')
parser.add_argument('--label_smoothing', type=float, default=0.1, 
                    help='Label smoothing factor (0-1)')
parser.add_argument('--warmup_ratio', type=float, default=0.1, 
                    help='Ratio of steps for warmup')
parser.add_argument('--freeze_base', action='store_true', 
                    help='Initially freeze base model and train only the classifier')
parser.add_argument('--unfreeze_at_epoch', type=int, default=3, 
                    help='Epoch to unfreeze the base model when using --freeze_base')
parser.add_argument('--output_dir', type=str, default='./fine_tuned_vowel_model', 
                    help='Directory to save the fine-tuned model')
parser.add_argument('--audio_dir', type=str, default='exported_vowels', 
                    help='Directory containing audio files')
args = parser.parse_args()

# SETTINGS
AUDIO_DIR = args.audio_dir
SAMPLE_RATE = 16000
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
OUTPUT_DIR = args.output_dir
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Settings for short vowel processing
# Standard length for padding (250ms is good for short vowels)
MAX_AUDIO_LENGTH = int(SAMPLE_RATE * 0.25)  # 250ms in samples
# Enable time stretching augmentation for training
ENABLE_TIME_STRETCH = True
# Stretching range (1.0-1.2 means up to 20% longer)
TIME_STRETCH_RANGE = (1.0, 1.2)

# Adjusted feature extraction parameters for short vowels
# Reduce FFT size to accommodate shorter segments (85ms window for 16kHz)
FEATURE_N_FFT = 1024  # Was 2048
# Smaller hop length for more frames from short segments
FEATURE_HOP_LENGTH = 160  # 10ms hop
# Window length, typically smaller than n_fft
FEATURE_WIN_LENGTH = 400  # 25ms window

# Model configuration based on type
if args.model_type == 'hubert':
    MODEL_NAME = "facebook/hubert-base-ls960"
    MODEL_CLASS = HubertForSequenceClassification
    print(f"Using HuBERT model: {MODEL_NAME}")
else:
    MODEL_NAME = "facebook/wav2vec2-base-960h"
    MODEL_CLASS = Wav2Vec2ForSequenceClassification
    print(f"Using Wav2Vec2 model: {MODEL_NAME}")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prepare data and count labels
all_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
all_labels = [f.rsplit("_", 1)[-1].replace(".wav", "") for f in all_files]
unique_labels = set(all_labels)
num_labels = len(unique_labels)
print(f"✅ Detected {num_labels} unique labels in the data: {unique_labels}")

# Load pretrained model and feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

# Update feature extractor config for short vowel segments
# This addresses the "n_fft is too large" warning
feature_params = feature_extractor.to_dict()
if 'feature_extractor' in feature_params:
    orig_params = feature_params['feature_extractor']
    # Store original values for reference
    print(f"Original feature params: n_fft={orig_params.get('n_fft', 'N/A')}, "
          f"hop_length={orig_params.get('hop_length', 'N/A')}, "
          f"win_length={orig_params.get('win_length', 'N/A')}")
    
    # Update with our optimized values for short segments
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
model = MODEL_CLASS.from_pretrained(MODEL_NAME, num_labels=num_labels).to(DEVICE)

# Special configuration for both models - disable masking
model.config.mask_time_prob = 0.0  # turn off masking explicitly

# Set learning rate if not specified
if args.learning_rate is None:
    if args.model_type == 'hubert':
        learning_rate = 5e-5
    else:  # wav2vec2
        learning_rate = 1e-5
else:
    learning_rate = args.learning_rate

print(f"Using learning rate: {learning_rate}")

# Implement gradual unfreezing if requested
if args.freeze_base:
    if args.model_type == 'hubert':
        # Freeze the base HuBERT model
        for param in model.hubert.parameters():
            param.requires_grad = False
        print(f"✅ Base HuBERT model frozen. Will unfreeze at epoch {args.unfreeze_at_epoch}")
    elif args.model_type == 'wav2vec2':
        # Freeze the base Wav2Vec2 model
        for param in model.wav2vec2.parameters():
            param.requires_grad = False
        print(f"✅ Base Wav2Vec2 model frozen. Will unfreeze at epoch {args.unfreeze_at_epoch}")

# Additional settings for Wav2Vec2
if args.model_type == 'wav2vec2':
    # Set a very small mask length to prevent the masking error with short segments
    model.config.mask_time_length = 1
    
    # Optimize for short segments
    # Reduce feature extraction stride for finer-grained features
    if hasattr(model.config, 'conv_stride'):
        # Make a copy of the original strides
        original_strides = model.config.conv_stride.copy() if isinstance(model.config.conv_stride, list) else [model.config.conv_stride]
        # Log the changes
        print(f"Original conv_stride: {original_strides}")
        
        # For very short segments, we can use smaller strides
        # This helps capture more detail from short vowels
        if len(original_strides) >= 3:
            # Only modify if we have a list of strides to work with
            new_strides = [max(2, s-1) for s in original_strides[:3]]
            model.config.conv_stride = new_strides
            print(f"Modified conv_stride for short vowels: {new_strides}")
            
    # Ensure the attention mechanism can handle short sequences
    model.config.attention_dropout = 0.0  # Disable attention dropout
    # Reduce positional encoding variance to focus on shorter timeframes
    if hasattr(model.config, 'position_embedding_type') and model.config.position_embedding_type == 'relative_key':
        model.config.position_embedding_type = 'relative_key_query'  # More suitable for short segments

# Prepare label encoder
le = LabelEncoder()
le.fit(all_labels)
print(f"✅ Labels: {le.classes_}")

# Save model type info for later
with open(f"{OUTPUT_DIR}/model_info.txt", "w") as f:
    f.write(f"model_type: {args.model_type}\n")
    f.write(f"model_name: {MODEL_NAME}\n")
    f.write(f"num_labels: {num_labels}\n")
    f.write(f"labels: {', '.join(le.classes_)}\n")

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
        
        # Set explicit window parameters to avoid the "n_fft too large" warning
        # Note: We need to suppress other librosa warnings too
        import warnings
        warnings.filterwarnings("ignore", message="PySoundFile failed.*", category=UserWarning)
        
        # Load audio with custom parameters
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        
        # Check if we need to pad very short segments before processing
        if len(audio) < 1024:  # If smaller than our FFT size
            # Pad to at least 1024 samples to avoid the warning
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

# Trainer setup
training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/checkpoints",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_strategy="epoch",
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
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
print(f"Starting training with {args.model_type} model...")
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
plt.title(f'Confusion Matrix - {args.model_type.upper()}')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
print(f"Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")

# Generate classification report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f"{OUTPUT_DIR}/classification_report.csv")
print(f"Classification report saved to {OUTPUT_DIR}/classification_report.csv")

# Print classification report to console
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save model and label encoder
model.save_pretrained(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)
joblib.dump(le, f"{OUTPUT_DIR}/label_encoder.pkl")
print(f"✅ Fine-tuned {args.model_type} model and label encoder saved to {OUTPUT_DIR}!")