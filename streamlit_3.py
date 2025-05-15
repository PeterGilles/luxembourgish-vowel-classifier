import streamlit as st
import numpy as np
import torch
import librosa
import tempfile
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import joblib
from st_audiorec import st_audiorec
from huggingface_hub import hf_hub_download

# SETTINGS
REPO_ID = "pgilles/vowel-classifier-hubert2"
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and components from Hugging Face Hub
st.title("🎙 Vowel Classifier (HuBERT Fine-Tuned)")
st.write("Upload a WAV file or record your vowel live.")


@st.cache_resource  # Cache models to avoid reloading
def load_model():
    # Download model files from Hugging Face Hub
    # First load label encoder to determine number of classes
    label_encoder_path = hf_hub_download(repo_id=REPO_ID, filename="label_encoder.pkl")
    label_encoder = joblib.load(label_encoder_path)
    
    # Download model and check configuration
    model = HubertForSequenceClassification.from_pretrained(REPO_ID)
    model_num_labels = model.config.num_labels
    encoder_num_labels = len(label_encoder.classes_)
    
    if model_num_labels != encoder_num_labels:
        st.warning(f"⚠️ Model mismatch detected: Model has {model_num_labels} output classes but label encoder has {encoder_num_labels} classes.\n"
                  f"This suggests the model was trained with fewer labels than currently exist in the data.\n"
                  f"You may need to retrain the model with the updated finetune_huBERT.py script.")
    
    st.write(f"Model configured for {model_num_labels} labels")
    st.write(f"Label encoder has {encoder_num_labels} classes: {', '.join(label_encoder.classes_)}")
    
    model.eval()
    model.to(DEVICE)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(REPO_ID)
    
    return model, feature_extractor, label_encoder

model, feature_extractor, label_encoder = load_model()

# Input source selection
audio_source = st.radio("Choose input source:", ["Upload WAV file", "Record from microphone"])

audio_data = None

if audio_source == "Upload WAV file":
    uploaded_file = st.file_uploader("Upload your .wav file", type=["wav"])
    if uploaded_file is not None:
        audio_data, _ = librosa.load(uploaded_file, sr=SAMPLE_RATE)
        st.audio(uploaded_file, format='audio/wav')

elif audio_source == "Record from microphone":
    st.write("Click the red circle to record, then stop. Prediction will run automatically.")
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav.write(wav_audio_data)
            tmp_wav_path = tmp_wav.name
        audio_data, _ = librosa.load(tmp_wav_path, sr=SAMPLE_RATE)
        st.audio(wav_audio_data, format='audio/wav')

# Prediction logic
if audio_data is not None:
    # Extract features
    inputs = feature_extractor(
        audio_data,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    # Run prediction
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        pred_id = np.argmax(probabilities)
        
        # Make sure we don't go out of bounds
        if pred_id < len(label_encoder.classes_):
            pred_label = label_encoder.inverse_transform([pred_id])[0]
            st.success(f"🔊 Predicted Vowel: **{pred_label}**")
        else:
            st.error(f"Prediction error: model predicted class {pred_id} but label encoder only has {len(label_encoder.classes_)} classes")
            pred_label = "unknown"

        # Show confidence scores - handle mismatch between model output and label encoder
        st.subheader("Confidence Scores")
        
        model_num_labels = model.config.num_labels
        encoder_num_labels = len(label_encoder.classes_)
        
        if model_num_labels == encoder_num_labels:
            # Perfect match - show all scores
            for label, prob in zip(label_encoder.classes_, probabilities):
                st.write(f"**{label}**: {prob:.2%}")
        else:
            # Mismatch - show a warning and only display scores for available classes
            st.warning(f"⚠️ Showing scores for only {model_num_labels} classes due to model mismatch")
            for i, prob in enumerate(probabilities):
                if i < encoder_num_labels:
                    label = label_encoder.classes_[i]
                    st.write(f"**{label}**: {prob:.2%}")
                else:
                    st.write(f"**Unknown class {i}**: {prob:.2%}")
