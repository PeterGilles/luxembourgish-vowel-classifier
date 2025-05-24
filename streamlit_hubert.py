# Fix for torch._classes issue with Streamlit file watcher
import os
import sys

# Disable Streamlit's file watcher for torch modules
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import torch
# Additional fix for torch._classes
if hasattr(torch, '_classes'):
    # Create a dummy __path__ that won't cause issues
    class DummyPath:
        def __iter__(self):
            return iter([])
        def _path(self):
            return []
    if hasattr(torch._classes, '__path__'):
        torch._classes.__path__ = DummyPath()

import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import shutil
import matplotlib.pyplot as plt
import joblib
from huggingface_hub import hf_hub_download
from transformers import (
    HubertForSequenceClassification, 
    Wav2Vec2FeatureExtractor
)
from st_audiorec import st_audiorec

# SETTINGS
HUBERT_REPO_ID = "pgilles/vowel-classifier-hubert"
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try to find the examples directory
EXAMPLE_DIR = "example_vowels"
if not os.path.exists(EXAMPLE_DIR):
    # Try relative paths for different environments
    possible_paths = [
        "exported_vowels",        # Original folder
        "../example_vowels",      # One directory up
        "../../example_vowels",   # Two directories up
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            EXAMPLE_DIR = path
            print(f"Found examples directory at: {EXAMPLE_DIR}")
            break

# Set up page configuration
st.set_page_config(
    page_title="🎙 Vowel Classifier",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for audio data persistence
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Remove unused session state variables
for key in ['synthetic_audio', 'synthetic_path']:
    if key in st.session_state:
        del st.session_state[key]

# App title and description
st.title("🎙 Luxembourgish Vowel Classifier")
st.write("Classify vowels using HuBERT model. Upload a WAV file, record your vowel live, or use example vowels.")

# Sidebar for information
with st.sidebar:
    st.header("About")
    st.write("This app uses a fine-tuned HuBERT model to classify Luxembourgish vowels.")
    st.write("**Model**: Fine-tuned HuBERT for phonetic classification")
    st.write("**Training data**: Vowel segments (90-300ms) from the Schnëssen corpus")
    
    # Microphone tips
    st.header("Microphone Tips")
    st.markdown("""
    If you're having trouble with the microphone recording:
    
    1. Use Chrome browser for best compatibility
    2. Record at least 0.5 seconds of audio
    3. Speak clearly and directly into the microphone
    4. Hold a sustained vowel sound rather than a short one
    5. Try refreshing the page if recording fails
    """)
    
    # Add debug mode toggle
    st.header("Settings")
    debug_mode = st.checkbox("Debug mode", value=st.session_state.debug_mode)
    st.session_state.debug_mode = debug_mode

# Model loading function with cache
@st.cache_resource
def load_model():
    try:
        st.sidebar.info(f"Loading model from {HUBERT_REPO_ID}...")
        
        # First load label encoder to determine number of classes
        try:
            label_encoder_path = hf_hub_download(repo_id=HUBERT_REPO_ID, filename="label_encoder.pkl")
            label_encoder = joblib.load(label_encoder_path)
            num_classes = len(label_encoder.classes_)
            st.sidebar.success(f"✅ Label encoder loaded with {num_classes} classes")
        except Exception as e:
            st.sidebar.error(f"Error loading label encoder: {str(e)}")
            raise
            
        # Try to load model info if available
        try:
            model_info_path = hf_hub_download(repo_id=HUBERT_REPO_ID, filename="model_info.txt")
            with open(model_info_path, 'r') as f:
                model_info = {line.split(': ')[0]: line.split(': ')[1].strip() 
                            for line in f.readlines() if ': ' in line}
            st.sidebar.write(f"**Model details:** {model_info.get('model_name', 'HuBERT')}")
            st.sidebar.write(f"**Vowel classes:** {num_classes}")
            st.sidebar.write(f"**Labels:** {', '.join(label_encoder.classes_)}")
        except Exception as e:
            st.sidebar.warning(f"Could not load model info: {str(e)}")
            st.sidebar.write(f"**Vowel classes:** {num_classes}")
        
        # Download model
        try:
            st.sidebar.info("Loading model architecture...")
            model = HubertForSequenceClassification.from_pretrained(HUBERT_REPO_ID)
            
            # Check model configuration
            if hasattr(model.config, 'num_labels'):
                model_num_labels = model.config.num_labels
                if model_num_labels != num_classes:
                    st.sidebar.warning(f"⚠️ Model has {model_num_labels} output classes but label encoder has {num_classes} classes")
            
            model.eval()
            model.to(DEVICE)
            st.sidebar.success("✅ Model loaded successfully")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            raise
            
        # Load feature extractor
        try:
            st.sidebar.info("Loading feature extractor...")
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_REPO_ID)
            st.sidebar.success("✅ Feature extractor loaded")
        except Exception as e:
            st.sidebar.error(f"Error loading feature extractor: {str(e)}")
            raise
        
        return {
            "model": model,
            "feature_extractor": feature_extractor,
            "label_encoder": label_encoder,
            "num_classes": num_classes
        }
    except Exception as e:
        st.error(f"Critical error loading model: {str(e)}")
        return None

# Load model
hubert_components = load_model()
if hubert_components:
    st.sidebar.success("✅ HuBERT model loaded")

# Input source selection
audio_source = st.radio("Choose input source:", ["Upload WAV file", "Record from microphone", "Use Example Vowels"])

audio_data = None
audio_file_path = None

if audio_source == "Upload WAV file":
    st.write("You can upload your own audio file of a vowel sound (.wav format).")
    st.write("💡 **Tip:** For best results, use a clear recording of a single sustained vowel sound.")
    
    uploaded_file = st.file_uploader("Upload your .wav file", type=["wav"])
    if uploaded_file is not None:
        # Use a unique identifier for the upload to track changes
        file_id = hash(uploaded_file.getvalue())
        
        # Check if this is a new upload to avoid reprocessing
        if 'last_file_id' not in st.session_state or file_id != st.session_state.last_file_id:
            st.session_state.last_file_id = file_id
            
            try:
                # Save to a temporary file (needed for librosa)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    tmp_wav.write(uploaded_file.getvalue())
                    tmp_wav_path = tmp_wav.name
                
                # Load audio data
                audio_data, sr = librosa.load(tmp_wav_path, sr=SAMPLE_RATE)
                
                # Debug information
                if st.session_state.debug_mode:
                    duration = len(audio_data) / sr
                    st.write(f"File duration: {duration:.2f} seconds")
                    st.write(f"Audio shape: {audio_data.shape}")
                    st.write(f"Sample rate: {sr} Hz")
                    
                    # Plot waveform
                    fig, ax = plt.subplots(figsize=(10, 2))
                    plt.plot(audio_data)
                    plt.title("Waveform")
                    plt.xlabel("Sample")
                    plt.ylabel("Amplitude")
                    st.pyplot(fig)
                
                # Flag that this needs prediction
                st.session_state.should_predict = True
                
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"Error processing file: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                else:
                    st.error("There was an error processing your audio file. Please try a different file.")
                audio_data = None
        
        # Always display the audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Add a button to run prediction
        if st.button("Classify Vowel"):
            st.session_state.should_predict = True

elif audio_source == "Record from microphone":
    st.write("Click the record button below to capture your vowel sound.")
    st.write("💡 **Tip:** Hold a sustained vowel sound for at least 1 second for best results.")
    
    # Record audio
    wav_audio_data = st_audiorec()
    
    if wav_audio_data is not None:
        try:
            # Save to a temporary file (needed for librosa)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                tmp_wav.write(wav_audio_data)
                tmp_wav_path = tmp_wav.name
            
            # Load audio data
            audio_data, sr = librosa.load(tmp_wav_path, sr=SAMPLE_RATE)
            
            # Debug information
            if st.session_state.debug_mode:
                duration = len(audio_data) / sr
                st.write(f"Recording duration: {duration:.2f} seconds")
                st.write(f"Audio shape: {audio_data.shape}")
                st.write(f"Sample rate: {sr} Hz")
                
                # Plot waveform
                fig, ax = plt.subplots(figsize=(10, 2))
                plt.plot(audio_data)
                plt.title("Recorded Waveform")
                plt.xlabel("Sample")
                plt.ylabel("Amplitude")
                st.pyplot(fig)
            
            # Display audio player
            st.audio(wav_audio_data, format='audio/wav')
            
            # Set flags for prediction
            st.session_state.new_recording = True
            st.session_state.should_predict = True
            
        except Exception as e:
            if st.session_state.debug_mode:
                st.error(f"Error processing recording: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            else:
                st.error("There was an error processing your recording. Please try again.")
            audio_data = None

elif audio_source == "Use Example Vowels":
    # Create columns for organizing the controls
    col1, col2 = st.columns([3, 1])
    
    # Function to load example vowels
    @st.cache_data
    def load_example_vowels(max_per_class=5):
        vowel_examples = {}
        target_vowels = ["aː", "eː", "oː", "ɑɪ", "æːɪ", "ɜɪ", "əʊ", "ɑʊ", "æːʊ"]
        
        if not os.path.exists(EXAMPLE_DIR):
            st.warning(f"Example vowels directory '{EXAMPLE_DIR}' not found.")
            return vowel_examples
        
        # First check for standardized example files
        files = os.listdir(EXAMPLE_DIR)
        standard_examples_found = False
        
        # Look for standard example files first (example_aː.wav, etc.)
        for vowel in target_vowels:
            standard_file = f"example_{vowel}.wav"
            if standard_file in files:
                if vowel not in vowel_examples:
                    vowel_examples[vowel] = []
                vowel_examples[vowel].append(os.path.join(EXAMPLE_DIR, standard_file))
                standard_examples_found = True
        
        # If no standard examples found, fall back to the original method
        if not standard_examples_found:
            for file in files:
                if file.endswith(".wav"):
                    try:
                        vowel = file.rsplit("_", 1)[-1].replace(".wav", "")
                        if vowel in target_vowels:
                            if vowel not in vowel_examples:
                                vowel_examples[vowel] = []
                            
                            if len(vowel_examples[vowel]) < max_per_class:
                                vowel_examples[vowel].append(os.path.join(EXAMPLE_DIR, file))
                    except Exception as e:
                        print(f"Error processing file {file}: {str(e)}")
        
        return vowel_examples
    
    vowel_examples = load_example_vowels()
    
    # Initialize variables that need to be accessible outside the column scope
    selected_vowel = None
    
    with col1:
        # First select vowel category
        if vowel_examples:
            vowel_categories = sorted(list(vowel_examples.keys()))
            selected_vowel = st.selectbox("Select vowel category", vowel_categories)
            
            # Then select specific example
            if selected_vowel and vowel_examples[selected_vowel]:
                example_files = vowel_examples[selected_vowel]
                
                # Use standard example if available
                standard_example = next((f for f in example_files if f.endswith(f"example_{selected_vowel}.wav")), None)
                
                if standard_example:
                    # If we have a standard example, just use that one
                    audio_file_path = standard_example
                    file_description = "Standard example"
                else:
                    # Otherwise let the user select
                    example_options = [os.path.basename(f) for f in example_files]
                    selected_example = st.selectbox("Select specific example", example_options)
                    
                    # Get the full path of the selected example
                    selected_idx = example_options.index(selected_example)
                    audio_file_path = example_files[selected_idx]
                    file_description = selected_example
                
                # Load the audio data
                audio_data, _ = librosa.load(audio_file_path, sr=SAMPLE_RATE)
                
                # Show additional info about the sample
                st.info(f"Example: {file_description}\nVowel: {selected_vowel}\nDuration: {len(audio_data)/SAMPLE_RATE:.3f}s")
        else:
            st.warning(f"No example vowels found. Make sure the '{EXAMPLE_DIR}' directory exists and contains WAV files.")
    
    with col2:
        # Display audio player in the second column if we have a selected file
        if audio_file_path:
            st.write("Play audio:")
            st.audio(audio_file_path, format='audio/wav')


# Prediction function
def predict_vowel(audio_data, components):
    try:
        debug_mode = st.session_state.get('debug_mode', False)
        
        # Make sure audio is not too short (silently pad)
        min_samples = SAMPLE_RATE * 0.25  # Minimum 250ms of audio
        if len(audio_data) < min_samples:
            if debug_mode:
                st.write(f"Audio duration: {len(audio_data)/SAMPLE_RATE:.3f}s (padding added)")
            padding = int(min_samples - len(audio_data))
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
        
        # Make sure audio is not too long (helps with performance)
        max_samples = SAMPLE_RATE * 2  # Maximum 2s of audio
        if len(audio_data) > max_samples:
            if debug_mode:
                st.write(f"Audio duration: {len(audio_data)/SAMPLE_RATE:.1f}s (trimming to center)")
            # Take the middle segment for more consistent predictions
            middle = len(audio_data) // 2
            half_window = int(min_samples // 2)
            audio_data = audio_data[middle-half_window:middle+half_window]
        
        # Extract features
        try:
            inputs = components["feature_extractor"](
                audio_data,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)
        except Exception as feat_error:
            if debug_mode:
                st.error(f"Error extracting features: {str(feat_error)}")
            # Try with a different approach - resample and pad
            audio_data = librosa.resample(audio_data, orig_sr=SAMPLE_RATE, target_sr=SAMPLE_RATE)
            audio_data = np.pad(audio_data, (0, SAMPLE_RATE), mode='constant')
            inputs = components["feature_extractor"](
                audio_data,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)

        # Run prediction
        with torch.no_grad():
            # Get model output
            logits = components["model"](**inputs).logits
            
            # Check if logits dimensions match expected number of classes
            model_num_classes = logits.shape[-1]
            expected_num_classes = components["num_classes"]
            
            if model_num_classes != expected_num_classes and debug_mode:
                st.warning(f"Model output dimension ({model_num_classes}) doesn't match expected classes ({expected_num_classes})")
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
            pred_id = np.argmax(probabilities)
            
            # Make sure pred_id is valid for our label encoder
            if pred_id < len(components["label_encoder"].classes_):
                pred_label = components["label_encoder"].inverse_transform([pred_id])[0]
            else:
                pred_label = f"unknown (class {pred_id})"
                if debug_mode:
                    st.warning(f"Model predicted class {pred_id} which is not in the label encoder")
                
            return pred_label, probabilities, components["label_encoder"].classes_
            
    except Exception as e:
        if debug_mode:
            st.error(f"Error during prediction: {str(e)}")
        else:
            st.error("An error occurred during prediction")
        # Return fallback values
        return "error", np.zeros(len(components["label_encoder"].classes_)), components["label_encoder"].classes_

# Save audio data to session state if it exists and it's a new recording
if audio_data is not None:
    if audio_source == "Record from microphone" and not st.session_state.get('new_recording', False):
        # Don't update audio data if it's not a new recording (prevents auto-prediction)
        pass
    else:
        st.session_state.audio_data = audio_data
        
        # For microphone recording, add a flag to indicate we need to run prediction
        if audio_source == "Record from microphone":
            st.session_state.should_predict = True
        
# Reset the new recording flag
if hasattr(st.session_state, 'new_recording'):
    st.session_state.new_recording = False

# Decide whether to run prediction
run_prediction = False

if audio_source == "Upload WAV file" and audio_data is not None:
    # Always run prediction for uploads
    run_prediction = True
elif audio_source == "Use Example Vowels" and audio_data is not None:
    # Always run prediction for examples
    run_prediction = True
elif audio_source == "Record from microphone":
    # For microphone, only run if we have the should_predict flag
    run_prediction = st.session_state.get('should_predict', False)
    if run_prediction:
        # Reset the flag after using it
        st.session_state.should_predict = False

# Debug output when debug mode is enabled
if st.session_state.get('debug_mode', False):
    st.write(f"Debug: audio_source = {audio_source}")
    st.write(f"Debug: audio_data is not None = {audio_data is not None}")
    st.write(f"Debug: run_prediction = {run_prediction}")
    st.write(f"Debug: hubert_components loaded = {hubert_components is not None}")
    if audio_source == "Use Example Vowels":
        st.write(f"Debug: selected_vowel = {selected_vowel if 'selected_vowel' in locals() else 'Not defined'}")

# Prediction logic
if run_prediction and (audio_data is not None or st.session_state.get('audio_data') is not None) and hubert_components:
    # Use audio data from session state if current audio_data is None
    prediction_audio = audio_data if audio_data is not None else st.session_state.audio_data
    
    # Run prediction
    pred_label, probabilities, classes = predict_vowel(prediction_audio, hubert_components)
    
    # Simplify results to just show the prediction with highest confidence
    # Ensure arrays are the same length
    if len(classes) != len(probabilities):
        # Adjust probabilities to match classes length
        if len(probabilities) > len(classes):
            probabilities = probabilities[:len(classes)]
        else:
            probabilities = np.pad(probabilities, (0, len(classes) - len(probabilities)), 'constant')
    
    # Find the highest confidence score
    max_confidence = np.max(probabilities)
    
    # Check the source of audio and determine the true label
    true_label = None
    
    # For example vowels
    if audio_source == "Use Example Vowels" and 'selected_vowel' in locals() and selected_vowel is not None:
        true_label = selected_vowel
    
    # Display results with comparison if we have a true label
    if true_label:
        is_correct = pred_label == true_label
        
        if is_correct:
            st.success(f"✅ Correct! Predicted Vowel: **{pred_label}** (Confidence: {max_confidence:.2%})")
        else:
            st.error(f"❌ Incorrect. Predicted: **{pred_label}** (Confidence: {max_confidence:.2%}), True label: **{true_label}**")
    else:
        st.success(f"🔊 Predicted Vowel: **{pred_label}** (Confidence: {max_confidence:.2%})")

# Add information about the model
with st.expander("📊 About the Model"):
    st.write("""
    ### HuBERT Model for Vowel Classification
    
    This app uses a fine-tuned version of the HuBERT (Hidden-Unit BERT) model to classify Luxembourgish vowels.
    
    **HuBERT** learns speech representations by predicting masked frames, similar to how BERT works for text.
    This approach is particularly effective for phonetic discrimination tasks like vowel classification.
    
    The model has been fine-tuned on a dataset of Luxembourgish vowel segments from the Schnëssen corpus,
    focusing on 9 different vowel categories.
    
    For more information about HuBERT, see:
    [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447)
    """)

# Add usage instructions
with st.expander("📝 How to Use"):
    st.write("""
    1. Choose an input source:
       - **Upload a WAV file** with a vowel recording
       - **Record from microphone** to record your own vowel live
       - **Use Example Vowels** to select from existing samples
    
    2. View the prediction and confidence score
    
    3. For example vowels, the app will show whether the prediction matches the true label
    
    4. Enable debug mode in the sidebar to see detailed processing information
    
    The confidence score shows how certain the model is about its prediction.
    """)
