import streamlit as st
import numpy as np
import torch
import librosa
import tempfile
import os
import pandas as pd
import altair as alt
import scipy.signal
import soundfile as sf
from scipy.io import wavfile
from transformers import (
    HubertForSequenceClassification, 
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor
)
import joblib
from st_audiorec import st_audiorec
from huggingface_hub import hf_hub_download

# SETTINGS
HUBERT_REPO_ID = "pgilles/vowel-classifier-hubert2"
WAV2VEC2_REPO_ID = "pgilles/vowel-classifier-wav2vec2"  # Update this with your wav2vec2 repo
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up page configuration
st.set_page_config(
    page_title="🎙 Vowel Classifier",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🎙 Vowel Classifier Comparison")
st.write("Compare HuBERT and Wav2Vec2 models for vowel classification. Upload a WAV file or record your vowel live.")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("Settings")
    compare_models = st.checkbox("Compare both models", value=True)
    
    if not compare_models:
        model_type = st.radio("Select model to use:", ["HuBERT", "Wav2Vec2"])
    else:
        st.info("Both models will be used for comparison")
        model_type = "both"
    
    # Show model information
    st.subheader("Model Information")
    st.write("**HuBERT**: Fine-tuned HuBERT model for vowel classification")
    st.write("**Wav2Vec2**: Fine-tuned Wav2Vec2 model for vowel classification")
    

# Model loading function with cache
@st.cache_resource
def load_model(model_type="hubert"):
    if model_type == "hubert":
        repo_id = HUBERT_REPO_ID
        model_class = HubertForSequenceClassification
    else:  # wav2vec2
        repo_id = WAV2VEC2_REPO_ID
        model_class = Wav2Vec2ForSequenceClassification
    
    try:
        # First load label encoder to determine number of classes
        label_encoder_path = hf_hub_download(repo_id=repo_id, filename="label_encoder.pkl")
        label_encoder = joblib.load(label_encoder_path)
        
        # Try to load model info if available
        try:
            model_info_path = hf_hub_download(repo_id=repo_id, filename="model_info.txt")
            with open(model_info_path, 'r') as f:
                model_info = {line.split(': ')[0]: line.split(': ')[1].strip() 
                             for line in f.readlines() if ': ' in line}
            st.sidebar.write(f"**{model_type.upper()} info:** {model_info.get('model_name', 'unknown')}")
        except:
            st.sidebar.write(f"No detailed info available for {model_type} model")
        
        # Download model
        model = model_class.from_pretrained(repo_id)
        model.eval()
        model.to(DEVICE)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(repo_id)
        
        return {
            "model": model,
            "feature_extractor": feature_extractor,
            "label_encoder": label_encoder,
            "num_classes": len(label_encoder.classes_)
        }
    except Exception as e:
        st.error(f"Error loading {model_type} model: {str(e)}")
        return None

# Load models based on selection
if model_type == "both" or model_type == "HuBERT":
    hubert_components = load_model("hubert")
    if hubert_components:
        st.sidebar.success("✅ HuBERT model loaded")

if model_type == "both" or model_type == "Wav2Vec2":
    wav2vec2_components = load_model("wav2vec2")
    if wav2vec2_components:
        st.sidebar.success("✅ Wav2Vec2 model loaded")

# Input source selection
audio_source = st.radio("Choose input source:", ["Upload WAV file", "Record from microphone", "Example vowels", "Synthetic vowels"])

audio_data = None
audio_file_path = None

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

elif audio_source == "Example vowels":
    # Create columns for organizing the controls
    col1, col2 = st.columns([3, 1])
    
    # Function to load example vowels from the training data
    @st.cache_data
    def load_example_vowels(max_per_class=5):
        vowel_examples = {}
        exported_vowels_dir = "exported_vowels"
        
        if os.path.exists(exported_vowels_dir):
            files = os.listdir(exported_vowels_dir)
            # Group by vowel type
            for file in files:
                if file.endswith(".wav"):
                    vowel = file.rsplit("_", 1)[-1].replace(".wav", "")
                    if vowel not in vowel_examples:
                        vowel_examples[vowel] = []
                    
                    if len(vowel_examples[vowel]) < max_per_class:
                        vowel_examples[vowel].append(os.path.join(exported_vowels_dir, file))
        else:
            st.warning(f"Example vowels directory '{exported_vowels_dir}' not found.")
        
        return vowel_examples
    
    vowel_examples = load_example_vowels()
    
    with col1:
        # First select vowel category
        if vowel_examples:
            vowel_categories = sorted(list(vowel_examples.keys()))
            selected_vowel = st.selectbox("Select vowel category", vowel_categories)
            
            # Then select specific example
            if selected_vowel and vowel_examples[selected_vowel]:
                example_files = vowel_examples[selected_vowel]
                example_options = [os.path.basename(f) for f in example_files]
                selected_example = st.selectbox("Select specific example", example_options)
                
                # Get the full path of the selected example
                selected_idx = example_options.index(selected_example)
                audio_file_path = example_files[selected_idx]
                
                # Load the audio data
                audio_data, _ = librosa.load(audio_file_path, sr=SAMPLE_RATE)
                
                # Show additional info about the sample
                st.info(f"Selected example: {selected_example}\nTrue label: {selected_vowel}\nDuration: {len(audio_data)/SAMPLE_RATE:.3f}s")
        else:
            st.warning("No example vowels found. Make sure the 'exported_vowels' directory exists and contains WAV files.")
    
    with col2:
        # Display audio player in the second column if we have a selected file
        if audio_file_path:
            st.write("Play audio:")
            st.audio(audio_file_path, format='audio/wav')

elif audio_source == "Synthetic vowels":
    st.subheader("Generate Synthetic Vowel")
    
    # Function to generate synthetic vowels based on formant frequencies
    def generate_synthetic_vowel(f1, f2, f3=None, duration=0.2, sample_rate=SAMPLE_RATE):
        """Generate a synthetic vowel using formant synthesis"""
        # Time axis
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Fundamental frequency (pitch) with slight variation for naturalness
        f0 = 120  # Base frequency in Hz (typical male voice)
        f0_variation = np.sin(2 * np.pi * 3 * t) * 5  # Slight vibrato
        f0_with_variation = f0 + f0_variation
        
        # Generate harmonics
        num_harmonics = 10
        signal = np.zeros_like(t)
        
        # Add harmonics with decreasing amplitude
        for i in range(1, num_harmonics + 1):
            harmonic_freq = f0_with_variation * i
            amplitude = 1.0 / i  # Amplitude decreases with harmonic number
            signal += amplitude * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Apply formant filters
        def resonator(signal, center_freq, bandwidth):
            nyquist = sample_rate / 2
            norm_center = center_freq / nyquist
            norm_bandwidth = bandwidth / nyquist
            b, a = scipy.signal.iirpeak(norm_center, norm_bandwidth)
            return scipy.signal.lfilter(b, a, signal)
        
        # Apply each formant filter
        signal = resonator(signal, f1, 80)
        signal = resonator(signal, f2, 100)
        if f3 is not None:
            signal = resonator(signal, f3, 120)
        
        # Apply fade in/out to avoid clicks
        fade_samples = int(0.01 * sample_rate)  # 10ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        signal[:fade_samples] *= fade_in
        signal[-fade_samples:] *= fade_out
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.9
        
        return signal
    
    # Define common vowel formant frequencies (in Hz)
    # These are approximate and can be adjusted
    vowel_formants = {
        "i": {"f1": 280, "f2": 2250, "f3": 2890, "name": "IPA: i (ee in 'deep')"},
        "I": {"f1": 400, "f2": 1920, "f3": 2560, "name": "IPA: ɪ (i in 'bit')"},
        "e": {"f1": 400, "f2": 2200, "f3": 2800, "name": "IPA: e (a in 'bait')"},
        "E": {"f1": 570, "f2": 1800, "f3": 2400, "name": "IPA: ɛ (e in 'bet')"},
        "ae": {"f1": 660, "f2": 1720, "f3": 2410, "name": "IPA: æ (a in 'bat')"},
        "a": {"f1": 730, "f2": 1090, "f3": 2440, "name": "IPA: a (a in 'caught')"},
        "u": {"f1": 300, "f2": 870, "f3": 2240, "name": "IPA: u (oo in 'boot')"},
        "U": {"f1": 440, "f2": 1020, "f3": 2240, "name": "IPA: ʊ (oo in 'foot')"},
        "o": {"f1": 450, "f2": 800, "f3": 2830, "name": "IPA: o (o in 'boat')"},
        "ai": {"f1": 750, "f2": 1400, "f3": 2780, "name": "IPA: aɪ (dipthong in 'eye')"}
    }
    
    # Add Luxembourg-specific vowels that match your dataset
    lux_vowels = {
        "aː": {"f1": 800, "f2": 1300, "f3": 2500, "name": "IPA: aː (Luxembourgish long a)"},
        "eː": {"f1": 380, "f2": 2300, "f3": 2800, "name": "IPA: eː (Luxembourgish long e)"},
        "oː": {"f1": 400, "f2": 750, "f3": 2600, "name": "IPA: oː (Luxembourgish long o)"},
        "æːɪ": {"f1": 600, "f2": 1850, "f3": 2500, "name": "IPA: æːɪ (Luxembourgish dipthong)"},
        "æːʊ": {"f1": 600, "f2": 1200, "f3": 2400, "name": "IPA: æːʊ (Luxembourgish dipthong)"},
        "ɑɪ": {"f1": 700, "f2": 1500, "f3": 2600, "name": "IPA: ɑɪ (Luxembourgish dipthong)"},
        "ɑʊ": {"f1": 700, "f2": 1100, "f3": 2500, "name": "IPA: ɑʊ (Luxembourgish dipthong)"},
        "əʊ": {"f1": 500, "f2": 1100, "f3": 2500, "name": "IPA: əʊ (Luxembourgish dipthong)"},
        "ɜɪ": {"f1": 550, "f2": 1700, "f3": 2450, "name": "IPA: ɜɪ (Luxembourgish dipthong)"}
    }
    
    # Combine all vowels, giving priority to Luxembourg-specific ones
    all_vowels = {**vowel_formants, **lux_vowels}
    
    # Create columns for controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Option to select from predefined vowels or customize
        vowel_option = st.radio("Select option:", ["Choose from predefined vowels", "Customize formants"])
        
        selected_vowel = None
        
        if vowel_option == "Choose from predefined vowels":
            # Group vowels by type
            vowel_groups = {
                "Luxembourgish Vowels": [k for k in lux_vowels.keys()],
                "Standard Vowels": [k for k in vowel_formants.keys()]
            }
            
            # Let user select vowel group first
            vowel_group = st.selectbox("Vowel group:", list(vowel_groups.keys()))
            
            # Then select specific vowel from that group
            vowel_list = vowel_groups[vowel_group]
            vowel_labels = [f"{v} - {all_vowels[v]['name']}" for v in vowel_list]
            selected_label = st.selectbox("Select vowel:", vowel_labels)
            
            # Extract vowel key from selected label
            selected_vowel = vowel_list[vowel_labels.index(selected_label)]
            vowel_data = all_vowels[selected_vowel]
            
            f1 = vowel_data["f1"]
            f2 = vowel_data["f2"]
            f3 = vowel_data.get("f3", None)
            
            # Display the selected formants
            st.info(f"Selected vowel: {selected_vowel}\nF1: {f1} Hz, F2: {f2} Hz, F3: {f3 if f3 else 'Not used'} Hz")
            
        else:  # Customize formants
            # Let user adjust formant frequencies
            f1 = st.slider("F1 (Hz)", 200, 1000, 500, 10)
            f2 = st.slider("F2 (Hz)", 600, 2500, 1500, 10)
            f3 = st.slider("F3 (Hz)", 2000, 3500, 2500, 10)
            use_f3 = st.checkbox("Apply F3", value=True)
            selected_vowel = "custom"
            
            if not use_f3:
                f3 = None
        
        # Duration control
        duration = st.slider("Duration (seconds)", 0.1, 0.5, 0.2, 0.05)
        
        # Generate button
        if st.button("Generate Vowel"):
            # Generate the synthetic vowel
            synthetic_audio = generate_synthetic_vowel(f1, f2, f3, duration, SAMPLE_RATE)
            
            # Save to a temporary file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, f"synthetic_vowel_{selected_vowel}.wav")
            sf.write(temp_path, synthetic_audio, SAMPLE_RATE)
            
            # Load as audio data for prediction
            audio_data = synthetic_audio
            audio_file_path = temp_path
            
            st.success(f"✅ Generated synthetic vowel{': ' + selected_vowel if selected_vowel != 'custom' else ''}")
    
    with col2:
        # Display audio player if file is generated
        if 'audio_file_path' in locals() and audio_file_path:
            st.write("Play audio:")
            st.audio(audio_file_path, format='audio/wav')

# Prediction function
def predict_vowel(audio_data, components):
    # Extract features
    inputs = components["feature_extractor"](
        audio_data,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    # Run prediction
    with torch.no_grad():
        logits = components["model"](**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        pred_id = np.argmax(probabilities)
        
        # Make sure we don't go out of bounds
        if pred_id < components["num_classes"]:
            pred_label = components["label_encoder"].inverse_transform([pred_id])[0]
        else:
            pred_label = "unknown"
            
        return pred_label, probabilities, components["label_encoder"].classes_

# Prediction logic
if audio_data is not None:
    if model_type == "both":
        # Create a two-column layout for comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("HuBERT Model")
            if hubert_components:
                pred_label, probabilities, classes = predict_vowel(audio_data, hubert_components)
                
                # Check if this is from an example vowel or synthetic vowel so we can show if prediction is correct
                if (audio_source == "Example vowels" or audio_source == "Synthetic vowels") and 'selected_vowel' in locals():
                    true_label = selected_vowel
                    is_correct = pred_label == true_label
                    
                    if is_correct:
                        st.success(f"✅ Correct! Predicted Vowel: **{pred_label}**")
                    else:
                        st.error(f"❌ Incorrect. Predicted: **{pred_label}**, True label: **{true_label}**")
                else:
                    st.success(f"🔊 Predicted Vowel: **{pred_label}**")
                
                # Create DataFrame for visualization
                df = pd.DataFrame({
                    'Vowel': classes,
                    'Confidence': probabilities
                })
                
                # Sort by confidence
                df = df.sort_values('Confidence', ascending=False)
                
                # Create bar chart with enhanced coloring for example or synthetic vowels
                if (audio_source == "Example vowels" or audio_source == "Synthetic vowels") and 'selected_vowel' in locals():
                    # Use a custom color scheme that highlights both prediction and true label
                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X('Confidence:Q', axis=alt.Axis(format='.0%')),
                        y=alt.Y('Vowel:N', sort='-x'),
                        color=alt.condition(
                            alt.datum.Vowel == pred_label,
                            alt.value('orange'),  # Predicted vowel
                            alt.condition(
                                alt.datum.Vowel == selected_vowel,
                                alt.value('green'),  # True vowel (if different from prediction)
                                alt.value('steelblue')  # Other vowels
                            )
                        ),
                        tooltip=['Vowel', alt.Tooltip('Confidence:Q', format='.2%')]
                    ).properties(
                        title='Confidence Scores (Orange: Predicted, Green: True)',
                        width=300
                    )
                else:
                    # Standard color scheme for non-example inputs
                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X('Confidence:Q', axis=alt.Axis(format='.0%')),
                        y=alt.Y('Vowel:N', sort='-x'),
                        color=alt.condition(
                            alt.datum.Vowel == pred_label,
                            alt.value('orange'),
                            alt.value('steelblue')
                        ),
                        tooltip=['Vowel', alt.Tooltip('Confidence:Q', format='.2%')]
                    ).properties(
                        title='Confidence Scores',
                        width=300
                    )
                
                st.altair_chart(chart, use_container_width=True)
                
                # Also show numeric values
                st.write("**Confidence Scores:**")
                for i, (vowel, conf) in enumerate(zip(df['Vowel'], df['Confidence'])):
                    st.write(f"**{vowel}**: {conf:.2%}")
        
        with col2:
            st.subheader("Wav2Vec2 Model")
            if wav2vec2_components:
                pred_label, probabilities, classes = predict_vowel(audio_data, wav2vec2_components)
                
                # Check if this is from an example vowel or synthetic vowel so we can show if prediction is correct
                if (audio_source == "Example vowels" or audio_source == "Synthetic vowels") and 'selected_vowel' in locals():
                    true_label = selected_vowel
                    is_correct = pred_label == true_label
                    
                    if is_correct:
                        st.success(f"✅ Correct! Predicted Vowel: **{pred_label}**")
                    else:
                        st.error(f"❌ Incorrect. Predicted: **{pred_label}**, True label: **{true_label}**")
                else:
                    st.success(f"🔊 Predicted Vowel: **{pred_label}**")
                
                # Create DataFrame for visualization
                df = pd.DataFrame({
                    'Vowel': classes,
                    'Confidence': probabilities
                })
                
                # Sort by confidence
                df = df.sort_values('Confidence', ascending=False)
                
                # Create bar chart with enhanced coloring for example or synthetic vowels
                if (audio_source == "Example vowels" or audio_source == "Synthetic vowels") and 'selected_vowel' in locals():
                    # Use a custom color scheme that highlights both prediction and true label
                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X('Confidence:Q', axis=alt.Axis(format='.0%')),
                        y=alt.Y('Vowel:N', sort='-x'),
                        color=alt.condition(
                            alt.datum.Vowel == pred_label,
                            alt.value('orange'),  # Predicted vowel
                            alt.condition(
                                alt.datum.Vowel == selected_vowel,
                                alt.value('green'),  # True vowel (if different from prediction)
                                alt.value('steelblue')  # Other vowels
                            )
                        ),
                        tooltip=['Vowel', alt.Tooltip('Confidence:Q', format='.2%')]
                    ).properties(
                        title='Confidence Scores (Orange: Predicted, Green: True)',
                        width=300
                    )
                else:
                    # Standard color scheme for non-example inputs
                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X('Confidence:Q', axis=alt.Axis(format='.0%')),
                        y=alt.Y('Vowel:N', sort='-x'),
                        color=alt.condition(
                            alt.datum.Vowel == pred_label,
                            alt.value('orange'),
                            alt.value('steelblue')
                        ),
                        tooltip=['Vowel', alt.Tooltip('Confidence:Q', format='.2%')]
                    ).properties(
                        title='Confidence Scores',
                        width=300
                    )
                
                st.altair_chart(chart, use_container_width=True)
                
                # Also show numeric values
                st.write("**Confidence Scores:**")
                for i, (vowel, conf) in enumerate(zip(df['Vowel'], df['Confidence'])):
                    st.write(f"**{vowel}**: {conf:.2%}")
    
    else:  # Single model view
        if model_type == "HuBERT" and hubert_components:
            pred_label, probabilities, classes = predict_vowel(audio_data, hubert_components)
        elif model_type == "Wav2Vec2" and wav2vec2_components:
            pred_label, probabilities, classes = predict_vowel(audio_data, wav2vec2_components)
        else:
            st.error(f"Selected model {model_type} is not available")
            st.stop()
        
        # Check if this is from an example vowel or synthetic vowel so we can show if prediction is correct
        if (audio_source == "Example vowels" or audio_source == "Synthetic vowels") and 'selected_vowel' in locals():
            true_label = selected_vowel
            is_correct = pred_label == true_label
            
            if is_correct:
                st.success(f"✅ Correct! Predicted Vowel: **{pred_label}**")
            else:
                st.error(f"❌ Incorrect. Predicted: **{pred_label}**, True label: **{true_label}**")
        else:
            st.success(f"🔊 Predicted Vowel: **{pred_label}**")
        
        # Create DataFrame for visualization
        df = pd.DataFrame({
            'Vowel': classes,
            'Confidence': probabilities
        })
        
        # Sort by confidence
        df = df.sort_values('Confidence', ascending=False)
        
        # Create bar chart with enhanced coloring for example or synthetic vowels
        if (audio_source == "Example vowels" or audio_source == "Synthetic vowels") and 'selected_vowel' in locals():
            # Use a custom color scheme that highlights both prediction and true label
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Confidence:Q', axis=alt.Axis(format='.0%')),
                y=alt.Y('Vowel:N', sort='-x'),
                color=alt.condition(
                    alt.datum.Vowel == pred_label,
                    alt.value('orange'),  # Predicted vowel
                    alt.condition(
                        alt.datum.Vowel == selected_vowel,
                        alt.value('green'),  # True vowel (if different from prediction)
                        alt.value('steelblue')  # Other vowels
                    )
                ),
                tooltip=['Vowel', alt.Tooltip('Confidence:Q', format='.2%')]
            ).properties(
                title='Confidence Scores (Orange: Predicted, Green: True Label)',
                width=600,
                height=400
            )
        else:
            # Standard color scheme for non-example inputs
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Confidence:Q', axis=alt.Axis(format='.0%')),
                y=alt.Y('Vowel:N', sort='-x'),
                color=alt.condition(
                    alt.datum.Vowel == pred_label,
                    alt.value('orange'),
                    alt.value('steelblue')
                ),
                tooltip=['Vowel', alt.Tooltip('Confidence:Q', format='.2%')]
            ).properties(
                title='Confidence Scores',
                width=600,
                height=400
            )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Also show numeric values
        st.write("**Confidence Scores:**")
        for i, (vowel, conf) in enumerate(zip(df['Vowel'], df['Confidence'])):
            st.write(f"**{vowel}**: {conf:.2%}")

# Add information about the models
with st.expander("📊 About the Models"):
    st.write("""
    ### HuBERT vs Wav2Vec2
    
    **HuBERT (Hidden-Unit BERT)**
    - Learns speech representations by predicting masked frames
    - Uses a BERT-like self-supervised approach
    - May perform better for phonetic discrimination tasks
    
    **Wav2Vec2**
    - Uses contrastive learning on raw audio
    - Learns to differentiate real vs. distracted audio samples
    - Often performs well for accent/dialect classification
    
    Both models are state-of-the-art for speech processing and have been fine-tuned specifically for vowel classification.
    """)

# Add usage instructions
with st.expander("📝 How to Use"):
    st.write("""
    1. Choose an input source (upload or record)
    2. Select whether to use a single model or compare both
    3. View the predictions and confidence scores
    4. Try different vowel sounds to compare accuracy
    
    The confidence scores show how certain each model is about its prediction.
    """)