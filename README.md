# Luxembourgish Vowel Classifier

This is a proof of concept to train an acoustic classifier for vowels. The ultimate aim is to use this in an app to assess pronunciations of language learners. The project focuses specifically on Luxembourgish vowels, which present unique challenges due to their distinctive phonetic characteristics and the language's complex vowel system. By leveraging state-of-the-art machine learning techniques, this classifier aims to provide accurate and real-time feedback to language learners, helping them improve their pronunciation of Luxembourgish vowels. The system is designed to be both educational and practical, serving as a valuable tool for language learning and phonetic research.
<img width="1259" alt="Bildschirmfoto 2025-05-24 um 22 11 59" src="https://github.com/user-attachments/assets/e688311e-0cde-4300-a94f-98e57f282fb9" />

An AI-powered vowel classification web application for Luxembourgish using a fine-tuned HuBERT transformer model. This Streamlit application provides real-time vowel classification with multiple input methods.


## 🎯 Features

- **HuBERT Transformer Model**: State-of-the-art speech recognition model fine-tuned for Luxembourgish vowels
- **Multiple Input Methods**:
  - 📁 Upload WAV audio files
  - 🎤 Record live from microphone
  - 📋 Use pre-existing example vowels
- **Real-time Classification**: Instant vowel prediction with confidence scores
- **Debug Mode**: Detailed audio processing visualization and information
- **User-friendly Interface**: Clean Streamlit web interface

## 🔊 Supported Vowels

The system classifies 9 Luxembourgish vowel categories:
- `aː` (long a)
- `eː` (long e) 
- `oː` (long o)
- `ɑɪ` 
- `æːɪ` 
- `ɜɪ` 
- `əʊ` 
- `ɑʊ` 
- `æːʊ` 

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PeterGilles/luxembourgish-vowel-classifier.git
cd luxembourgish-vowel-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run streamlit_hubert.py
```

The application will open in your browser at `http://localhost:8501`.

## 🖥️ How to Use

1. **Choose Input Method**:
   - **Upload WAV file**: Select a `.wav` file containing a vowel sound
   - **Record from microphone**: Click the record button and speak a vowel
   - **Use Example Vowels**: Select from pre-loaded vowel samples

2. **Get Prediction**: The app automatically classifies the vowel and shows:
   - Predicted vowel category
   - Confidence percentage
   - For examples: accuracy comparison with true label

3. **Debug Mode**: Enable in the sidebar for detailed information:
   - Audio waveform visualization
   - Processing parameters
   - Technical details

## 🔬 Model Details

- **Base Model**: Pre-trained HuBERT from Hugging Face (`facebook/hubert-base-ls960`)
- **Fine-tuning**: Custom sequence classification head for Luxembourgish vowels
- **Training Data**: 27,283 vowel segments from the Schnëssen corpus with the following distribution:
  - `aː`: 7,905 samples
  - `eː`: 4,812 samples  
  - `oː`: 3,703 samples
  - `ɜɪ`: 3,384 samples
  - `ɑɪ`: 2,588 samples
  - `æːɪ`: 1,924 samples
  - `əʊ`: 1,157 samples
  - `ɑʊ`: 1,001 samples
  - `æːʊ`: 809 samples
- **Sample Rate**: 16kHz
- **Input Length**: 90-300ms audio segments (max 250ms for training)
- **Training Configuration**:
  - **Epochs**: 8 (default)
  - **Batch Size**: 8
  - **Learning Rate**: 5e-5
  - **Optimizer**: AdamW with weight decay (0.01)
  - **Augmentation**: Time stretching (0-20% speed variation)
  - **Train/Validation Split**: 80/20
  - **Loss Function**: Cross-entropy with optional class weighting and label smoothing (0.1)
- **Audio Processing**:
  - **Feature Extraction**: Wav2Vec2FeatureExtractor with optimized parameters
  - **FFT Size**: 1024
  - **Hop Length**: 160 samples
  - **Window Length**: 400 samples
  - **Padding**: Max length with consistent strategy
- **Training Features**:
  - Automatic masking disabled for vowel classification
  - Class weighting option for imbalanced data
  - Best model selection based on validation accuracy
  - Confusion matrix and classification report generation

## 🎵 Audio Requirements

For best results:
- **File format**: WAV files
- **Duration**: At least 0.5 seconds of sustained vowel sound
- **Quality**: Clear recording without background noise
- **Content**: Single vowel sound (not diphthongs or consonants)

## 🛠️ Technical Architecture

The application uses:
- **Streamlit**: Web framework for the user interface
- **HuBERT**: Transformer-based speech representation model
- **Librosa**: Audio processing and analysis
- **PyTorch**: Deep learning framework
- **st_audiorec**: Browser-based audio recording

## 📞 Contact

Peter Gilles - [@PeterGilles](https://github.com/PeterGilles)

Project Link: [https://github.com/PeterGilles/luxembourgish-vowel-classifier](https://github.com/PeterGilles/luxembourgish-vowel-classifier)

## 🔗 Model

The fine-tuned HuBERT model is available on Hugging Face:
[pgilles/vowel-classifier-hubert](https://huggingface.co/pgilles/vowel-classifier-hubert)
