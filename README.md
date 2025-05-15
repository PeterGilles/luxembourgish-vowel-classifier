# Luxembourgish Vowel Classifier

![Vowel Classifier Demo](https://img.shields.io/badge/demo-vowel_classifier-blue?style=for-the-badge)
![HuBERT Model](https://img.shields.io/badge/model-HuBERT-green?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge)

A deep learning-based vowel classification system for Luxembourgish vowels using the HuBERT model. This application can identify 9 Luxembourgish vowels from audio input.

## 🎙️ Features

- **Vowel Classification**: Identify 9 Luxembourgish vowels from audio input
- **Interactive Interface**: User-friendly Streamlit web application 
- **Multiple Input Methods**:
  - Upload WAV files
  - Record audio directly from microphone
  - Select from example vowels
  - Generate synthetic vowels with customizable formant frequencies
- **Visualization**: Clear visualization of confidence scores and predictions
- **Synthetic Vowel Generation**: Create vowel sounds based on customizable formant frequencies

## 📋 Supported Vowels

The classifier can identify the following Luxembourgish vowels:

| IPA Symbol | Description             |
|------------|-------------------------|
| aː         | Long a                  |
| eː         | Long e                  |
| oː         | Long o                  |
| æːɪ        | Diphthong (æː + ɪ)      |
| æːʊ        | Diphthong (æː + ʊ)      |
| ɑɪ         | Diphthong (ɑ + ɪ)       |
| ɑʊ         | Diphthong (ɑ + ʊ)       |
| əʊ         | Diphthong (ə + ʊ)       |
| ɜɪ         | Diphthong (ɜ + ɪ)       |

## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Setup

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/luxembourgish-vowel-classifier.git
   cd luxembourgish-vowel-classifier
   ```

2. **Create a virtual environment (recommended)**:
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Streamlit App

```
streamlit run streamlit_hubert.py
```

This will launch the application in your default web browser.

### Using the App

1. **Choose input method**:
   - Upload a WAV file
   - Record a vowel using your microphone
   - Select from example vowels
   - Generate a synthetic vowel

2. **View prediction results**:
   - The app will display the predicted vowel
   - Confidence scores for all vowel classes
   - Visual bar chart representation

3. **Synthetic Vowel Generation**:
   - Select predefined vowels or customize formants
   - Adjust F1, F2, and F3 frequencies
   - Control duration
   - Generate and test vowel sounds

## 🧠 Model Details

This project uses a fine-tuned HuBERT (Hidden-Unit BERT) model for vowel classification. HuBERT is a self-supervised speech representation model that learns by predicting masked units.

- **Base Model**: facebook/hubert-base-ls960
- **Training Data**: Vowel segments (90-300ms) from the Schnëssen corpus
- **Model Repository**: [pgilles/vowel-classifier-hubert](https://huggingface.co/pgilles/vowel-classifier-hubert)

## 🛠️ Technical Components

- **Fine-tuning Script**: `finetune_hubert_improved.py` for training custom models
- **Streamlit Application**: `streamlit_hubert.py` for interactive usage
- **Upload Script**: `upload_hubert.py` for pushing models to Hugging Face

## 📝 License

This project is released under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- The [Schnëssen](https://infolux.uni.lu/schnessen/) project for providing the audio corpus.
- The [HuggingFace Transformers](https://huggingface.co/transformers/) library for providing pre-trained models.
- The [Streamlit](https://streamlit.io/) team for their excellent interactive app framework.