# Luxembourgish Vowel Classifier

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
- `ɑɪ` (ai diphthong)
- `æːɪ` (long ae + i)
- `ɜɪ` (schwa + i)
- `əʊ` (schwa + ou)
- `ɑʊ` (a + ou)
- `æːʊ` (long ae + ou)

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

- **Base Model**: Pre-trained HuBERT from Hugging Face
- **Fine-tuning**: Custom sequence classification head for Luxembourgish vowels
- **Training Data**: Vowel segments from the Schnëssen corpus
- **Sample Rate**: 16kHz
- **Input Length**: 90-300ms audio segments

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

## 📁 Files

- `streamlit_hubert.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `exported_vowels/`: Example vowel audio files
- `README.md`: This documentation

## 🚀 Deployment

The application can be deployed to:
- **Streamlit Cloud**: Connect your GitHub repository
- **Heroku**: Use the provided configuration
- **Docker**: Build and run in containers
- **Local**: Run on any machine with Python

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Schnëssen Corpus**: Source of training data for Luxembourgish vowels
- **Hugging Face**: Pre-trained HuBERT model and hosting platform
- **Streamlit**: Web application framework
- **Research Community**: Advances in speech recognition and phonetics

## 📞 Contact

Peter Gilles - [@PeterGilles](https://github.com/PeterGilles)

Project Link: [https://github.com/PeterGilles/luxembourgish-vowel-classifier](https://github.com/PeterGilles/luxembourgish-vowel-classifier)

## 🔗 Model

The fine-tuned HuBERT model is available on Hugging Face:
[pgilles/vowel-classifier-hubert](https://huggingface.co/pgilles/vowel-classifier-hubert)

---

**Note**: This application is part of computational phonetics research for Luxembourgish language technology.