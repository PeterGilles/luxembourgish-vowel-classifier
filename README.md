# Luxembourgish Vowel Classifier

An AI-powered vowel classification system for Luxembourgish using state-of-the-art machine learning models. This project combines traditional CNN architectures with modern transformer-based models (HuBERT) to classify vowel sounds with high accuracy.

## 🎯 Features

- **Multiple Model Architectures**: CNN-based and HuBERT transformer-based vowel classifiers
- **Interactive Web Interface**: Streamlit application with three input methods:
  - Upload WAV files
  - Record live from microphone
  - Use pre-existing example vowels
- **Real-time Classification**: Instant vowel classification with confidence scores
- **Debug Mode**: Detailed audio processing information and waveform visualization
- **Model Comparison**: Side-by-side comparison of different classifier approaches

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
- R (for data preparation)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PeterGilles/luxembourgish-vowel-classifier.git
cd luxembourgish-vowel-classifier
```

2. Set up Python environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
pip install -r requirements_streamlit.txt
```

3. Set up R environment (for data preparation):
```bash
Rscript -e "renv::restore()"
```

### Running the Applications

#### HuBERT-based Classifier (Recommended)
```bash
streamlit run streamlit_hubert.py
```

#### CNN-based Classifier
```bash
streamlit run streamlit_3.py
```

#### Model Comparison
```bash
streamlit run streamlit_compare.py
```

## 📁 Project Structure

```
luxembourgish-vowel-classifier/
├── streamlit_hubert.py          # Main HuBERT-based Streamlit app
├── streamlit_3.py               # CNN-based Streamlit app
├── streamlit_compare.py         # Model comparison app
├── vowel_classifier.py          # CNN model training script
├── finetune_huBERT.py          # HuBERT fine-tuning script
├── finetune_hubert_improved.py # Improved HuBERT training
├── vowel_classifier_prep.R     # R script for data preparation
├── upload2HF.py               # Hugging Face model upload
├── exported_vowels/           # Audio data directory
├── requirements_streamlit.txt # Python dependencies
├── requirements_hf.txt       # Hugging Face specific deps
└── README.md                 # This file
```

## 🔬 Model Architecture

### HuBERT Model
- **Base Model**: Pre-trained HuBERT from Hugging Face
- **Fine-tuning**: Sequence classification head for vowel categories
- **Features**: Self-supervised speech representation learning
- **Advantages**: Better generalization, state-of-the-art performance

### CNN Model
- **Architecture**: Convolutional layers with batch normalization
- **Input**: Mel spectrograms from audio
- **Features**: Traditional computer vision approach adapted for audio
- **Advantages**: Faster inference, smaller model size

## 🎵 Data Pipeline

1. **Audio Extraction**: `vowel_classifier_prep.R` extracts vowel segments from the Schnëssen corpus
2. **Feature Processing**: 
   - CNN: Converts audio to mel spectrograms
   - HuBERT: Uses pre-trained audio feature extractor
3. **Training**: Both models trained with class balancing for uneven vowel distributions
4. **Deployment**: Models uploaded to Hugging Face Hub for easy deployment

## 🖥️ Web Interface Features

### Input Methods
- **File Upload**: Support for WAV audio files
- **Live Recording**: Browser-based microphone recording using `st_audiorec`
- **Example Vowels**: Pre-loaded sample vowels for testing

### Output
- **Prediction**: Primary vowel classification
- **Confidence Score**: Model certainty percentage
- **Accuracy Check**: For example vowels, shows if prediction matches true label
- **Debug Information**: Optional detailed processing information

## 🛠️ Development

### Training New Models

#### CNN Model
```bash
python vowel_classifier.py
```

#### HuBERT Model
```bash
python finetune_huBERT.py
```

### Data Preparation
```bash
Rscript vowel_classifier_prep.R
```

### Model Upload to Hugging Face
```bash
python upload2HF.py
```

## 📊 Performance

The HuBERT-based model typically achieves higher accuracy due to its pre-training on large-scale speech data and self-supervised learning approach. The CNN model offers faster inference and is suitable for resource-constrained environments.

## 🔧 Configuration

### Model Settings
- **Sample Rate**: 16kHz (standardized for both models)
- **Audio Length**: 90-300ms vowel segments
- **Batch Size**: Configurable in training scripts
- **Learning Rate**: Optimized for each architecture

### Streamlit Configuration
- **File Watcher**: Disabled for torch compatibility
- **Debug Mode**: Toggle in sidebar for detailed information
- **Auto-prediction**: Enabled for seamless user experience

## 🚀 Deployment

### Local Development
Run any of the Streamlit applications locally for development and testing.

### Production Deployment
The applications can be deployed to:
- Streamlit Cloud
- Heroku
- Docker containers
- Any Python web hosting service

See deployment-specific documentation in:
- `DEPLOYMENT_INSTRUCTIONS.md`
- `STREAMLIT_DEPLOYMENT.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Schnëssen Corpus**: Training data source for Luxembourgish vowels
- **Hugging Face**: Pre-trained models and hosting platform
- **emuR Package**: R tools for phonetic data manipulation
- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework

## 📞 Contact

Peter Gilles - [@PeterGilles](https://github.com/PeterGilles)

Project Link: [https://github.com/PeterGilles/luxembourgish-vowel-classifier](https://github.com/PeterGilles/luxembourgish-vowel-classifier)

## 🔗 Related Models

- [HuBERT Model on Hugging Face](https://huggingface.co/pgilles/vowel-classifier-hubert)
- [Fine-tuned Models Collection](https://huggingface.co/pgilles)

---

**Note**: This project is part of research in computational phonetics and Luxembourgish language technology. For academic use, please cite appropriately.