# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a vowel classification system that uses machine learning to identify different vowel sounds in audio recordings. The project includes:

1. A data preparation script in R that exports vowel audio segments from a larger corpus
2. A PyTorch CNN-based vowel classifier
3. A fine-tuned HuBERT model (transformer-based) for vowel classification
4. A Streamlit web application for real-time vowel classification

## Key Files

- `vowel_classifier_prep.R`: R script for extracting vowel segments from a larger audio corpus
- `vowel_classifier.py`: Original CNN-based classifier using PyTorch
- `finetune_huBERT.py`: Script to fine-tune a pre-trained HuBERT model for vowel classification
- `streamlit_3.py`: Streamlit web application for interactive vowel classification
- `upload2HF.py`: Script to upload the trained model to Hugging Face Hub

## Commands

### Environment Setup

```bash
# Set up R environment with renv
Rscript -e "renv::restore()"

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
pip install -r requirements.txt  # If available, otherwise install packages manually
```

### Running the Vowel Classifier

```bash
# Run the basic CNN classifier
python vowel_classifier.py

# Fine-tune the HuBERT model
python finetune_huBERT.py

# Upload the model to Hugging Face
python upload2HF.py

# Run the Streamlit web application
streamlit run streamlit_3.py
```

## Data Workflow

1. **Data Preparation**: `vowel_classifier_prep.R` extracts vowel segments from a larger audio corpus using the emuR package and exports them as WAV files to the `exported_vowels` directory.

2. **Feature Extraction**: Both classifier approaches extract features from audio:
   - CNN model: Converts audio to Mel spectrograms
   - HuBERT model: Uses pre-trained audio transformers

3. **Model Training**:
   - `vowel_classifier.py`: Trains a CNN model with class balancing
   - `finetune_huBERT.py`: Fine-tunes a pre-trained HuBERT model from Hugging Face

4. **Model Deployment**:
   - The fine-tuned model is uploaded to Hugging Face Hub
   - The Streamlit app downloads and uses the model for inference

## Architecture Notes

- The project uses both traditional CNNs and transformer-based models for audio classification
- The Streamlit app supports both file upload and microphone recording for real-time classification
- The system classifies Luxembourgish vowels (e.g., "oː", "aː", "eː", "ɑɪ", "æːɪ", etc.)
- Class balancing is implemented to handle uneven distribution of vowel occurrences

## Dependencies

- Python libraries: PyTorch, librosa, transformers, streamlit, st_audiorec
- R packages: emuR, dplyr, reticulate, purrr, stringr

## Important Notes

- Audio data is sourced from the "exported_vowels" directory
- The model can be used via Streamlit or programmatically
- Fine-tuned models are saved locally before being uploaded to Hugging Face