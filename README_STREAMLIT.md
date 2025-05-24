---
title: Luxembourgish Vowel Classifier
emoji: 🎙️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.25.0
app_file: app_streamlit.py
pinned: false
---

# Luxembourgish Vowel Classifier

![Luxembourgish](https://img.shields.io/badge/language-Luxembourgish-red)
![HuBERT](https://img.shields.io/badge/model-HuBERT-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-blue)

An interactive app for classifying Luxembourgish vowels using a fine-tuned HuBERT model.

## About

This application allows you to upload audio of a Luxembourgish vowel sound and have it classified into one of 9 vowel categories. The underlying model is a fine-tuned HuBERT (Hidden-Unit BERT) speech model, which has been trained on a dataset of Luxembourgish vowel segments.

## Supported Vowels

The classifier can identify the following Luxembourgish vowels:

| IPA Symbol | Description          |
|------------|----------------------|
| aː         | Long a               |
| eː         | Long e               |
| oː         | Long o               |
| æːɪ        | Diphthong (æː + ɪ)   |
| æːʊ        | Diphthong (æː + ʊ)   |
| ɑɪ         | Diphthong (ɑ + ɪ)    |
| ɑʊ         | Diphthong (ɑ + ʊ)    |
| əʊ         | Diphthong (ə + ʊ)    |
| ɜɪ         | Diphthong (ɜ + ɪ)    |

## How to Use

1. **Option 1 - Upload:** Upload your own WAV file containing a vowel sound
   - WAV files should be clear recordings of a single vowel sound
   - Click the "Classify Vowel" button after uploading
   
2. **Option 2 - Use Examples:** Choose from our collection of example vowels
   - Select a vowel category from the dropdown menu
   - Choose a specific example to analyze
   
3. **View Results:**
   - The app will classify the vowel and show the predicted vowel type
   - You'll see confidence scores for all vowel classes, indicating the model's certainty
   - For example vowels, you'll also see if the prediction matches the true vowel

## Technical Details

- **Model:** Fine-tuned HuBERT (facebook/hubert-base-ls960)
- **Training Data:** Vowel segments (90-300ms) from the Schnëssen corpus
- **Input:** Audio recordings of vowel sounds
- **Output:** Vowel classification with confidence scores

## Credits

- Model trained by Prof. Dr. Peter Gilles, University of Luxembourg
- Data from the [Schnëssen](https://infolux.uni.lu/schnessen/) project
- HuBERT model developed by Facebook AI Research

## Citation

If you use this model in your research, please cite:

```
@misc{gilles2023luxembourgish,
  author = {Gilles, Peter},
  title = {Luxembourgish Vowel Classifier},
  year = {2023},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/spaces/pgilles/luxembourgish-vowel-classifier}}
}
```