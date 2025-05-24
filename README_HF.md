---
title: Luxembourgish Vowel Classifier
emoji: 🎙️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.0
app_file: app.py
pinned: false
---

# Luxembourgish Vowel Classifier

![Luxembourgish](https://img.shields.io/badge/language-Luxembourgish-red)
![HuBERT](https://img.shields.io/badge/model-HuBERT-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-blue)

An interactive app for classifying Luxembourgish vowels using a fine-tuned HuBERT model.

## About

This application allows you to record or upload audio of a Luxembourgish vowel sound and have it classified into one of 9 vowel categories. The underlying model is a fine-tuned HuBERT (Hidden-Unit BERT) speech model, which has been trained on a dataset of Luxembourgish vowel segments.

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

1. **Option 1 - Use an Example:** Click one of the example vowel buttons to load a pre-recorded example
2. **Option 2 - Record Your Own:** Click the "Record" button to record your vowel sound
3. **Option 3 - Upload:** Upload your own WAV file containing a vowel sound
4. The app will automatically classify the vowel and show the predicted vowel type
5. You'll see confidence scores for all vowel classes, indicating the model's certainty

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