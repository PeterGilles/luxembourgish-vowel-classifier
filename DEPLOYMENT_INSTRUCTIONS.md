# Luxembourgish Vowel Classifier - Deployment Instructions

This document provides instructions for deploying the Luxembourgish Vowel Classifier to Hugging Face Spaces.

## Files Overview

1. **Main Application Files:**
   - `app.py` - Entry point for Hugging Face Spaces
   - `gradio_app.py` - Main Gradio application logic
   - `requirements_hf.txt` - Dependencies for Hugging Face Spaces
   - `README_HF.md` - Hugging Face README with Space configuration

2. **Example Files:**
   - `example_vowels/` - Directory containing example WAV files for each vowel
   - `setup.py` and `MANIFEST.in` - Ensure example files are included in deployment
   - `.gitattributes` - Handles WAV files correctly in Git

## Deployment Steps

### 1. Prepare for Deployment

Make sure you have:
- Created a Hugging Face account (if you don't have one already)
- Installed Git and Git LFS to handle the audio files
- Installed the Hugging Face CLI (`pip install huggingface_hub`)

### 2. Initialize the Space

```bash
# Login to Hugging Face
huggingface-cli login

# Create a new Space
huggingface-cli repo create luxembourgish-vowel-classifier --type space --space-sdk gradio
```

### 3. Set Up Git Repository

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/luxembourgish-vowel-classifier

# Move into the repository
cd luxembourgish-vowel-classifier

# Initialize Git LFS
git lfs install
```

### 4. Copy Files to the Repository

```bash
# Copy the main application files
cp /path/to/app.py .
cp /path/to/gradio_app.py .
cp /path/to/requirements_hf.txt requirements.txt  # Note the name change
cp /path/to/README_HF.md README.md  # Note the name change

# Copy example files
mkdir -p example_vowels
cp /path/to/example_vowels/*.wav example_vowels/

# Copy gitattributes
cp /path/to/.gitattributes .
```

### 5. Commit and Push

```bash
# Add all files
git add .

# Commit changes
git commit -m "Initial app deployment with example vowels"

# Push to Hugging Face
git push
```

### 6. Verify Deployment

- Go to https://huggingface.co/spaces/YOUR_USERNAME/luxembourgish-vowel-classifier
- Wait for the build to complete
- Test the application with the example vowels and recording feature

## Troubleshooting

If you encounter issues with the examples not being found:

1. Check that the example files are correctly pushed to Hugging Face
2. Verify the path in `gradio_app.py` for finding examples
3. Look at the Spaces logs for any error messages
4. If needed, update the app and push changes again

## Updating the Space

To make updates to your Space:

1. Make changes to the files locally
2. Commit the changes (`git commit -m "Description of changes"`)
3. Push to Hugging Face (`git push`)