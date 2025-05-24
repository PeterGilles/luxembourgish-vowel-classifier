# Setting Up and Deploying to Hugging Face Spaces

This document provides step-by-step instructions for setting up the Luxembourgish Vowel Classifier and deploying it to Hugging Face Spaces.

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements_hf.txt
```

### 2. Test the Gradio app locally

```bash
python gradio_app.py
```

This will start a local server, and you should be able to access the app in your browser at `http://127.0.0.1:7860/`.

## Deploying to Hugging Face Spaces

### 1. Create a new Space on Hugging Face

1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
2. Click on "Create new Space"
3. Enter details:
   - Owner: Your username (pgilles)
   - Space name: "luxembourgish-vowel-classifier"
   - License: Choose appropriate license (e.g., MIT)
   - SDK: Gradio
   - Visibility: Public (or as preferred)
4. Click "Create Space"

### 2. Prepare your files for upload

Make sure you have the following files ready in your local repository:
- `app.py`: Entry point for Hugging Face Spaces
- `gradio_app.py`: Main application code
- `requirements.txt`: Renamed from `requirements_hf.txt`
- `README.md`: Renamed from `README_HF.md`
- `.gitattributes`: For handling large files if you're including audio examples

### 3. Upload your files to Hugging Face

#### Option 1: Using Git

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/pgilles/luxembourgish-vowel-classifier

# Copy your files to the cloned directory
cp app.py gradio_app.py requirements_hf.txt README_HF.md .gitattributes /path/to/cloned/repo

# Rename files
cd /path/to/cloned/repo
mv requirements_hf.txt requirements.txt
mv README_HF.md README.md

# If you're including audio examples, copy them as well
cp -r exported_vowels /path/to/cloned/repo/

# Commit and push
git add .
git commit -m "Initial commit for Luxembourgish Vowel Classifier"
git push
```

#### Option 2: Using the Hugging Face Web Interface

1. Go to your newly created Space
2. Click on "Files and versions"
3. Upload each file individually using the web interface
4. Rename `requirements_hf.txt` to `requirements.txt`
5. Rename `README_HF.md` to `README.md`

### 4. Configure your Space (if needed)

1. Go to your Space settings
2. Adjust hardware requirements (if needed)
3. Set any environment variables (if needed)
4. Configure privacy settings

### 5. Build and Deploy

Hugging Face will automatically build and deploy your Space when you push your files. You can monitor the build process in the "Settings" tab.

## Notes

- Make sure your `app.py` is correctly importing from `gradio_app.py`
- The `requirements.txt` file must list all required packages
- The app might take a few minutes to start if it needs to download the model
- If you're having issues, check the Hugging Face Spaces documentation for more information