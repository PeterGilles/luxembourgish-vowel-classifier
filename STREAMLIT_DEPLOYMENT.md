# Deploying the Streamlit Version to Hugging Face Spaces

This guide provides step-by-step instructions for deploying the Streamlit version of the Luxembourgish Vowel Classifier to Hugging Face Spaces.

## Files Overview

For a Streamlit deployment, you'll need the following files:

1. **Main Files:**
   - `app_streamlit.py` - The entry point for HF Spaces
   - `streamlit_hubert.py` - The main Streamlit application
   - `requirements_streamlit.txt` (renamed to `requirements.txt` when deploying)
   - `README_STREAMLIT.md` (renamed to `README.md` when deploying)

2. **Example Vowels:**
   - `example_vowels/` directory with vowel examples
   - `.gitattributes` for handling large audio files

## Deployment Steps

### 1. Prepare Your Environment

Make sure you have:
- A Hugging Face account
- Git and Git LFS installed
- Hugging Face CLI installed (`pip install huggingface_hub`)

### 2. Create a Hugging Face Space

```bash
# Login to Hugging Face
huggingface-cli login

# Create a new Space (replace YOUR_USERNAME with your Hugging Face username)
huggingface-cli repo create luxembourgish-vowel-classifier-streamlit --type space --space-sdk streamlit
```

### 3. Clone and Set Up the Repository

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/luxembourgish-vowel-classifier-streamlit

# Move into the repository
cd luxembourgish-vowel-classifier-streamlit

# Initialize Git LFS for handling audio files
git lfs install
git lfs track "*.wav"
```

### 4. Copy Files to the Repository

```bash
# Copy the main application files
cp /path/to/app_streamlit.py app.py
cp /path/to/streamlit_hubert.py .
cp /path/to/requirements_streamlit.txt requirements.txt
cp /path/to/README_STREAMLIT.md README.md

# Create example_vowels directory and copy the example files
mkdir -p example_vowels
cp /path/to/example_vowels/*.wav example_vowels/

# Copy .gitattributes file
cp /path/to/.gitattributes .
```

### 5. Commit and Push

```bash
# Stage all files
git add .

# Commit changes
git commit -m "Initial deployment of Streamlit app with example vowels"

# Push to Hugging Face
git push
```

### 6. Verify the Deployment

- Visit your Space at: `https://huggingface.co/spaces/YOUR_USERNAME/luxembourgish-vowel-classifier-streamlit`
- Wait for the build to complete (this may take a few minutes)
- Test the application by:
  - Using the example vowels
  - Recording your own vowels
  - Uploading a vowel audio file

## Troubleshooting

If you encounter issues:

1. **Missing Dependencies:**
   - Check the Hugging Face Space logs
   - Make sure all required packages are in `requirements.txt`
   - The app requires `matplotlib` and other dependencies listed in `requirements_streamlit.txt`

2. **Example Vowels Not Found:**
   - Verify that the audio files were pushed correctly using Git LFS
   - Check the file paths in the app
   - Make sure the `example_vowels` directory exists and contains the example files

3. **Build Failures:**
   - Look at the build logs in the Hugging Face Space UI
   - Make sure your `app.py` file is correctly set up as the entry point
   - Check for syntax errors in the Python files

4. **Space Taking Too Long to Start:**
   - The first build might take some time, especially if it needs to download models
   - Be patient during the initial startup

5. **Audio File Upload Issues:**
   - Make sure you're uploading valid WAV files
   - Try shorter audio clips (1-2 seconds) of a sustained vowel sound
   - If a file doesn't work, try a different one or use the example vowels

6. **Audio Processing Errors:**
   - The app has built-in handling for short audio clips
   - If you see errors, try enabling debug mode in the sidebar
   - The app automatically pads short clips and trims very long ones

7. **Browser Compatibility:**
   - The app should work in all modern browsers
   - Chrome or Firefox are recommended for best performance
   - If you encounter issues, try clearing your browser cache

## Switching Between Gradio and Streamlit

You can have both versions running as separate Spaces on Hugging Face. Just make sure to:

1. Use different Space names (e.g., `luxembourgish-vowel-classifier-gradio` and `luxembourgish-vowel-classifier-streamlit`)
2. Use the correct configuration in the README.md file (see SDK configuration at the top)
3. Use the appropriate app entry point and requirements file

## Updating the Space

To update your Space after making changes:

1. Make your changes locally
2. Commit the changes: `git commit -m "Description of updates"`
3. Push to Hugging Face: `git push`