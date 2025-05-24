#!/usr/bin/env python

# This is a simple entry point for Hugging Face Spaces
# It imports and launches the Gradio app

from gradio_app import demo

# Launch the demo with Gradio's default settings for Spaces
if __name__ == "__main__":
    demo.launch()