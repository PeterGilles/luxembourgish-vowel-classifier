#!/usr/bin/env python3
"""
Launcher script for the Luxembourgish Vowel Classifier Streamlit app.
This script provides a simple entry point to run the application.
"""

import os
import sys
import streamlit.web.cli as stcli

def main():
    """Run the Streamlit app."""
    # Get the absolute path to the app directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app", "streamlit_hubert.py")
    
    # Check if the app file exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find the app file at {app_path}")
        sys.exit(1)
    
    # Launch the Streamlit app
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()