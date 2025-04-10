# story_generator_standalone.py
# This file imports and reuses all functionality from cd.py
# It provides a single entry point that replicates the original monolithic behavior

# Import all functions and variables from the original cd.py file
from cd import (
    # API and configuration
    safety_settings, 
    
    # Core generation functions
    generate_prompt,
    retry_api_call,
    retry_story_generation,
    generate,
    collect_complete_story,
    
    # Google Drive utilities
    download_file_from_google_drive,
    test_google_drive_api,
    upload_text_file_to_drive,
    
    # Metadata and asset generation
    generate_seo_metadata,
    default_seo_metadata,
    generate_thumbnail
)

import os
import sys
import tempfile
import base64
import datetime
import json
import re
import numpy as np
import soundfile as sf
from IPython.display import display, HTML, Image, Audio
from PIL import Image as PILImage
import subprocess

# Main function that replicates the original cd.py execution
def main():
    """Execute the story generation pipeline just like the original cd.py"""
    print("--- Starting generation (attempting 16:9 via prompt) ---")
    
    # Use the same approach as original cd.py - calling retry_story_generation directly
    retry_story_generation(use_prompt_generator=True)
    
    print("--- Generation function finished ---")

# Check API Key
if __name__ == "__main__":
    api_key_check = os.environ.get("GEMINI_API_KEY")
    if not api_key_check:
        print("🛑 ERROR: Environment variable GEMINI_API_KEY is not set.")
        print("💡 TIP: Uncomment and set your API key above, or run this in a cell before running this script:")
        print("    os.environ['GEMINI_API_KEY'] = 'YOUR_API_KEY_HERE'")
        raise ValueError("API Key not found in environment.")
    else:
        print(f"✅ Found API Key: ...{api_key_check[-4:]}")
        
    # Execute the main function
    main()
