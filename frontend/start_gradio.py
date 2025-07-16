#!/usr/bin/env python3
"""
Simple startup script for the Gradio frontend
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import and run the Gradio app
from frontend.app import create_interface

if __name__ == "__main__":
    print("🎨 Starting Gradio frontend...")
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        debug=False,
        quiet=False
    ) 