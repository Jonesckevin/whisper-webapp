#!/usr/bin/env python3

import subprocess
import sys
import os

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("Installing Whisper and dependencies...")
    
    # Core dependencies
    dependencies = [
        "torch",
        "openai-whisper"  # This is the correct package name for Whisper
    ]
    
    for dep in dependencies:
        install_package(dep)
    
    # Verify installation
    try:
        import whisper
        print("Whisper successfully installed!")
        
        # Check CUDA availability
        import torch
        if torch.cuda.is_available():
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Whisper will use CPU only.")
            
    except ImportError as e:
        print(f"Error: {e}")
        print("Installation failed. Please try installing manually:")
        print("pip install openai-whisper")

if __name__ == "__main__":
    main()
