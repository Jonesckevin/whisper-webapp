#!/usr/bin/env python3

import subprocess
import sys
import os

def install_dependencies():
    """Install all required dependencies for the enhanced audio transcription script."""
    
    print("Installing dependencies for enhanced audio transcription...")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt not found in current directory!")
        return False
    
    try:
        # Install PyTorch with CUDA support (if available)
        print("Installing PyTorch with CUDA support...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "torch", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)
        
        # Install other requirements
        print("Installing other requirements...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("=" * 60)
        print("All dependencies installed successfully!")
        print("=" * 60)
        
        # Test if CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠ CUDA not available. Will use CPU processing.")
        except ImportError:
            print("⚠ Could not import torch to check CUDA status.")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = install_dependencies()
    if not success:
        sys.exit(1)
