#!/usr/bin/env python3

import os
import sys
import platform
import subprocess
import zipfile
import shutil
import tempfile
from pathlib import Path

def is_ffmpeg_installed():
    """Check if ffmpeg is already installed and available in PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=False
        )
        return True
    except FileNotFoundError:
        return False

def download_file(url, target_path):
    """Download a file from URL to the target path."""
    import requests
    
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return target_path

def install_ffmpeg_windows():
    """Download and install FFmpeg for Windows."""
    # Using gyan.dev builds which are well-maintained
    ffmpeg_url = "https://github.com/GyanD/codexffmpeg/releases/download/6.0/ffmpeg-6.0-essentials_build.zip"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the zip file
        zip_path = os.path.join(temp_dir, "ffmpeg.zip")
        download_file(ffmpeg_url, zip_path)
        
        # Extract the zip file
        print("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the bin directory
        extracted_dir = next(Path(temp_dir).glob("ffmpeg-*"))
        bin_dir = extracted_dir / "bin"
        
        # Create the destination directory if it doesn't exist
        script_dir = Path(__file__).parent.absolute()
        ffmpeg_dir = script_dir / "ffmpeg"
        ffmpeg_dir.mkdir(exist_ok=True)
        
        # Copy the files to the destination
        for file in bin_dir.glob("*"):
            shutil.copy(file, ffmpeg_dir)
        
        print(f"FFmpeg installed to: {ffmpeg_dir}")
        
        # Add to PATH for the current session
        os.environ["PATH"] += os.pathsep + str(ffmpeg_dir.absolute())
        
        # Suggest adding to permanent PATH
        print("\nTo use FFmpeg from any command prompt, add this directory to your PATH:")
        print(f"{ffmpeg_dir.absolute()}")
        print("\nCommand to add to PATH (run in an administrator Command Prompt):")
        print(f'setx /M PATH "%PATH%;{ffmpeg_dir.absolute()}"')

def install_ffmpeg_linux():
    """Install FFmpeg using the system package manager on Linux."""
    if shutil.which("apt-get"):  # Debian/Ubuntu
        print("Installing FFmpeg via apt...")
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "ffmpeg"], check=True)
    elif shutil.which("yum"):  # CentOS/RHEL
        print("Installing FFmpeg via yum...")
        subprocess.run(["sudo", "yum", "install", "-y", "ffmpeg"], check=True)
    elif shutil.which("dnf"):  # Fedora
        print("Installing FFmpeg via dnf...")
        subprocess.run(["sudo", "dnf", "install", "-y", "ffmpeg"], check=True)
    elif shutil.which("pacman"):  # Arch Linux
        print("Installing FFmpeg via pacman...")
        subprocess.run(["sudo", "pacman", "-S", "--noconfirm", "ffmpeg"], check=True)
    else:
        print("Could not detect package manager. Please install FFmpeg manually.")
        print("Try: https://ffmpeg.org/download.html")
        return False
    return True

def install_ffmpeg_macos():
    """Install FFmpeg using Homebrew on macOS."""
    if shutil.which("brew"):
        print("Installing FFmpeg via Homebrew...")
        subprocess.run(["brew", "install", "ffmpeg"], check=True)
    else:
        print("Homebrew not found. Installing Homebrew first...")
        brew_install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        subprocess.run(brew_install_cmd, shell=True, check=True)
        print("Installing FFmpeg via Homebrew...")
        subprocess.run(["brew", "install", "ffmpeg"], check=True)
    return True

def main():
    # Check if FFmpeg is already installed
    if is_ffmpeg_installed():
        print("FFmpeg is already installed and available in your PATH.")
        return

    print("FFmpeg not found. Installing FFmpeg...")
    
    # Install FFmpeg based on operating system
    system = platform.system()
    if system == "Windows":
        try:
            import requests
        except ImportError:
            print("Installing required dependency: requests")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
            import requests
        install_ffmpeg_windows()
    elif system == "Linux":
        install_ffmpeg_linux()
    elif system == "Darwin":  # macOS
        install_ffmpeg_macos()
    else:
        print(f"Unsupported operating system: {system}")
        print("Please install FFmpeg manually: https://ffmpeg.org/download.html")
        return
    
    # Verify installation
    if is_ffmpeg_installed():
        print("\nFFmpeg installed successfully!")
        # Show the version of FFmpeg
        subprocess.run(["ffmpeg", "-version"])
    else:
        print("\nFFmpeg installation may have failed or it's not in your PATH.")
        print("Please install FFmpeg manually: https://ffmpeg.org/download.html")

if __name__ == "__main__":
    main()
