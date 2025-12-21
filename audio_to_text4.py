#!/usr/bin/env python3

import os
import sys
import argparse
import tempfile
import time
from pathlib import Path
import speech_recognition as sr
from pydub import AudioSegment
import logging
import whisper
import torch
import glob
from typing import List, Optional

# Import VideoFileClip for video audio extraction
from moviepy import VideoFileClip

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_audio_video_files(directory: str) -> List[str]:
    """Get all audio and video files from the specified directory."""
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.aac', '*.ogg', '*.m4a', '*.amr']
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    
    files = []
    for ext in audio_extensions + video_extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    
    return sorted([os.path.basename(f) for f in files])

def select_whisper_model_interactive() -> str:
    """Interactive Whisper model selection."""
    models = {
        '1': ('tiny', 'Fastest, lowest quality (~39 MB)'),
        '2': ('base', 'Balanced speed/quality (~74 MB) [Default]'),
        '3': ('small', 'Good quality, slower (~244 MB)'),
        '4': ('medium', 'Better quality, much slower (~769 MB)'),
        '5': ('large', 'Best quality, very slow (~1550 MB)')
    }
    
    print("\n" + "="*60)
    print("SELECT WHISPER MODEL SIZE:")
    print("="*60)
    
    for key, (model, description) in models.items():
        print(f" {key}. {model:8} - {description}")
    
    print("="*60)
    
    while True:
        choice = input("Select model (1-5) or press Enter for default (base): ").strip()
        
        if choice == "":
            return "base"
        elif choice in models:
            selected_model = models[choice][0]
            print(f"\nSelected: {selected_model}")
            return selected_model
        else:
            print("Invalid choice. Please select 1-5 or press Enter for default.")

def select_file_interactive(directory: str) -> Optional[str]:
    """Interactive file selection from available audio/video files."""
    files = get_audio_video_files(directory)
    
    if not files:
        logger.error("No audio or video files found in the current directory.")
        return None
    
    print("\n" + "="*50)
    print("AUDIO/VIDEO FILES FOUND:")
    print("="*50)
    
    for i, file in enumerate(files, 1):
        file_path = os.path.join(directory, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"{i:2d}. {file:<30} ({file_size:.1f} MB)")
    
    print("="*50)
    
    while True:
        try:
            choice = input(f"\nSelect a file (1-{len(files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(files):
                selected_file = files[index]
                print(f"\nSelected: {selected_file}")
                return os.path.join(directory, selected_file)
            else:
                print(f"Please enter a number between 1 and {len(files)}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None

def setup_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """Setup Whisper model with GPU acceleration if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        logger.info(f"🚀 GPU ACCELERATION ENABLED!")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Clear GPU cache for optimal performance
        torch.cuda.empty_cache()
    else:
        logger.info("⚠️  Using CPU for processing (CUDA not available)")
        logger.info("Note: GPU processing would be significantly faster")
    
    logger.info(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size, device=device)
    
    if device == "cuda":
        logger.info(f"✅ Model loaded on GPU successfully")
    
    return model

def extract_audio_from_video(video_path, temp_dir):
    """Extract audio from video file and save as WAV with GPU acceleration."""
    logger.info(f"Extracting audio from video file: {video_path}")
    
    # Use GPU acceleration if available
    try:
        video = VideoFileClip(str(video_path))
        temp_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        video.audio.write_audiofile(
            temp_audio_path,
            codec='pcm_s16le'
        )
        video.close()  # Properly close the video file
        return temp_audio_path
    except Exception as e:
        logger.error(f"Error extracting audio from video: {e}")
        raise

def convert_amr_to_wav(amr_path, temp_dir):
    """Convert AMR file to WAV format for better compatibility with Whisper."""
    logger.info(f"Converting AMR file to WAV: {amr_path}")
    
    try:
        # Load AMR file using pydub (requires ffmpeg for AMR support)
        audio = AudioSegment.from_file(amr_path, format="amr")
        
        # Convert to WAV format with standard parameters optimized for speech recognition
        temp_audio_path = os.path.join(temp_dir, "converted_amr_audio.wav")
        audio.export(
            temp_audio_path,
            format="wav",
            parameters=["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"]  # 16kHz, mono, 16-bit
        )
        
        logger.info(f"AMR conversion completed: {temp_audio_path}")
        return temp_audio_path
        
    except Exception as e:
        error_msg = f"Error converting AMR file: {e}"
        if "ffmpeg" in str(e).lower():
            error_msg += "\nNote: AMR file support requires FFmpeg to be installed and accessible."
            error_msg += "\nPlease install FFmpeg from https://ffmpeg.org/download.html"
        logger.error(error_msg)
        raise

def convert_video_to_audio(input_path: str, output_path: str, audio_format: str = "wav", audio_quality: str = "high") -> bool:
    """
    Convert video file to audio file with specified format and quality.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output audio file
        audio_format: Output audio format (wav, mp3, flac, aac)
        audio_quality: Quality setting (low, medium, high)
    """
    try:
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            logger.error(f"Input video file does not exist: {input_path}")
            return False
        
        logger.info(f"Converting video to audio: {input_path} -> {output_path}")
        logger.info(f"Format: {audio_format.upper()}, Quality: {audio_quality}")
        
        # Load video file
        video = VideoFileClip(str(input_path))
        audio = video.audio
        
        if audio is None:
            logger.error("Video file contains no audio track")
            video.close()
            return False
        
        # Quality settings
        quality_settings = {
            "low": {"bitrate": "64k", "ar": 22050},
            "medium": {"bitrate": "128k", "ar": 44100}, 
            "high": {"bitrate": "320k", "ar": 48000}
        }
        
        settings = quality_settings.get(audio_quality, quality_settings["high"])
        
        # Format-specific parameters
        if audio_format.lower() == "mp3":
            audio.write_audiofile(
                str(output_path),
                codec='mp3',
                bitrate=settings["bitrate"],
                ffmpeg_params=["-ar", str(settings["ar"])]
            )
        elif audio_format.lower() == "wav":
            audio.write_audiofile(
                str(output_path),
                codec='pcm_s16le',  # Uncompressed WAV
                ffmpeg_params=["-ar", str(settings["ar"])]
            )
        elif audio_format.lower() == "flac":
            audio.write_audiofile(
                str(output_path),
                codec='flac',
                ffmpeg_params=["-ar", str(settings["ar"])]
            )
        elif audio_format.lower() == "aac":
            audio.write_audiofile(
                str(output_path),
                codec='aac',
                bitrate=settings["bitrate"],
                ffmpeg_params=["-ar", str(settings["ar"])]
            )
        else:
            logger.error(f"Unsupported audio format: {audio_format}")
            video.close()
            return False
        
        # Clean up
        video.close()
        
        # Get file sizes for logging
        input_size = input_path.stat().st_size / (1024 * 1024)  # MB
        output_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        logger.info(f"Conversion completed successfully!")
        logger.info(f"Input size: {input_size:.2f} MB")
        logger.info(f"Output size: {output_size:.2f} MB")
        logger.info(f"Compression ratio: {(output_size/input_size)*100:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting video to audio: {e}")
        return False

def transcribe_with_whisper(audio_path: str, model: whisper.Whisper, output_path: str) -> bool:
    """
    Transcribe audio using Whisper AI model with GPU acceleration.
    This is much faster and more accurate than traditional speech recognition.
    """
    try:
        logger.info(f"Starting Whisper transcription of: {audio_path}")
        start_time = time.time()
        
        # CUDA optimization settings
        use_fp16 = torch.cuda.is_available()
        if use_fp16:
            logger.info("🔥 Using FP16 precision for faster GPU processing")
        
        # Transcribe with Whisper with optimal settings
        result = model.transcribe(
            audio_path,
            fp16=use_fp16,  # Use FP16 for GPU acceleration
            verbose=True,
            # Additional optimization parameters
            temperature=0,  # Use greedy decoding for better performance
            best_of=1,     # Single pass for speed
            beam_size=1    # Single beam for speed
        )
        
        # Extract segments with timestamps from result
        segments = result["segments"]
        
        # Format segments with timestamps
        formatted_output = []
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            
            # Format timestamps as [MM:SS.fff --> MM:SS.fff]
            start_formatted = f"{int(start_time//60):02d}:{start_time%60:06.3f}"
            end_formatted = f"{int(end_time//60):02d}:{end_time%60:06.3f}"
            
            formatted_line = f"[{start_formatted} --> {end_formatted}]  {text}"
            formatted_output.append(formatted_line)
        
        # Join all formatted segments with newlines
        transcribed_text = "\n".join(formatted_output)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcribed_text)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Whisper transcription completed in {elapsed_time:.2f} seconds")
        logger.info(f"Transcribed text length: {len(transcribed_text)} characters")
        logger.info(f"Transcription saved to: {output_path}")
        
        # Display detected language
        if "language" in result:
            logger.info(f"Detected language: {result['language']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during Whisper transcription: {e}")
        return False

def process_large_audio_fallback(audio_path, output_path, engine="google", chunk_size_ms=60000):
    """
    Process large audio files by splitting into time-based chunks
    and transcribing each chunk.
    """
    recognizer = sr.Recognizer()
    
    # Load the audio file
    logger.info(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    
    # Get audio properties
    duration_sec = len(audio) / 1000
    logger.info(f"Audio duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")
    
    # Calculate the number of chunks
    num_chunks = len(audio) // chunk_size_ms + (1 if len(audio) % chunk_size_ms > 0 else 0)
    logger.info(f"Splitting audio into {num_chunks} chunks of {chunk_size_ms/1000} seconds each")
    
    # Process each chunk
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for i in range(num_chunks):
            chunk_start = i * chunk_size_ms
            chunk_end = min((i + 1) * chunk_size_ms, len(audio))
            
            logger.info(f"Processing chunk {i+1}/{num_chunks} ({chunk_start/1000:.1f}s to {chunk_end/1000:.1f}s)")
            
            # Extract chunk
            audio_chunk = audio[chunk_start:chunk_end]
            
            # Export chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                chunk_filename = tmp_file.name
            
            audio_chunk.export(chunk_filename, format="wav")
            
            # Transcribe the chunk
            with sr.AudioFile(chunk_filename) as source:
                audio_data = recognizer.record(source)
                
                try:
                    if engine == "google":
                        text = recognizer.recognize_google(audio_data)
                    elif engine == "sphinx":
                        text = recognizer.recognize_sphinx(audio_data)
                    # Add more engines as needed
                    
                    if text:
                        logger.info(f"Transcribed text length: {len(text)} characters")
                        
                        # Write immediately to file to save progress
                        outfile.write(text + " ")
                        outfile.flush()
                    else:
                        logger.warning("No text transcribed for this chunk")
                
                except sr.UnknownValueError:
                    logger.warning("Speech recognition could not understand audio in this chunk")
                except sr.RequestError as e:
                    logger.error(f"Could not request results from speech recognition service: {e}")
                except Exception as e:
                    logger.error(f"Error during transcription: {e}")
            
            # Clean up the temporary file
            try:
                os.unlink(chunk_filename)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {chunk_filename}: {e}")

def convert_audio_to_text(input_path: str, output_path: str, model_size: str = "base", use_whisper: bool = True):
    """
    Main function to convert audio to text using AI models
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        logger.error(f"Error: Input file '{input_path}' does not exist.")
        return False
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Check file type and handle accordingly
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        is_video = input_path.suffix.lower() in video_extensions
        is_amr = input_path.suffix.lower() == '.amr'
        
        audio_path = str(input_path)
        if is_video:
            audio_path = extract_audio_from_video(input_path, temp_dir)
        elif is_amr:
            audio_path = convert_amr_to_wav(input_path, temp_dir)
        
        start_time = time.time()
        logger.info(f"Starting transcription of '{input_path}'...")
        
        # Get file size in MB
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        if use_whisper:
            # Use Whisper AI model (recommended)
            model = setup_whisper_model(model_size)
            success = transcribe_with_whisper(audio_path, model, output_path)
        else:
            # Fallback to traditional speech recognition
            logger.info("Using traditional speech recognition (slower and less accurate)")
            success = process_large_audio_fallback(audio_path, output_path)
        
        elapsed_time = time.time() - start_time
        if success:
            logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
            logger.info(f"Transcription saved to '{output_path}'")
        else:
            logger.error("Transcription failed!")
    
    return success

def main():
    parser = argparse.ArgumentParser(
        description='Convert audio or video to text using AI models, or extract audio from video',
        epilog='Supported formats: MP3, WAV, FLAC, AAC, OGG, M4A, AMR (audio) | MP4, AVI, MOV, MKV, FLV, WMV (video)'
    )
    parser.add_argument('input', nargs='?', help='Input audio or video file path (optional - will prompt if not provided)')
    parser.add_argument('-o', '--output', help='Output file path (text for transcription, audio for conversion)')
    parser.add_argument('-m', '--model', choices=['tiny', 'base', 'small', 'medium', 'large'], 
                       default='base', help='Whisper model size (default: base)')
    parser.add_argument('--legacy', action='store_true', 
                       help='Use legacy speech recognition instead of Whisper')
    parser.add_argument('-e', '--engine', choices=['google', 'sphinx'], default='google',
                       help='Speech recognition engine for legacy mode (default: google)')
    
    # Video conversion options
    parser.add_argument('--convert-only', action='store_true',
                       help='Convert video to audio only (no transcription)')
    parser.add_argument('--audio-format', choices=['wav', 'mp3', 'flac', 'aac'], default='wav',
                       help='Output audio format for video conversion (default: wav)')
    parser.add_argument('--audio-quality', choices=['low', 'medium', 'high'], default='high',
                       help='Audio quality for conversion (default: high)')
    
    args = parser.parse_args()
    
    # Interactive file selection if no input provided
    if not args.input:
        print("No input file specified. Let's select one from the current directory.")
        current_dir = os.getcwd()
        input_file = select_file_interactive(current_dir)
        if not input_file:
            logger.info("No file selected. Exiting.")
            return
    else:
        input_file = args.input
    
    # Handle video-to-audio conversion mode
    if args.convert_only:
        input_path = Path(input_file)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        if input_path.suffix.lower() not in video_extensions:
            logger.error("Video conversion mode requires a video file input")
            return
        
        # Set default output path for audio conversion
        if not args.output:
            stem = input_path.stem
            output_path = input_path.parent / f"{stem}.{args.audio_format}"
        else:
            output_path = Path(args.output)
        
        # Display conversion settings
        print(f"\n{'='*60}")
        print("VIDEO TO AUDIO CONVERSION:")
        print(f"{'='*60}")
        print(f"Input video:    {input_file}")
        print(f"Output audio:   {output_path}")
        print(f"Audio format:   {args.audio_format.upper()}")
        print(f"Audio quality:  {args.audio_quality.title()}")
        print(f"{'='*60}")
        
        # Confirm before proceeding
        confirm = input("\nProceed with conversion? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("Conversion cancelled by user.")
            return
        
        # Perform conversion
        success = convert_video_to_audio(input_file, output_path, args.audio_format, args.audio_quality)
        
        if success:
            print(f"\n{'='*60}")
            print("CONVERSION COMPLETED SUCCESSFULLY!")
            print(f"Audio saved to: {output_path}")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("CONVERSION FAILED!")
            print("Check the logs above for error details.")
            print(f"{'='*60}")
        
        return
    
    # Interactive model selection if using Whisper and model not specified via command line
    model_size = args.model
    if not args.legacy:
        # Check if model was explicitly set via command line (not just default)
        if len([arg for arg in sys.argv if arg in ['-m', '--model']]) == 0:
            model_size = select_whisper_model_interactive()
        else:
            model_size = args.model
    
    # Set default output path if not provided
    if not args.output:
        input_path = Path(input_file)
        if not args.legacy:
            # Create filename with model info: filename_whisper_base.txt
            stem = input_path.stem  # filename without extension
            output_path = input_path.parent / f"{stem}_whisper_{model_size}.txt"
        else:
            # Simple .txt extension for legacy mode
            output_path = input_path.with_suffix('.txt')
    else:
        output_path = args.output
    
    # Display settings
    print(f"\n{'='*60}")
    print("TRANSCRIPTION SETTINGS:")
    print(f"{'='*60}")
    print(f"Input file:    {input_file}")
    print(f"Output file:   {output_path}")
    if not args.legacy:
        print(f"AI Model:      Whisper ({model_size})")
        print(f"GPU Support:   {'Yes' if torch.cuda.is_available() else 'No'}")
    else:
        print(f"Engine:        Legacy ({args.engine})")
    print(f"{'='*60}")
    
    # Confirm before proceeding
    confirm = input("\nProceed with transcription? (y/n): ").strip().lower()
    if confirm != 'y':
        logger.info("Transcription cancelled by user.")
        return
    
    try:
        # Start transcription
        success = convert_audio_to_text(
            input_file, 
            output_path, 
            model_size=model_size,
            use_whisper=not args.legacy
        )
        
        if success:
            print(f"\n{'='*60}")
            print("TRANSCRIPTION COMPLETED SUCCESSFULLY!")
            print(f"Output saved to: {output_path}")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("TRANSCRIPTION FAILED!")
            print("Check the logs above for error details.")
            print(f"{'='*60}")
    except KeyboardInterrupt:
        print("\nTranscription cancelled by user.")
        logger.info("Transcription cancelled by user.")
        return

if __name__ == "__main__":
    # Enhanced CUDA support check
    print("\n" + "="*50)
    print("🖥️  SYSTEM STATUS")
    print("="*50)
    
    if torch.cuda.is_available():
        print("🚀 CUDA ACCELERATION: ENABLED")
        print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        print("⚡ Whisper will use GPU acceleration for faster processing!")
    else:
        print("⚠️  CUDA ACCELERATION: DISABLED")
        print("💻 Processing will use CPU only")
        print("📝 Note: Install CUDA-enabled PyTorch for GPU acceleration")
        
    print("="*50)
    main()
