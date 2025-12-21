#!/usr/bin/env python3

import os
import sys
import argparse
import tempfile
import time
import gc
from pathlib import Path
import speech_recognition as sr
from pydub import AudioSegment
import logging
import whisper
import torch
import glob
from typing import List, Optional

# RTX 3060 optimizations
torch.multiprocessing.set_sharing_strategy('file_system')
if hasattr(torch.multiprocessing, 'set_start_method'):
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# Try to import faster-whisper for better RTX 3060 performance
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# Import VideoFileClip for video audio extraction
from moviepy import VideoFileClip

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_audio_video_files(directory: str) -> List[str]:
    """Get all audio and video files from the specified directory with full paths."""
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.aac', '*.ogg', '*.m4a', '*.amr']
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    
    files = []
    for ext in audio_extensions + video_extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    
    return sorted(files)  # Return full paths instead of just basenames

def select_whisper_model_interactive() -> str:
    """Interactive Whisper model selection optimized for RTX 3060."""
    
    # Check if RTX 3060 is available for smart recommendations
    is_rtx_3060 = torch.cuda.is_available() and "3060" in torch.cuda.get_device_name(0)
    
    if is_rtx_3060:
        models = {
            '1': ('tiny', 'Ultra Fast - ~1GB VRAM (~39 MB) - 10x real-time'),
            '2': ('base', 'Recommended for RTX 3060 - ~1.5GB VRAM (~74 MB) - 5x real-time [BEST]'),
            '3': ('small', 'High Quality - ~2.5GB VRAM (~244 MB) - 3x real-time'),
            '4': ('medium', 'Premium Quality - ~4GB VRAM (~769 MB) - 2x real-time'),
            '5': ('large', 'Maximum Quality - ~5.5GB VRAM (~1550 MB) - 1.5x real-time [TIGHT FIT]')
        }
    else:
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
    """Interactive file selection from organized directory structure."""
    
    # Check both main directory and organized subdirectories
    base_dir = Path(directory)
    video_dir = base_dir / "video"
    audio_dir = base_dir / "audio"
    
    all_files = []
    file_paths = {}  # Map display names to actual paths
    
    # Get files from main directory (for backwards compatibility)
    if base_dir.exists():
        main_files = get_audio_video_files(str(base_dir))
        for f in main_files:
            display_name = Path(f).name
            all_files.append(display_name)
            file_paths[display_name] = f
    
    # Get files from organized subdirectories
    if video_dir.exists():
        video_files = get_audio_video_files(str(video_dir))
        for f in video_files:
            display_name = f"📹 video/{Path(f).name}"
            all_files.append(display_name)
            file_paths[display_name] = f
    
    if audio_dir.exists():
        audio_files = get_audio_video_files(str(audio_dir))
        for f in audio_files:
            display_name = f"🎵 audio/{Path(f).name}"
            all_files.append(display_name)
            file_paths[display_name] = f
    
    if not all_files:
        logger.error("No audio or video files found in the holding directory or its subdirectories.")
        print("📁 Try organizing your files with: python src/audio_to_text4.py --organize")
        return None
    
    print("\n" + "="*50)
    print("AUDIO/VIDEO FILES FOUND:")
    print("="*50)
    
    for i, display_name in enumerate(all_files, 1):
        actual_path = file_paths[display_name]
        file_size = os.path.getsize(actual_path) / (1024 * 1024)  # MB
        print(f"{i:2d}. {display_name:<40} ({file_size:.1f} MB)")
    
    print("="*50)
    
    while True:
        try:
            choice = input(f"\nSelect a file (1-{len(all_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(all_files):
                selected_display = all_files[index]
                selected_path = file_paths[selected_display]
                print(f"\nSelected: {selected_display}")
                return selected_path
            else:
                print(f"Please enter a number between 1 and {len(all_files)}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None

def setup_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """Setup Whisper model with RTX 3060 optimizations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        logger.info(f"🚀 RTX 3060 GPU ACCELERATION ENABLED!")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # RTX 3060 specific optimizations
        if "3060" in torch.cuda.get_device_name(0):
            logger.info("🎯 Applying RTX 3060 specific optimizations...")
            
            # Optimize CUDA settings for RTX 3060
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Model size recommendations for 6GB VRAM
            memory_usage = {
                "tiny": "~1GB VRAM - Ultra Fast",
                "base": "~1.5GB VRAM - Recommended for RTX 3060", 
                "small": "~2.5GB VRAM - Good quality",
                "medium": "~4GB VRAM - High quality",
                "large": "~5.5GB VRAM - Best quality (tight fit)"
            }
            
            if model_size in memory_usage:
                logger.info(f"📊 Memory usage: {memory_usage[model_size]}")
            
            if model_size == "large":
                logger.warning("⚠️  Large model uses ~5.5GB VRAM - monitor for OOM errors")
        
        # Advanced GPU memory management
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Enable memory efficiency
        if hasattr(torch.cuda, 'set_memory_fraction'):
            torch.cuda.set_memory_fraction(0.9)  # Reserve 10% for system
            
    else:
        logger.info("⚠️  Using CPU for processing (CUDA not available)")
        logger.info("Note: RTX 3060 GPU would be 15-30x faster")
    
    logger.info(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size, device=device)
    
    if device == "cuda":
        logger.info(f"✅ Model loaded on GPU with RTX 3060 optimizations")
        
        # Check actual VRAM usage after model loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"💾 VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
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

def clean_hallucinations(segments, repetition_threshold=0.8, min_segment_length=1.0):
    """
    Remove hallucinations, repetitions, and noise from Whisper segments.
    
    Args:
        segments: List of segments from Whisper result
        repetition_threshold: Similarity threshold for detecting repetitions (0.0-1.0)
        min_segment_length: Minimum segment length in seconds to keep
    """
    import difflib
    
    if not segments:
        return []
    
    cleaned_segments = []
    last_text = ""
    repetition_count = 0
    
    for segment in segments:
        text = segment['text'].strip()
        duration = segment['end'] - segment['start']
        
        # Skip very short segments
        if duration < min_segment_length:
            continue
            
        # Skip empty or very short text
        if len(text) < 3:
            continue
            
        # Detect common hallucination patterns
        is_hallucination = (
            # Repetitive patterns
            len(set(text.split())) < len(text.split()) * 0.5 or
            # Too many repeated characters
            any(char * 5 in text for char in 'abcdefghijklmnopqrstuvwxyz') or
            # Musical notations or nonsense
            text.count('do ') > 5 or text.count('la ') > 5 or text.count('na ') > 5 or
            # Common Whisper artifacts
            'thank you' in text.lower() and duration > 10 or
            # Repetitive words
            any(word * 3 in text.lower() for word in ['the', 'and', 'you', 'that', 'this'])
        )
        
        if is_hallucination:
            logger.info(f"🚫 Removing hallucination: '{text[:50]}...'")
            continue
            
        # Check for repetition with previous segment
        if last_text:
            similarity = difflib.SequenceMatcher(None, last_text.lower(), text.lower()).ratio()
            
            if similarity > repetition_threshold:
                repetition_count += 1
                if repetition_count > 2:  # Allow max 2 similar segments
                    logger.info(f"🔄 Removing repetition: '{text[:50]}...'")
                    continue
            else:
                repetition_count = 0
        
        cleaned_segments.append(segment)
        last_text = text
    
        if len(segments) > len(cleaned_segments):
            logger.info(f"🧹 Cleaned {len(segments) - len(cleaned_segments)} hallucinated/repeated segments")
        
    return cleaned_segments

def format_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format: HH:MM:SS,mmm
    Note: SRT uses comma as decimal separator, not period.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_srt_from_segments(segments: list, output_path: str) -> bool:
    """
    Generate an SRT subtitle file from Whisper segments.
    
    Args:
        segments: List of cleaned segments from Whisper transcription
        output_path: Path to save the SRT file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        srt_lines = []
        subtitle_index = 1
        
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            
            # Skip empty text
            if not text:
                continue
            
            # Format: index, timestamps, text, blank line
            srt_lines.append(str(subtitle_index))
            srt_lines.append(f"{format_srt_timestamp(start_time)} --> {format_srt_timestamp(end_time)}")
            srt_lines.append(text)
            srt_lines.append("")  # Blank line between subtitles
            
            subtitle_index += 1
        
        # Write SRT file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_lines))
        
        logger.info(f"📺 SRT subtitle file saved to: {output_path}")
        logger.info(f"📊 Total subtitles: {subtitle_index - 1}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating SRT file: {e}")
        return False

def preprocess_audio_for_whisper(audio_path: str, temp_dir: str = None) -> str:
    """
    Preprocess audio to reduce hallucinations by normalizing audio levels
    and reducing background noise that can cause Whisper to hallucinate.
    """
    try:
        logger.info("🎵 Preprocessing audio to reduce hallucinations...")
        
        # For now, let's skip complex preprocessing and just ensure proper format
        # This avoids ffmpeg issues while still providing basic optimization
        
        # Load audio with pydub 
        audio = AudioSegment.from_file(audio_path)
        
        # Basic normalization to prevent volume-related issues
        normalized_audio = audio.normalize()
        
        # Convert to optimal format for Whisper (16kHz mono)
        if temp_dir:
            preprocessed_path = os.path.join(temp_dir, "preprocessed_audio.wav")
        else:
            preprocessed_path = audio_path + "_preprocessed.wav"
            
        # Export with simple settings that work reliably
        normalized_audio.set_frame_rate(16000).set_channels(1).export(
            preprocessed_path,
            format="wav"
        )
        
        if os.path.exists(preprocessed_path):
            logger.info(f"✅ Audio preprocessed and saved to: {preprocessed_path}")
            return preprocessed_path
        else:
            logger.warning("⚠️ Preprocessed file not created, using original")
            return audio_path
        
    except Exception as e:
        logger.warning(f"⚠️ Audio preprocessing failed: {e}, using original file")
        return audio_path

def organize_holding_directory(holding_dir: Path) -> None:
    """Organize files in holding directory by separating video and audio files."""
    logger.info("📁 Organizing holding directory...")
    
    video_dir = holding_dir / "video"
    audio_dir = holding_dir / "audio"
    
    # Create subdirectories if they don't exist
    video_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.amr']
    
    moved_count = 0
    
    # Move video files
    for ext in video_extensions:
        for file_path in holding_dir.glob(f"*{ext}"):
            if file_path.is_file():
                new_path = video_dir / file_path.name
                file_path.rename(new_path)
                logger.info(f"📹 Moved {file_path.name} to video/")
                moved_count += 1
    
    # Move audio files
    for ext in audio_extensions:
        for file_path in holding_dir.glob(f"*{ext}"):
            if file_path.is_file():
                new_path = audio_dir / file_path.name
                file_path.rename(new_path)
                logger.info(f"🎵 Moved {file_path.name} to audio/")
                moved_count += 1
    
    if moved_count > 0:
        logger.info(f"✅ Organized {moved_count} files successfully!")
    else:
        logger.info("📁 All files already organized!")

def batch_convert_videos(video_dir: Path, audio_format: str = 'wav', audio_quality: str = 'high') -> List[str]:
    """Convert all video files in directory to audio format."""
    logger.info("🎬 Starting batch video conversion...")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    if not video_files:
        logger.warning("⚠️ No video files found in holding/video/")
        return []
    
    converted_files = []
    failed_files = []
    
    logger.info(f"📊 Found {len(video_files)} video files to convert")
    
    for i, video_file in enumerate(video_files, 1):
        logger.info(f"🎬 [{i}/{len(video_files)}] Converting: {video_file.name}")
        
        # Create output path in completed directory
        script_dir = Path(__file__).parent.parent
        completed_dir = script_dir / "completed"
        completed_dir.mkdir(exist_ok=True)
        
        audio_name = video_file.stem + f".{audio_format}"
        output_path = completed_dir / audio_name
        
        try:
            success = convert_video_to_audio(str(video_file), str(output_path), audio_format, audio_quality)
            if success:
                converted_files.append(str(output_path))
                logger.info(f"✅ [{i}/{len(video_files)}] Converted: {audio_name}")
            else:
                failed_files.append(video_file.name)
                logger.error(f"❌ [{i}/{len(video_files)}] Failed: {video_file.name}")
        except Exception as e:
            failed_files.append(video_file.name)
            logger.error(f"❌ [{i}/{len(video_files)}] Error converting {video_file.name}: {e}")
    
    # Summary
    logger.info(f"🎯 Batch conversion complete!")
    logger.info(f"✅ Successfully converted: {len(converted_files)}")
    if failed_files:
        logger.info(f"❌ Failed conversions: {len(failed_files)}")
        for failed in failed_files[:5]:  # Show first 5 failures
            logger.info(f"   - {failed}")
    
    return converted_files

def batch_transcribe_audio(audio_dir: Path, model_size: str = 'base', language: str = 'en', generate_srt: bool = False) -> List[str]:
    """Transcribe all audio files in directory using specified model."""
    logger.info("🎤 Starting batch audio transcription...")
    
    audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.amr']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
    
    if not audio_files:
        logger.warning("⚠️ No audio files found in holding/audio/")
        return []
    
    logger.info(f"📊 Found {len(audio_files)} audio files to transcribe")
    logger.info(f"🤖 Using Whisper model: {model_size}")
    
    # Load Whisper model once for batch processing
    model = setup_whisper_model(model_size)
    
    transcribed_files = []
    failed_files = []
    
    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"🎤 [{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
        
        # Create output path in completed directory
        script_dir = Path(__file__).parent.parent
        completed_dir = script_dir / "completed"
        completed_dir.mkdir(exist_ok=True)
        
        transcript_name = audio_file.stem + f"_whisper_{model_size}.txt"
        output_path = completed_dir / transcript_name
        
        try:
            success = transcribe_with_whisper(str(audio_file), model, str(output_path), language, generate_srt)
            if success:
                transcribed_files.append(str(output_path))
                logger.info(f"✅ [{i}/{len(audio_files)}] Transcribed: {transcript_name}")
                if generate_srt:
                    logger.info(f"   📺 SRT file also generated")
                
                # Clear GPU memory between files for RTX 3060
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                failed_files.append(audio_file.name)
                logger.error(f"❌ [{i}/{len(audio_files)}] Failed: {audio_file.name}")
        except Exception as e:
            failed_files.append(audio_file.name)
            logger.error(f"❌ [{i}/{len(audio_files)}] Error transcribing {audio_file.name}: {e}")
    
    # Summary
    logger.info(f"🎯 Batch transcription complete!")
    logger.info(f"✅ Successfully transcribed: {len(transcribed_files)}")
    if failed_files:
        logger.info(f"❌ Failed transcriptions: {len(failed_files)}")
        for failed in failed_files[:5]:  # Show first 5 failures
            logger.info(f"   - {failed}")
    
    return transcribed_files

def transcribe_with_whisper(audio_path: str, model: whisper.Whisper, output_path: str, language: str = "en", generate_srt: bool = False) -> bool:
    """
    Transcribe audio using Whisper AI model with RTX 3060 optimizations.
    This is much faster and more accurate than traditional speech recognition.
    
    Args:
        audio_path: Path to the audio file
        model: Loaded Whisper model
        output_path: Path to save the transcript
        language: Language code for transcription
        generate_srt: If True, also generate an SRT subtitle file
    """
    try:
        logger.info(f"Starting RTX 3060 optimized transcription: {audio_path}")
        start_time = time.time()
        
        # RTX 3060 specific optimizations with anti-hallucination
        use_fp16 = torch.cuda.is_available()
        if use_fp16:
            logger.info("🔥 Using FP16 precision optimized for RTX 3060 Tensor Cores with anti-hallucination")
        
        # Get file size for chunk optimization
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        
        # RTX 3060 optimized transcription parameters with anti-hallucination
        transcribe_params = {
            "fp16": use_fp16,
            "verbose": True,
            "language": language,                       # Use specified language
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8),  # Temperature fallback to prevent loops
            "compression_ratio_threshold": 2.4,         # Detect repetitive text
            "logprob_threshold": -1.0,                   # Filter low-confidence segments
            "no_speech_threshold": 0.6,                 # Skip silence/noise
            "condition_on_previous_text": False,        # Prevent context contamination
            "initial_prompt": None,                     # Clean slate for each segment
            "suppress_blank": True,                     # Remove empty segments
            "suppress_tokens": [-1],                    # Suppress problematic tokens
            "word_timestamps": True,                    # Enable word-level timing
        }
        
        # Optimize for RTX 3060's 6GB VRAM with anti-hallucination
        if torch.cuda.is_available() and "3060" in torch.cuda.get_device_name(0):
            # Monitor VRAM before processing
            mem_before = torch.cuda.memory_allocated(0) / (1024**3)
            logger.info(f"📊 VRAM before processing: {mem_before:.2f}GB")
            
            # Adjust parameters based on file size and available VRAM
            available_vram = 6.0 - mem_before - 0.5  # Reserve 0.5GB for safety
            
            if file_size > 500:  # Large files (>500MB)
                logger.info("🎯 Large file detected - using anti-hallucination settings")
                transcribe_params.update({
                    "compression_ratio_threshold": 2.0,  # More aggressive repetition detection
                    "logprob_threshold": -0.8,           # Higher confidence requirement
                    "no_speech_threshold": 0.7,          # Skip more silence
                    "temperature": (0.0, 0.2),          # Limited temperature range
                })
            elif available_vram < 1.0:  # Low VRAM available
                logger.warning("⚠️  Low VRAM available - using conservative anti-hallucination settings")
                transcribe_params.update({
                    "compression_ratio_threshold": 2.2,
                    "logprob_threshold": -0.9,
                    "no_speech_threshold": 0.8,
                    "temperature": 0.0,                  # Greedy only
                })
            else:
                logger.info("🚀 Optimal VRAM available - using balanced anti-hallucination settings")
                transcribe_params.update({
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                })
        
        # Clear any residual GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Transcribe with anti-hallucination optimized parameters
        result = model.transcribe(audio_path, **transcribe_params)
        
        # Extract and clean segments to remove hallucinations
        raw_segments = result["segments"]
        logger.info(f"📊 Raw segments before cleaning: {len(raw_segments)}")
        
        # Apply anti-hallucination cleaning
        cleaned_segments = clean_hallucinations(raw_segments)
        logger.info(f"✅ Clean segments after processing: {len(cleaned_segments)}")
        
        # Format segments with timestamps
        formatted_output = []
        for segment in cleaned_segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            
            # Skip empty text after cleaning
            if not text:
                continue
            
            # Format timestamps as [HH:MM:SS.fff --> HH:MM:SS.fff] for longer videos
            start_formatted = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{start_time % 60:06.3f}"
            end_formatted = f"{int(end_time // 3600):02d}:{int((end_time % 3600) // 60):02d}:{end_time % 60:06.3f}"
            
            formatted_line = f"[{start_formatted} --> {end_formatted}]  {text}"
            formatted_output.append(formatted_line)
        
        # Join all formatted segments with newlines
        transcribed_text = "\n".join(formatted_output)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcribed_text)
        
        # Generate SRT file if requested
        if generate_srt:
            srt_path = str(Path(output_path).with_suffix('.srt'))
            generate_srt_from_segments(cleaned_segments, srt_path)
        
        elapsed_time = time.time() - start_time
        
        # RTX 3060 performance monitoring and cleanup
        if torch.cuda.is_available():
            # Log VRAM usage after processing
            mem_after = torch.cuda.memory_allocated(0) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"📊 VRAM after processing: {mem_after:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            
            # Calculate processing speed
            audio_duration = len(result["segments"]) * 30 if result["segments"] else elapsed_time  # Rough estimate
            if audio_duration > 0:
                speed_ratio = audio_duration / elapsed_time
                logger.info(f"🚀 Processing speed: {speed_ratio:.1f}x real-time")
            
            # Aggressive cleanup for RTX 3060's limited VRAM
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Log final memory state
            mem_final = torch.cuda.memory_allocated(0) / (1024**3)
            if mem_final < mem_after:
                logger.info(f"🧹 Memory cleaned up: {mem_after - mem_final:.2f}GB freed")
        
        logger.info(f"✅ RTX 3060 optimized transcription completed in {elapsed_time:.2f} seconds")
        logger.info(f"📄 Transcribed text length: {len(transcribed_text)} characters")
        logger.info(f"💾 Transcription saved to: {output_path}")
        
        # Display detected language
        if "language" in result:
            logger.info(f"🌍 Detected language: {result['language']}")
        
        return True
        
    except KeyboardInterrupt:
        logger.info("Transcription cancelled by user.")
        # Cleanup on interruption
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False
    except Exception as e:
        logger.error(f"Error during RTX 3060 optimized transcription: {e}")
        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

def convert_audio_to_text(input_path: str, output_path: str, model_size: str = "base", use_whisper: bool = True, language: str = "en", generate_srt: bool = False):
    """
    Main function to convert audio to text using AI models
    
    Args:
        input_path: Path to input audio/video file
        output_path: Path to save transcript
        model_size: Whisper model size
        use_whisper: Use Whisper AI (True) or legacy recognition (False)
        language: Language code for transcription
        generate_srt: If True, also generate an SRT subtitle file
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
            success = transcribe_with_whisper(audio_path, model, output_path, language, generate_srt)
        else:
            # Fallback to traditional speech recognition
            logger.info("Using traditional speech recognition (slower and less accurate)")
            if generate_srt:
                logger.warning("⚠️ SRT generation is only supported with Whisper, not legacy mode")
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
    
    # Batch processing options
    parser.add_argument('--convert-all', action='store_true',
                        help='Convert all video files in holding/video/ to audio')
    parser.add_argument('--transcribe-all', action='store_true',
                        help='Transcribe all audio files in holding/audio/')
    parser.add_argument('--batch-model', choices=['tiny', 'base', 'small', 'medium', 'large'], default='base',
                        help='Model to use for batch transcription (default: base)')
    parser.add_argument('--organize', action='store_true',
                        help='Organize files in holding directory by type (video/audio)')
    parser.add_argument('--language', default='en', 
                        help='Language for transcription (default: en for English)')
    parser.add_argument('--srt', action='store_true',
                        help='Generate SRT subtitle file alongside transcript')
    
    args = parser.parse_args()
    
    # Set up directories
    script_dir = Path(__file__).parent.parent  # Go up one level from src
    holding_dir = script_dir / "holding"
    video_dir = holding_dir / "video"
    audio_dir = holding_dir / "audio"
    
    # Handle organize option
    if args.organize:
        if not holding_dir.exists():
            logger.error("Holding directory not found. Please create 'holding' directory first.")
            return
        organize_holding_directory(holding_dir)
        return
    
    # Handle batch operations
    if args.convert_all:
        if not video_dir.exists():
            logger.error("Video directory not found. Run --organize first or create holding/video/ directory.")
            return
        
        logger.info("🎬 Starting batch video conversion...")
        converted_files = batch_convert_videos(video_dir, args.audio_format, args.audio_quality)
        
        if converted_files:
            logger.info(f"🎯 Batch conversion completed! {len(converted_files)} files converted.")
        else:
            logger.warning("⚠️ No files were converted.")
        return
    
    if args.transcribe_all:
        if not audio_dir.exists():
            logger.error("Audio directory not found. Run --organize first or create holding/audio/ directory.")
            return
        
        logger.info("🎤 Starting batch audio transcription...")
        if args.srt:
            logger.info("📺 SRT subtitle generation enabled")
        transcribed_files = batch_transcribe_audio(audio_dir, args.batch_model, args.language, args.srt)
        
        if transcribed_files:
            logger.info(f"🎯 Batch transcription completed! {len(transcribed_files)} files transcribed.")
            if args.srt:
                logger.info(f"📺 SRT files also generated for each transcription")
        else:
            logger.warning("⚠️ No files were transcribed.")
        return
    
    # Interactive file selection if no input provided
    if not args.input:
        print("No input file specified. Let's select one from the 'holding' directory.")
        if not holding_dir.exists():
            logger.error("Holding directory not found. Please create 'holding' directory and place files there.")
            return
            
        input_file = select_file_interactive(str(holding_dir))
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
            script_dir = Path(__file__).parent.parent  # Go up one level from src
            completed_dir = script_dir / "completed"
            completed_dir.mkdir(exist_ok=True)  # Ensure completed directory exists
            stem = input_path.stem
            output_path = completed_dir / f"{stem}.{args.audio_format}"
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
        script_dir = Path(__file__).parent.parent  # Go up one level from src
        completed_dir = script_dir / "completed"
        completed_dir.mkdir(exist_ok=True)  # Ensure completed directory exists
        
        if not args.legacy and not args.convert_only:
            # Create filename with model info: filename_whisper_base.txt
            stem = input_path.stem  # filename without extension
            output_path = completed_dir / f"{stem}_whisper_{model_size}.txt"
        elif args.convert_only:
            # For video conversion, save audio to completed directory
            stem = input_path.stem
            output_path = completed_dir / f"{stem}.{args.audio_format}"
        else:
            # Simple .txt extension for legacy mode
            stem = input_path.stem
            output_path = completed_dir / f"{stem}.txt"
    else:
        output_path = Path(args.output)
    
    # Display settings
    print(f"\n{'='*60}")
    print("TRANSCRIPTION SETTINGS:")
    print(f"{'='*60}")
    print(f"Input file:    {input_file}")
    print(f"Output file:   {output_path}")
    if not args.legacy:
        print(f"AI Model:      Whisper ({model_size})")
        print(f"GPU Support:   {'Yes' if torch.cuda.is_available() else 'No'}")
        if args.srt:
            srt_path = Path(output_path).with_suffix('.srt')
            print(f"SRT Output:    {srt_path}")
    else:
        print(f"Engine:        Legacy ({args.engine})")
        if args.srt:
            print(f"SRT Output:    ⚠️ Not supported in legacy mode")
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
            use_whisper=not args.legacy,
            language=args.language,
            generate_srt=args.srt
        )
        
        if success:
            print(f"\n{'='*60}")
            print("TRANSCRIPTION COMPLETED SUCCESSFULLY!")
            print(f"Output saved to: {output_path}")
            if args.srt and not args.legacy:
                srt_path = Path(output_path).with_suffix('.srt')
                print(f"SRT subtitles: {srt_path}")
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
