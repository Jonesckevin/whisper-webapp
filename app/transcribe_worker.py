"""
Transcribe Worker Module for Whisper Transcription Web App

Handles the actual transcription work with:
- Progress tracking via segment callbacks
- Per-job model loading (saves VRAM)
- GPU memory cleanup after each job
- Results stored in SQLite database
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, '/app/src')

import torch
import whisper

from job_queue import job_queue
import db

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
UPLOAD_DIR = Path("/data/uploads")


def cleanup_gpu():
    """Clean up GPU memory after processing."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("🧹 GPU memory cleaned up")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("⚠️ Using CPU (no GPU available)")
    return device


def load_whisper_model(model_size: str):
    """Load Whisper model with optimizations."""
    device = get_device()
    
    logger.info(f"📥 Loading Whisper model: {model_size}")
    
    if device == "cuda":
        # RTX optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear any existing cache before loading
        cleanup_gpu()
    
    model = whisper.load_model(model_size, device=device)
    
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        logger.info(f"💾 Model loaded, VRAM used: {allocated:.2f}GB")
    
    return model


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_content(segments: list) -> str:
    """Generate SRT subtitle content from Whisper segments."""
    srt_lines = []
    subtitle_index = 1
    
    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()
        
        if not text:
            continue
        
        srt_lines.append(str(subtitle_index))
        srt_lines.append(f"{format_srt_timestamp(start_time)} --> {format_srt_timestamp(end_time)}")
        srt_lines.append(text)
        srt_lines.append("")
        
        subtitle_index += 1
    
    return "\n".join(srt_lines)


def process_job(job: dict):
    """
    Process a transcription job.
    
    This function is called by the job queue worker thread.
    It loads the model, transcribes the audio, and updates progress.
    Results are stored in the SQLite database.
    """
    model = None
    job_id = job['id']
    filename = job['filename']
    model_size = job['model']
    language = job['language']
    generate_srt = job.get('generate_srt', False)
    
    try:
        input_path = UPLOAD_DIR / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Check file size (ensure file is complete - not partial upload)
        file_size = input_path.stat().st_size
        if file_size == 0:
            raise ValueError("File is empty (possible partial upload)")
        
        logger.info(f"🎤 Starting transcription: {filename}")
        logger.info(f"   Model: {model_size}, Language: {language}, SRT: {generate_srt}")
        
        # Update progress
        job_queue.update_progress(job_id, 5, "Loading model...")
        
        # Load model (per-job loading to save VRAM when idle)
        model = load_whisper_model(model_size)
        
        job_queue.update_progress(job_id, 10, "Preparing audio...")
        
        # Transcription parameters
        use_fp16 = torch.cuda.is_available()
        transcribe_params = {
            "fp16": use_fp16,
            "language": language,
            "verbose": False,  # We'll track progress ourselves
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8),
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": False,
            "word_timestamps": True,
        }
        
        job_queue.update_progress(job_id, 15, "Transcribing audio...")
        
        # Run transcription
        start_time = time.time()
        result = model.transcribe(str(input_path), **transcribe_params)
        elapsed_time = time.time() - start_time
        
        job_queue.update_progress(job_id, 85, "Processing segments...")
        
        # Get segments
        segments = result.get("segments", [])
        logger.info(f"📊 Transcribed {len(segments)} segments in {elapsed_time:.1f}s")
        
        # Format transcript text
        formatted_lines = []
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            if text:
                start_fmt = f"{int(start // 3600):02d}:{int((start % 3600) // 60):02d}:{start % 60:06.3f}"
                end_fmt = f"{int(end // 3600):02d}:{int((end % 3600) // 60):02d}:{end % 60:06.3f}"
                formatted_lines.append(f"[{start_fmt} --> {end_fmt}]  {text}")
        
        transcript_text = "\n".join(formatted_lines)
        logger.info(f"📄 Transcript generated: {len(formatted_lines)} lines")
        
        # Generate SRT if requested
        srt_text = None
        if generate_srt:
            job_queue.update_progress(job_id, 92, "Generating subtitles...")
            srt_text = generate_srt_content(segments)
            logger.info(f"📺 SRT generated")
        
        job_queue.update_progress(job_id, 100, "Complete!")
        
        # Complete the job and store results in database
        db.complete_job(job_id, 'completed', transcript_text, srt_text)
        
        # Clean up uploaded file after successful transcription
        try:
            input_path.unlink()
            logger.info(f"🧹 Cleaned up upload file: {filename}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup upload file: {cleanup_error}")
        
        logger.info(f"✅ Job completed: {job_id}")
        
    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        logger.error(f"❌ Job failed: {job_id} - {error_msg}")
        db.complete_job(job_id, 'failed', error=error_msg)
    except torch.cuda.OutOfMemoryError:
        error_msg = "Out of GPU memory (VRAM). Try a smaller model or free up VRAM."
        logger.error(f"❌ Job failed: {job_id} - {error_msg}")
        db.complete_job(job_id, 'failed', error=error_msg)
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e):
            error_msg = f"GPU error: {str(e)}"
        else:
            error_msg = f"Runtime error: {str(e)}"
        logger.error(f"❌ Job failed: {job_id} - {error_msg}")
        db.complete_job(job_id, 'failed', error=error_msg)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ Job failed: {job_id} - {error_msg}")
        db.complete_job(job_id, 'failed', error=error_msg)
        
    finally:
        # Always clean up GPU memory
        del model
        cleanup_gpu()
