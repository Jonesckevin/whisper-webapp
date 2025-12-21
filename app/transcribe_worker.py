"""
Transcribe Worker Module for Whisper Transcription Web App

Handles the actual transcription work with:
- Progress tracking via segment callbacks
- Per-job model loading (saves VRAM)
- GPU memory cleanup after each job
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

from job_queue import Job, JobStatus, job_queue

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
UPLOAD_DIR = Path("/data/uploads")
COMPLETED_DIR = Path("/data/completed")


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


def generate_srt_file(segments: list, output_path: str) -> bool:
    """Generate SRT subtitle file from Whisper segments."""
    try:
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
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_lines))
        
        logger.info(f"📺 SRT file saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating SRT: {e}")
        return False


def process_job(job: Job):
    """
    Process a transcription job.
    
    This function is called by the job queue worker thread.
    It loads the model, transcribes the audio, and updates progress.
    """
    model = None
    
    try:
        input_path = UPLOAD_DIR / job.filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Check file size (ensure file is complete - not partial upload)
        file_size = input_path.stat().st_size
        if file_size == 0:
            raise ValueError("File is empty (possible partial upload)")
        
        logger.info(f"🎤 Starting transcription: {job.filename}")
        logger.info(f"   Model: {job.model}, Language: {job.language}, SRT: {job.generate_srt}")
        
        # Update progress
        job_queue.update_progress(job.id, 5, "Loading model...")
        
        # Load model (per-job loading to save VRAM when idle)
        model = load_whisper_model(job.model)
        
        job_queue.update_progress(job.id, 10, "Preparing audio...")
        
        # Transcription parameters
        use_fp16 = torch.cuda.is_available()
        transcribe_params = {
            "fp16": use_fp16,
            "language": job.language,
            "verbose": False,  # We'll track progress ourselves
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8),
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": False,
            "word_timestamps": True,
        }
        
        job_queue.update_progress(job.id, 15, "Transcribing audio...")
        
        # Run transcription
        start_time = time.time()
        result = model.transcribe(str(input_path), **transcribe_params)
        elapsed_time = time.time() - start_time
        
        job_queue.update_progress(job.id, 85, "Processing segments...")
        
        # Get segments
        segments = result.get("segments", [])
        logger.info(f"📊 Transcribed {len(segments)} segments in {elapsed_time:.1f}s")
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(job.filename).stem
        output_filename = f"{base_name}_whisper_{job.model}_{timestamp}.txt"
        output_path = COMPLETED_DIR / output_filename
        
        # Format and save transcript
        formatted_lines = []
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            if text:
                start_fmt = f"{int(start // 3600):02d}:{int((start % 3600) // 60):02d}:{start % 60:06.3f}"
                end_fmt = f"{int(end // 3600):02d}:{int((end % 3600) // 60):02d}:{end % 60:06.3f}"
                formatted_lines.append(f"[{start_fmt} --> {end_fmt}]  {text}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(formatted_lines))
        
        job.output_file = output_filename
        logger.info(f"📄 Transcript saved: {output_filename}")
        
        # Generate SRT if requested
        if job.generate_srt:
            job_queue.update_progress(job.id, 92, "Generating subtitles...")
            srt_filename = f"{base_name}_whisper_{job.model}_{timestamp}.srt"
            srt_path = COMPLETED_DIR / srt_filename
            if generate_srt_file(segments, str(srt_path)):
                job.srt_file = srt_filename
        
        job_queue.update_progress(job.id, 100, "Complete!")
        job.status = JobStatus.COMPLETED
        
        logger.info(f"✅ Job completed: {job.id}")
        
    except Exception as e:
        logger.error(f"❌ Job failed: {job.id} - {str(e)}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        
    finally:
        # Always clean up GPU memory
        del model
        cleanup_gpu()
