"""
Flask Web Application for Whisper Transcription

Features:
- File upload (max 5GB via web, larger files via network share)
- Job queue management with priority reordering
- Progress tracking with real-time updates
- Manual file deletion
- SQLite database for persistent storage
"""

import os
import sys
import hashlib
import time
from pathlib import Path
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, render_template, send_file, abort, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import database module
import db

# Import job queue and worker
from job_queue import job_queue, JobStatus
from transcribe_worker import process_job

# Configuration
MAX_UPLOAD_SIZE_GB = int(os.environ.get('MAX_UPLOAD_SIZE_GB', 5))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_GB * 1024 * 1024 * 1024

UPLOAD_DIR = Path("/data/uploads")

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'amr', 'wma'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# Track uploads in progress to prevent processing partial files
uploads_in_progress = set()

# Create Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_BYTES
CORS(app)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_info(filepath: Path) -> dict:
    """Get file information."""
    try:
        stat = filepath.stat()
        return {
            "name": filepath.name,
            "size": stat.st_size,
            "size_human": format_size(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": filepath.suffix.lower().lstrip('.'),
            "type": "audio" if filepath.suffix.lower().lstrip('.') in ALLOWED_AUDIO_EXTENSIONS else "video"
        }
    except Exception:
        return None


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def is_file_complete(filepath: Path) -> bool:
    """
    Check if a file upload is complete.
    Returns False if the file is currently being uploaded.
    """
    filename = filepath.name
    
    # Check if in our uploads_in_progress set
    if filename in uploads_in_progress:
        return False
    
    # Check if file size is stable (not still being written)
    try:
        size1 = filepath.stat().st_size
        time.sleep(0.5)  # Brief wait
        size2 = filepath.stat().st_size
        
        if size1 != size2:
            return False  # File is still being written
        
        if size1 == 0:
            return False  # Empty file
            
        return True
    except Exception:
        return False


# ============================================================================
# API Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html', 
                         max_upload_gb=MAX_UPLOAD_SIZE_GB)


@app.route('/api/config')
def get_config():
    """Get application configuration."""
    import torch
    from pathlib import Path
    
    gpu_info = None
    vram_usage = None
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
        
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_gb": total_memory
        }
        
        vram_usage = {
            "total_gb": total_memory,
            "allocated_gb": allocated_memory,
            "reserved_gb": reserved_memory,
            "free_gb": total_memory - reserved_memory
        }
    
    # Check which Whisper models are downloaded
    models_dir = Path.home() / '.cache' / 'whisper'
    model_files = {
        'tiny': 'tiny.pt',
        'base': 'base.pt',
        'small': 'small.pt',
        'medium': 'medium.pt',
        'large': 'large-v3.pt'  # Whisper uses large-v3 as default for 'large'
    }
    
    downloaded_models = {}
    if models_dir.exists():
        for model_id, filename in model_files.items():
            model_path = models_dir / filename
            downloaded_models[model_id] = model_path.exists()
    else:
        downloaded_models = {model_id: False for model_id in model_files.keys()}
    
    return jsonify({
        "max_upload_size_gb": MAX_UPLOAD_SIZE_GB,
        "max_upload_size_bytes": MAX_UPLOAD_SIZE_BYTES,
        "gpu_available": torch.cuda.is_available(),
        "gpu_info": gpu_info,
        "vram_usage": vram_usage,
        "downloaded_models": downloaded_models,
        "whisper_models": [
            {"id": "tiny", "name": "Tiny", "vram": "~1GB", "speed": "fastest"},
            {"id": "base", "name": "Base", "vram": "~1.5GB", "speed": "fast"},
            {"id": "small", "name": "Small", "vram": "~2.5GB", "speed": "medium"},
            {"id": "medium", "name": "Medium", "vram": "~4GB", "speed": "slow"},
            {"id": "large", "name": "Large", "vram": "~5.5GB", "speed": "slowest"}
        ]
    })


# ============================================================================
# File Management Routes
# ============================================================================

@app.route('/api/files', methods=['GET'])
def list_files():
    """List all files in uploads and completed directories."""
    uploads = []
    completed = []
    
    # List upload files
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file() and allowed_file(f.name):
                info = get_file_info(f)
                if info:
                    info['complete'] = is_file_complete(f)
                    uploads.append(info)
    
    return jsonify({
        "uploads": sorted(uploads, key=lambda x: x['modified'], reverse=True)
    })


@app.route('/api/files/refresh', methods=['POST'])
def refresh_files():
    """
    Refresh file list - useful when files are added via network share.
    Returns the updated file list.
    """
    return list_files()


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    filename = secure_filename(file.filename)
    filepath = UPLOAD_DIR / filename
    
    # Handle duplicate filenames
    counter = 1
    original_stem = filepath.stem
    while filepath.exists():
        filepath = UPLOAD_DIR / f"{original_stem}_{counter}{filepath.suffix}"
        counter += 1
    
    filename = filepath.name
    
    # Mark as upload in progress
    uploads_in_progress.add(filename)
    
    try:
        # Ensure directory exists
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file.save(str(filepath))
        
        # Verify file was saved completely
        if not filepath.exists() or filepath.stat().st_size == 0:
            raise Exception("File save failed or file is empty")
        
        db.add_log("SUCCESS", f"File uploaded: {filename} ({format_size(filepath.stat().st_size)})")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "size": filepath.stat().st_size,
            "size_human": format_size(filepath.stat().st_size)
        })
        
    except Exception as e:
        # Clean up partial file
        if filepath.exists():
            filepath.unlink()
        db.add_log("ERROR", f"File upload failed: {filename} - {str(e)}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Mark upload as complete
        uploads_in_progress.discard(filename)


@app.route('/api/files/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete a file from uploads directory."""
    filename = secure_filename(filename)
    
    # Check uploads directory
    upload_path = UPLOAD_DIR / filename
    if upload_path.exists():
        upload_path.unlink()
        db.add_log("INFO", f"File deleted from uploads: {filename}")
        return jsonify({"success": True, "deleted": filename, "from": "uploads"})
    
    return jsonify({"error": "File not found"}), 404


@app.route('/api/files/<path:filename>/download')
def download_file(filename):
    """Download a file."""
    filename = secure_filename(filename)
    
    # Check uploads
    filepath = UPLOAD_DIR / filename
    if filepath.exists():
        return send_file(
            str(filepath),
            as_attachment=True,
            download_name=filename
        )
    
    return jsonify({"error": "File not found"}), 404


@app.route('/api/files/<path:filename>/view')
def view_file(filename):
    """View text file contents - now served from database."""
    # This endpoint is kept for backwards compatibility but results are now in database
    return jsonify({"error": "Results are now stored in database. Use /api/results endpoints."}), 400


# ============================================================================
# Results Routes (Database-backed)
# ============================================================================

@app.route('/api/results', methods=['GET'])
def list_results():
    """Get all transcription results from database."""
    results = db.get_results()
    return jsonify({"results": results})


@app.route('/api/results/<job_id>/transcript')
def get_transcript(job_id):
    """Get transcript text for a specific job."""
    transcript = db.get_transcript(job_id)
    if transcript is None:
        return jsonify({"error": "Transcript not found"}), 404
    
    job = db.get_job(job_id)
    return jsonify({
        "job_id": job_id,
        "filename": job['filename'] if job else 'Unknown',
        "content": transcript
    })


@app.route('/api/results/<job_id>/srt')
def get_srt(job_id):
    """Get SRT text for a specific job."""
    srt = db.get_srt(job_id)
    if srt is None:
        return jsonify({"error": "SRT not found"}), 404
    
    job = db.get_job(job_id)
    return jsonify({
        "job_id": job_id,
        "filename": job['filename'] if job else 'Unknown',
        "content": srt
    })


@app.route('/api/results/<job_id>/download/<file_type>')
def download_result(job_id, file_type):
    """Download transcript or SRT as a file."""
    job = db.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    if file_type == 'transcript':
        content = db.get_transcript(job_id)
        suffix = '.txt'
    elif file_type == 'srt':
        content = db.get_srt(job_id)
        suffix = '.srt'
    else:
        return jsonify({"error": "Invalid file type"}), 400
    
    if content is None:
        return jsonify({"error": f"{file_type} not found"}), 404
    
    # Create filename from original filename
    base_name = Path(job['filename']).stem
    download_name = f"{base_name}{suffix}"
    
    return Response(
        content,
        mimetype='text/plain',
        headers={'Content-Disposition': f'attachment; filename="{download_name}"'}
    )


# ============================================================================
# Job Management Routes
# ============================================================================

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """Get all jobs (current, queued, completed)."""
    return jsonify(job_queue.get_all())


@app.route('/api/jobs', methods=['POST'])
def create_job():
    """Create a new transcription job."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    filename = data.get('filename')
    model = data.get('model', 'base')
    language = data.get('language', 'en')
    generate_srt = data.get('generate_srt', False)
    keep_file = data.get('keep_file', False)
    
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    
    # Verify file exists and is complete
    filepath = UPLOAD_DIR / secure_filename(filename)
    if not filepath.exists():
        return jsonify({"error": "File not found in uploads"}), 404
    
    if not is_file_complete(filepath):
        return jsonify({"error": "File upload is incomplete or in progress"}), 400
    
    # Validate model
    valid_models = ['tiny', 'base', 'small', 'medium', 'large']
    if model not in valid_models:
        return jsonify({"error": f"Invalid model. Choose from: {', '.join(valid_models)}"}), 400
    
    # Add job to queue
    job = job_queue.add(filename, model, language, generate_srt, keep_file)
    
    db.add_log("INFO", f"Job created: {filename} (model: {model}, language: {language})")
    
    return jsonify({
        "success": True,
        "job": job
    }), 201


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_route(job_id):
    """Get a specific job by ID."""
    job = job_queue.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route('/api/jobs/<job_id>/move-up', methods=['PUT'])
def move_job_up(job_id):
    """Move a job higher in the queue."""
    if job_queue.move_up(job_id):
        return jsonify({"success": True})
    return jsonify({"error": "Cannot move job (not found or already at top)"}), 400


@app.route('/api/jobs/<job_id>/move-down', methods=['PUT'])
def move_job_down(job_id):
    """Move a job lower in the queue."""
    if job_queue.move_down(job_id):
        return jsonify({"success": True})
    return jsonify({"error": "Cannot move job (not found or already at bottom)"}), 400


@app.route('/api/jobs/<job_id>', methods=['DELETE'])
def cancel_job(job_id):
    """Cancel a queued or running job."""
    # Try to cancel queued job first
    if job_queue.cancel(job_id):
        db.add_log("INFO", f"Job cancelled: {job_id}")
        return jsonify({"success": True, "cancelled": job_id})
    
    # Check if it's a running job
    running_job = db.get_running_job()
    if running_job and running_job['id'] == job_id:
        # Mark as cancelled in database
        db.complete_job(job_id, 'cancelled', error="Cancelled by user")
        db.add_log("WARNING", f"Running job cancelled: {job_id}")
        return jsonify({"success": True, "cancelled": job_id, "was_running": True})
    
    return jsonify({"error": "Cannot cancel job (not found or already completed)"}), 400


@app.route('/api/jobs/completed', methods=['DELETE'])
def clear_completed_jobs():
    """Clear all completed jobs from the queue."""
    try:
        count = job_queue.clear_completed()
        db.add_log("INFO", f"Cleared {count} completed jobs")
        return jsonify({"success": True, "cleared": count})
    except Exception as e:
        db.add_log("ERROR", f"Failed to clear completed jobs: {str(e)}")
        return jsonify({"error": "Failed to clear completed jobs"}), 500


@app.route('/api/jobs/<job_id>/delete', methods=['DELETE'])
def delete_completed_job(job_id):
    """Delete a specific completed job."""
    try:
        if db.delete_job(job_id):
            db.add_log("INFO", f"Deleted job: {job_id}")
            return jsonify({"success": True, "deleted": job_id})
        return jsonify({"error": "Job not found or not completed"}), 404
    except Exception as e:
        db.add_log("ERROR", f"Failed to delete job {job_id}: {str(e)}")
        return jsonify({"error": "Failed to delete job"}), 500


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get application logs from database."""
    return jsonify(db.get_logs())


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        "error": f"File too large. Maximum upload size is {MAX_UPLOAD_SIZE_GB}GB via web interface. "
                 f"For larger files, copy directly to the uploads folder via network share."
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# Application Startup
# ============================================================================

def job_completion_callback(job):
    """Callback function called when a job completes."""
    status = job.get('status', '')
    filename = job.get('filename', 'Unknown')
    model = job.get('model', 'Unknown')
    error = job.get('error', '')
    
    if status == 'completed':
        db.add_log("SUCCESS", f"Job completed: {filename} (model: {model})")
    elif status == 'failed':
        db.add_log("ERROR", f"Job failed: {filename} - {error}")
    elif status == 'cancelled':
        db.add_log("WARNING", f"Job cancelled: {filename}")

def init_app():
    """Initialize the application."""
    # Initialize database
    db.init_database()
    
    # Create upload directory
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set up job processor and completion callback
    job_queue.set_processor(process_job)
    job_queue.set_completion_callback(job_completion_callback)
    job_queue.start()
    
    db.add_log("INFO", "Whisper Transcription Web App started")
    
    print("=" * 60)
    print("🎤 Whisper Transcription Web App")
    print("=" * 60)
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"💾 Database: {db.DATABASE_PATH}")
    print(f"📦 Max upload size: {MAX_UPLOAD_SIZE_GB}GB")
    
    import torch
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        db.add_log("INFO", f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected, using CPU")
        db.add_log("WARNING", "No GPU detected, using CPU")
    print("=" * 60)


# Initialize on import
init_app()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
