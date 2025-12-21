"""
Flask Web Application for Whisper Transcription

Features:
- File upload (max 5GB via web, larger files via network share)
- Job queue management with priority reordering
- Progress tracking with real-time updates
- Manual file deletion
- Refresh to detect externally added files
"""

import os
import sys
import hashlib
import time
from pathlib import Path
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import job queue and worker
from job_queue import job_queue, JobStatus
from transcribe_worker import process_job

# Configuration
MAX_UPLOAD_SIZE_GB = int(os.environ.get('MAX_UPLOAD_SIZE_GB', 5))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_GB * 1024 * 1024 * 1024

UPLOAD_DIR = Path("/data/uploads")
COMPLETED_DIR = Path("/data/completed")

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


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/config')
def get_config():
    """Get application configuration."""
    import torch
    
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    
    return jsonify({
        "max_upload_size_gb": MAX_UPLOAD_SIZE_GB,
        "max_upload_size_bytes": MAX_UPLOAD_SIZE_BYTES,
        "gpu_available": torch.cuda.is_available(),
        "gpu_info": gpu_info,
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
    
    # List completed files
    if COMPLETED_DIR.exists():
        for f in COMPLETED_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in ['.txt', '.srt']:
                info = get_file_info(f)
                if info:
                    completed.append(info)
    
    return jsonify({
        "uploads": sorted(uploads, key=lambda x: x['modified'], reverse=True),
        "completed": sorted(completed, key=lambda x: x['modified'], reverse=True)
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
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Mark upload as complete
        uploads_in_progress.discard(filename)


@app.route('/api/files/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete a file from uploads or completed directory."""
    filename = secure_filename(filename)
    
    # Check uploads directory
    upload_path = UPLOAD_DIR / filename
    if upload_path.exists():
        upload_path.unlink()
        return jsonify({"success": True, "deleted": filename, "from": "uploads"})
    
    # Check completed directory
    completed_path = COMPLETED_DIR / filename
    if completed_path.exists():
        completed_path.unlink()
        return jsonify({"success": True, "deleted": filename, "from": "completed"})
    
    return jsonify({"error": "File not found"}), 404


@app.route('/api/files/<path:filename>/download')
def download_file(filename):
    """Download a file."""
    filename = secure_filename(filename)
    
    # Check completed first, then uploads
    for directory in [COMPLETED_DIR, UPLOAD_DIR]:
        filepath = directory / filename
        if filepath.exists():
            return send_file(
                str(filepath),
                as_attachment=True,
                download_name=filename
            )
    
    return jsonify({"error": "File not found"}), 404


@app.route('/api/files/<path:filename>/view')
def view_file(filename):
    """View text file contents."""
    filename = secure_filename(filename)
    
    # Only allow viewing text files
    if not filename.endswith(('.txt', '.srt')):
        return jsonify({"error": "Only text files can be viewed"}), 400
    
    filepath = COMPLETED_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({
            "filename": filename,
            "content": content,
            "size": len(content)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    job = job_queue.add(filename, model, language, generate_srt)
    
    return jsonify({
        "success": True,
        "job": job.to_dict()
    }), 201


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get a specific job by ID."""
    job = job_queue.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job.to_dict())


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
    """Cancel a queued job."""
    if job_queue.cancel(job_id):
        return jsonify({"success": True, "cancelled": job_id})
    return jsonify({"error": "Cannot cancel job (not found or already running)"}), 400


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

def init_app():
    """Initialize the application."""
    # Create directories
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set up job processor and start queue
    job_queue.set_processor(process_job)
    job_queue.start()
    
    print("=" * 60)
    print("🎤 Whisper Transcription Web App")
    print("=" * 60)
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"📁 Completed directory: {COMPLETED_DIR}")
    print(f"📦 Max upload size: {MAX_UPLOAD_SIZE_GB}GB")
    
    import torch
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected, using CPU")
    print("=" * 60)


# Initialize on import
init_app()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
