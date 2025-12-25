"""
Job Queue Module for Whisper Transcription Web App

Database-backed job queue with:
- Single job processing (one at a time for GPU VRAM efficiency)
- Priority reordering via queue_position
- Progress tracking
- Thread-safe operations using SQLite
"""

import threading
import uuid
import time
from typing import Optional, Dict, Callable
from enum import Enum

import db


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobQueue:
    """
    Database-backed job queue for managing transcription jobs.
    
    Jobs are processed one at a time to avoid GPU VRAM conflicts.
    All job data is stored in SQLite for persistence.
    """
    
    def __init__(self):
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._job_processor: Optional[Callable] = None
        self._completion_callback: Optional[Callable] = None
        self._startup_cleanup_done = False
        
    def set_processor(self, processor: Callable):
        """Set the function that processes jobs."""
        self._job_processor = processor
    
    def set_completion_callback(self, callback: Callable):
        """Set a callback function to be called when a job completes."""
        self._completion_callback = callback
        
    def start(self):
        """Start the job queue worker thread."""
        if self._running:
            return
        self._running = True
        self._startup_cleanup_done = False  # Reset flag on start
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
    def stop(self):
        """Stop the job queue worker thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            
    def _worker_loop(self):
        """Main worker loop that processes jobs sequentially."""
        # Check for interrupted jobs ONLY ONCE on startup
        if not self._startup_cleanup_done:
            self._startup_cleanup_done = True
            running_job = db.get_running_job()
            if running_job:
                # Job was running when server stopped - mark as interrupted
                db.complete_job(running_job['id'], error="Job interrupted by server restart")
                if self._completion_callback:
                    try:
                        self._completion_callback(running_job, 'failed')
                    except Exception:
                        pass
        
        while self._running:
            # Get next queued job
            job = db.get_next_queued_job()
            if job:
                self._process_job(job)
            else:
                time.sleep(1)  # Wait before checking again
                
    def _process_job(self, job: Dict):
        """Process a single job."""
        job_id = job['id']
        
        # Mark job as running
        db.start_job(job_id)
        
        try:
            if self._job_processor:
                # Create a job-like object for the processor
                self._job_processor(job)
            else:
                # No processor set, mark as failed
                db.complete_job(job_id, error="No job processor configured")
        except Exception as e:
            db.complete_job(job_id, error=str(e))
        
        # Get updated job for callback
        updated_job = db.get_job(job_id)
        
        # Call completion callback if set
        if self._completion_callback and updated_job:
            try:
                self._completion_callback(updated_job)
            except Exception:
                pass  # Don't let callback errors break the queue
                    
    def add(self, filename: str, model: str, language: str, generate_srt: bool, keep_file: bool = False) -> Dict:
        """Add a new job to the queue."""
        job_id = str(uuid.uuid4())[:8]
        return db.add_job(job_id, filename, model, language, generate_srt, keep_file)
        
    def get_all(self) -> Dict:
        """Get all jobs (current, queued, completed)."""
        return db.get_all_jobs()
            
    def get(self, job_id: str) -> Optional[Dict]:
        """Get a specific job by ID."""
        return db.get_job(job_id)
        
    def update_progress(self, job_id: str, progress: float, message: str = ""):
        """Update job progress (called by the processor)."""
        db.update_job_progress(job_id, progress, message)
                
    def move_up(self, job_id: str) -> bool:
        """Move a job higher in the queue."""
        return db.move_job_up(job_id)
        
    def move_down(self, job_id: str) -> bool:
        """Move a job lower in the queue."""
        return db.move_job_down(job_id)
        
    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job (cannot cancel running jobs)."""
        return db.cancel_job(job_id)
    
    def clear_completed(self) -> int:
        """Clear all completed jobs and return the count."""
        return db.clear_completed_jobs()


# Global job queue instance
job_queue = JobQueue()
