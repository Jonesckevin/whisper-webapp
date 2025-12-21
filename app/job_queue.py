"""
Job Queue Module for Whisper Transcription Web App

Implements an in-memory job queue with:
- Single job processing (one at a time for GPU VRAM efficiency)
- Priority reordering
- Progress tracking
- Thread-safe operations
"""

import threading
import uuid
import time
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a transcription job."""
    id: str
    filename: str
    model: str
    language: str
    generate_srt: bool
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    progress_message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_file: Optional[str] = None
    srt_file: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert job to dictionary for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        return data


class JobQueue:
    """
    Thread-safe job queue for managing transcription jobs.
    
    Jobs are processed one at a time to avoid GPU VRAM conflicts.
    """
    
    def __init__(self):
        self._queue: deque[Job] = deque()
        self._current_job: Optional[Job] = None
        self._completed_jobs: List[Job] = []
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._job_processor: Optional[Callable] = None
        
    def set_processor(self, processor: Callable):
        """Set the function that processes jobs."""
        self._job_processor = processor
        
    def start(self):
        """Start the job queue worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
    def stop(self):
        """Stop the job queue worker thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            
    def _worker_loop(self):
        """Main worker loop that processes jobs sequentially."""
        while self._running:
            job = self._get_next_job()
            if job:
                self._process_job(job)
            else:
                time.sleep(1)  # Wait before checking again
                
    def _get_next_job(self) -> Optional[Job]:
        """Get the next job from the queue."""
        with self._lock:
            if self._queue and self._current_job is None:
                job = self._queue.popleft()
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now().isoformat()
                self._current_job = job
                return job
        return None
        
    def _process_job(self, job: Job):
        """Process a single job."""
        try:
            if self._job_processor:
                self._job_processor(job)
            else:
                # No processor set, mark as failed
                job.status = JobStatus.FAILED
                job.error = "No job processor configured"
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
        finally:
            job.completed_at = datetime.now().isoformat()
            with self._lock:
                self._current_job = None
                self._completed_jobs.append(job)
                # Keep only last 100 completed jobs
                if len(self._completed_jobs) > 100:
                    self._completed_jobs = self._completed_jobs[-100:]
                    
    def add(self, filename: str, model: str, language: str, generate_srt: bool) -> Job:
        """Add a new job to the queue."""
        job = Job(
            id=str(uuid.uuid4())[:8],
            filename=filename,
            model=model,
            language=language,
            generate_srt=generate_srt
        )
        with self._lock:
            self._queue.append(job)
        return job
        
    def get_all(self) -> Dict:
        """Get all jobs (current, queued, completed)."""
        with self._lock:
            return {
                "current": self._current_job.to_dict() if self._current_job else None,
                "queued": [job.to_dict() for job in self._queue],
                "completed": [job.to_dict() for job in reversed(self._completed_jobs[-20:])]
            }
            
    def get(self, job_id: str) -> Optional[Job]:
        """Get a specific job by ID."""
        with self._lock:
            if self._current_job and self._current_job.id == job_id:
                return self._current_job
            for job in self._queue:
                if job.id == job_id:
                    return job
            for job in self._completed_jobs:
                if job.id == job_id:
                    return job
        return None
        
    def update_progress(self, job_id: str, progress: float, message: str = ""):
        """Update job progress (called by the processor)."""
        with self._lock:
            if self._current_job and self._current_job.id == job_id:
                self._current_job.progress = min(100.0, max(0.0, progress))
                self._current_job.progress_message = message
                
    def move_up(self, job_id: str) -> bool:
        """Move a job higher in the queue (lower index = higher priority)."""
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.id == job_id and i > 0:
                    # Swap with previous job
                    self._queue[i], self._queue[i-1] = self._queue[i-1], self._queue[i]
                    return True
        return False
        
    def move_down(self, job_id: str) -> bool:
        """Move a job lower in the queue."""
        with self._lock:
            queue_list = list(self._queue)
            for i, job in enumerate(queue_list):
                if job.id == job_id and i < len(queue_list) - 1:
                    # Swap with next job
                    queue_list[i], queue_list[i+1] = queue_list[i+1], queue_list[i]
                    self._queue = deque(queue_list)
                    return True
        return False
        
    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job (cannot cancel running jobs)."""
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.id == job_id:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now().isoformat()
                    self._completed_jobs.append(job)
                    del self._queue[i]
                    return True
        return False
        
    def get_queue_position(self, job_id: str) -> int:
        """Get the position of a job in the queue (1-based, 0 if not found or running)."""
        with self._lock:
            if self._current_job and self._current_job.id == job_id:
                return 0  # Currently running
            for i, job in enumerate(self._queue):
                if job.id == job_id:
                    return i + 1
        return -1  # Not found


# Global job queue instance
job_queue = JobQueue()
