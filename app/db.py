"""
SQLite Database Module for Whisper Transcription Web App

Provides persistent storage for:
- Job queue (current, queued, completed jobs)
- Application logs
- Transcription results (text content stored in database)
"""

import sqlite3
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from contextlib import contextmanager

# Database path - mapped volume for persistence
DATABASE_PATH = Path("/data/db/sqlite.db")

# Thread-local storage for connections
_local = threading.local()

def get_connection() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = sqlite3.connect(
            str(DATABASE_PATH),
            check_same_thread=False,
            timeout=30.0
        )
        _local.connection.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        _local.connection.execute("PRAGMA journal_mode=WAL")
        _local.connection.execute("PRAGMA foreign_keys=ON")
    return _local.connection

@contextmanager
def get_db():
    """Context manager for database operations."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise

def init_database():
    """Initialize database schema."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with get_db() as conn:
        # Jobs table - holds all jobs with status and queue position
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                model TEXT NOT NULL,
                language TEXT NOT NULL,
                generate_srt BOOLEAN NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'queued',
                progress REAL DEFAULT 0.0,
                progress_message TEXT DEFAULT '',
                queue_position INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                transcript_text TEXT,
                srt_text TEXT,
                error TEXT
            )
        """)
        
        # Create indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_queue_position ON jobs(queue_position)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_completed_at ON jobs(completed_at)")
        
        # Logs table - application logs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)")

# =============================================================================
# Job Operations
# =============================================================================

def add_job(job_id: str, filename: str, model: str, language: str, generate_srt: bool) -> Dict:
    """Add a new job to the queue."""
    with get_db() as conn:
        # Get next queue position (max + 1, or 1 if empty)
        result = conn.execute(
            "SELECT COALESCE(MAX(queue_position), 0) + 1 as next_pos FROM jobs WHERE status = 'queued'"
        ).fetchone()
        queue_position = result['next_pos']
        
        created_at = datetime.now().isoformat()
        
        conn.execute("""
            INSERT INTO jobs (id, filename, model, language, generate_srt, status, queue_position, created_at)
            VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)
        """, (job_id, filename, model, language, generate_srt, queue_position, created_at))
        
        return get_job(job_id)

def get_job(job_id: str) -> Optional[Dict]:
    """Get a job by ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

def get_all_jobs() -> Dict[str, Any]:
    """Get all jobs organized by status."""
    with get_db() as conn:
        # Current job (running)
        current_row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'running' LIMIT 1"
        ).fetchone()
        current = dict(current_row) if current_row else None
        
        # Queued jobs (ordered by queue_position)
        queued_rows = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY queue_position ASC"
        ).fetchall()
        queued = [dict(row) for row in queued_rows]
        
        # Completed jobs (last 1000, newest first)
        completed_rows = conn.execute("""
            SELECT * FROM jobs 
            WHERE status IN ('completed', 'failed', 'cancelled')
            ORDER BY completed_at DESC
            LIMIT 1000
        """).fetchall()
        completed = []
        for row in completed_rows:
            job = dict(row)
            # Add convenience fields for frontend
            job['has_transcript'] = bool(job.get('transcript_text'))
            job['has_srt'] = bool(job.get('srt_text'))
            completed.append(job)
        
        return {
            "current": current,
            "queued": queued,
            "completed": completed
        }

def get_next_queued_job() -> Optional[Dict]:
    """Get the next job in queue (lowest queue_position)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY queue_position ASC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

def start_job(job_id: str) -> bool:
    """Mark a job as running."""
    with get_db() as conn:
        started_at = datetime.now().isoformat()
        cursor = conn.execute("""
            UPDATE jobs 
            SET status = 'running', started_at = ?, queue_position = NULL, progress = 0, progress_message = 'Starting...'
            WHERE id = ? AND status = 'queued'
        """, (started_at, job_id))
        return cursor.rowcount > 0

def update_job_progress(job_id: str, progress: float, message: str = "") -> bool:
    """Update job progress."""
    with get_db() as conn:
        cursor = conn.execute("""
            UPDATE jobs 
            SET progress = ?, progress_message = ?
            WHERE id = ? AND status = 'running'
        """, (min(100.0, max(0.0, progress)), message, job_id))
        return cursor.rowcount > 0

def complete_job(job_id: str, status: str = 'completed', transcript_text: str = None, srt_text: str = None, error: str = None) -> bool:
    """Mark a job as completed, failed, or cancelled."""
    with get_db() as conn:
        completed_at = datetime.now().isoformat()
        # Status can be 'completed', 'failed', or 'cancelled'
        progress = 100 if status == 'completed' else 0
        
        cursor = conn.execute("""
            UPDATE jobs 
            SET status = ?, completed_at = ?, transcript_text = ?, srt_text = ?, error = ?, 
                progress = ?, progress_message = ?, queue_position = NULL
            WHERE id = ?
        """, (status, completed_at, transcript_text, srt_text, error, progress, 
              error if error else 'Complete', job_id))
        return cursor.rowcount > 0
        return cursor.rowcount > 0

def cancel_job(job_id: str) -> bool:
    """Cancel a queued job."""
    with get_db() as conn:
        completed_at = datetime.now().isoformat()
        cursor = conn.execute("""
            UPDATE jobs 
            SET status = 'cancelled', completed_at = ?, queue_position = NULL
            WHERE id = ? AND status = 'queued'
        """, (completed_at, job_id))
        return cursor.rowcount > 0

def move_job_up(job_id: str) -> bool:
    """Move a job higher in the queue (swap with previous)."""
    with get_db() as conn:
        # Get current job's position
        current = conn.execute(
            "SELECT queue_position FROM jobs WHERE id = ? AND status = 'queued'", (job_id,)
        ).fetchone()
        
        if not current or current['queue_position'] is None:
            return False
        
        current_pos = current['queue_position']
        
        # Find job with next lower position
        prev_job = conn.execute("""
            SELECT id, queue_position FROM jobs 
            WHERE status = 'queued' AND queue_position < ?
            ORDER BY queue_position DESC LIMIT 1
        """, (current_pos,)).fetchone()
        
        if not prev_job:
            return False  # Already at top
        
        # Swap positions
        conn.execute("UPDATE jobs SET queue_position = ? WHERE id = ?", (prev_job['queue_position'], job_id))
        conn.execute("UPDATE jobs SET queue_position = ? WHERE id = ?", (current_pos, prev_job['id']))
        
        return True

def move_job_down(job_id: str) -> bool:
    """Move a job lower in the queue (swap with next)."""
    with get_db() as conn:
        # Get current job's position
        current = conn.execute(
            "SELECT queue_position FROM jobs WHERE id = ? AND status = 'queued'", (job_id,)
        ).fetchone()
        
        if not current or current['queue_position'] is None:
            return False
        
        current_pos = current['queue_position']
        
        # Find job with next higher position
        next_job = conn.execute("""
            SELECT id, queue_position FROM jobs 
            WHERE status = 'queued' AND queue_position > ?
            ORDER BY queue_position ASC LIMIT 1
        """, (current_pos,)).fetchone()
        
        if not next_job:
            return False  # Already at bottom
        
        # Swap positions
        conn.execute("UPDATE jobs SET queue_position = ? WHERE id = ?", (next_job['queue_position'], job_id))
        conn.execute("UPDATE jobs SET queue_position = ? WHERE id = ?", (current_pos, next_job['id']))
        
        return True

def clear_completed_jobs() -> int:
    """Clear all completed jobs. Returns count deleted."""
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM jobs WHERE status IN ('completed', 'failed', 'cancelled')"
        )
        return cursor.rowcount

def delete_job(job_id: str) -> bool:
    """Delete a specific completed job."""
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM jobs WHERE id = ? AND status IN ('completed', 'failed', 'cancelled')", 
            (job_id,)
        )
        return cursor.rowcount > 0

def get_running_job() -> Optional[Dict]:
    """Get the currently running job, if any."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE status = 'running' LIMIT 1").fetchone()
        return dict(row) if row else None

# =============================================================================
# Log Operations
# =============================================================================

MAX_LOG_ENTRIES = 1000

def add_log(level: str, message: str):
    """Add a log entry."""
    with get_db() as conn:
        timestamp = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)",
            (timestamp, level, message)
        )
        
        # Cleanup old logs if over limit
        conn.execute("""
            DELETE FROM logs WHERE id NOT IN (
                SELECT id FROM logs ORDER BY id DESC LIMIT ?
            )
        """, (MAX_LOG_ENTRIES,))

def get_logs(limit: int = 1000) -> List[Dict]:
    """Get logs, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT timestamp, level, message FROM logs ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

def clear_logs() -> int:
    """Clear all logs. Returns count deleted."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM logs")
        return cursor.rowcount

# =============================================================================
# Results Operations
# =============================================================================

def get_results() -> List[Dict]:
    """Get all completed jobs with transcripts (results)."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, filename, model, language, generate_srt, completed_at, 
                   transcript_text, srt_text
            FROM jobs 
            WHERE status = 'completed' AND transcript_text IS NOT NULL
            ORDER BY completed_at DESC
        """).fetchall()
        
        results = []
        for row in rows:
            result = dict(row)
            # Add file info for UI compatibility
            result['has_transcript'] = bool(result.get('transcript_text'))
            result['has_srt'] = bool(result.get('srt_text'))
            results.append(result)
        
        return results

def get_transcript(job_id: str) -> Optional[str]:
    """Get transcript text for a job."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT transcript_text FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return row['transcript_text'] if row else None

def get_srt(job_id: str) -> Optional[str]:
    """Get SRT text for a job."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT srt_text FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return row['srt_text'] if row else None
