/**
 * Whisper Transcription Web App - Frontend JavaScript
 * 
 * Features:
 * - File upload with drag & drop
 * - Job queue management
 * - Real-time progress updates
 * - File management with refresh and delete
 */

// ============================================================================
// Configuration
// ============================================================================

const API_BASE = '/api';
const POLL_INTERVAL = 2000; // 2 seconds

let currentDeleteFile = null;
let currentViewFile = null;

// ============================================================================
// Utility Functions
// ============================================================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

function formatDate(isoString) {
    if (!isoString) return '-';
    const date = new Date(isoString);
    return date.toLocaleString();
}

function getFileIcon(type, extension) {
    if (type === 'audio') {
        return '🎵';
    } else if (type === 'video') {
        return '🎬';
    } else if (extension === 'txt') {
        return '📄';
    } else if (extension === 'srt') {
        return '📺';
    }
    return '📁';
}

// ============================================================================
// Tab Navigation
// ============================================================================

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        
        // Update buttons
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(`${tabId}-tab`).classList.add('active');
        
        // Refresh data when switching tabs
        if (tabId === 'upload') {
            loadFiles();
        } else if (tabId === 'transcribe') {
            loadFileSelect();
        } else if (tabId === 'queue') {
            loadJobs();
        } else if (tabId === 'results') {
            loadResults();
        }
    });
});

// ============================================================================
// File Upload
// ============================================================================

const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const uploadProgress = document.getElementById('upload-progress');
const uploadProgressFill = document.getElementById('upload-progress-fill');
const uploadStatus = document.getElementById('upload-status');

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    
    const files = Array.from(e.dataTransfer.files);
    uploadFiles(files);
});

fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    uploadFiles(files);
    fileInput.value = ''; // Reset input
});

async function uploadFiles(files) {
    for (const file of files) {
        await uploadFile(file);
    }
    loadFiles();
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    uploadProgress.classList.remove('hidden');
    uploadProgressFill.style.width = '0%';
    uploadStatus.textContent = `Uploading ${file.name}...`;
    
    try {
        const xhr = new XMLHttpRequest();
        
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = (e.loaded / e.total) * 100;
                uploadProgressFill.style.width = `${percent}%`;
                uploadStatus.textContent = `Uploading ${file.name}... ${Math.round(percent)}%`;
            }
        });
        
        await new Promise((resolve, reject) => {
            xhr.onload = () => {
                if (xhr.status === 200 || xhr.status === 201) {
                    const response = JSON.parse(xhr.responseText);
                    showToast(`✅ Uploaded: ${response.filename}`, 'success');
                    resolve(response);
                } else {
                    const error = JSON.parse(xhr.responseText);
                    showToast(`❌ ${error.error}`, 'error');
                    reject(new Error(error.error));
                }
            };
            xhr.onerror = () => reject(new Error('Upload failed'));
            xhr.open('POST', `${API_BASE}/upload`);
            xhr.send(formData);
        });
        
    } catch (error) {
        showToast(`❌ Upload failed: ${error.message}`, 'error');
    } finally {
        setTimeout(() => {
            uploadProgress.classList.add('hidden');
        }, 1000);
    }
}

// ============================================================================
// File Management
// ============================================================================

async function loadFiles() {
    try {
        const response = await fetch(`${API_BASE}/files`);
        const data = await response.json();
        
        renderUploadsList(data.uploads);
    } catch (error) {
        console.error('Failed to load files:', error);
    }
}

function renderUploadsList(files) {
    const container = document.getElementById('uploads-list');
    
    if (!files || files.length === 0) {
        container.innerHTML = '<p class="no-jobs">No files uploaded yet</p>';
        return;
    }
    
    container.innerHTML = files.map(file => `
        <div class="file-item">
            <div class="file-info">
                <span class="file-icon">${getFileIcon(file.type, file.extension)}</span>
                <div class="file-details">
                    <div class="file-name">${file.name}</div>
                    <div class="file-meta">
                        ${file.size_human} • ${formatDate(file.modified)}
                        ${!file.complete ? '<span class="status-badge status-incomplete">Incomplete</span>' : ''}
                    </div>
                </div>
            </div>
            <div class="file-actions">
                <button class="btn btn-icon" onclick="deleteFile('${file.name}', 'uploads')" title="Delete">🗑️</button>
            </div>
        </div>
    `).join('');
}

async function loadResults() {
    try {
        const response = await fetch(`${API_BASE}/files`);
        const data = await response.json();
        
        renderResultsList(data.completed);
    } catch (error) {
        console.error('Failed to load results:', error);
    }
}

function renderResultsList(files) {
    const container = document.getElementById('results-list');
    
    if (!files || files.length === 0) {
        container.innerHTML = '<p class="no-jobs">No transcripts yet. Process some files!</p>';
        return;
    }
    
    container.innerHTML = files.map(file => `
        <div class="file-item">
            <div class="file-info">
                <span class="file-icon">${getFileIcon(null, file.extension)}</span>
                <div class="file-details">
                    <div class="file-name">${file.name}</div>
                    <div class="file-meta">${file.size_human} • ${formatDate(file.modified)}</div>
                </div>
            </div>
            <div class="file-actions">
                <button class="btn btn-small btn-secondary" onclick="viewFile('${file.name}')" title="View">👁️ View</button>
                <button class="btn btn-small btn-secondary" onclick="downloadFile('${file.name}')" title="Download">⬇️</button>
                <button class="btn btn-icon" onclick="deleteFile('${file.name}', 'completed')" title="Delete">🗑️</button>
            </div>
        </div>
    `).join('');
}

async function refreshFiles() {
    try {
        await fetch(`${API_BASE}/files/refresh`, { method: 'POST' });
        await loadFiles();
        showToast('📁 File list refreshed', 'success');
    } catch (error) {
        showToast('❌ Failed to refresh files', 'error');
    }
}

function deleteFile(filename, source) {
    currentDeleteFile = { filename, source };
    document.getElementById('delete-filename').textContent = filename;
    document.getElementById('delete-modal').classList.remove('hidden');
}

async function confirmDelete() {
    if (!currentDeleteFile) return;
    
    try {
        const response = await fetch(`${API_BASE}/files/${currentDeleteFile.filename}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showToast(`🗑️ Deleted: ${currentDeleteFile.filename}`, 'success');
            loadFiles();
            loadResults();
            loadFileSelect();
        } else {
            const error = await response.json();
            showToast(`❌ ${error.error}`, 'error');
        }
    } catch (error) {
        showToast('❌ Delete failed', 'error');
    } finally {
        closeDeleteModal();
    }
}

function closeDeleteModal() {
    document.getElementById('delete-modal').classList.add('hidden');
    currentDeleteFile = null;
}

async function viewFile(filename) {
    try {
        const response = await fetch(`${API_BASE}/files/${filename}/view`);
        const data = await response.json();
        
        if (response.ok) {
            currentViewFile = filename;
            document.getElementById('modal-title').textContent = filename;
            document.getElementById('modal-text').textContent = data.content;
            document.getElementById('text-modal').classList.remove('hidden');
        } else {
            showToast(`❌ ${data.error}`, 'error');
        }
    } catch (error) {
        showToast('❌ Failed to load file', 'error');
    }
}

function downloadFile(filename) {
    window.open(`${API_BASE}/files/${filename}/download`, '_blank');
}

function closeTextModal() {
    document.getElementById('text-modal').classList.add('hidden');
    currentViewFile = null;
}

// ============================================================================
// Transcribe Form
// ============================================================================

async function loadFileSelect() {
    try {
        const response = await fetch(`${API_BASE}/files`);
        const data = await response.json();
        
        const select = document.getElementById('file-select');
        const currentValue = select.value;
        
        // Only include complete files
        const completeFiles = data.uploads.filter(f => f.complete);
        
        select.innerHTML = '<option value="">-- Select a file --</option>' +
            completeFiles.map(file => 
                `<option value="${file.name}">${getFileIcon(file.type, file.extension)} ${file.name} (${file.size_human})</option>`
            ).join('');
        
        // Restore selection if still valid
        if (currentValue && completeFiles.some(f => f.name === currentValue)) {
            select.value = currentValue;
        }
    } catch (error) {
        console.error('Failed to load file select:', error);
    }
}

document.getElementById('transcribe-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const filename = document.getElementById('file-select').value;
    const model = document.getElementById('model-select').value;
    const language = document.getElementById('language-input').value.trim() || 'en';
    const generateSrt = document.getElementById('srt-checkbox').checked;
    
    if (!filename) {
        showToast('⚠️ Please select a file', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename,
                model,
                language,
                generate_srt: generateSrt
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showToast(`✅ Job added to queue: ${data.job.id}`, 'success');
            document.getElementById('file-select').value = '';
            
            // Switch to queue tab
            document.querySelector('[data-tab="queue"]').click();
        } else {
            showToast(`❌ ${data.error}`, 'error');
        }
    } catch (error) {
        showToast('❌ Failed to create job', 'error');
    }
});

// ============================================================================
// Job Queue
// ============================================================================

async function loadJobs() {
    try {
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();
        
        renderCurrentJob(data.current);
        renderQueuedJobs(data.queued);
        renderCompletedJobs(data.completed);
    } catch (error) {
        console.error('Failed to load jobs:', error);
    }
}

function renderCurrentJob(job) {
    const container = document.getElementById('current-job');
    
    if (!job) {
        container.innerHTML = '<p class="no-job">No job currently running</p>';
        return;
    }
    
    container.innerHTML = `
        <div class="job-card">
            <div class="job-header">
                <span class="job-title">${job.filename}</span>
                <span class="status-badge status-running">Running</span>
            </div>
            <div class="job-meta">
                Model: ${job.model.toUpperCase()} • Language: ${job.language} • 
                SRT: ${job.generate_srt ? '✓' : '✗'} • Started: ${formatDate(job.started_at)}
            </div>
            <div class="job-progress">
                <div class="job-progress-text">
                    <span>${job.progress_message || 'Processing...'}</span>
                    <span>${Math.round(job.progress)}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${job.progress}%"></div>
                </div>
            </div>
        </div>
    `;
}

function renderQueuedJobs(jobs) {
    const container = document.getElementById('queued-jobs');
    
    if (!jobs || jobs.length === 0) {
        container.innerHTML = '<p class="no-jobs">No jobs in queue</p>';
        return;
    }
    
    container.innerHTML = jobs.map((job, index) => `
        <div class="job-card">
            <div class="job-header">
                <span class="job-title">#${index + 1} ${job.filename}</span>
                <span class="status-badge status-queued">Queued</span>
            </div>
            <div class="job-meta">
                Model: ${job.model.toUpperCase()} • Language: ${job.language} • 
                SRT: ${job.generate_srt ? '✓' : '✗'} • Added: ${formatDate(job.created_at)}
            </div>
            <div class="job-actions">
                <button class="btn btn-small btn-secondary" onclick="moveJobUp('${job.id}')" ${index === 0 ? 'disabled' : ''}>⬆️ Up</button>
                <button class="btn btn-small btn-secondary" onclick="moveJobDown('${job.id}')" ${index === jobs.length - 1 ? 'disabled' : ''}>⬇️ Down</button>
                <button class="btn btn-small btn-danger" onclick="cancelJob('${job.id}')">❌ Cancel</button>
            </div>
        </div>
    `).join('');
}

function renderCompletedJobs(jobs) {
    const container = document.getElementById('completed-jobs');
    
    if (!jobs || jobs.length === 0) {
        container.innerHTML = '<p class="no-jobs">No completed jobs yet</p>';
        return;
    }
    
    container.innerHTML = jobs.map(job => {
        const statusClass = job.status === 'completed' ? 'status-completed' : 
                           job.status === 'failed' ? 'status-failed' : 'status-cancelled';
        const statusText = job.status.charAt(0).toUpperCase() + job.status.slice(1);
        
        return `
            <div class="job-card">
                <div class="job-header">
                    <span class="job-title">${job.filename}</span>
                    <span class="status-badge ${statusClass}">${statusText}</span>
                </div>
                <div class="job-meta">
                    Model: ${job.model.toUpperCase()} • Completed: ${formatDate(job.completed_at)}
                    ${job.output_file ? ` • <a href="#" onclick="viewFile('${job.output_file}'); return false;">📄 View Transcript</a>` : ''}
                    ${job.srt_file ? ` • <a href="#" onclick="viewFile('${job.srt_file}'); return false;">📺 View SRT</a>` : ''}
                </div>
                ${job.error ? `<div class="job-error">Error: ${job.error}</div>` : ''}
            </div>
        `;
    }).join('');
}

async function moveJobUp(jobId) {
    try {
        await fetch(`${API_BASE}/jobs/${jobId}/move-up`, { method: 'PUT' });
        loadJobs();
    } catch (error) {
        showToast('❌ Failed to move job', 'error');
    }
}

async function moveJobDown(jobId) {
    try {
        await fetch(`${API_BASE}/jobs/${jobId}/move-down`, { method: 'PUT' });
        loadJobs();
    } catch (error) {
        showToast('❌ Failed to move job', 'error');
    }
}

async function cancelJob(jobId) {
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}`, { method: 'DELETE' });
        if (response.ok) {
            showToast('✅ Job cancelled', 'success');
            loadJobs();
        } else {
            const error = await response.json();
            showToast(`❌ ${error.error}`, 'error');
        }
    } catch (error) {
        showToast('❌ Failed to cancel job', 'error');
    }
}

// ============================================================================
// Modal Event Handlers
// ============================================================================

document.getElementById('modal-close').addEventListener('click', closeTextModal);
document.getElementById('modal-close-btn').addEventListener('click', closeTextModal);
document.getElementById('modal-download').addEventListener('click', () => {
    if (currentViewFile) {
        downloadFile(currentViewFile);
    }
});

document.getElementById('delete-modal-close').addEventListener('click', closeDeleteModal);
document.getElementById('delete-cancel').addEventListener('click', closeDeleteModal);
document.getElementById('delete-confirm').addEventListener('click', confirmDelete);

// Close modals on backdrop click
document.getElementById('text-modal').addEventListener('click', (e) => {
    if (e.target.id === 'text-modal') closeTextModal();
});
document.getElementById('delete-modal').addEventListener('click', (e) => {
    if (e.target.id === 'delete-modal') closeDeleteModal();
});

// ============================================================================
// Refresh Buttons
// ============================================================================

document.getElementById('refresh-uploads-btn').addEventListener('click', refreshFiles);
document.getElementById('refresh-results-btn').addEventListener('click', async () => {
    await loadResults();
    showToast('📁 Results refreshed', 'success');
});

// ============================================================================
// GPU Status & Config
// ============================================================================

async function loadConfig() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        const config = await response.json();
        
        const gpuStatus = document.getElementById('gpu-status');
        if (config.gpu_available) {
            gpuStatus.textContent = `GPU: ${config.gpu_info.name} (${config.gpu_info.memory_gb.toFixed(1)}GB)`;
            gpuStatus.style.color = '#22c55e';
        } else {
            gpuStatus.textContent = 'GPU: Not available (using CPU)';
            gpuStatus.style.color = '#f59e0b';
        }
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

// ============================================================================
// Polling & Initialization
// ============================================================================

// Poll for job updates
setInterval(() => {
    const queueTab = document.getElementById('queue-tab');
    if (queueTab.classList.contains('active')) {
        loadJobs();
    }
}, POLL_INTERVAL);

// Initial load
document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
    loadFiles();
    loadJobs();
});
