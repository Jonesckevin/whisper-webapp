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

// Cache for queue data to prevent flickering
let cachedQueueData = {
    current: '',
    queued: '',
    completed: ''
};

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
        } else if (tabId === 'logs') {
            loadLogs();
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
        const response = await fetch(`${API_BASE}/results`);
        const data = await response.json();
        
        renderResultsList(data.results);
    } catch (error) {
        console.error('Failed to load results:', error);
    }
}

function renderResultsList(results) {
    const container = document.getElementById('results-list');
    
    if (!results || results.length === 0) {
        container.innerHTML = '<p class="no-jobs">No transcripts yet. Process some files!</p>';
        return;
    }
    
    container.innerHTML = results.map(result => `
        <div class="file-item">
            <div class="file-info">
                <span class="file-icon">📄</span>
                <div class="file-details">
                    <div class="file-name">${result.filename}</div>
                    <div class="file-meta">Model: ${result.model.toUpperCase()} • ${formatDate(result.completed_at)}</div>
                </div>
            </div>
            <div class="file-actions">
                <button class="btn btn-small btn-secondary" onclick="viewTranscript('${result.id}')" title="View Transcript">📄 View</button>
                ${result.has_srt ? `<button class="btn btn-small btn-secondary" onclick="viewSrt('${result.id}')" title="View SRT">📺 SRT</button>` : ''}
                <button class="btn btn-small btn-secondary" onclick="downloadResult('${result.id}', 'transcript')" title="Download Transcript">⬇️ TXT</button>
                ${result.has_srt ? `<button class="btn btn-small btn-secondary" onclick="downloadResult('${result.id}', 'srt')" title="Download SRT">⬇️ SRT</button>` : ''}
            </div>
        </div>
    `).join('');
}

async function viewTranscript(jobId) {
    try {
        const response = await fetch(`${API_BASE}/results/${jobId}/transcript`);
        const data = await response.json();
        
        if (response.ok) {
            currentViewFile = `transcript-${jobId}`;
            document.getElementById('modal-title').textContent = data.filename;
            document.getElementById('modal-text').textContent = data.content;
            document.getElementById('text-modal').classList.remove('hidden');
        } else {
            showToast(`❌ ${data.error}`, 'error');
        }
    } catch (error) {
        showToast('❌ Failed to load transcript', 'error');
    }
}

async function viewSrt(jobId) {
    try {
        const response = await fetch(`${API_BASE}/results/${jobId}/srt`);
        const data = await response.json();
        
        if (response.ok) {
            currentViewFile = `srt-${jobId}`;
            document.getElementById('modal-title').textContent = `${data.filename} (SRT)`;
            document.getElementById('modal-text').textContent = data.content;
            document.getElementById('text-modal').classList.remove('hidden');
        } else {
            showToast(`❌ ${data.error}`, 'error');
        }
    } catch (error) {
        showToast('❌ Failed to load SRT', 'error');
    }
}

async function copyTranscript(jobId) {
    try {
        const response = await fetch(`${API_BASE}/results/${jobId}/transcript`);
        const data = await response.json();
        
        if (response.ok) {
            await navigator.clipboard.writeText(data.content);
            showToast('📋 Copied to clipboard', 'success');
        } else {
            showToast(`❌ ${data.error}`, 'error');
        }
    } catch (error) {
        showToast('❌ Failed to copy', 'error');
    }
}

function downloadResult(jobId, fileType) {
    window.open(`${API_BASE}/results/${jobId}/download/${fileType}`, '_blank');
}

function filterResults() {
    const searchTerm = document.getElementById('results-search').value.toLowerCase();
    const items = document.querySelectorAll('#results-list .file-item');
    
    items.forEach(item => {
        const filename = item.querySelector('.file-name').textContent.toLowerCase();
        item.style.display = filename.includes(searchTerm) ? '' : 'none';
    });
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
    const keepFile = document.getElementById('keep-file-checkbox').checked;
    
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
                generate_srt: generateSrt,
                keep_file: keepFile
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
        
        // Normalize data to ensure consistent comparison
        const currentStr = JSON.stringify(data.current || null);
        const queuedStr = JSON.stringify(data.queued || []);
        const completedStr = JSON.stringify(data.completed || []);
        
        // Update queue statistics
        const queuedCount = (data.queued || []).length;
        const completedCount = (data.completed || []).filter(j => j.status === 'completed').length;
        const failedCount = (data.completed || []).filter(j => j.status === 'failed' || j.status === 'cancelled').length;
        
        document.getElementById('stat-queued').textContent = `⏳ ${queuedCount}`;
        document.getElementById('stat-completed').textContent = `✅ ${completedCount}`;
        document.getElementById('stat-failed').textContent = `❌ ${failedCount}`;
        
        // Update DOM only if data has actually changed
        if (currentStr !== cachedQueueData.current) {
            cachedQueueData.current = currentStr;
            renderCurrentJob(data.current);
        }
        if (queuedStr !== cachedQueueData.queued) {
            cachedQueueData.queued = queuedStr;
            renderQueuedJobs(data.queued);
        }
        if (completedStr !== cachedQueueData.completed) {
            cachedQueueData.completed = completedStr;
            renderCompletedJobs(data.completed);
        }
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
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <span class="status-badge status-running">Running</span>
                    <button class="btn btn-small btn-danger" onclick="cancelRunningJob('${job.id}')" title="Cancel job">❌ Cancel</button>
                </div>
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
        
        // Calculate duration
        let duration = '';
        if (job.started_at && job.completed_at) {
            const start = new Date(job.started_at);
            const end = new Date(job.completed_at);
            const seconds = Math.round((end - start) / 1000);
            const minutes = Math.floor(seconds / 60);
            const secs = seconds % 60;
            duration = minutes > 0 ? `${minutes}m ${secs}s` : `${secs}s`;
        }
        
        return `
            <div class="job-card">
                <div class="job-header">
                    <span class="job-title">${job.filename}</span>
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        <span class="status-badge ${statusClass}">${statusText}</span>
                        <button class="btn btn-icon" onclick="deleteCompletedJob('${job.id}')" title="Delete job">🗑️</button>
                    </div>
                </div>
                <div class="job-meta">
                    Model: ${job.model.toUpperCase()} • Completed: ${formatDate(job.completed_at)}
                    ${duration ? ` • Duration: ${duration}` : ''}
                </div>
                ${job.has_transcript || job.has_srt ? `
                    <div class="job-actions" style="margin-top: 0.5rem;">
                        ${job.has_transcript ? `
                            <button class="btn btn-small btn-secondary" onclick="viewTranscript('${job.id}')">📄 View</button>
                            <button class="btn btn-small btn-secondary" onclick="copyTranscript('${job.id}')">📋 Copy</button>
                            <button class="btn btn-small btn-secondary" onclick="downloadResult('${job.id}', 'transcript')">⬇️ TXT</button>
                        ` : ''}
                        ${job.has_srt ? `
                            <button class="btn btn-small btn-secondary" onclick="viewSrt('${job.id}')">📺 SRT</button>
                            <button class="btn btn-small btn-secondary" onclick="downloadResult('${job.id}', 'srt')">⬇️ SRT</button>
                        ` : ''}
                    </div>
                ` : ''}
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

async function cancelRunningJob(jobId) {
    if (!confirm('Cancel this running job? Progress will be lost.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}`, { method: 'DELETE' });
        if (response.ok) {
            showToast('✅ Running job cancelled', 'success');
            loadJobs();
        } else {
            const error = await response.json();
            showToast(`❌ ${error.error}`, 'error');
        }
    } catch (error) {
        showToast('❌ Failed to cancel job', 'error');
    }
}

async function cancelRunningJob(jobId) {
    if (!confirm('Cancel this running job? Progress will be lost.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}`, { method: 'DELETE' });
        if (response.ok) {
            showToast('✅ Running job cancelled', 'success');
            loadJobs();
        } else {
            const error = await response.json();
            showToast(`❌ ${error.error}`, 'error');
        }
    } catch (error) {
        showToast('❌ Failed to cancel job', 'error');
    }
}

async function clearCompletedJobs() {
    if (!confirm('Clear all completed jobs? This action cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/jobs/completed`, { method: 'DELETE' });
        if (response.ok) {
            showToast('✅ Completed jobs cleared', 'success');
            cachedQueueData.completed = ''; // Clear cache to force re-render
            loadJobs();
            loadResults();
        } else {
            const error = await response.json();
            showToast(`❌ ${error.error}`, 'error');
        }
    } catch (error) {
        showToast('❌ Failed to clear completed jobs', 'error');
    }
}

async function deleteCompletedJob(jobId) {
    if (!confirm('Delete this job and its results? This cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}/delete`, { method: 'DELETE' });
        if (response.ok) {
            showToast('✅ Job deleted', 'success');
            cachedQueueData.completed = ''; // Clear cache to force re-render
            loadJobs();
            loadResults();
        } else {
            const error = await response.json();
            showToast(`❌ ${error.error}`, 'error');
        }
    } catch (error) {
        showToast('❌ Failed to delete job', 'error');
    }
}

// ============================================================================
// Modal Event Handlers
// ============================================================================

document.getElementById('modal-close').addEventListener('click', closeTextModal);
document.getElementById('modal-close-btn').addEventListener('click', closeTextModal);
document.getElementById('modal-copy').addEventListener('click', async () => {
    const text = document.getElementById('modal-text').textContent;
    try {
        await navigator.clipboard.writeText(text);
        showToast('📋 Copied to clipboard', 'success');
    } catch (error) {
        showToast('❌ Failed to copy', 'error');
    }
});
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
// Unified Refresh Function
// ============================================================================

async function refreshAll() {
    try {
        // Reset cache to force re-render
        cachedQueueData = {
            current: '',
            queued: '',
            completed: ''
        };
        
        // Refresh all data
        await Promise.all([
            loadFiles(),
            loadJobs(),
            loadResults(),
            loadLogs()
        ]);
        
        // Also refresh file select dropdown
        loadFileSelect();
        
        showToast('🔄 All data refreshed', 'success');
    } catch (error) {
        showToast('❌ Failed to refresh data', 'error');
    }
}

// ============================================================================
// GPU Status & Config
// ============================================================================

async function loadConfig() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        const config = await response.json();
        
        const gpuStatus = document.getElementById('gpu-status');
        if (config.gpu_available) {
            let statusText = `GPU: ${config.gpu_info.name} (${config.gpu_info.memory_gb.toFixed(1)}GB)`;
            if (config.vram_usage) {
                const used = config.vram_usage.reserved_gb;
                const total = config.vram_usage.total_gb;
                const percent = (used / total * 100).toFixed(0);
                statusText += ` | VRAM: ${used.toFixed(1)}/${total.toFixed(1)}GB (${percent}%)`;
            }
            gpuStatus.textContent = statusText;
            gpuStatus.style.color = '#22c55e';
        } else {
            gpuStatus.textContent = 'GPU: Not available (using CPU)';
            gpuStatus.style.color = '#f59e0b';
        }
        
        // Update model download status
        if (config.downloaded_models) {
            updateModelStatus(config.downloaded_models);
        }
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

function updateModelStatus(downloadedModels) {
    const modelRows = document.querySelectorAll('#model-table-body tr');
    const modelOrder = ['tiny', 'base', 'small', 'medium', 'large'];
    
    modelRows.forEach((row, index) => {
        const modelId = modelOrder[index];
        const statusCell = row.querySelector('.model-status');
        
        if (statusCell) {
            if (downloadedModels[modelId]) {
                statusCell.innerHTML = '<span style="color: #22c55e;">✅ Downloaded</span>';
            } else {
                statusCell.innerHTML = '<span style="color: #f59e0b;">📥 Will download</span>';
            }
        }
    });
}

function updateModelStatus(downloadedModels) {
    const modelRows = document.querySelectorAll('#model-table-body tr');
    const modelOrder = ['tiny', 'base', 'small', 'medium', 'large'];
    
    modelRows.forEach((row, index) => {
        const modelId = modelOrder[index];
        const statusCell = row.querySelector('.model-status');
        
        if (statusCell) {
            if (downloadedModels[modelId]) {
                statusCell.innerHTML = '<span style="color: #22c55e;">✅ Downloaded</span>';
            } else {
                statusCell.innerHTML = '<span style="color: #f59e0b;">📥 Will download</span>';
            }
        }
    });
}

// ============================================================================
// Logs
// ============================================================================

async function loadLogs() {
    try {
        const response = await fetch(`${API_BASE}/logs`);
        const logs = await response.json();
        
        const container = document.getElementById('logs-container');
        
        if (!logs || logs.length === 0) {
            container.innerHTML = '<p class="no-jobs">No logs yet</p>';
            return;
        }
        
        // Display logs in reverse chronological order (newest first)
        container.innerHTML = logs.map(log => `
            <div class="log-entry">
                <span class="log-timestamp">${formatDate(log.timestamp)}</span>
                <span class="log-level ${log.level}">${log.level}</span>
                <span class="log-message">${log.message}</span>
            </div>
        `).join('');
        
        // Auto-scroll to top (newest)
        container.scrollTop = 0;
    } catch (error) {
        console.error('Failed to load logs:', error);
    }
}

async function downloadLogs() {
    try {
        const response = await fetch(`${API_BASE}/logs`);
        const logs = await response.json();
        
        if (!logs || logs.length === 0) {
            showToast('❌ No logs to download', 'error');
            return;
        }
        
        // Format logs as plain text
        const logText = logs.map(log => 
            `${new Date(log.timestamp).toISOString()} [${log.level}] ${log.message}`
        ).join('\n');
        
        // Create and download file
        const blob = new Blob([logText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `whisper-logs-${new Date().toISOString().replace(/:/g, '-')}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showToast('💾 Logs downloaded', 'success');
    } catch (error) {
        showToast('❌ Failed to download logs', 'error');
    }
}

// ============================================================================
// Initialization
// ============================================================================

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ignore if typing in input fields
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
    }
    
    // Esc - close modals
    if (e.key === 'Escape') {
        closeTextModal();
        closeDeleteModal();
    }
    // R - refresh all
    else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        refreshAll();
    }
    // U - upload tab
    else if (e.key === 'u' || e.key === 'U') {
        e.preventDefault();
        document.querySelector('[data-tab="upload"]').click();
    }
});

// Initial load
document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
    loadFiles();
    loadJobs();
});
