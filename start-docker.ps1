<# 
.SYNOPSIS
    Start the Whisper Transcription Docker container

.DESCRIPTION
    This script starts the Whisper Transcription tool in either:
    - Web mode: Multi-user web interface with job queue
    - CLI mode: Batch processing of files in data/uploads

.EXAMPLE
    .\start-docker.ps1
    .\start-docker.ps1 -Mode web
    .\start-docker.ps1 -Mode cli -Model medium
#>

param(
    [Parameter()]
    [ValidateSet('web', 'cli', 'menu')]
    [string]$Mode = 'menu',
    
    [Parameter()]
    [ValidateSet('tiny', 'base', 'small', 'medium', 'large')]
    [string]$Model = 'base',
    
    [Parameter()]
    [switch]$NoSrt,
    
    [Parameter()]
    [switch]$Build
)

# Ensure we're in the right directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Create data directories if they don't exist
$DataDirs = @(
    ".\data\uploads",
    ".\data\completed"
)

foreach ($dir in $DataDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# Check for Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH"
    exit 1
}

# Check for NVIDIA Docker support
$NvidiaSupport = $false
try {
    $nvidiaSmi = docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        $NvidiaSupport = $true
    }
} catch {
    Write-Warning "NVIDIA Docker support not detected. GPU acceleration will not be available."
}

function Show-Menu {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Whisper Transcription Docker Setup" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    if ($NvidiaSupport) {
        Write-Host "  GPU Support: " -NoNewline
        Write-Host "ENABLED" -ForegroundColor Green
    } else {
        Write-Host "  GPU Support: " -NoNewline
        Write-Host "DISABLED (CPU only)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "Select mode:" -ForegroundColor Yellow
    Write-Host "  1. Web Mode - Start web interface (http://localhost:8080)"
    Write-Host "  2. CLI Mode - Batch process files in data/uploads/"
    Write-Host "  3. Build Only - Build Docker image without running"
    Write-Host "  4. Stop - Stop running containers"
    Write-Host "  5. Exit"
    Write-Host ""
    
    $choice = Read-Host "Enter choice (1-5)"
    return $choice
}

function Start-WebMode {
    Write-Host ""
    Write-Host "Starting Web Mode..." -ForegroundColor Cyan
    Write-Host ""
    
    if ($Build) {
        Write-Host "Building Docker image..." -ForegroundColor Yellow
        docker-compose build
    }
    
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  Web interface started successfully!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "  URL: http://localhost:8080" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  Upload files via:" -ForegroundColor Yellow
        Write-Host "    - Web interface drag & drop (max 5GB)" -ForegroundColor White
        Write-Host "    - Copy to: $ScriptDir\data\uploads\" -ForegroundColor White
        Write-Host ""
        Write-Host "  Completed transcripts saved to:" -ForegroundColor Yellow
        Write-Host "    $ScriptDir\data\completed\" -ForegroundColor White
        Write-Host ""
        Write-Host "  To stop: docker-compose down" -ForegroundColor Gray
        Write-Host ""
        
        # Open browser
        Start-Process "http://localhost:8080"
    } else {
        Write-Error "Failed to start Docker container"
    }
}

function Start-CLIMode {
    param([string]$SelectedModel, [bool]$GenerateSrt)
    
    Write-Host ""
    Write-Host "Starting CLI Mode..." -ForegroundColor Cyan
    Write-Host "  Model: $SelectedModel" -ForegroundColor Yellow
    Write-Host "  Generate SRT: $GenerateSrt" -ForegroundColor Yellow
    Write-Host ""
    
    # Check for files
    $uploadFiles = Get-ChildItem -Path ".\data\uploads" -File -ErrorAction SilentlyContinue
    if (-not $uploadFiles) {
        Write-Warning "No files found in data/uploads/"
        Write-Host "Please add audio/video files to: $ScriptDir\data\uploads\" -ForegroundColor Yellow
        return
    }
    
    Write-Host "Found $($uploadFiles.Count) file(s) to process:" -ForegroundColor Green
    foreach ($file in $uploadFiles) {
        Write-Host "  - $($file.Name)" -ForegroundColor White
    }
    Write-Host ""
    
    $confirm = Read-Host "Proceed with transcription? (Y/N)"
    if ($confirm -notmatch '^[Yy]') {
        Write-Host "Cancelled."
        return
    }
    
    # Set environment variables
    $env:WHISPER_MODEL = $SelectedModel
    $env:GENERATE_SRT = if ($GenerateSrt) { "true" } else { "" }
    
    if ($Build) {
        Write-Host "Building Docker image..." -ForegroundColor Yellow
        docker-compose -f docker-compose.cli.yml build
    }
    
    docker-compose -f docker-compose.cli.yml up --abort-on-container-exit
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  CLI transcription complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Results saved to: $ScriptDir\data\completed\" -ForegroundColor Yellow
}

function Stop-Containers {
    Write-Host "Stopping containers..." -ForegroundColor Yellow
    docker-compose down
    docker-compose -f docker-compose.cli.yml down 2>$null
    Write-Host "Containers stopped." -ForegroundColor Green
}

function Build-Image {
    Write-Host "Building Docker image..." -ForegroundColor Yellow
    docker-compose build
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build complete!" -ForegroundColor Green
    } else {
        Write-Error "Build failed"
    }
}

# Main execution
if ($Mode -eq 'menu') {
    $choice = Show-Menu
    
    switch ($choice) {
        "1" {
            $Build = $true
            Start-WebMode
        }
        "2" {
            $Build = $true
            
            Write-Host ""
            Write-Host "Select Whisper model:" -ForegroundColor Yellow
            Write-Host "  1. tiny   - Ultra fast, basic quality (~1GB VRAM)"
            Write-Host "  2. base   - Recommended balance (~1.5GB VRAM)"
            Write-Host "  3. small  - Better quality (~2.5GB VRAM)"
            Write-Host "  4. medium - High quality (~4GB VRAM)"
            Write-Host "  5. large  - Best quality (~5.5GB VRAM)"
            
            $modelChoice = Read-Host "Enter choice (1-5, default 2)"
            $modelMap = @{
                "1" = "tiny"
                "2" = "base"
                "3" = "small"
                "4" = "medium"
                "5" = "large"
            }
            $selectedModel = if ($modelMap.ContainsKey($modelChoice)) { $modelMap[$modelChoice] } else { "base" }
            
            $srtChoice = Read-Host "Generate SRT subtitles? (Y/N, default Y)"
            $generateSrt = $srtChoice -notmatch '^[Nn]'
            
            Start-CLIMode -SelectedModel $selectedModel -GenerateSrt $generateSrt
        }
        "3" {
            Build-Image
        }
        "4" {
            Stop-Containers
        }
        "5" {
            Write-Host "Exiting..."
            exit 0
        }
        default {
            Write-Host "Invalid choice" -ForegroundColor Red
        }
    }
} elseif ($Mode -eq 'web') {
    Start-WebMode
} elseif ($Mode -eq 'cli') {
    Start-CLIMode -SelectedModel $Model -GenerateSrt (-not $NoSrt)
}
