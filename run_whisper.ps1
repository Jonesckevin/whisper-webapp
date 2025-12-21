# PowerShell script to activate Python venv, run audio_to_text4.py, then deactivate

# Set execution policy for this session to allow script execution
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# Set the path to your virtual environment and script
$venvPath = ".\.venv_312"
$pythonScript = "src\audio_to_text4.py"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

# Activate the virtual environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (-Not (Test-Path $activateScript)) {
    Write-Error "Virtual environment activation script not found at $activateScript"
    $resp = Read-Host "Activation script not found. Create a new virtual environment at $venvPath now? [Y/N]"
    if ($resp -match '^[Yy]') {
        try {
            if (Get-Command py -ErrorAction SilentlyContinue) {
                & py -3.12 -m venv $venvPath
            }
            elseif (Get-Command python -ErrorAction SilentlyContinue) {
                & python -m venv $venvPath
            }
            else {
                throw "Neither 'py' nor 'python' was found in PATH."
            }
        }
        catch {
            Write-Error "Failed to create virtual environment: $($_.Exception.Message)"
            return
        }

        $newActivate = Join-Path $venvPath "Scripts\Activate.ps1"
        if (Test-Path $newActivate) {
            Write-Host "Virtual environment created at $venvPath. Re-running script..."
            $self = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.MyCommand.Path }
            $shell = if ($PSVersionTable.PSEdition -eq 'Core') { 'pwsh' } else { 'powershell' }
            Start-Process -FilePath $shell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$self`"" -Wait
            exit 0
        }
        else {
            Write-Error "Activation script still not found after venv creation."
        }
    }
    else {
        Write-Host "User declined to create virtual environment."
    }
    exit 1
}

Write-Host "Activating virtual environment..."
& $activateScript

# Use the full path to the venv's python executable
& $pythonExe -m pip install --upgrade pip

## Run requirements installation
$requirementsFile = ".\src\requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Installing requirements from $requirementsFile..."
    & $pythonExe -m pip install -r $requirementsFile
}
else {
    Write-Host "No requirements.txt found, skipping installation."
}

## Check for CUDA support and fix if needed
Write-Host "`nChecking CUDA availability..." -ForegroundColor Cyan
$cudaCheck = & $pythonExe -c "import torch; print('CUDA_AVAILABLE' if torch.cuda.is_available() else 'CUDA_MISSING'); print('BUILD:', torch.version.cuda if torch.version.cuda else 'cpu')" 2>&1

if ($cudaCheck -match "CUDA_MISSING" -or $cudaCheck -match "BUILD: cpu") {
    # Check if NVIDIA GPU is available
    $nvidiaSmi = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $nvidiaSmi) {
        Write-Host "⚠️  NVIDIA GPU detected ($nvidiaSmi) but PyTorch CUDA is not available!" -ForegroundColor Yellow
        Write-Host "   PyTorch was installed as CPU-only version." -ForegroundColor Yellow
        Write-Host ""
        $fixCuda = Read-Host "Would you like to install PyTorch with CUDA support? [Y/N]"
        if ($fixCuda -match '^[Yy]') {
            Write-Host "🔧 Uninstalling CPU-only PyTorch..." -ForegroundColor Cyan
            & $pythonExe -m pip uninstall torch torchvision torchaudio -y
            
            Write-Host "🔧 Installing PyTorch with CUDA 12.8 support..." -ForegroundColor Cyan
            & $pythonExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
            
            # Verify installation
            $cudaVerify = & $pythonExe -c "import torch; print('SUCCESS' if torch.cuda.is_available() else 'FAILED')" 2>&1
            if ($cudaVerify -match "SUCCESS") {
                Write-Host "✅ PyTorch CUDA installation successful!" -ForegroundColor Green
                $gpuName = & $pythonExe -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
                Write-Host "   GPU detected: $gpuName" -ForegroundColor Green
            }
            else {
                Write-Host "❌ CUDA still not available. You may need to:" -ForegroundColor Red
                Write-Host "   1. Update your NVIDIA drivers" -ForegroundColor Yellow
                Write-Host "   2. Install CUDA Toolkit from NVIDIA" -ForegroundColor Yellow
                Write-Host "   Continuing with CPU mode..." -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "Skipping CUDA installation. Transcription will use CPU (slower)." -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "ℹ️  No NVIDIA GPU detected. Using CPU for transcription." -ForegroundColor Yellow
    }
}
else {
    $gpuName = & $pythonExe -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    Write-Host "✅ CUDA is available! GPU: $gpuName" -ForegroundColor Green
}

# Ask user what they want to do
Write-Host "`nCurtis Transcription Tool - Enhanced Edition" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Choose an option:" -ForegroundColor Yellow
Write-Host "1. 🎤 Transcribe single audio/video to text" -ForegroundColor Green
Write-Host "2. 🎬 Convert single video to audio" -ForegroundColor Yellow
Write-Host "3. 🚀 BATCH: Convert ALL videos to audio" -ForegroundColor Magenta
Write-Host "4. 🚀 BATCH: Transcribe ALL audio files" -ForegroundColor Magenta
Write-Host "5. 📁 Organize files (separate video/audio)" -ForegroundColor Cyan
Write-Host "6. ❌ Exit" -ForegroundColor Red

$choice = Read-Host "`nEnter your choice (1-6)"

switch ($choice) {
    "1" {
        Write-Host "🎤 Running single file transcription mode..." -ForegroundColor Green
        $srtChoice = Read-Host "Generate SRT subtitle file alongside transcript? [Y/N]"
        if ($srtChoice -match '^[Yy]') {
            Write-Host "📺 SRT subtitle generation enabled" -ForegroundColor Cyan
            & $pythonExe $pythonScript --srt
        } else {
            & $pythonExe $pythonScript
        }
    }
    "2" {
        Write-Host "🎬 Running single video-to-audio conversion mode..." -ForegroundColor Yellow
        & $pythonExe $pythonScript --convert-only
    }
    "3" {
        Write-Host "🚀 Starting batch video conversion..." -ForegroundColor Magenta
        Write-Host "Converting all videos in holding/video/ to audio..." -ForegroundColor Yellow
        & $pythonExe $pythonScript --convert-all
    }
    "4" {
        Write-Host "🚀 Starting batch audio transcription..." -ForegroundColor Magenta
        
        # Ask for model selection for batch processing
        Write-Host "`nSelect Whisper model for batch transcription:" -ForegroundColor Cyan
        Write-Host "1. tiny   - Ultra fast, basic quality" -ForegroundColor White
        Write-Host "2. base   - Recommended for RTX 3060 (balanced)" -ForegroundColor Green
        Write-Host "3. small  - High quality, slower" -ForegroundColor Yellow
        Write-Host "4. medium - Premium quality, much slower" -ForegroundColor Yellow
        Write-Host "5. large  - Maximum quality, very slow" -ForegroundColor Red
        
        $modelChoice = Read-Host 'Enter model choice (1-5, default 2)'
        
        $modelMap = @{
            "1" = "tiny"
            "2" = "base"
            "3" = "small"
            "4" = "medium"
            "5" = "large"
        }
        
        $selectedModel = if ($modelMap.ContainsKey($modelChoice)) { $modelMap[$modelChoice] } else { "base" }
        
        Write-Host "Using model: $selectedModel" -ForegroundColor Green
        
        $srtChoice = Read-Host "`nGenerate SRT subtitle files for each transcription? [Y/N]"
        $srtFlag = ""
        if ($srtChoice -match '^[Yy]') {
            Write-Host "📺 SRT subtitle generation enabled" -ForegroundColor Cyan
            $srtFlag = "--srt"
        }
        
        Write-Host "Transcribing all audio files in holding/audio/..." -ForegroundColor Yellow
        if ($srtFlag) {
            & $pythonExe $pythonScript --transcribe-all --batch-model $selectedModel $srtFlag
        } else {
            & $pythonExe $pythonScript --transcribe-all --batch-model $selectedModel
        }
    }
    "5" {
        Write-Host "📁 Organizing files in holding directory..." -ForegroundColor Cyan
        & $pythonExe $pythonScript --organize
    }
    "6" {
        Write-Host "❌ Exiting..." -ForegroundColor Red
        deactivate
        exit 0
    }
    default {
        Write-Host "Invalid choice. Running default transcription mode..." -ForegroundColor Yellow
        & $pythonExe $pythonScript
    }
}

# Deactivate the virtual environment
Write-Host "Deactivating virtual environment..."
deactivate