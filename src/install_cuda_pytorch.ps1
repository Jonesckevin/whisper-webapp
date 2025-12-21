# PowerShell script to install CUDA-enabled PyTorch for better Whisper performance

Write-Host "Installing CUDA-enabled PyTorch for GPU acceleration..." -ForegroundColor Green

# Activate virtual environment
$venvPath = ".\.venv_312"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..."
    & $activateScript
    
    # Uninstall CPU-only PyTorch if present
    Write-Host "Removing CPU-only PyTorch..."
    python -m pip uninstall torch torchvision torchaudio -y
    
    # Install CUDA-enabled PyTorch
    Write-Host "Installing CUDA-enabled PyTorch..."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Test CUDA availability
    Write-Host "Testing CUDA availability..."
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only')"
    
    Write-Host "CUDA PyTorch installation completed!" -ForegroundColor Green
} else {
    Write-Host "Virtual environment not found. Please run run.ps1 first to create the environment." -ForegroundColor Red
}