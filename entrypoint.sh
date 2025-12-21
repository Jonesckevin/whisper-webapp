#!/bin/bash
# Whisper Transcription Web App - Entrypoint Script
# Handles optional model pre-download on first run

set -e

echo "🚀 Starting Whisper Transcription Web App..."

# Check if models should be pre-downloaded
if [ "${PRELOAD_WHISPER_MODELS}" = "true" ]; then
    echo "📥 Pre-downloading Whisper models..."
    echo "   This will take a few minutes on first run..."
    
    python3.12 -c "
import whisper
import sys

models = ['tiny', 'base', 'small', 'medium', 'large']
print('Downloading models: ' + ', '.join(models))

for i, model_name in enumerate(models, 1):
    print(f'[{i}/{len(models)}] Downloading {model_name}...')
    try:
        whisper.load_model(model_name, download_root='/root/.cache/whisper')
        print(f'✅ {model_name} downloaded')
    except Exception as e:
        print(f'❌ Failed to download {model_name}: {e}', file=sys.stderr)
        sys.exit(1)

print('✅ All Whisper models pre-downloaded and ready')
"
else
    echo "⏩ Skipping model pre-download (PRELOAD_WHISPER_MODELS not set)"
    echo "   Models will be downloaded on-demand when first used"
fi

echo "🎬 Starting application services..."

# Start supervisord
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
