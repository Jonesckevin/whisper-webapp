#!/bin/bash
# Start the Whisper Transcription Docker container
#
# Usage:
#   ./start-docker.sh           # Interactive menu
#   ./start-docker.sh web       # Start web mode
#   ./start-docker.sh cli       # Start CLI mode with defaults
#   ./start-docker.sh cli medium --srt  # CLI with medium model and SRT

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
MODE="${1:-menu}"
MODEL="${2:-base}"
GENERATE_SRT="true"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create data directories
mkdir -p data/uploads data/completed

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check for NVIDIA Docker support
NVIDIA_SUPPORT=false
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    NVIDIA_SUPPORT=true
fi

show_menu() {
    echo ""
    echo -e "${CYAN}========================================"
    echo "  Whisper Transcription Docker Setup"
    echo -e "========================================${NC}"
    echo ""
    
    if [ "$NVIDIA_SUPPORT" = true ]; then
        echo -e "  GPU Support: ${GREEN}ENABLED${NC}"
    else
        echo -e "  GPU Support: ${YELLOW}DISABLED (CPU only)${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}Select mode:${NC}"
    echo "  1. Web Mode - Start web interface (http://localhost:8080)"
    echo "  2. CLI Mode - Batch process files in data/uploads/"
    echo "  3. Build Only - Build Docker image without running"
    echo "  4. Stop - Stop running containers"
    echo "  5. Exit"
    echo ""
    
    read -p "Enter choice (1-5): " choice
    echo "$choice"
}

start_web_mode() {
    echo ""
    echo -e "${CYAN}Starting Web Mode...${NC}"
    echo ""
    
    docker-compose build
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================"
        echo "  Web interface started successfully!"
        echo -e "========================================${NC}"
        echo ""
        echo -e "  ${CYAN}URL: http://localhost:8080${NC}"
        echo ""
        echo -e "  ${YELLOW}Upload files via:${NC}"
        echo "    - Web interface drag & drop (max 5GB)"
        echo "    - Copy to: $SCRIPT_DIR/data/uploads/"
        echo ""
        echo -e "  ${YELLOW}Completed transcripts saved to:${NC}"
        echo "    $SCRIPT_DIR/data/completed/"
        echo ""
        echo -e "  To stop: docker-compose down"
        echo ""
        
        # Try to open browser
        if command -v xdg-open &> /dev/null; then
            xdg-open "http://localhost:8080" 2>/dev/null || true
        elif command -v open &> /dev/null; then
            open "http://localhost:8080" 2>/dev/null || true
        fi
    else
        echo -e "${RED}Failed to start Docker container${NC}"
        exit 1
    fi
}

start_cli_mode() {
    local selected_model="$1"
    local generate_srt="$2"
    
    echo ""
    echo -e "${CYAN}Starting CLI Mode...${NC}"
    echo -e "  ${YELLOW}Model: $selected_model${NC}"
    echo -e "  ${YELLOW}Generate SRT: $generate_srt${NC}"
    echo ""
    
    # Check for files
    file_count=$(find data/uploads -maxdepth 1 -type f | wc -l)
    if [ "$file_count" -eq 0 ]; then
        echo -e "${YELLOW}Warning: No files found in data/uploads/${NC}"
        echo "Please add audio/video files to: $SCRIPT_DIR/data/uploads/"
        return
    fi
    
    echo -e "${GREEN}Found $file_count file(s) to process:${NC}"
    ls -1 data/uploads/
    echo ""
    
    read -p "Proceed with transcription? (Y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy] ]]; then
        echo "Cancelled."
        return
    fi
    
    # Set environment variables
    export WHISPER_MODEL="$selected_model"
    export GENERATE_SRT="$generate_srt"
    
    docker-compose -f docker-compose.cli.yml build
    docker-compose -f docker-compose.cli.yml up --abort-on-container-exit
    
    echo ""
    echo -e "${GREEN}========================================"
    echo "  CLI transcription complete!"
    echo -e "========================================${NC}"
    echo ""
    echo -e "  ${YELLOW}Results saved to: $SCRIPT_DIR/data/completed/${NC}"
}

stop_containers() {
    echo -e "${YELLOW}Stopping containers...${NC}"
    docker-compose down 2>/dev/null || true
    docker-compose -f docker-compose.cli.yml down 2>/dev/null || true
    echo -e "${GREEN}Containers stopped.${NC}"
}

build_image() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker-compose build
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Build complete!${NC}"
    else
        echo -e "${RED}Build failed${NC}"
        exit 1
    fi
}

# Main execution
case "$MODE" in
    menu)
        choice=$(show_menu)
        
        case "$choice" in
            1)
                start_web_mode
                ;;
            2)
                echo ""
                echo -e "${YELLOW}Select Whisper model:${NC}"
                echo "  1. tiny   - Ultra fast, basic quality (~1GB VRAM)"
                echo "  2. base   - Recommended balance (~1.5GB VRAM)"
                echo "  3. small  - Better quality (~2.5GB VRAM)"
                echo "  4. medium - High quality (~4GB VRAM)"
                echo "  5. large  - Best quality (~5.5GB VRAM)"
                
                read -p "Enter choice (1-5, default 2): " model_choice
                
                case "$model_choice" in
                    1) selected_model="tiny" ;;
                    3) selected_model="small" ;;
                    4) selected_model="medium" ;;
                    5) selected_model="large" ;;
                    *) selected_model="base" ;;
                esac
                
                read -p "Generate SRT subtitles? (Y/N, default Y): " srt_choice
                if [[ "$srt_choice" =~ ^[Nn] ]]; then
                    generate_srt="false"
                else
                    generate_srt="true"
                fi
                
                start_cli_mode "$selected_model" "$generate_srt"
                ;;
            3)
                build_image
                ;;
            4)
                stop_containers
                ;;
            5)
                echo "Exiting..."
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                ;;
        esac
        ;;
    web)
        start_web_mode
        ;;
    cli)
        start_cli_mode "$MODEL" "$GENERATE_SRT"
        ;;
    stop)
        stop_containers
        ;;
    build)
        build_image
        ;;
    *)
        echo "Usage: $0 [web|cli|stop|build|menu]"
        exit 1
        ;;
esac
