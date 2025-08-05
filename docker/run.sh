#!/bin/bash
# Run script for pygSQuiG Docker containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RUN_MODE="jupyter"
USE_GPU=false
MOUNT_CODE=true
CONFIG_FILE=""
DATA_DIR="./data"
NOTEBOOKS_DIR="./notebooks"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            RUN_MODE="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --no-mount)
            MOUNT_CODE=false
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE      Run mode: jupyter, shell, script, test (default: jupyter)"
            echo "  --gpu            Use GPU-enabled image"
            echo "  --no-mount       Don't mount code directories"
            echo "  --config FILE    Config file for script mode"
            echo "  --data-dir DIR   Data directory to mount (default: ./data)"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run Jupyter Lab"
            echo "  $0 --mode shell              # Interactive shell"
            echo "  $0 --mode script --config configs/example.yml"
            echo "  $0 --mode test               # Run tests"
            echo "  $0 --gpu --mode jupyter      # GPU-enabled Jupyter"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set image name based on GPU flag
if [ "$USE_GPU" = true ]; then
    IMAGE_NAME="pygsquig-gpu:latest"
    PLATFORM_ENV="-e JAX_PLATFORM_NAME=gpu"
    GPU_FLAGS="--gpus all"
    echo -e "${YELLOW}Using GPU-enabled image${NC}"
else
    IMAGE_NAME="pygsquig:latest"
    PLATFORM_ENV="-e JAX_PLATFORM_NAME=cpu"
    GPU_FLAGS=""
    echo -e "${YELLOW}Using CPU-only image${NC}"
fi

# Create directories if they don't exist
mkdir -p "$DATA_DIR" "$NOTEBOOKS_DIR"

# Build volume mount arguments
VOLUME_ARGS="-v $(realpath $DATA_DIR):/home/pygsquig/data"
VOLUME_ARGS="$VOLUME_ARGS -v $(realpath $NOTEBOOKS_DIR):/home/pygsquig/notebooks"

if [ "$MOUNT_CODE" = true ]; then
    VOLUME_ARGS="$VOLUME_ARGS -v $PROJECT_ROOT/pygsquig:/home/pygsquig/app/pygsquig:ro"
    VOLUME_ARGS="$VOLUME_ARGS -v $PROJECT_ROOT/examples:/home/pygsquig/app/examples:ro"
    VOLUME_ARGS="$VOLUME_ARGS -v $PROJECT_ROOT/tests:/home/pygsquig/app/tests:ro"
    
    if [ -d "$PROJECT_ROOT/configs" ]; then
        VOLUME_ARGS="$VOLUME_ARGS -v $PROJECT_ROOT/configs:/home/pygsquig/configs:ro"
    fi
fi

# Common docker run arguments
DOCKER_ARGS="--rm -it $GPU_FLAGS $PLATFORM_ENV -e JAX_ENABLE_X64=true $VOLUME_ARGS"

# Run based on mode
case $RUN_MODE in
    jupyter)
        echo -e "${GREEN}Starting Jupyter Lab...${NC}"
        echo -e "${BLUE}Access at: http://localhost:8888${NC}"
        docker run $DOCKER_ARGS \
            -p 8888:8888 \
            "$IMAGE_NAME" \
            jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
        ;;
        
    shell)
        echo -e "${GREEN}Starting interactive shell...${NC}"
        docker run $DOCKER_ARGS \
            "$IMAGE_NAME" \
            /bin/bash
        ;;
        
    script)
        if [ -z "$CONFIG_FILE" ]; then
            echo -e "${RED}Error: --config required for script mode${NC}"
            exit 1
        fi
        
        # Check if config file exists
        if [ ! -f "$CONFIG_FILE" ]; then
            echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
            exit 1
        fi
        
        # Get absolute path of config file
        CONFIG_ABS=$(realpath "$CONFIG_FILE")
        CONFIG_DIR=$(dirname "$CONFIG_ABS")
        CONFIG_NAME=$(basename "$CONFIG_ABS")
        
        echo -e "${GREEN}Running simulation with config: $CONFIG_FILE${NC}"
        docker run $DOCKER_ARGS \
            -v "$CONFIG_DIR:/home/pygsquig/run_config:ro" \
            "$IMAGE_NAME" \
            python -m pygsquig.scripts.run "/home/pygsquig/run_config/$CONFIG_NAME"
        ;;
        
    test)
        echo -e "${GREEN}Running tests...${NC}"
        docker run $DOCKER_ARGS \
            -v "$PROJECT_ROOT:/home/pygsquig/test_root:ro" \
            -w /home/pygsquig/test_root \
            "$IMAGE_NAME" \
            python -m pytest tests/ -v
        ;;
        
    *)
        echo -e "${RED}Unknown mode: $RUN_MODE${NC}"
        echo "Valid modes: jupyter, shell, script, test"
        exit 1
        ;;
esac