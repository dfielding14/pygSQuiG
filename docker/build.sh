#!/bin/bash
# Build script for pygSQuiG Docker images

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="cpu"
IMAGE_TAG="latest"
NO_CACHE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            BUILD_TYPE="gpu"
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu        Build GPU-enabled image"
            echo "  --tag TAG    Specify image tag (default: latest)"
            echo "  --no-cache   Build without cache"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set build parameters based on type
if [ "$BUILD_TYPE" == "gpu" ]; then
    DOCKERFILE="Dockerfile.gpu"
    IMAGE_NAME="pygsquig-gpu"
    echo -e "${YELLOW}Building GPU-enabled image...${NC}"
else
    DOCKERFILE="Dockerfile"
    IMAGE_NAME="pygsquig"
    echo -e "${YELLOW}Building CPU-only image...${NC}"
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Build the image
echo -e "${GREEN}Building ${IMAGE_NAME}:${IMAGE_TAG}...${NC}"
docker build $NO_CACHE \
    -f "$DOCKERFILE" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:latest" \
    .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo -e "${GREEN}Image tagged as:${NC}"
    echo "  - ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "  - ${IMAGE_NAME}:latest"
    
    # Show image size
    echo -e "\n${YELLOW}Image size:${NC}"
    docker images "${IMAGE_NAME}:${IMAGE_TAG}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi

# Optional: Test the image
echo -e "\n${YELLOW}Testing image...${NC}"
docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python -c "import pygsquig; print('pygSQuiG version:', pygsquig.__version__)"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image test passed!${NC}"
else
    echo -e "${RED}✗ Image test failed!${NC}"
    exit 1
fi

echo -e "\n${GREEN}Done! You can now run the container with:${NC}"
echo "  docker run -it --rm -p 8888:8888 ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  or"
echo "  docker-compose up pygsquig"