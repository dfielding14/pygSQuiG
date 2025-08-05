# Docker Support for pygSQuiG

This directory contains Docker configurations for running pygSQuiG in containerized environments, ensuring reproducibility and ease of deployment.

## Quick Start

### Building the Container

```bash
# Build CPU-only image (default)
./docker/build.sh

# Build GPU-enabled image
./docker/build.sh --gpu

# Build with specific tag
./docker/build.sh --tag v1.0.0

# Build without cache
./docker/build.sh --no-cache
```

### Running the Container

```bash
# Start Jupyter Lab (default)
./docker/run.sh

# Start interactive shell
./docker/run.sh --mode shell

# Run simulation with config file
./docker/run.sh --mode script --config configs/example.yml

# Run tests
./docker/run.sh --mode test

# Use GPU-enabled container
./docker/run.sh --gpu
```

## Available Images

### CPU Image (`pygsquig:latest`)
- Based on `python:3.11-slim-bullseye`
- Includes JAX with CPU support
- Minimal footprint (~500MB)
- Suitable for development and small-scale simulations

### GPU Image (`pygsquig-gpu:latest`)
- Based on `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- Includes JAX with CUDA 12 support
- GPU memory management utilities
- Suitable for large-scale simulations

## Docker Compose

For more complex setups, use Docker Compose:

```bash
# Start Jupyter Lab service
docker-compose up pygsquig

# Run simulation
docker-compose run pygsquig-run configs/example.yml

# Run tests
docker-compose up pygsquig-test

# Run with GPU support
docker-compose --profile gpu up pygsquig
```

## Volume Mounts

The containers mount several directories for development:

| Host Directory | Container Path | Purpose |
|----------------|----------------|---------|
| `./pygsquig` | `/home/pygsquig/app/pygsquig` | Source code (read-only) |
| `./examples` | `/home/pygsquig/app/examples` | Example scripts (read-only) |
| `./tests` | `/home/pygsquig/app/tests` | Test files (read-only) |
| `./data` | `/home/pygsquig/data` | Output data (read-write) |
| `./notebooks` | `/home/pygsquig/notebooks` | Jupyter notebooks (read-write) |
| `./configs` | `/home/pygsquig/configs` | Configuration files (read-only) |

## Environment Variables

### JAX Configuration
- `JAX_PLATFORM_NAME`: `cpu` or `gpu`
- `JAX_ENABLE_X64`: Enable 64-bit precision (default: `true`)
- `XLA_PYTHON_CLIENT_PREALLOCATE`: GPU memory preallocation (default: `false`)
- `XLA_PYTHON_CLIENT_MEM_FRACTION`: GPU memory fraction (default: `0.8`)

### pygSQuiG Configuration
- `PYGSQUIG_LOG_LEVEL`: Logging level (default: `INFO`)

## Use Cases

### 1. Development Environment

Start Jupyter Lab for interactive development:

```bash
./docker/run.sh
# Access at http://localhost:8888
```

### 2. Running Simulations

Run a simulation with configuration file:

```bash
# Create config file
cat > configs/turbulence.yml << EOF
simulation:
  N: 512
  L: 6.283185307179586
  alpha: 1.0
  nu_p: 1e-8
  p: 8
  
forcing:
  type: ring
  kf: 40.0
  epsilon: 0.1
  
time:
  t_final: 100.0
  dt: 0.001
  
output:
  save_interval: 1.0
  filename: "turbulence_run"
EOF

# Run simulation
./docker/run.sh --mode script --config configs/turbulence.yml
```

### 3. Batch Processing

Use Docker Compose for parallel runs:

```yaml
# docker-compose.override.yml
version: '3.8'

services:
  sim1:
    extends:
      service: pygsquig-run
    command: ["configs/run1.yml"]
    
  sim2:
    extends:
      service: pygsquig-run
    command: ["configs/run2.yml"]
```

Then run: `docker-compose up sim1 sim2`

### 4. GPU Acceleration

For GPU-accelerated simulations:

```bash
# Build GPU image
./docker/build.sh --gpu

# Run with GPU
./docker/run.sh --gpu --mode script --config configs/large_simulation.yml
```

### 5. Testing

Run the test suite in a clean environment:

```bash
./docker/run.sh --mode test
```

## Advanced Usage

### Custom Dockerfile

Create a custom Dockerfile for specific needs:

```dockerfile
# Dockerfile.custom
FROM pygsquig:latest

# Add custom dependencies
RUN pip install additional-package

# Add custom scripts
COPY my_analysis.py /home/pygsquig/

# Set custom entrypoint
ENTRYPOINT ["python", "/home/pygsquig/my_analysis.py"]
```

### Multi-Stage Builds

The Dockerfiles use multi-stage builds to minimize image size:

1. **Builder stage**: Installs all build dependencies
2. **Runtime stage**: Contains only runtime requirements

### Security

- Containers run as non-root user `pygsquig`
- Read-only mounts for code directories
- No unnecessary system packages

## Troubleshooting

### Common Issues

1. **GPU not detected**
   - Ensure NVIDIA Docker runtime is installed
   - Check GPU availability: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

2. **Permission denied errors**
   - Ensure data directories have correct permissions
   - Run: `chmod -R 755 data/ notebooks/`

3. **Out of memory**
   - For GPU: Adjust `XLA_PYTHON_CLIENT_MEM_FRACTION`
   - For CPU: Increase Docker memory limit

4. **Jupyter token required**
   - Check container logs: `docker logs pygsquig_dev`
   - Token is displayed in startup messages

### Debugging

Enter a running container:

```bash
# Find container ID
docker ps

# Enter container
docker exec -it <container_id> /bin/bash
```

Check logs:

```bash
# View logs
docker logs pygsquig_dev

# Follow logs
docker logs -f pygsquig_dev
```

## Best Practices

1. **Version Control**: Tag images with version numbers
2. **Data Persistence**: Always mount data directories
3. **Resource Limits**: Set appropriate CPU/memory limits
4. **Clean Up**: Remove unused images with `docker system prune`

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Docker Build

on: [push, pull_request]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: ./docker/build.sh --tag ${{ github.sha }}
        
      - name: Run tests in Docker
        run: ./docker/run.sh --mode test
```

## Contributing

When adding new dependencies:

1. Update `pyproject.toml`
2. Rebuild Docker images
3. Test both CPU and GPU images
4. Update documentation

## Future Enhancements

- [ ] Kubernetes deployment manifests
- [ ] Docker Hub automated builds
- [ ] Multi-architecture support (ARM64)
- [ ] Development container for VS Code
- [ ] Singularity/Apptainer support