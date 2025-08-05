# Multi-stage build for efficient pygSQuiG container
# Stage 1: Build stage with all dependencies
FROM python:3.11-slim-bullseye AS builder

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY pyproject.toml README.md ./
COPY pygsquig ./pygsquig

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install JAX with CPU support (can be changed to GPU)
# For GPU support, use: pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --upgrade "jax[cpu]"

# Install the package and dependencies
RUN pip install -e .

# Install additional scientific computing tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipython \
    matplotlib-inline

# Stage 2: Runtime stage with minimal footprint
FROM python:3.11-slim-bullseye AS runtime

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORM_NAME=cpu
ENV JAX_ENABLE_X64=true

# Create non-root user for security
RUN useradd -m -s /bin/bash pygsquig
USER pygsquig
WORKDIR /home/pygsquig

# Copy the application code
COPY --chown=pygsquig:pygsquig . /home/pygsquig/app

# Set working directory to app
WORKDIR /home/pygsquig/app

# Expose ports for Jupyter
EXPOSE 8888

# Default command runs jupyter lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Alternative entrypoint for running scripts
# ENTRYPOINT ["python", "-m", "pygsquig.scripts.run"]