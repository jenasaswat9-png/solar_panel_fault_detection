# Solar Panel Fault Detection - Docker Image
# Multi-stage build for production deployment

# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# Stage 2: Production
# ==========================================
FROM python:3.10-slim as production

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/checkpoints results logs

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_TYPE=resnet50
ENV MODEL_WEIGHTS=models/checkpoints/best_model.pth
ENV PYTHONUNBUFFERED=1

# Expose ports for API and Dashboard
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ==========================================
# Stage 3: Development
# ==========================================
FROM production as development

# Install additional development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    pytest \
    pytest-cov \
    black \
    flake8 \
    isort

# Set development environment
ENV ENVIRONMENT=development

# Default command for development
CMD ["/bin/bash"]

# ==========================================
# Stage 4: Training
# ==========================================
FROM production as training

# Install training-specific dependencies
RUN pip install --no-cache-dir \
    wandb \
    mlflow \
    tensorboard

# Set training environment
ENV ENVIRONMENT=training

# Default command for training
CMD ["python", "src/train.py", "--model", "yolo"]
