FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PIP_PREFER_BINARY=1
ENV PYTHONPATH=/app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install "numpy<2" && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/model_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["sh", "-c", "PYTHONPATH=/app uvicorn backend.app:app --host 0.0.0.0 --port 8000"]
