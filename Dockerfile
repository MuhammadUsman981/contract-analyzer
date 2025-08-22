FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for faster builds
ENV PIP_PREFER_BINARY=1
ENV PYTHONPATH=/app

# Copy requirements
COPY requirements-production.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements-production.txt

# Create necessary directories
RUN mkdir -p /app/data /app/model_cache

# Copy application code
COPY backend ./backend
COPY frontend ./frontend
COPY configs ./configs
COPY data ./data

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "backend/app.py"]
