FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PIP_PREFER_BINARY=1
ENV PYTHONPATH=/app

# Install only essential packages to reduce memory usage
RUN pip install --upgrade pip && \
    pip install fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-dotenv==1.0.0 \
    aiofiles==23.2.0 \
    python-multipart==0.0.6 \
    requests==2.31.0

# Copy application code
COPY backend ./backend
COPY configs ./configs

# Create a minimal app for demo
RUN echo 'from fastapi import FastAPI\nfrom fastapi.responses import JSONResponse\nimport datetime\n\napp = FastAPI(title="Contract Analyzer", description="AI-powered legal document analysis")\n\n@app.get("/")\nasync def root():\n    return {"message": "Contract Analyzer API", "status": "running", "timestamp": datetime.datetime.now()}\n\n@app.get("/health")\nasync def health():\n    return {"status": "healthy", "timestamp": datetime.datetime.now()}\n\n@app.post("/analyze")\nasync def analyze_contract():\n    return {"message": "Contract analysis feature coming soon", "status": "demo_mode"}\n\nif __name__ == "__main__":\n    import uvicorn\n    uvicorn.run(app, host="0.0.0.0", port=8000)' > backend/minimal_app.py

# Expose port
EXPOSE 8000

# Start the minimal application
CMD uvicorn backend.minimal_app:app --host 0.0.0.0 --port ${PORT:-8000}
