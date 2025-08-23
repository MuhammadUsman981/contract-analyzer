FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PIP_PREFER_BINARY=1
ENV PYTHONPATH=/app

# Copy Railway-specific requirements
COPY requirements-railway.txt .

# Install lightweight packages
RUN pip install --upgrade pip && \
    pip install -r requirements-railway.txt

# Copy application code
COPY backend ./backend
COPY configs ./configs

# Create start script to handle PORT variable
RUN echo '#!/bin/bash\nPORT=${PORT:-8000}\nexec uvicorn backend.railway_app:app --host 0.0.0.0 --port $PORT' > start.sh && \
    chmod +x start.sh

# Create the lightweight railway app
RUN echo 'from fastapi import FastAPI\nfrom fastapi.responses import JSONResponse\nimport datetime\nimport os\n\napp = FastAPI(\n    title="Contract Analyzer API",\n    description="AI-powered legal document analysis",\n    version="1.0.0"\n)\n\n@app.get("/")\nasync def root():\n    return {\n        "message": "Contract Analyzer API",\n        "status": "running",\n        "version": "1.0.0",\n        "timestamp": datetime.datetime.now().isoformat(),\n        "endpoints": {\n            "health": "/health",\n            "docs": "/docs",\n            "analyze": "/analyze"\n        }\n    }\n\n@app.get("/health")\nasync def health():\n    return {\n        "status": "healthy",\n        "timestamp": datetime.datetime.now().isoformat(),\n        "service": "contract-analyzer",\n        "environment": "production",\n        "memory_usage": "optimized"\n    }\n\n@app.post("/analyze")\nasync def analyze_contract():\n    return {\n        "message": "Contract analysis endpoint",\n        "status": "demo_mode",\n        "note": "This is a lightweight deployment. Full ML analysis available in development environment.",\n        "timestamp": datetime.datetime.now().isoformat(),\n        "demo_features": [\n            "API structure demonstration",\n            "Health monitoring",\n            "Documentation generation",\n            "Production-ready deployment"\n        ]\n    }\n\nif __name__ == "__main__":\n    import uvicorn\n    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))' > backend/railway_app.py

# Expose port
EXPOSE 8000

# Use exec form with shell script to handle environment variables
CMD ["./start.sh"]
