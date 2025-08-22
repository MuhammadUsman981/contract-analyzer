"""
FastAPI Deployment for CPU-Safe Legal Document Analyzer
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import uvicorn
import logging
import json
import os
from datetime import datetime
import PyPDF2
import io
import psutil

# Absolute import for legal_analyzer_cpu - import the actual available names
from backend.legal_analyzer_cpu import (
    CPUSafeRAGLegalAnalyzer,
    init_wandb_from_env,
    SystemMonitor,
    CPUConfig,
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal-analyzer-api")

# ------------------ Pydantic Models ------------------

class ContractAnalysisRequest(BaseModel):
    text: str = Field(..., description="Contract text to analyze")
    analysis_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    success: bool
    analysis_id: str
    metadata: Dict[str, Any]
    executive_summary: Dict[str, Any]
    clause_analysis: List[Dict[str, Any]]
    identified_entities: Dict[str, List[Dict[str, Any]]]
    analysis_duration: float
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    system_info: Dict[str, Any]
    model_loaded: bool

# ------------------ Globals ------------------

analyzer = None
system_monitor = None

# ------------------ Analyzer Init ------------------

# Update KB path to be relative to backend folder
kb_path = os.path.join(os.path.dirname(__file__), "legal_knowledge_base")

def initialize_analyzer():
    """Initialize analyzer with shared KB cache"""
    global analyzer, system_monitor
    try:
        logger.info("Initializing Legal Document Analyzer...")

        # Load KB from shared cache dir
        # kb_path = os.getenv("KB_PATH", "legal_knowledge_base")

        # Initialize analyzer (loads embeddings + FAISS from cache if available)
        analyzer = CPUSafeRAGLegalAnalyzer(knowledge_base_path=kb_path)

        # W&B + system monitor
        init_wandb_from_env(default_project="legal-analyzer-api")
        system_monitor = SystemMonitor()
        system_monitor.start()

        logger.info(f"Analyzer ready (using KB from {kb_path})")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return False

# ------------------ FastAPI App ------------------

app = FastAPI(
    title="Legal Document Analyzer API",
    description="CPU-optimized legal contract analysis with shared cache",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    success = initialize_analyzer()
    if not success:
        logger.error("⚠️ Analyzer failed to initialize - API degraded")

@app.on_event("shutdown")
async def shutdown_event():
    global system_monitor
    if system_monitor:
        system_monitor.stop()

# ------------------ Endpoints ------------------

@app.get("/", response_model=Dict[str, Any])
async def root():
    return {
        "message": "Legal Document Analyzer API",
        "version": "1.1.0",
        "docs": "/docs",
        "endpoints": {
            "analyze_text": "/analyze/text",
            "analyze_pdf": "/analyze/pdf",
            "health": "/health",
            "config": "/config"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    global analyzer
    memory = psutil.virtual_memory()
    system_info = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": memory.percent,
        "cpu_count": os.cpu_count(),
        "omp_threads": CPUConfig.OMP_NUM_THREADS,
        "model_name": CPUConfig.MODEL_NAME,
        "device": "cpu"
    }
    return HealthResponse(
        status="healthy" if analyzer else "degraded",
        timestamp=datetime.now().isoformat(),
        system_info=system_info,
        model_loaded=analyzer is not None
    )

@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: ContractAnalysisRequest):
    global analyzer
    if analyzer is None:
        raise HTTPException(503, "Analyzer not initialized")

    if not request.text.strip():
        raise HTTPException(400, "Empty contract text")

    analysis_id = request.analysis_id or f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        result = analyzer.analyze_contract(request.text, analysis_id)
        return AnalysisResponse(
            success=True,
            analysis_id=analysis_id,
            metadata=result["metadata"],
            executive_summary=result["executive_summary"],
            clause_analysis=result["clause_analysis"],
            identified_entities=result["identified_entities"],
            analysis_duration=result["analysis_duration"],
            message="Analysis complete"
        )
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.post("/analyze/pdf", response_model=AnalysisResponse)
async def analyze_pdf(file: UploadFile = File(...)):
    global analyzer
    if analyzer is None:
        raise HTTPException(503, "Analyzer not initialized")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF supported")

    try:
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)
        if not text.strip():
            raise HTTPException(400, "No text extracted")

        analysis_id = f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = analyzer.analyze_contract(text, analysis_id)
        return AnalysisResponse(
            success=True,
            analysis_id=analysis_id,
            metadata={**result["metadata"], "source_file": file.filename},
            executive_summary=result["executive_summary"],
            clause_analysis=result["clause_analysis"],
            identified_entities=result["identified_entities"],
            analysis_duration=result["analysis_duration"],
            message=f"Analysis complete for {file.filename}"
        )
    except Exception as e:
        raise HTTPException(500, f"PDF analysis failed: {str(e)}")

@app.get("/config")
async def get_config():
    return {
        "model": CPUConfig.MODEL_NAME,
        "max_length": CPUConfig.MAX_LENGTH,
        "batch_size": CPUConfig.BATCH_SIZE,
        "threads": CPUConfig.OMP_NUM_THREADS,
        "device": "cpu"
    }

# ------------------ Helpers ------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PyPDF2.PdfReader(pdf_file)
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {i+1} ---\n{page_text}\n"
    return text

# ------------------ Entry ------------------

if __name__ == "__main__":
    # Use the app object directly so running:
    #   python -m backend.app   (from repo root) or
    #   python backend/app.py   (not recommended) both work
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
