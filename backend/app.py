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

def initialize_analyzer():
    """Initialize analyzer with optional KB cache"""
    global analyzer, system_monitor
    try:
        logger.info("Initializing Legal Document Analyzer...")

        # Set KB path (optional)
        kb_path = os.path.join(os.path.dirname(__file__), "legal_knowledge_base")
        kb_path_exists = os.path.exists(kb_path)
        
        logger.info(f"Knowledge base path: {kb_path} (exists: {kb_path_exists})")

        # Initialize analyzer with optional knowledge base
        if kb_path_exists:
            analyzer = CPUSafeRAGLegalAnalyzer(knowledge_base_path=kb_path)
            logger.info("Analyzer initialized with knowledge base")
        else:
            analyzer = CPUSafeRAGLegalAnalyzer()
            logger.info("Analyzer initialized without knowledge base (will work with pattern matching only)")

        # W&B + system monitor
        init_wandb_from_env(default_project="legal-analyzer-api")
        system_monitor = SystemMonitor()
        system_monitor.start()

        logger.info("Analyzer ready")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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

@app.post("/test")
async def test_analysis():
    """Test endpoint with sample contract"""
    sample_contract = """
    This Service Agreement is entered into between Company A and Company B.
    
    Payment Terms: Invoices shall be paid within 30 days of receipt. Late payments will incur a 2% monthly penalty fee.
    
    Termination: Either party may terminate this agreement with 30 days written notice. Immediate termination is allowed for material breach.
    
    Confidentiality: Both parties agree to keep all proprietary information confidential for a period of 5 years.
    
    Liability: Company A's liability shall be limited to the amount paid under this agreement. Neither party shall be liable for consequential damages.
    
    Governing Law: This agreement shall be governed by the laws of California.
    """
    
    global analyzer
    if analyzer is None:
        raise HTTPException(503, "Analyzer not initialized")
    
    try:
        result = analyzer.analyze_contract(sample_contract, "test_analysis")
        return result
    except Exception as e:
        raise HTTPException(500, f"Test analysis failed: {str(e)}")

@app.get("/config")
async def get_config():
    return {
        "model": CPUConfig.MODEL_NAME,
        "max_length": CPUConfig.MAX_LENGTH,
        "batch_size": CPUConfig.BATCH_SIZE,
        "threads": CPUConfig.OMP_NUM_THREADS,
        "device": "cpu"
    }

@app.post("/debug/clauses")
async def debug_clause_detection(request: ContractAnalysisRequest):
    """Debug endpoint to see clause detection details"""
    global analyzer
    if analyzer is None:
        raise HTTPException(503, "Analyzer not initialized")
    
    # Get the improved detection results with debug info
    text = request.text
    debug_info = {
        "text_length": len(text),
        "word_count": len(text.split()),
        "paragraphs": len([p for p in text.split('\n') if p.strip()]),
    }
    
    # Test each clause pattern
    clause_debug = {}
    for clause_type, clause_info in analyzer.CLAUSE_PATTERNS.items():
        keywords = clause_info["keywords"]
        found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
        clause_debug[clause_type] = {
            "keywords_total": len(keywords),
            "keywords_found": found_keywords,
            "keywords_found_count": len(found_keywords)
        }
    
    # Run actual detection
    clauses = analyzer._improved_clause_detection(text)
    
    return {
        "debug_info": debug_info,
        "clause_patterns_debug": clause_debug,
        "detected_clauses": clauses,
        "total_detected": len(clauses)
    }

@app.post("/debug/risk-assessment")
async def debug_risk_assessment(request: ContractAnalysisRequest):
    """Debug endpoint to trace risk assessment"""
    global analyzer
    if analyzer is None:
        raise HTTPException(503, "Analyzer not initialized")
    
    # Test with sample high-risk text
    high_risk_text = """
    Payment Terms: Payment is due immediately upon receipt. Late payments will incur a 5% monthly penalty fee.
    Termination: This agreement may be terminated immediately without notice by either party.
    Liability: Company shall have unlimited liability for all damages and losses.
    """
    
    print("=== DEBUGGING RISK ASSESSMENT ===")
    clauses = analyzer._improved_clause_detection(high_risk_text)
    high, medium, low, score = analyzer._calculate_overall_risk_score(clauses, high_risk_text)
    
    return {
        "sample_text": high_risk_text,
        "detected_clauses": clauses,
        "risk_counts": {"high": high, "medium": medium, "low": low},
        "overall_score": score,
        "user_text_analysis": {
            "clauses": analyzer._improved_clause_detection(request.text),
            "risk_breakdown": analyzer._calculate_overall_risk_score(
                analyzer._improved_clause_detection(request.text), 
                request.text
            )
        }
    }

# ------------------ Helpers ------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF with proper page handling"""
    text_parts = []
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PyPDF2.PdfReader(pdf_file)
    
    pages_with_content = 0
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():  # Only process pages with actual content
                text_parts.append(page_text.strip())
                pages_with_content += 1
        except Exception as e:
            logger.warning(f"Could not extract text from page {i+1}: {str(e)}")
            continue
    
    logger.info(f"Extracted text from {pages_with_content} pages out of {len(reader.pages)} total pages")
    return "\n\n".join(text_parts)  # Join with double newlines

# ------------------ Entry ------------------

if __name__ == "__main__":
    # Use the app object directly so running:
    #   python -m backend.app   (from repo root) or
    #   python backend/app.py   (not recommended) both work
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
