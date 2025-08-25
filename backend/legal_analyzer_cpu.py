#!/usr/bin/env python3
"""
CPU-Safe Legal Analyzer (RAG) — single-file core module
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import sys
import json
import time
import logging
import warnings
import re
import io

# Scientific computing
import numpy as np

# ML Libraries
import faiss  # faiss-cpu
from sentence_transformers import SentenceTransformer

# System monitoring
import psutil

# W&B (optional)
try:
    import wandb
except ImportError:
    wandb = None

# PyPDF2 for PDF handling
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Torch is optional, only to tune CPU threading if present
try:
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception:
    torch = None

# ---------------- Logging & Warnings ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal-analyzer")
warnings.filterwarnings("ignore")


# ====================================================
# CPU & Runtime Config
# ====================================================
@dataclass
class CPUConfig:
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 8
    OMP_NUM_THREADS: int = max(1, os.cpu_count() or 1)
    
    @staticmethod
    def set_cpu_optimizations():
        """Set CPU optimization environment variables"""
        os.environ["OMP_NUM_THREADS"] = str(CPUConfig.OMP_NUM_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(CPUConfig.OMP_NUM_THREADS)
        os.environ["NUMEXPR_NUM_THREADS"] = str(CPUConfig.OMP_NUM_THREADS)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


CPUConfig.set_cpu_optimizations()


# ====================================================
# W&B helper
# ====================================================
def init_wandb_from_env(default_project: str = "legal-analyzer") -> None:
    """Initialize W&B if WANDB_API_KEY present"""
    if wandb is None or os.environ.get("WANDB_DISABLED", "").lower() == "true":
        return
    
    mode = os.getenv("WANDB_MODE", "online")
    project = os.getenv("WANDB_PROJECT", default_project)
    try:
        wandb.init(project=project, mode=mode)
    except Exception as e:
        logger.warning(f"W&B init failed: {e}")


# ====================================================
# System monitor (lightweight)
# ====================================================
class SystemMonitor:
    def __init__(self, interval_sec: float = 5.0):
        self.interval_sec = interval_sec
        self.running = False

    def start(self) -> None:
        self.running = True
        logger.info("System monitor started")

    def stop(self) -> None:
        self.running = False
        logger.info("System monitor stopped")

    def log_once(self, prefix: str = "system/") -> None:
        try:
            memory = psutil.virtual_memory()
            logger.info(f"CPU: {psutil.cpu_percent()}%, Memory: {memory.percent}%")
        except Exception as e:
            logger.warning(f"Monitor failed: {e}")


# ====================================================
# Knowledge Base Builder
# ====================================================
class CPUSafeLegalKnowledgeBase:
    """Builds & caches a CPU-friendly KB"""

    def __init__(self, model_name: str = CPUConfig.MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.knowledge_entries = []
        self.embeddings = None
        self.faiss_index = None

    def build_vector_embeddings(self, texts: List[str]) -> np.ndarray:
        """Build embeddings for texts"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device="cpu")
        
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.embeddings = embeddings.astype("float32")
        return self.embeddings

    def create_faiss_index(self) -> faiss.Index:
        """Create FAISS index"""
        if self.embeddings is None:
            raise ValueError("No embeddings available")
        
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.embeddings)
        return self.faiss_index

    def save_cache(self, output_dir: Union[str, Path]) -> None:
        """Save cache to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save knowledge entries
        with open(output_dir / "knowledge_entries.json", "w") as f:
            json.dump(self.knowledge_entries, f)
        
        # Save embeddings
        np.save(output_dir / "embeddings.npy", self.embeddings)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(output_dir / "faiss.index"))

    @staticmethod
    def load_cache(cache_dir: Union[str, Path]) -> Tuple[List[Dict], np.ndarray, faiss.Index]:
        """Load cached knowledge base"""
        cache_dir = Path(cache_dir)
        
        # Load knowledge entries
        with open(cache_dir / "knowledge_entries.json", "r") as f:
            knowledge_entries = json.load(f)
        
        # Load embeddings
        embeddings = np.load(cache_dir / "embeddings.npy")
        
        # Load FAISS index
        faiss_index = faiss.read_index(str(cache_dir / "faiss.index"))
        
        return knowledge_entries, embeddings, faiss_index


# ====================================================
# RAG Analyzer
# ====================================================
class CPUSafeRAGLegalAnalyzer:
    """Enhanced RAG Legal Analyzer with improved clause detection"""
    
    # Updated clause patterns with better keywords and more realistic patterns
    CLAUSE_PATTERNS: Dict[str, Dict[str, Any]] = {
        "Payment": {
            "keywords": ["payment", "pay", "invoice", "fee", "cost", "amount", "due", "net", "days"],
            "base_risk": "MEDIUM",
            "risk_factors": {
                "late_fees": ["late fee", "interest", "penalty", "overdue"],
                "payment_terms": ["net 30", "net 60", "net 90", "immediate", "upon receipt"],
                "penalties": ["penalty", "fine", "charge"]
            }
        },
        "Termination": {
            "keywords": ["terminate", "termination", "end", "expire", "cancel", "notice", "breach"],
            "base_risk": "HIGH", 
            "risk_factors": {
                "immediate_termination": ["immediate", "without notice", "at will"],
                "penalty_terms": ["penalty", "liquidated damages", "forfeit"],
                "notice_period": ["30 days", "notice period", "reasonable notice"]
            }
        },
        "Liability": {
            "keywords": ["liable", "liability", "responsible", "damages", "loss", "harm", "indemnify"],
            "base_risk": "HIGH",
            "risk_factors": {
                "unlimited_liability": ["unlimited", "no limitation", "full liability"],
                "exclusions": ["consequential", "indirect", "punitive"],
                "caps": ["limited to", "shall not exceed", "maximum"]
            }
        },
        "Confidentiality": {
            "keywords": ["confidential", "proprietary", "secret", "disclose", "non-disclosure"],
            "base_risk": "MEDIUM",
            "risk_factors": {
                "broad_scope": ["all information", "any information", "perpetual"],
                "penalties": ["injunctive relief", "monetary damages"]
            }
        },
        "Intellectual Property": {
            "keywords": ["intellectual property", "copyright", "patent", "trademark", "license", "ownership"],
            "base_risk": "HIGH",
            "risk_factors": {
                "broad_assignment": ["all rights", "work for hire", "assign"],
                "licensing_terms": ["exclusive", "perpetual", "irrevocable"]
            }
        },
        "Governing Law": {
            "keywords": ["governing law", "jurisdiction", "court", "venue", "disputes"],
            "base_risk": "LOW",
            "risk_factors": {
                "foreign_jurisdiction": ["foreign", "international"],
                "exclusive_venue": ["exclusive jurisdiction", "sole venue"]
            }
        }
    }

    def __init__(self, model_name: str = CPUConfig.MODEL_NAME, knowledge_base_path: Optional[str] = None):
        """Initialize the analyzer with optional knowledge base"""
        self.model_name = model_name
        self.knowledge_base_path = knowledge_base_path
        self.model = None
        self.knowledge_entries = []
        self.embeddings = None
        self.faiss_index = None
        
        # Initialize the sentence transformer model
        try:
            self.model = SentenceTransformer(self.model_name, device="cpu")
            logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}")
            self.model = None
        
        # Load knowledge base if provided and exists
        if knowledge_base_path and Path(knowledge_base_path).exists():
            try:
                self.knowledge_entries, self.embeddings, self.faiss_index = CPUSafeLegalKnowledgeBase.load_cache(knowledge_base_path)
                logger.info(f"Loaded knowledge base from {knowledge_base_path}")
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")
                self.knowledge_entries = []
                self.embeddings = None
                self.faiss_index = None
        else:
            logger.info("No knowledge base provided or path doesn't exist - running without RAG")

    def _retrieve_legal_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve relevant legal context if knowledge base is available"""
        if not self.model or not self.faiss_index or not self.knowledge_entries:
            return {
                "status": "no_kb",
                "message": "No knowledge base available",
                "retrieved_docs": []
            }
        
        try:
            # Encode query
            query_embedding = self.model.encode([query], normalize_embeddings=True).astype("float32")
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.knowledge_entries)))
            
            retrieved_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.knowledge_entries):
                    doc = self.knowledge_entries[idx]
                    retrieved_docs.append({
                        "text": doc.get("text", "")[:200] + "...",
                        "score": float(score),
                        "id": doc.get("id", idx)
                    })
            
            return {
                "status": "success",
                "message": f"Retrieved {len(retrieved_docs)} relevant documents",
                "retrieved_docs": retrieved_docs
            }
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return {
                "status": "error",
                "message": f"Retrieval failed: {str(e)}",
                "retrieved_docs": []
            }

    def _extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract named entities from text"""
        entities = {
            "ORGANIZATION": [],
            "PERSON": [],
            "DATE": [],
            "MONEY": [],
            "LOCATION": []
        }
        
        # Organizations (simple pattern)
        org_pattern = r'\b([A-Z][a-zA-Z\s&,\.]{2,50}(?:Inc|LLC|Corp|Company|Ltd|Co\.))\b'
        for match in re.finditer(org_pattern, text):
            entities["ORGANIZATION"].append({
                "text": match.group(1),
                "confidence": 0.7
            })
        
        # Dates (simple pattern)
        date_pattern = r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+ \d{1,2}, \d{4})\b'
        for match in re.finditer(date_pattern, text):
            entities["DATE"].append({
                "text": match.group(1),
                "confidence": 0.8
            })
        
        # Money (simple pattern)
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        for match in re.finditer(money_pattern, text):
            entities["MONEY"].append({
                "text": match.group(0),
                "confidence": 0.9
            })
        
        return entities

    def _get_recommendations(self, clause_type: str, risk_level: str) -> List[str]:
        """Get recommendations based on clause type and risk level"""
        recommendations = {
            "Payment": {
                "HIGH": ["Consider negotiating longer payment terms", "Review penalty clauses carefully"],
                "MEDIUM": ["Ensure payment terms are clearly defined", "Consider adding dispute resolution process"],
                "LOW": ["Standard payment terms appear reasonable"]
            },
            "Termination": {
                "HIGH": ["Negotiate longer notice periods", "Add specific breach definitions"],
                "MEDIUM": ["Review termination conditions", "Consider mutual termination clauses"],
                "LOW": ["Termination terms appear balanced"]
            },
            "Liability": {
                "HIGH": ["Strongly consider liability caps", "Review insurance requirements"],
                "MEDIUM": ["Clarify liability scope", "Consider mutual liability limitations"],
                "LOW": ["Liability terms appear reasonable"]
            },
            "Confidentiality": {
                "HIGH": ["Review scope of confidential information", "Consider time limitations"],
                "MEDIUM": ["Ensure mutual confidentiality obligations", "Clarify permitted disclosures"],
                "LOW": ["Confidentiality terms appear standard"]
            },
            "Intellectual Property": {
                "HIGH": ["Carefully review IP ownership terms", "Consider retaining background IP rights"],
                "MEDIUM": ["Clarify IP ownership and licensing", "Review work-for-hire provisions"],
                "LOW": ["IP terms appear reasonable"]
            },
            "Governing Law": {
                "HIGH": ["Consider jurisdiction implications", "Review dispute resolution mechanisms"],
                "MEDIUM": ["Ensure chosen jurisdiction is appropriate", "Consider arbitration clauses"],
                "LOW": ["Governing law terms appear standard"]
            }
        }
        
        return recommendations.get(clause_type, {}).get(risk_level, ["Review clause carefully"])

    def _assess_clause_risk(self, text: str, clause_type: str, clause_info: Dict[str, Any]) -> str:
        """Simplified and more effective risk assessment"""
        text_lower = text.lower()
        base_risk = clause_info["base_risk"]
        
        # Count risk-increasing factors
        high_risk_indicators = 0
        low_risk_indicators = 0
        
        # Define high-risk terms that should escalate risk
        high_risk_terms = {
            "Payment": ["penalty", "late fee", "interest", "immediate", "upon receipt", "forfeit"],
            "Termination": ["immediate", "without notice", "at will", "liquidated damages", "penalty"],
            "Liability": ["unlimited", "no limitation", "full liability", "entire liability", "sole liability"],
            "Confidentiality": ["perpetual", "all information", "any information", "injunctive relief"],
            "Intellectual Property": ["all rights", "work for hire", "assign", "transfer", "exclusive", "perpetual"],
            "Governing Law": ["foreign", "international", "exclusive jurisdiction"]
        }
        
        # Define low-risk terms that should reduce risk
        low_risk_terms = {
            "Payment": ["net 30", "net 45", "reasonable time", "good faith"],
            "Termination": ["30 days notice", "60 days notice", "reasonable notice", "material breach"],
            "Liability": ["limited to", "shall not exceed", "cap", "maximum", "consequential damages excluded"],
            "Confidentiality": ["reasonable", "standard", "mutual"],
            "Intellectual Property": ["background rights", "pre-existing", "retained"],
            "Governing Law": ["mutual consent", "arbitration"]
        }
        
        # Check for high-risk indicators
        clause_high_risk_terms = high_risk_terms.get(clause_type, [])
        for term in clause_high_risk_terms:
            if term in text_lower:
                high_risk_indicators += 1
        
        # Check for low-risk indicators  
        clause_low_risk_terms = low_risk_terms.get(clause_type, [])
        for term in clause_low_risk_terms:
            if term in text_lower:
                low_risk_indicators += 1
        
        # Simple decision logic
        net_risk = high_risk_indicators - low_risk_indicators
        
        # Adjust based on clause type and risk indicators
        if clause_type in ["Liability", "Termination", "Intellectual Property"]:
            # These are inherently higher risk clause types
            if net_risk >= 1 or high_risk_indicators >= 2:
                return "HIGH"
            elif net_risk <= -1 or low_risk_indicators >= 2:
                return "LOW"
            else:
                return "MEDIUM"
        
        elif clause_type in ["Payment", "Confidentiality"]:
            # These are medium risk by default
            if net_risk >= 2 or high_risk_indicators >= 3:
                return "HIGH"
            elif net_risk <= -1 or low_risk_indicators >= 2:
                return "LOW"
            else:
                return "MEDIUM"
        
        else:  # Governing Law and others
            # These are lower risk by default
            if net_risk >= 2:
                return "HIGH" 
            elif net_risk >= 1:
                return "MEDIUM"
            else:
                return "LOW"

    def _improved_clause_detection(self, text: str) -> List[Dict[str, Any]]:
        """Improved clause detection with better thresholds"""
        found_clauses = []
        text_lower = text.lower()
        
        print(f"DEBUG: Analyzing text of length {len(text)} characters")
        
        # Split text into meaningful chunks
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 10]
        if paragraphs:
            chunks.extend(paragraphs)
        
        # Also split by sentences for better coverage
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 10]
        chunks.extend(sentences)
        
        print(f"DEBUG: Created {len(chunks)} text chunks to analyze")
        
        for clause_type, clause_info in self.CLAUSE_PATTERNS.items():
            keywords = clause_info["keywords"]
            best_match = None
            highest_score = 0
            
            print(f"DEBUG: Checking {clause_type} clause with keywords: {keywords}")
            
            for chunk in chunks:
                chunk_lower = chunk.lower()
                
                # Count keyword matches
                keyword_matches = sum(1 for keyword in keywords if keyword in chunk_lower)
                
                if keyword_matches > 0:  # At least one keyword must match
                    # Calculate scores
                    keyword_score = keyword_matches / len(keywords)
                    
                    # Check for risk factors
                    risk_factors_found = []
                    for factor_type, factors in clause_info.get("risk_factors", {}).items():
                        for factor in factors:
                            if factor in chunk_lower:
                                risk_factors_found.append(factor)
                    
                    # Calculate confidence - much simpler
                    confidence = 0.5 + (keyword_matches * 0.1) + (len(risk_factors_found) * 0.05)
                    confidence = min(0.95, confidence)
                    
                    # Combined score
                    total_score = keyword_score + (len(risk_factors_found) * 0.1)
                    
                    print(f"DEBUG: {clause_type} - Keywords found: {keyword_matches}, Risk factors: {len(risk_factors_found)}, Score: {total_score}")
                    
                    # Much lower threshold - if we find keywords, include it
                    if total_score > 0.1 and keyword_matches >= 1:
                        if total_score > highest_score:
                            highest_score = total_score
                            best_match = {
                                "clause_type": clause_type,
                                "confidence": round(confidence, 2),
                                "context": chunk[:300] + "..." if len(chunk) > 300 else chunk,
                                "keyword_matches": keyword_matches,
                                "risk_factors_found": risk_factors_found[:3]
                            }
            
            if best_match:
                # Assess risk level
                risk_level = self._assess_clause_risk(
                    best_match["context"], 
                    clause_type, 
                    clause_info
                )
                
                print(f"DEBUG: {clause_type} clause detected with risk level: {risk_level}")
                
                best_match.update({
                    "risk_level": risk_level,
                    "risk_factors": best_match["risk_factors_found"],
                    "recommendations": self._get_recommendations(clause_type, risk_level),
                    "legal_definition": f"Identified {clause_type} clause with {risk_level.lower()} risk level"
                })
                
                # Clean up debug fields
                del best_match["risk_factors_found"]
                
                found_clauses.append(best_match)
        
        print(f"DEBUG: Total clauses found: {len(found_clauses)}")
        return found_clauses

    def _calculate_overall_risk_score(self, clause_analysis: List[Dict[str, Any]], text: str) -> Tuple[int, int, int, float]:
        """Simplified risk scoring that actually produces varied results"""
        
        # Count clauses by risk level
        high = sum(1 for c in clause_analysis if c.get("risk_level") == "HIGH")
        medium = sum(1 for c in clause_analysis if c.get("risk_level") == "MEDIUM") 
        low = sum(1 for c in clause_analysis if c.get("risk_level") == "LOW")
        total_clauses = len(clause_analysis)
        
        print(f"DEBUG: Risk counts - High: {high}, Medium: {medium}, Low: {low}, Total: {total_clauses}")
        
        # If no clauses detected, do text-based risk assessment
        if total_clauses == 0:
            text_lower = text.lower()
            risk_score = 20  # Base risk for any contract
            
            # Check for high-risk terms in the text
            high_risk_keywords = ["penalty", "forfeit", "liquidated damages", "unlimited liability", 
                                 "immediate termination", "without notice", "exclusive", "perpetual",
                                 "all rights", "work for hire"]
            
            risk_terms_found = sum(1 for term in high_risk_keywords if term in text_lower)
            risk_score += risk_terms_found * 10
            
            return 0, 0, 0, min(85, risk_score)
        
        # Calculate score based on clause risk distribution
        if high > 0:
            # Contracts with high-risk clauses should score 60-90
            base_score = 60
            base_score += high * 10  # Each high-risk clause adds 10 points
            base_score += medium * 5  # Each medium-risk clause adds 5 points
            base_score += low * 2    # Each low-risk clause adds 2 points
            final_score = min(90, base_score)
            
        elif medium > 0:
            # Contracts with only medium/low risk should score 30-60
            base_score = 35
            base_score += medium * 8  # Each medium-risk clause adds 8 points
            base_score += low * 3     # Each low-risk clause adds 3 points
            final_score = min(65, base_score)
            
        else:
            # Contracts with only low-risk clauses should score 15-40
            base_score = 20
            base_score += low * 5     # Each low-risk clause adds 5 points
            final_score = min(40, base_score)
        
        print(f"DEBUG: Final risk score: {final_score}")
        return high, medium, low, round(final_score, 1)

    def analyze_contract(self, contract_text: str, analysis_id: str) -> Dict[str, Any]:
        """Main analysis method with improved detection"""
        t0 = time.time()
        
        # Use improved clause detection
        clauses = self._improved_clause_detection(contract_text)
        
        # RAG retrieval (if KB available)
        retrieval = self._retrieve_legal_context(contract_text, top_k=5)
        
        # Entity extraction
        entities = self._extract_entities(contract_text)
        
        # Enhanced risk scoring
        high, medium, low, overall = self._calculate_overall_risk_score(clauses, contract_text)
        
        # Generate key findings
        key_findings = []
        if clauses:
            key_findings.append(f"Identified {len(clauses)} legal clause types")
            if high > 0:
                key_findings.append(f"Found {high} high-risk clause(s) requiring attention")
            if medium > 0:
                key_findings.append(f"Found {medium} medium-risk clause(s) for review")
        else:
            key_findings.append("No specific legal clauses automatically identified")
            key_findings.append("Manual review recommended for comprehensive analysis")
        
        result = {
            "success": True,
            "analysis_id": analysis_id,
            "metadata": {
                "analysis_id": analysis_id,
                "analysis_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "contract_stats": {
                    "word_count": len(contract_text.split()),
                    "char_count": len(contract_text),
                    "paragraph_count": len([p for p in contract_text.split('\n') if p.strip()]),
                },
                "retrieval": retrieval,
            },
            "executive_summary": {
                "total_clauses_identified": len(clauses),
                "high_risk_items": high,
                "medium_risk_items": medium,
                "low_risk_items": low,
                "overall_risk_score": overall,
                "key_findings": key_findings,
            },
            "clause_analysis": clauses,
            "identified_entities": entities,
            "analysis_duration": time.time() - t0,
            "message": "Analysis complete",
        }
        
        logger.info(f"Analysis complete: {len(clauses)} clauses found, risk score: {overall}%")
        return result

    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'model') and self.model:
            del self.model
        self.model = None
        self.knowledge_entries = []
        self.embeddings = None
        self.faiss_index = None


# ====================================================
# (Optional) Tiny PDF text extractor
# ====================================================
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    if PyPDF2 is None:
        raise ImportError("PyPDF2 not available")
    
    text_parts = []
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PyPDF2.PdfReader(pdf_file)
    
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text.strip())
        except Exception:
            continue
    
    return "\n\n".join(text_parts)
