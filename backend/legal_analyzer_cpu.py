#!/usr/bin/env python3
"""
CPU-Safe Legal Analyzer (RAG) — single-file core module

Exposes:
- CPUConfig
- init_wandb_from_env
- SystemMonitor
- CPUSafeLegalKnowledgeBase
- CPUSafeRAGLegalAnalyzer

Designed to work with:
- legal_analyzer_cli.py (build-kb / analyze / config)
- your FastAPI app (imports these same classes)
"""

from __future__ import annotations

import os
import re
import io
import json
import time
import math
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# --- Third-party libs (all CPU-friendly) ---
import psutil
import faiss  # faiss-cpu
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import wandb

# Torch is optional, only to tune CPU threading if present
try:
    import torch
except Exception:  # pragma: no cover
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
    MODEL_NAME: str = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "512"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "64"))
    OMP_NUM_THREADS: int = int(os.getenv("OMP_NUM_THREADS", str(max(1, os.cpu_count() or 1))))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "2e-5"))
    EPOCHS: int = int(os.getenv("EPOCHS", "1"))

    @staticmethod
    def set_cpu_optimizations() -> None:
        # Environment knobs for BLAS/OpenMP backends
        os.environ.setdefault("OMP_NUM_THREADS", str(CPUConfig.OMP_NUM_THREADS))
        os.environ.setdefault("MKL_NUM_THREADS", str(CPUConfig.OMP_NUM_THREADS))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPUConfig.OMP_NUM_THREADS))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPUConfig.OMP_NUM_THREADS))
        if torch is not None:
            try:
                torch.set_num_threads(CPUConfig.OMP_NUM_THREADS)
            except Exception:
                pass


CPUConfig.set_cpu_optimizations()


# ====================================================
# W&B helper
# ====================================================
def init_wandb_from_env(default_project: str = "legal-analyzer") -> None:
    """
    Initialize W&B if WANDB_API_KEY present. Respects WANDB_MODE=offline.
    Safe to call multiple times.
    """
    if os.environ.get("WANDB_DISABLED", "").lower() == "true":
        return
    mode = os.getenv("WANDB_MODE", "online")
    project = os.getenv("WANDB_PROJECT", default_project)
    try:
        # If already initialized, do nothing
        if wandb.run is not None:
            return
        wandb.init(project=project, mode=mode)
        logger.info(f"W&B initialized: project={project}, mode={mode}")
    except Exception as e:
        logger.warning(f"W&B init skipped: {e}")


# ====================================================
# System monitor (lightweight)
# ====================================================
class SystemMonitor:
    def __init__(self, interval_sec: float = 5.0):
        self.interval_sec = interval_sec
        self.monitoring = False

    def start(self) -> None:
        self.monitoring = True

    def stop(self) -> None:
        self.monitoring = False

    def log_once(self, prefix: str = "system/") -> None:
        try:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)
            if wandb.run:
                wandb.log(
                    {
                        f"{prefix}cpu_percent": cpu,
                        f"{prefix}memory_percent": mem.percent,
                        f"{prefix}memory_used_gb": mem.used / (1024 ** 3),
                        f"{prefix}memory_available_gb": mem.available / (1024 ** 3),
                    }
                )
        except Exception:
            pass


# ====================================================
# Knowledge Base Builder
# ====================================================
class CPUSafeLegalKnowledgeBase:
    """
    Builds & caches a CPU-friendly KB:
      - knowledge_entries.json (list of small dicts)
      - embeddings.npy
      - faiss.index

    Typical flow:
      kb = CPUSafeLegalKnowledgeBase(model_name=...)
      kb.build_vector_embeddings(texts)
      kb.create_faiss_index()
      kb.save_cache(output_dir)

    Later for loading (handled automatically by analyzer):
      CPUSafeLegalKnowledgeBase.load_cache(dir)
    """

    def __init__(self, model_name: str = CPUConfig.MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device="cpu")
        self.knowledge_entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None

    # ---------- Build ----------
    def build_vector_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Build embeddings for given texts and populate knowledge_entries 1-to-1.
        """
        if not texts:
            raise ValueError("No texts provided for embeddings.")

        t0 = time.time()
        logger.info(f"Encoding {len(texts)} texts on CPU...")

        # Encode in batches (SentenceTransformer handles batching internally)
        embeddings = self.model.encode(
            texts,
            batch_size=min(CPUConfig.BATCH_SIZE, 256),
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        self.embeddings = embeddings
        # Create slim entries referencing the original text
        self.knowledge_entries = [{"id": i, "text": texts[i]} for i in range(len(texts))]

        logger.info(f"Embeddings shape: {embeddings.shape}; took {time.time() - t0:.2f}s")
        if wandb.run:
            wandb.log(
                {
                    "embedding/final_shape": str(embeddings.shape),
                    "embedding/total_time": time.time() - t0,
                }
            )
        return embeddings

    def create_faiss_index(self) -> faiss.Index:
        """
        Create an Inner Product (cosine with normalized vectors) index.
        """
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings available; call build_vector_embeddings first.")
        dim = int(self.embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(self.embeddings)
        self.faiss_index = index
        logger.info(f"FAISS index created with {self.embeddings.shape[0]} vectors.")
        return index

    # ---------- Cache ----------
    def save_cache(self, output_dir: Union[str, Path]) -> None:
        """
        Writes knowledge_entries.json, embeddings.npy, faiss.index
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.knowledge_entries is None:
            self.knowledge_entries = []

        with open(out / "knowledge_entries.json", "w", encoding="utf-8") as f:
            json.dump(self.knowledge_entries, f, ensure_ascii=False, indent=2)

        if self.embeddings is None:
            raise ValueError("embeddings is None; cannot save cache")
        np.save(out / "embeddings.npy", self.embeddings)

        if self.faiss_index is None:
            raise ValueError("faiss_index is None; cannot save cache")
        faiss.write_index(self.faiss_index, str(out / "faiss.index"))

    @staticmethod
    def load_cache(kb_dir: Union[str, Path]) -> Tuple[List[Dict[str, Any]], np.ndarray, faiss.Index]:
        """
        Load KB cache artifacts from a directory.
        """
        kb_path = Path(kb_dir)
        entries_p = kb_path / "knowledge_entries.json"
        emb_p = kb_path / "embeddings.npy"
        faiss_p = kb_path / "faiss.index"

        if not (entries_p.exists() and emb_p.exists() and faiss_p.exists()):
            raise FileNotFoundError(f"KB cache not found or incomplete at: {kb_path}")

        with open(entries_p, "r", encoding="utf-8") as f:
            entries = json.load(f)
        embeddings = np.load(emb_p).astype("float32")
        index = faiss.read_index(str(faiss_p))

        if len(entries) != embeddings.shape[0]:
            logger.warning(
                f"KB mismatch: entries={len(entries)} vs embeddings={embeddings.shape[0]}. "
                f"Proceeding, but retrieval results may not align perfectly."
            )

        return entries, embeddings, index


# ====================================================
# RAG Analyzer (loads KB; retrieval + light heuristics)
# ====================================================
class CPUSafeRAGLegalAnalyzer:
    """
    Loads a saved KB and performs retrieval + simple clause tagging + NER.

    Expected KB layout at knowledge_base_path:
      - knowledge_entries.json
      - embeddings.npy
      - faiss.index
    """

    CLAUSE_PATTERNS: Dict[str, Dict[str, Any]] = {
        "Termination": {
            "keywords": ["terminate", "termination", "notice period", "for cause", "for convenience"],
            "base_risk": "HIGH",
            "risk_factors": {
                "immediate_termination": ["immediate", "without notice", "at will"],
                "penalty_terms": ["penalty", "liquidated damages", "forfeit"],
                "notice_period": ["30 days", "notice period", "reasonable notice"]
            }
        },
        "Confidentiality": {
            "keywords": ["confidential", "non-disclosure", "nda", "proprietary"],
            "base_risk": "MEDIUM",
            "risk_factors": {
                "broad_scope": ["all information", "any information", "perpetual"],
                "penalties": ["injunctive relief", "monetary damages", "specific performance"]
            }
        },
        "Liability": {
            "keywords": ["liability", "indemnif", "hold harmless", "limitation of liability"],
            "base_risk": "HIGH",
            "risk_factors": {
                "unlimited_liability": ["unlimited", "no limitation", "full liability"],
                "exclusions": ["consequential damages", "indirect damages", "punitive"],
                "caps": ["limited to", "shall not exceed", "maximum liability"]
            }
        },
        "Governing Law": {
            "keywords": ["governing law", "jurisdiction", "venue"],
            "base_risk": "LOW",
            "risk_factors": {
                "foreign_jurisdiction": ["foreign", "international", "arbitration"],
                "exclusive_venue": ["exclusive jurisdiction", "sole venue"]
            }
        },
        "Payment": {
            "keywords": ["payment", "invoice", "fees", "compensation", "net terms"],
            "base_risk": "MEDIUM",
            "risk_factors": {
                "late_fees": ["late fee", "interest", "penalty"],
                "payment_terms": ["net 30", "net 60", "net 90", "immediate"],
                "currency_risk": ["foreign currency", "exchange rate"]
            }
        },
        "IP": {
            "keywords": ["intellectual property", "license", "licence", "ownership", "assign", "patent", "trademark"],
            "base_risk": "HIGH",
            "risk_factors": {
                "broad_assignment": ["all rights", "work for hire", "assign all"],
                "licensing_terms": ["exclusive license", "perpetual", "irrevocable"],
                "infringement": ["indemnify", "defend", "infringement claims"]
            }
        },
        "Force Majeure": {
            "keywords": ["force majeure", "act of god", "unforeseeable circumstances"],
            "base_risk": "LOW",
            "risk_factors": {
                "broad_definition": ["including but not limited to", "any cause"],
                "notice_requirements": ["immediate notice", "written notice"]
            }
        },
        "Data Protection": {
            "keywords": ["personal data", "gdpr", "privacy", "data protection"],
            "base_risk": "HIGH",
            "risk_factors": {
                "regulatory_compliance": ["gdpr", "ccpa", "hipaa"],
                "breach_notification": ["data breach", "security incident"],
                "international_transfer": ["cross-border", "international transfer"]
            }
        }
    }

    def _calculate_clause_confidence(self, text: str, keywords: List[str], risk_factors: Dict[str, List[str]]) -> float:
        """Calculate confidence based on keyword density and context"""
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in keywords if kw in text_lower)
        base_confidence = min(0.9, 0.3 + (keyword_matches * 0.15))
        risk_factor_boost = 0
        for factor_type, factors in risk_factors.items():
            factor_matches = sum(1 for factor in factors if factor in text_lower)
            if factor_matches > 0:
                risk_factor_boost += 0.1 * factor_matches
        final_confidence = min(0.95, base_confidence + risk_factor_boost)
        return round(final_confidence, 3)

    def _assess_clause_risk(self, text: str, clause_type: str, clause_info: Dict[str, Any]) -> str:
        """Assess actual risk level based on clause content and context"""
        text_lower = text.lower()
        base_risk = clause_info["base_risk"]
        risk_factors = clause_info.get("risk_factors", {})
        escalation_score = 0
        mitigation_score = 0
        for factor_type, factors in risk_factors.items():
            for factor in factors:
                if factor in text_lower:
                    if factor_type in ["unlimited_liability", "broad_assignment", "immediate_termination", "penalty_terms"]:
                        escalation_score += 2
                    elif factor_type in ["broad_scope", "exclusive_venue", "late_fees"]:
                        escalation_score += 1
                    elif factor_type in ["caps", "exclusions", "notice_period"]:
                        mitigation_score += 1
        if clause_type == "Liability":
            if any(phrase in text_lower for phrase in ["unlimited", "no limitation", "entire liability"]):
                escalation_score += 3
            elif any(phrase in text_lower for phrase in ["limited to", "shall not exceed"]):
                mitigation_score += 2
        elif clause_type == "Termination":
            if any(phrase in text_lower for phrase in ["without cause", "at will", "immediate"]):
                escalation_score += 2
            elif any(phrase in text_lower for phrase in ["reasonable notice", "30 days notice"]):
                mitigation_score += 1
        elif clause_type == "IP":
            if any(phrase in text_lower for phrase in ["work for hire", "assign all rights"]):
                escalation_score += 3
            elif "license" in text_lower and "revocable" in text_lower:
                mitigation_score += 1
        net_score = escalation_score - mitigation_score
        if base_risk == "LOW":
            if net_score >= 3:
                return "HIGH"
            elif net_score >= 1:
                return "MEDIUM"
            return "LOW"
        elif base_risk == "MEDIUM":
            if net_score >= 2:
                return "HIGH"
            elif net_score <= -2:
                return "LOW"
            return "MEDIUM"
        else:  # HIGH
            if net_score <= -3:
                return "MEDIUM"
            elif net_score <= -1:
                return "MEDIUM"
            return "HIGH"

    def _get_recommendations(self, clause_type: str, risk_level: str) -> List[str]:
        """Get recommendations based on clause type and risk level"""
        recommendations = {
            "Termination": {
                "HIGH": ["Review termination conditions", "Negotiate notice periods", "Consider mutual termination clauses"],
                "MEDIUM": ["Clarify termination procedures", "Define reasonable notice"],
                "LOW": ["Standard termination provisions acceptable"]
            },
            "Liability": {
                "HIGH": ["Negotiate liability caps", "Add mutual indemnification", "Review insurance requirements"],
                "MEDIUM": ["Consider liability limitations", "Review indemnification scope"],
                "LOW": ["Standard liability terms acceptable"]
            },
            "IP": {
                "HIGH": ["Limit IP assignment scope", "Negotiate licensing terms", "Protect background IP"],
                "MEDIUM": ["Clarify IP ownership", "Review licensing grants"],
                "LOW": ["Standard IP provisions acceptable"]
            },
            "Payment": {
                "HIGH": ["Negotiate payment terms", "Add late fee protections", "Consider escrow arrangements"],
                "MEDIUM": ["Clarify payment schedules", "Review currency terms"],
                "LOW": ["Standard payment terms acceptable"]
            },
            "Confidentiality": {
                "HIGH": ["Limit confidentiality scope", "Add mutual obligations", "Define exceptions clearly"],
                "MEDIUM": ["Clarify confidential information definition", "Review term duration"],
                "LOW": ["Standard confidentiality terms acceptable"]
            }
        }
        return recommendations.get(clause_type, {}).get(risk_level, ["Review clause carefully"])

    def __init__(self, knowledge_base_path: Union[str, Path] = "legal_knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.model_name = CPUConfig.MODEL_NAME
        self.model = SentenceTransformer(self.model_name, device="cpu")
        self.knowledge_entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        try:
            self.ner = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=-1,  # CPU
            )
        except Exception as e:
            logger.warning(f"NER model failed to load; continuing without NER: {e}")
            self.ner = None
        self._load_kb_if_available()

    def _load_kb_if_available(self) -> None:
        try:
            entries, embeddings, index = CPUSafeLegalKnowledgeBase.load_cache(self.knowledge_base_path)
            self.knowledge_entries = entries
            self.embeddings = embeddings
            self.faiss_index = index
            logger.info(f"KB loaded: {len(self.knowledge_entries)} entries.")
        except FileNotFoundError:
            logger.warning(
                f"Knowledge base not found at {self.knowledge_base_path}. "
                f"RAG will run with empty KB."
            )
        except Exception as e:
            logger.error(f"Failed to load KB from {self.knowledge_base_path}: {e}")

    def _encode_query(self, text: str) -> np.ndarray:
        emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        return emb

    def _retrieve_legal_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.faiss_index is None or self.embeddings is None or len(self.knowledge_entries) == 0:
            return []
        q_emb = self._encode_query(query)
        scores, idx = self.faiss_index.search(q_emb, min(top_k, self.embeddings.shape[0]))
        scores = scores[0]
        idx = idx[0]
        results: List[Dict[str, Any]] = []
        for rank, (i, s) in enumerate(zip(idx, scores), 1):
            if i < 0 or i >= len(self.knowledge_entries):
                continue
            entry = self.knowledge_entries[i]
            results.append(
                {
                    "rank": rank,
                    "kb_id": entry.get("id", i),
                    "text": entry.get("text", ""),
                    "relevance_score": float(s),
                }
            )
        return results

    @staticmethod
    def _simple_clause_tagging(text: str) -> List[Dict[str, Any]]:
        """Enhanced clause tagging with proper risk assessment"""
        analyzer = CPUSafeRAGLegalAnalyzer()
        found: List[Dict[str, Any]] = []
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        for clause_type, clause_info in analyzer.CLAUSE_PATTERNS.items():
            keywords = clause_info["keywords"]
            matching_sentences = []
            for sentence in sentences:
                if any(kw in sentence.lower() for kw in keywords):
                    matching_sentences.append(sentence.strip())
            if matching_sentences:
                context_text = " ".join(matching_sentences[:3])
                confidence = analyzer._calculate_clause_confidence(
                    context_text, keywords, clause_info.get("risk_factors", {})
                )
                risk_level = analyzer._assess_clause_risk(
                    context_text, clause_type, clause_info
                )
                found.append({
                    "clause_type": clause_type,
                    "risk_level": risk_level,
                    "confidence": confidence,
                    "context": context_text[:200] + "..." if len(context_text) > 200 else context_text,
                    "legal_definition": f"Analysis of {clause_type} clause with {risk_level.lower()} risk assessment",
                    "risk_factors": [
                        factor for factor_type, factors in clause_info.get("risk_factors", {}).items()
                        for factor in factors if factor in context_text.lower()
                    ][:5],
                    "recommendations": analyzer._get_recommendations(clause_type, risk_level)
                })
        return found

    def _extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        if not self.ner:
            return {}
        ents = self.ner(text)
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for e in ents:
            label = e.get("entity_group", "MISC")
            buckets.setdefault(label, []).append(
                {
                    "text": e.get("word"),
                    "confidence": float(e.get("score", 0.0)),
                    "start": int(e.get("start", 0)),
                    "end": int(e.get("end", 0)),
                }
            )
        return buckets

    @staticmethod
    def _risk_score(clause_analysis: List[Dict[str, Any]]) -> Tuple[int, int, int, float]:
        """Enhanced risk scoring with LOW/MEDIUM/HIGH counts"""
        if not clause_analysis:
            return 0, 0, 0, 0.0
        high = sum(1 for c in clause_analysis if c.get("risk_level") == "HIGH")
        medium = sum(1 for c in clause_analysis if c.get("risk_level") == "MEDIUM")
        low = sum(1 for c in clause_analysis if c.get("risk_level") == "LOW")
        total_clauses = len(clause_analysis)
        weighted_score = (high * 100 + medium * 50 + low * 10)
        max_possible_score = total_clauses * 100
        if max_possible_score > 0:
            normalized_score = (weighted_score / max_possible_score) * 100
        else:
            normalized_score = 0.0
        return high, medium, low, round(normalized_score, 1)

    def analyze_contract(self, contract_text: str, analysis_id: str) -> Dict[str, Any]:
        t0 = time.time()
        retrieval = self._retrieve_legal_context(contract_text, top_k=5)
        clauses = self._simple_clause_tagging(contract_text)
        entities = self._extract_entities(contract_text)
        high, medium, low, overall = self._risk_score(clauses)
        result = {
            "success": True,
            "analysis_id": analysis_id,
            "metadata": {
                "analysis_id": analysis_id,
                "analysis_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "contract_stats": {
                    "word_count": len(contract_text.split()),
                    "char_count": len(contract_text),
                    "section_count": 1,
                },
                "retrieval": retrieval,
            },
            "executive_summary": {
                "total_clauses_identified": len(clauses),
                "high_risk_items": high,
                "medium_risk_items": medium,
                "low_risk_items": low,
                "overall_risk_score": overall,
                "key_findings": (
                    ["No specific legal clauses identified"] if not clauses else
                    [f"Identified {len(clauses)} clause types with varying risk levels",
                     f"High risk: {high}, Medium risk: {medium}, Low risk: {low}"]
                ),
            },
            "clause_analysis": clauses,
            "identified_entities": entities,
            "analysis_duration": time.time() - t0,
            "message": "Analysis complete",
        }
        if wandb.run:
            wandb.log(
                {
                    "sample/clauses_found": len(clauses),
                    "sample/entities_found": sum(len(v) for v in entities.values()) if entities else 0,
                    "sample/input_word_count": len(contract_text.split()),
                    "kb/entries_loaded": len(self.knowledge_entries),
                    "faiss/has_index": int(self.faiss_index is not None),
                    "analysis/duration_sec": result["analysis_duration"],
                }
            )
        logger.info(f"RAG analysis done: {analysis_id} in {result['analysis_duration']:.2f}s")
        return result

    def cleanup(self) -> None:
        pass

# ====================================================
# (Optional) Tiny PDF text extractor — used by CLI if needed
# ====================================================
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Minimal PDF text extraction using PyPDF2 when available.
    """
    try:
        import PyPDF2
    except Exception:
        logger.warning("PyPDF2 not installed; cannot parse PDF.")
        return ""
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
            if page_text:
                text += f"\n--- Page {i+1} ---\n{page_text}"
        except Exception:
            continue
    return text.strip()
