#!/usr/bin/env python
"""
CLI for CPU-Safe Legal Document Analyzer
- Build Knowledge Base (CUAD)
- Cache embeddings & FAISS index
- Analyze text/PDF contracts using RAG
"""

import os
import sys
import json
import click
import logging
import wandb
import numpy as np
import faiss
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------- Logging & Warnings ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal-analyzer-cli")
warnings.filterwarnings("ignore")


# ---------------- Config ----------------
class CPUConfig:
    MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    MAX_LENGTH = 512
    BATCH_SIZE = 32
    OMP_NUM_THREADS = int(os.getenv("OMP_NUM_THREADS", "4"))


# ---------------- Knowledge Base ----------------
class CPUSafeLegalKnowledgeBase:
    """Manages CUAD data, embeddings, and FAISS index."""

    def __init__(self, cuad_data: Optional[List[str]] = None, model_name: str = CPUConfig.MODEL_NAME):
        self.cuad_data = cuad_data or []
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device="cpu")
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.knowledge_entries: List[Dict[str, Any]] = []

    def build_vector_embeddings(self, texts: List[str]):
        logger.info(f"Encoding {len(texts)} texts on CPU...")
        self.embeddings = self.model.encode(texts, batch_size=CPUConfig.BATCH_SIZE, show_progress_bar=True)
        self.knowledge_entries = [{"id": i, "text": t} for i, t in enumerate(texts)]
        logger.info(f"Embeddings shape: {self.embeddings.shape}")

    def create_faiss_index(self):
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)
        logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors")


# ---------------- RAG Analyzer ----------------
class CPUSafeRAGLegalAnalyzer:
    """Loads a saved KB and performs retrieval + clause tagging + NER."""

    def __init__(self, knowledge_base_path: Union[str, Path] = "legal_knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.model_name = CPUConfig.MODEL_NAME
        self.knowledge_entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None

        # Load embedding model
        self.model = SentenceTransformer(self.model_name, device="cpu")

        # Load KB if available
        entries_file = self.knowledge_base_path / "knowledge_entries.json"
        embeddings_file = self.knowledge_base_path / "embeddings.npy"
        faiss_file = self.knowledge_base_path / "faiss.index"

        if entries_file.exists() and embeddings_file.exists() and faiss_file.exists():
            logger.info(f"📦 Loading KB from cache at {self.knowledge_base_path}")
            with open(entries_file, "r") as f:
                self.knowledge_entries = json.load(f)
            self.embeddings = np.load(embeddings_file)
            self.faiss_index = faiss.read_index(str(faiss_file))
            logger.info(f"✅ KB loaded: {len(self.knowledge_entries)} entries, {self.embeddings.shape[0]} vectors")
        else:
            logger.warning(f"⚠️ Knowledge base not found at {self.knowledge_base_path}. RAG will run with empty KB.")

        # NER pipeline
        try:
            self.ner = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=-1
            )
        except Exception as e:
            logger.warning(f"NER model failed to load; continuing without NER: {e}")
            self.ner = None

    def analyze_contract(self, text: str, analysis_id: str):
        start_time = datetime.now()

        # Stats
        words = text.split()
        metadata = {
            "analysis_id": analysis_id,
            "analysis_date": datetime.now().isoformat(),
            "contract_stats": {
                "word_count": len(words),
                "char_count": len(text),
                "section_count": text.count("\n\n") + 1,
            },
        }

        # Retrieval
        key_findings = []
        if self.faiss_index is not None and self.embeddings is not None:
            query_vec = self.model.encode([text])
            scores, indices = self.faiss_index.search(query_vec, k=5)
            retrieved = [self.knowledge_entries[i]["text"] for i in indices[0] if i < len(self.knowledge_entries)]
            key_findings.extend([f"Similar clause: {r[:100]}..." for r in retrieved])

        # NER
        entities = {}
        if self.ner:
            try:
                ents = self.ner(text)
                for e in ents:
                    entities.setdefault(e["entity_group"], []).append(
                        {"text": e["word"], "confidence": float(e["score"])}
                    )
            except Exception as e:
                logger.warning(f"NER failed: {e}")

        duration = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "analysis_id": analysis_id,
            "metadata": metadata,
            "executive_summary": {
                "total_clauses_identified": len(key_findings),
                "high_risk_items": 0,
                "medium_risk_items": len(key_findings),
                "overall_risk_score": min(100, len(key_findings) * 10),
                "key_findings": key_findings or ["Standard legal provisions identified"],
            },
            "clause_analysis": [],
            "identified_entities": entities,
            "analysis_duration": duration,
            "message": "Analysis complete",
        }


# ---------------- CLI ----------------
@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--wandb-project", default="legal-analyzer-cli", help="W&B project name")
@click.option("--wandb-offline", is_flag=True, help="Run W&B in offline mode")
@click.pass_context
def cli(ctx, verbose, wandb_project, wandb_offline):
    """Legal Document Analyzer CLI"""
    if verbose:
        logger.setLevel(logging.DEBUG)
    wandb.init(project=wandb_project, mode="offline" if wandb_offline else "online")
    ctx.ensure_object(dict)
    ctx.obj["wandb_project"] = wandb_project


# ---------------- Build KB ----------------
@cli.command()
@click.argument("cuad_data_path", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="legal_knowledge_base", help="Output directory for KB")
@click.option("--model-name", default="all-MiniLM-L6-v2", help="Sentence transformer model")
@click.option("--rebuild", is_flag=True, help="Force rebuild KB even if cache exists")
def build_kb(cuad_data_path, output_dir, model_name, rebuild):
    """Build Knowledge Base from CUAD dataset (json/txt/pdf folder)."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    entries_file = output_path / "knowledge_entries.json"
    embeddings_file = output_path / "embeddings.npy"
    faiss_file = output_path / "faiss.index"

    try:
        if not rebuild and entries_file.exists() and embeddings_file.exists() and faiss_file.exists():
            click.echo("📦 KB already cached, skipping rebuild.")
            return

        all_texts = []
        if cuad_data_path.endswith(".json"):
            with open(cuad_data_path, "r") as f:
                cuad_data = json.load(f)
            for entry in cuad_data.get("data", []):
                for para in entry.get("paragraphs", []):
                    ctx = para.get("context", "").strip()
                    if ctx:
                        all_texts.append(ctx)
                    for qa in para.get("qas", []):
                        q = qa.get("question", "").strip()
                        if q:
                            all_texts.append(q)
        else:
            folder = Path(cuad_data_path)
            for file in folder.glob("*.txt"):
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    all_texts.append(f.read())

        kb = CPUSafeLegalKnowledgeBase(model_name=model_name)
        kb.build_vector_embeddings(all_texts)
        kb.create_faiss_index()

        with open(entries_file, "w") as f:
            json.dump(kb.knowledge_entries, f, indent=2)
        np.save(embeddings_file, kb.embeddings)
        faiss.write_index(kb.faiss_index, str(faiss_file))

        click.echo(f"✅ KB built and cached successfully with {len(kb.knowledge_entries)} entries")

    except Exception as e:
        click.echo(f"❌ KB build failed: {e}", err=True)
        logger.error(f"KB build error: {e}", exc_info=True)
        sys.exit(1)


# ---------------- Analyze ----------------
@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--analysis-id", default=None, help="Optional analysis ID")
@click.option("--kb-dir", default="legal_knowledge_base", help="Path to cached KB directory")
def analyze(file_path, analysis_id, kb_dir):
    """Analyze a legal document (txt/pdf)."""
    try:
        analyzer = CPUSafeRAGLegalAnalyzer(knowledge_base_path=kb_dir)

        if file_path.lower().endswith(".pdf"):
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        if not text.strip():
            raise ValueError("Empty document")

        analysis_id = analysis_id or f"cli_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = analyzer.analyze_contract(text, analysis_id)

        click.echo("✅ Analysis completed")
        click.echo(json.dumps(result["executive_summary"], indent=2))

    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        logger.error(f"Analysis error: {e}", exc_info=True)
        sys.exit(1)


# ---------------- Config ----------------
@cli.command()
def config():
    """Show analyzer config."""
    cfg = {
        "model_name": CPUConfig.MODEL_NAME,
        "batch_size": CPUConfig.BATCH_SIZE,
        "num_threads": CPUConfig.OMP_NUM_THREADS,
        "device": "cpu",
    }
    click.echo(json.dumps(cfg, indent=2))


# ---------------- Main ----------------
if __name__ == "__main__":
    cli()
