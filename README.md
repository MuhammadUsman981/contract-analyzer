# ⚖️ Legal Contract Analyzer

AI-powered legal contract analysis tool that identifies risks, analyzes clauses, and provides actionable recommendations. Built with FastAPI backend and Streamlit frontend.

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-00a393) ![Streamlit](https://img.shields.io/badge/Streamlit-Latest-ff4b4b)

## 🌟 Features

- **🤖 AI-Powered Analysis**: Advanced NLP models for clause identification and risk assessment
- **📊 Risk Scoring**: Comprehensive risk analysis with detailed explanations
- **📑 Clause Detection**: Identifies and categorizes different types of legal clauses
- **🔍 Entity Recognition**: Extracts key entities (people, organizations, dates, amounts)
- **📄 Multi-Format Support**: Handles both PDF and text file uploads
- **🎨 Interactive UI**: Beautiful Streamlit interface with real-time analysis
- **☁️ Cloud Ready**: Deployable on Railway (backend) and Streamlit Cloud (frontend)

## 🚀 Quick Start

### Live Deployments
- **Backend API**: [Railway API](https://contract-analyzer-production-a7ea.up.railway.app)
- **API Docs**: [API Documentation](https://contract-analyzer-production-a7ea.up.railway.app/docs)

### Local Development

```bash
# Clone repository
git clone https://github.com/MuhammadUsman981/contract-analyzer.git
cd contract-analyzer

# Setup environment
conda create -n contract-analyzer python=3.11
conda activate contract-analyzer
pip install -r requirements.txt

# Run backend (Terminal 1)
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000


📁 Project Structure

contract-analyzer/
├── backend/                    # FastAPI backend
│   ├── app.py                 # Main FastAPI application
│   ├── legal_analyzer_cpu.py  # Core analysis engine
│   └── legal_knowledge_base/  # Legal knowledge and patterns
├── frontend/                  # Streamlit frontend
│   ├── streamlit_app.py      # Main Streamlit application
├── configs/                   # Configuration files
│   └── .env                  # Environment variables
├── requirements.txt          # Python dependencies
├── requirements-railway.txt  # Railway-specific dependencies
├── requirements-streamlit.txt # Streamlit Cloud dependencies
└── Dockerfile               # Docker configuration

# W&B Configuration
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=legal-analyzer

# Model Configuration
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DEVICE=cpu

# Hugging Face
HF_TOKEN=your_hf_token

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO


📊 API Endpoints
POST /analyze/text
{
  "text": "Your contract text here..."
}

Response
{
  "executive_summary": {
    "total_clauses_identified": 5,
    "high_risk_items": 2,
    "medium_risk_items": 1,
    "low_risk_items": 2,
    "overall_risk_score": 73.5,
    "key_findings": ["Non-compete clause is overly restrictive"]
  },
  "clause_analysis": [...],
  "identified_entities": {...}
}

🚀 Deployment
Railway Backend
Connect GitHub repo to Railway
Set environment variables from .env
Deploy automatically
Streamlit Cloud Frontend
Connect GitHub repo to Streamlit Cloud
Main file: streamlit_app.py
Requirements: requirements-streamlit.txt

# Run frontend (Terminal 2)
streamlit run frontend/streamlit_app.py

🧪 Testing
# CLI testing
python backend/legal_analyzer_cli.py --input "sample contract text"

# API testing
curl -X POST "http://localhost:8000/analyze/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your contract text here"}'


🔍 Supported Analysis
Risk Levels: High, Medium, Low Clause Types: Non-compete, Termination, Confidentiality, Compensation, IP Rights, Governing Law Entities: PERSON, ORG, DATE, MONEY, GPE

