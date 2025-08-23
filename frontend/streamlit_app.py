# import streamlit as st
# import requests
# from pathlib import Path
# from PyPDF2 import PdfReader
# from legal_analyzer_cpu import CPUSafeRAGLegalAnalyzer

# # Initialize result dictionary at the start
# result = {
#     "executive_summary": {
#         "total_clauses_identified": 0,
#         "high_risk_items": 0,
#         "medium_risk_items": 0,
#         "low_risk_items": 0,
#         "overall_risk_score": 0
#     },
#     "clause_analysis": [],
#     "identified_entities": {}
# }

# API_URL = "http://127.0.0.1:8000/analyze/text"

# # Initialize the analyzer
# analyzer = CPUSafeRAGLegalAnalyzer()

# st.set_page_config(
#     page_title="Legal Contract Analyzer",
#     page_icon="⚖️",
#     layout="wide"
# )

# # ------------------------------
# # Sidebar
# # ------------------------------
# st.sidebar.title("⚙️ Settings")
# api_url = st.sidebar.text_input("API URL", API_URL)
# risk_threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)
# show_entities = st.sidebar.checkbox("Show Named Entities", value=True)

# st.sidebar.markdown("---")
# st.sidebar.info("Backend must be running with `uvicorn app:app --reload`")

# # ------------------------------
# # Main UI
# # ------------------------------
# st.title("⚖️ Legal Contract Analyzer")
# st.write("Upload or paste a contract to analyze clauses, risks, and entities.")

# # Input choice
# tab1, tab2 = st.tabs(["📂 Upload File", "✍️ Paste Text"])

# contract_text = None

# with tab1:
#     uploaded_file = st.file_uploader("Upload contract (TXT or PDF)", type=["txt", "pdf"])
#     if uploaded_file:
#         try:
#             if uploaded_file.name.lower().endswith('.pdf'):
#                 st.info("Processing PDF file...")
#                 pdf_reader = PdfReader(uploaded_file)
#                 text_content = []
#                 for page in pdf_reader.pages:
#                     text_content.append(page.extract_text() or "")
#                 contract_text = "\n".join(text_content)
#                 if not contract_text.strip():
#                     st.error("Could not extract text from PDF. Please ensure the PDF contains searchable text.")
#                 else:
#                     st.success(f"Successfully extracted {len(text_content)} pages from PDF")
#             else:
#                 contract_text = uploaded_file.read().decode('utf-8')
#         except Exception as e:
#             st.error(f"Error processing file: {str(e)}")
#             contract_text = None

# with tab2:
#     pasted_text = st.text_area("Paste contract text here", height=250)
#     if pasted_text:
#         contract_text = pasted_text

# # ------------------------------
# # Run Analysis
# # ------------------------------
# if st.button("🔍 Analyze Contract") and contract_text:
#     with st.spinner("Analyzing with Legal Analyzer..."):
#         # Debug: Show text being analyzed
#         st.info(f"Analyzing text ({len(contract_text)} characters)...")
        
#         # Local analysis fallback
#         try:
#             # First try API
#             payload = {"text": contract_text}
#             try:
#                 st.info(f"Attempting API request to {api_url}")
#                 res = requests.post(api_url, json=payload, timeout=30)
#                 res.raise_for_status()
#                 result = res.json()
#                 st.success("API analysis successful!")
#             except Exception as e:
#                 st.warning(f"API request failed: {str(e)}. Falling back to local analysis...")
#                 # Fallback to local analysis
#                 result = analyzer.analyze_contract(contract_text, "streamlit-analysis")
                
#             # Debug: Show raw result
#             with st.expander("Debug: Raw Analysis Result"):
#                 st.json(result)
                
#         except Exception as e:
#             st.error(f"Analysis failed: {str(e)}")
#             st.stop()

#     # Continue only if we have valid results
#     if result and "executive_summary" in result:
#         st.success("✅ Analysis complete")
        
#         # ---------------- Summary Metrics ----------------
#         try:
#             cols = st.columns(5)
#             cols[0].metric("Total Clauses", result["executive_summary"]["total_clauses_identified"])
#             cols[1].metric("High Risk", result["executive_summary"]["high_risk_items"], delta_color="inverse")
#             cols[2].metric("Medium Risk", result["executive_summary"]["medium_risk_items"])
#             cols[3].metric("Low Risk", result["executive_summary"].get("low_risk_items", 0), delta_color="normal")
#             cols[4].metric("Risk Score", f"{result['executive_summary']['overall_risk_score']:.1f}%")
#         except Exception as e:
#             st.error(f"Error displaying metrics: {str(e)}")

#         # Risk level indicator
#         risk_score = result["executive_summary"]["overall_risk_score"]
#         if risk_score >= 70:
#             st.error(f"🚨 High Risk Contract (Score: {risk_score:.1f}%)")
#         elif risk_score >= 40:
#             st.warning(f"⚠️ Medium Risk Contract (Score: {risk_score:.1f}%)")
#         else:
#             st.success(f"✅ Low Risk Contract (Score: {risk_score:.1f}%)")

#         st.markdown("**Key Findings:**")
#         for finding in result["executive_summary"].get("key_findings", []):
#             st.info(f"🔎 {finding}")

#         # ---------------- Clause Analysis ----------------
#         if result["clause_analysis"]:
#             st.subheader("📑 Detailed Clause Analysis")
            
#             # Group clauses by risk level
#             high_risk = [c for c in result["clause_analysis"] if c.get("risk_level") == "HIGH"]
#             medium_risk = [c for c in result["clause_analysis"] if c.get("risk_level") == "MEDIUM"]
#             low_risk = [c for c in result["clause_analysis"] if c.get("risk_level") == "LOW"]
            
#             # Display clauses by risk level
#             if high_risk:
#                 st.markdown("### 🚨 High Risk Clauses")
#                 for clause in high_risk:
#                     with st.expander(f"🔴 {clause['clause_type']} (Confidence: {clause['confidence']:.2f})", expanded=True):
#                         st.markdown(f"**Risk Level:** {clause['risk_level']}")
#                         st.markdown(f"**Context:** {clause.get('context', 'N/A')}")
#                         if clause.get('risk_factors'):
#                             st.markdown("**Risk Factors Identified:**")
#                             for factor in clause['risk_factors']:
#                                 st.markdown(f"- {factor}")
#                         if clause.get('recommendations'):
#                             st.markdown("**Recommendations:**")
#                             for rec in clause['recommendations']:
#                                 st.markdown(f"- {rec}")
            
#             # Medium risk clauses
#             if medium_risk:
#                 st.markdown("### ⚠️ Medium Risk Clauses")
#                 for idx, clause in enumerate(medium_risk, 1):
#                     with st.expander(f"🟡 {clause['clause_type']} (Confidence: {clause['confidence']:.2f})"):
#                         st.markdown(f"**Risk Level:** {clause['risk_level']}")
#                         st.markdown(f"**Context:** {clause.get('context', 'N/A')}")
                        
#                         if clause.get('risk_factors'):
#                             st.markdown("**Risk Factors:**")
#                             for factor in clause['risk_factors']:
#                                 st.markdown(f"- {factor}")
                        
#                         if clause.get('recommendations'):
#                             st.markdown("**Recommendations:**")
#                             for rec in clause['recommendations']:
#                                 st.markdown(f"- {rec}")
            
#             # Low risk clauses
#             if low_risk:
#                 st.markdown("### ✅ Low Risk Clauses")
#                 for idx, clause in enumerate(low_risk, 1):
#                     with st.expander(f"🟢 {clause['clause_type']} (Confidence: {clause['confidence']:.2f})"):
#                         st.markdown(f"**Risk Level:** {clause['risk_level']}")
#                         st.markdown(f"**Context:** {clause.get('context', 'N/A')}")
                        
#                         if clause.get('recommendations'):
#                             st.markdown("**Recommendations:**")
#                             for rec in clause['recommendations']:
#                                 st.markdown(f"- {rec}")

#             # ---------------- Entities ----------------
#             if show_entities and result.get("identified_entities"):
#                 st.subheader("🧾 Named Entities")
#                 for ent_type, ents in result["identified_entities"].items():
#                     st.markdown(f"**{ent_type}**")
#                     for ent in ents:
#                         st.write(f"- {ent['text']} (conf: {ent['confidence']:.2f})")

# else:
#     st.warning("👆 Upload or paste a contract, then click **Analyze Contract**")

import sys
from pathlib import Path

# Ensure project root is on sys.path so `import backend.*` works both locally and in container
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import backend analyzer using package name
from backend.legal_analyzer_cpu import CPUSafeRAGLegalAnalyzer

import streamlit as st
import requests
import json
import time
import json
import time
from PyPDF2 import PdfReader

# API URL (points to your Railway backend)
API_URL = "https://contract-analyzer-production-a7ea.up.railway.app/analyze"

# Set to localhost for local development
LOCAL_API_URL = "http://localhost:8000/analyze/text"  # Note: Changed to the correct endpoint
RAILWAY_API_URL = "https://contract-analyzer-production-a7ea.up.railway.app/analyze"

st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("⚙️ Settings")
use_local = st.sidebar.checkbox("Use Local Backend", value=True)
api_url = st.sidebar.text_input("API URL", LOCAL_API_URL if use_local else RAILWAY_API_URL)
show_entities = st.sidebar.checkbox("Show Named Entities", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Backend running on Railway Cloud")

# ------------------------------
# Main UI
# ------------------------------
st.title("⚖️ Legal Contract Analyzer")
st.write("Upload or paste a contract to analyze clauses, risks, and entities.")

# Input choice
tab1, tab2 = st.tabs(["📂 Upload File", "✍️ Paste Text"])

contract_text = None

with tab1:
    uploaded_file = st.file_uploader("Upload contract (TXT or PDF)", type=["txt", "pdf"])
    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith('.pdf'):
                st.info("Processing PDF file...")
                pdf_reader = PdfReader(uploaded_file)
                text_content = [page.extract_text() or "" for page in pdf_reader.pages]
                contract_text = "\n".join(text_content)
                if not contract_text.strip():
                    st.error("Could not extract text from PDF. Please ensure the PDF contains searchable text.")
                else:
                    st.success(f"Successfully extracted {len(text_content)} pages from PDF")
            else:
                contract_text = uploaded_file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            contract_text = None

with tab2:
    pasted_text = st.text_area("Paste contract text here", height=250)
    if pasted_text:
        contract_text = pasted_text

# Sample contract
st.markdown("### 📝 Try Sample Contract")
sample_contract = """
EMPLOYMENT AGREEMENT

This Employment Agreement is entered into between ABC Corp. and John Doe.

1. TERM: This agreement shall commence on January 1, 2024, and continue for a period of two years.
2. COMPENSATION: Employee shall receive a base salary of $75,000 per year.
3. TERMINATION: Either party may terminate this agreement with 30 days written notice.
4. CONFIDENTIALITY: Employee agrees to maintain confidentiality of all proprietary information.
5. NON-COMPETE: Employee agrees not to compete with Company for 12 months after termination.
"""

if st.button("📋 Use Sample Contract"):
    contract_text = sample_contract
    st.success("Sample contract loaded!")

# ------------------------------
# Run Analysis
# ------------------------------
if st.button("🔍 Analyze Contract", type="primary") and contract_text:
    with st.spinner("Analyzing contract with Railway backend..."):
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update status
            status_text.text("Sending request to Railway API...")
            progress_bar.progress(10)
            
            # Prepare and send request
            payload = {"text": contract_text}
            start_time = time.time()
            
            response = requests.post(api_url, json=payload, timeout=120)
            
            # Update progress
            status_text.text("Processing response...")
            progress_bar.progress(70)
            
            if response.status_code == 200:
                result = response.json()
                progress_bar.progress(100)
                status_text.text(f"Analysis complete in {time.time() - start_time:.2f} seconds!")
                st.success("✅ Analysis complete!")
                
                # ---------------- Summary Metrics ----------------
                try:
                    cols = st.columns(5)
                    cols[0].metric("Total Clauses", result["executive_summary"]["total_clauses_identified"])
                    cols[1].metric("High Risk", result["executive_summary"]["high_risk_items"], delta_color="inverse")
                    cols[2].metric("Medium Risk", result["executive_summary"]["medium_risk_items"])
                    cols[3].metric("Low Risk", result["executive_summary"].get("low_risk_items", 0), delta_color="normal")
                    cols[4].metric("Risk Score", f"{result['executive_summary']['overall_risk_score']:.1f}%")
                except Exception as e:
                    st.error(f"Error displaying metrics: {str(e)}")

                # Risk level indicator
                risk_score = result["executive_summary"]["overall_risk_score"]
                if risk_score >= 70:
                    st.error(f"🚨 High Risk Contract (Score: {risk_score:.1f}%)")
                elif risk_score >= 40:
                    st.warning(f"⚠️ Medium Risk Contract (Score: {risk_score:.1f}%)")
                else:
                    st.success(f"✅ Low Risk Contract (Score: {risk_score:.1f}%)")

                # Key Findings
                st.markdown("**Key Findings:**")
                for finding in result["executive_summary"].get("key_findings", []):
                    st.info(f"🔎 {finding}")

                # ---------------- Clause Analysis ----------------
                if result["clause_analysis"]:
                    st.subheader("📑 Detailed Clause Analysis")
                    
                    # Group clauses
                    high_risk = [c for c in result["clause_analysis"] if c.get("risk_level") == "HIGH"]
                    medium_risk = [c for c in result["clause_analysis"] if c.get("risk_level") == "MEDIUM"]
                    low_risk = [c for c in result["clause_analysis"] if c.get("risk_level") == "LOW"]

                    # High Risk
                    if high_risk:
                        st.markdown("### 🚨 High Risk Clauses")
                        for clause in high_risk:
                            with st.expander(f"🔴 {clause['clause_type']} (Confidence: {clause['confidence']:.2f})", expanded=True):
                                st.markdown(f"**Risk Level:** {clause['risk_level']}")
                                st.markdown(f"**Context:** {clause.get('context', 'N/A')}")
                                if clause.get('risk_factors'):
                                    st.markdown("**Risk Factors Identified:**")
                                    for factor in clause['risk_factors']:
                                        st.markdown(f"- {factor}")
                                if clause.get('recommendations'):
                                    st.markdown("**Recommendations:**")
                                    for rec in clause['recommendations']:
                                        st.markdown(f"- {rec}")

                    # Medium Risk
                    if medium_risk:
                        st.markdown("### ⚠️ Medium Risk Clauses")
                        for clause in medium_risk:
                            with st.expander(f"🟡 {clause['clause_type']} (Confidence: {clause['confidence']:.2f})"):
                                st.markdown(f"**Risk Level:** {clause['risk_level']}")
                                st.markdown(f"**Context:** {clause.get('context', 'N/A')}")
                                if clause.get('risk_factors'):
                                    st.markdown("**Risk Factors:**")
                                    for factor in clause['risk_factors']:
                                        st.markdown(f"- {factor}")
                                if clause.get('recommendations'):
                                    st.markdown("**Recommendations:**")
                                    for rec in clause['recommendations']:
                                        st.markdown(f"- {rec}")

                    # Low Risk
                    if low_risk:
                        st.markdown("### ✅ Low Risk Clauses")
                        for clause in low_risk:
                            with st.expander(f"🟢 {clause['clause_type']} (Confidence: {clause['confidence']:.2f})"):
                                st.markdown(f"**Risk Level:** {clause['risk_level']}")
                                st.markdown(f"**Context:** {clause.get('context', 'N/A')}")
                                if clause.get('recommendations'):
                                    st.markdown("**Recommendations:**")
                                    for rec in clause['recommendations']:
                                        st.markdown(f"- {rec}")

                # ---------------- Entities ----------------
                if show_entities and result.get("identified_entities"):
                    st.subheader("🧾 Named Entities")
                    for ent_type, ents in result["identified_entities"].items():
                        st.markdown(f"**{ent_type}**")
                        for ent in ents:
                            st.write(f"- {ent['text']} (conf: {ent['confidence']:.2f})")
                
                # Raw JSON in expandable section
                with st.expander("🔍 View Raw Analysis JSON"):
                    st.json(result)
                
            else:
                st.error(f"API returned status code: {response.status_code}")
                st.error(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The analysis might be taking longer than expected.")
            st.info("Please try with a shorter document or try again later.")
            
        except requests.exceptions.ConnectionError:
            st.error("🔌 Connection error. Could not connect to the Railway API.")
            st.info("Please check if the API is running and accessible.")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("Note: This app connects to a Railway backend API for contract analysis.")

else:
    st.warning("👆 Upload/paste a contract or use the sample, then click **Analyze Contract**")

# Footer
st.markdown("---")
st.markdown("**Legal Contract Analyzer** | Powered by Railway + Streamlit Cloud")