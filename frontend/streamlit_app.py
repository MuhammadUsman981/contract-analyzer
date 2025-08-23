import streamlit as st
import requests
import json
import time
from PyPDF2 import PdfReader

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
st.sidebar.info(f"Backend: {'Local' if use_local else 'Railway'}")

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

# ------------------------------
# Run Analysis
# ------------------------------
if st.button("🔍 Analyze Contract", type="primary") and contract_text:
    with st.spinner("Analyzing contract..."):
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update status
            status_text.text(f"Sending request to {'local' if use_local else 'Railway'} API...")
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
            st.error("🔌 Connection error. Could not connect to the API.")
            st.info(f"Please check if the {'local' if use_local else 'Railway'} API is running and accessible.")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

else:
    st.warning("👆 Upload/paste a contract and click **Analyze Contract**")

# Footer
st.markdown("---")
st.markdown("**Legal Contract Analyzer** | Running locally")
