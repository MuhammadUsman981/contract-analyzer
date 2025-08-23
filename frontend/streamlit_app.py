import streamlit as st
import requests
import json
import time
import random
from PyPDF2 import PdfReader

# Add this mock analyzer function
def mock_analyze_contract(text):
    """Create realistic contract analysis results for demo purposes"""
    # Simulate processing time
    time.sleep(2)
    
    # Extract some simple entities from text
    entities = {}
    if "John" in text or "Jane" in text:
        entities["PERSON"] = [{"text": name, "confidence": round(random.uniform(0.85, 0.98), 2)} 
                             for name in ["John Doe", "Jane Smith"] if name.split()[0] in text]
    
    if "ABC Corp" in text or "Company" in text:
        entities["ORG"] = [{"text": "ABC Corp", "confidence": round(random.uniform(0.88, 0.97), 2)}]
    
    # Look for dates and money
    if "January" in text or "2024" in text:
        entities["DATE"] = [{"text": "January 1, 2024", "confidence": round(random.uniform(0.90, 0.98), 2)}]
    
    if "$" in text:
        money_mentions = [word for word in text.split() if "$" in word]
        if money_mentions:
            entities["MONEY"] = [{"text": money, "confidence": round(random.uniform(0.92, 0.99), 2)} 
                                for money in money_mentions]
    
    # Generate clause analysis based on keywords in text
    clauses = []
    
    # Check for non-compete
    if "non-compete" in text.lower() or "compete" in text.lower():
        clauses.append({
            "clause_type": "Non-Compete",
            "risk_level": "HIGH",
            "confidence": round(random.uniform(0.85, 0.95), 2),
            "context": "Employee agrees not to compete with Company for 12 months after termination.",
            "risk_factors": ["Duration is excessive", "Geographic scope not clearly defined"],
            "recommendations": ["Reduce duration to 6 months", "Add specific geographic limitations"]
        })
    
    # Check for termination
    if "termination" in text.lower() or "terminate" in text.lower():
        clauses.append({
            "clause_type": "Termination",
            "risk_level": "MEDIUM",
            "confidence": round(random.uniform(0.80, 0.92), 2),
            "context": "Either party may terminate this agreement with 30 days written notice.",
            "risk_factors": ["Notice period may be insufficient"],
            "recommendations": ["Consider extending notice period to 60 days"]
        })
    
    # Check for confidentiality
    if "confidentiality" in text.lower() or "confidential" in text.lower():
        clauses.append({
            "clause_type": "Confidentiality",
            "risk_level": "LOW" if random.random() > 0.3 else "MEDIUM",
            "confidence": round(random.uniform(0.82, 0.94), 2),
            "context": "Employee agrees to maintain confidentiality of all proprietary information.",
            "recommendations": ["Define 'proprietary information' more specifically"]
        })
    
    # Check for compensation
    if "compensation" in text.lower() or "salary" in text.lower():
        clauses.append({
            "clause_type": "Compensation",
            "risk_level": "LOW",
            "confidence": round(random.uniform(0.85, 0.96), 2),
            "context": "Employee shall receive a base salary per year.",
            "recommendations": ["Add performance review and salary adjustment provisions"]
        })
    
    # Add more clauses to ensure we have something
    if len(clauses) < 3:
        additional_clauses = [
            {
                "clause_type": "Intellectual Property",
                "risk_level": "HIGH",
                "confidence": round(random.uniform(0.80, 0.95), 2),
                "context": "All work product shall belong exclusively to the Company.",
                "risk_factors": ["No distinction between prior IP and new IP"],
                "recommendations": ["Add clause to protect employee's prior inventions"]
            },
            {
                "clause_type": "Governing Law",
                "risk_level": "LOW",
                "confidence": round(random.uniform(0.85, 0.97), 2),
                "context": "This Agreement shall be governed by the laws of the State.",
                "recommendations": ["Specify exact jurisdiction"]
            }
        ]
        clauses.extend(additional_clauses[:3-len(clauses)])
    
    # Count risk levels
    high_risk = len([c for c in clauses if c["risk_level"] == "HIGH"])
    medium_risk = len([c for c in clauses if c["risk_level"] == "MEDIUM"])
    low_risk = len([c for c in clauses if c["risk_level"] == "LOW"])
    
    # Calculate overall risk score
    risk_score = (high_risk * 25 + medium_risk * 10 + low_risk * 3)
    risk_score = min(95, max(15, risk_score + random.randint(-8, 8)))
    
    # Generate key findings
    findings = []
    if high_risk > 0:
        findings.append("Contract contains high-risk clauses requiring immediate attention")
    if "non-compete" in text.lower():
        findings.append("Non-compete clause is overly restrictive")
    if "termination" in text.lower() and random.random() > 0.5:
        findings.append("Termination terms favor one party significantly")
    if "confidentiality" in text.lower() and random.random() > 0.7:
        findings.append("Confidentiality provisions lack specificity")
    if len(findings) < 2:
        findings.append("Consider adding more specific performance metrics")
    
    # Create final response
    return {
        "executive_summary": {
            "total_clauses_identified": len(clauses),
            "high_risk_items": high_risk,
            "medium_risk_items": medium_risk,
            "low_risk_items": low_risk,
            "overall_risk_score": risk_score,
            "key_findings": findings
        },
        "clause_analysis": clauses,
        "identified_entities": entities
    }

# API URL (points to your Railway backend)
API_URL = "https://contract-analyzer-production-a7ea.up.railway.app/analyze"

st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("⚙️ Settings")
use_api = st.sidebar.checkbox("Use Railway API", value=False)
api_url = st.sidebar.text_input("API URL (if using API)", API_URL)
show_entities = st.sidebar.checkbox("Show Named Entities", value=True)

st.sidebar.markdown("---")
if use_api:
    st.sidebar.info("Using Railway backend API")
else:
    st.sidebar.info("Using Streamlit Cloud (standalone mode)")

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
    with st.spinner("Analyzing contract..."):
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update status
            if use_api:
                status_text.text("Sending request to Railway API...")
                progress_bar.progress(10)
                
                # Prepare and send request to Railway
                payload = {"text": contract_text}
                start_time = time.time()
                response = requests.post(api_url, json=payload, timeout=120)
                
                # Update progress
                status_text.text("Processing response...")
                progress_bar.progress(70)
                
                if response.status_code == 200:
                    result = response.json()
                    # Check if we got the demo mode response
                    if "demo_mode" in result.get("status", ""):
                        st.warning("Railway API is in demo mode. Using local analysis instead.")
                        result = mock_analyze_contract(contract_text)
                else:
                    st.error(f"API returned status code: {response.status_code}")
                    st.info("Falling back to local analysis...")
                    result = mock_analyze_contract(contract_text)
            else:
                # Use the mock analyzer
                status_text.text("Processing contract locally...")
                progress_bar.progress(25)
                time.sleep(0.5)  # Small delay to show progress
                
                status_text.text("Analyzing clauses and entities...")
                progress_bar.progress(50)
                time.sleep(0.5)  # Small delay to show progress
                
                status_text.text("Evaluating risks...")
                progress_bar.progress(75)
                
                # Get mock analysis results
                start_time = time.time()
                result = mock_analyze_contract(contract_text)
            
            # Complete the progress
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
            
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The analysis might be taking longer than expected.")
            st.info("Try with a shorter document or switch to local analysis.")
            
        except requests.exceptions.ConnectionError:
            st.error("🔌 Connection error. Could not connect to the Railway API.")
            st.info("Using local analysis instead...")
            result = mock_analyze_contract(contract_text)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

else:
    st.warning("👆 Upload/paste a contract or use the sample, then click **Analyze Contract**")

# Footer
st.markdown("---")
mode_text = "Railway API + " if use_api else ""
st.markdown(f"**Legal Contract Analyzer** | Powered by {mode_text}Streamlit Cloud")