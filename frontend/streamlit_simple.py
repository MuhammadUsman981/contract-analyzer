import streamlit as st
import requests
import json
from typing import Dict, Any

# Configure Streamlit page
st.set_page_config(
    page_title="Contract Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# Your Railway API URL
API_BASE_URL = "https://contract-analyzer-production-a7ea.up.railway.app"

st.title("⚖️ Legal Contract Analyzer")
st.write("AI-Powered Legal Document Analysis - Connected to Railway Backend")

# Sidebar with API status
st.sidebar.title("🔗 Backend Status")

try:
    response = requests.get(f"{API_BASE_URL}/health", timeout=10)
    if response.status_code == 200:
        health_data = response.json()
        st.sidebar.success("✅ Backend Online")
        st.sidebar.json(health_data)
    else:
        st.sidebar.error("❌ Backend Offline")
except Exception as e:
    st.sidebar.error(f"❌ Connection Error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.info("**API Endpoints:**")
st.sidebar.write(f"• [Main API]({API_BASE_URL})")
st.sidebar.write(f"• [API Docs]({API_BASE_URL}/docs)")
st.sidebar.write(f"• [Health Check]({API_BASE_URL}/health)")

# Main content
st.markdown("## 📄 Contract Analysis")

# Input methods
tab1, tab2 = st.tabs(["✍️ Text Input", "🔗 API Testing"])

with tab1:
    st.markdown("### Input Contract Text")
    contract_text = st.text_area(
        "Paste your contract text here:",
        height=200,
        placeholder="Enter contract text for analysis..."
    )
    
    if st.button("🔍 Analyze Contract", type="primary"):
        if contract_text.strip():
            with st.spinner("Analyzing contract..."):
                try:
                    # Call your Railway API
                    response = requests.post(
                        f"{API_BASE_URL}/analyze",
                        json={"text": contract_text},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Analysis Complete!")
                        
                        # Display results
                        st.markdown("### 📊 Analysis Results")
                        st.json(result)
                        
                    else:
                        st.error(f"❌ Analysis failed: {response.status_code}")
                        st.write(response.text)
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter some contract text to analyze.")

with tab2:
    st.markdown("### 🧪 API Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Health Endpoint"):
            try:
                response = requests.get(f"{API_BASE_URL}/health")
                st.json(response.json())
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("Test Main Endpoint"):
            try:
                response = requests.get(f"{API_BASE_URL}/")
                st.json(response.json())
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Sample contract for testing
st.markdown("### 📝 Sample Contract for Testing")
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
    st.text_area("Sample contract loaded:", value=sample_contract, height=150, key="sample")

# Footer
st.markdown("---")
st.markdown("**Created by Muhammad Usman** | [GitHub](https://github.com/MuhammadUsman981) | [Live API](https://contract-analyzer-production-a7ea.up.railway.app)")