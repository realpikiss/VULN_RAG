"""Streamlit frontend for VulnRAG pipeline"""
import streamlit as st
import logging
from io import StringIO
from pathlib import Path
from rag.core.pipeline import detect_vulnerability, generate_patch

st.set_page_config(page_title="VulnRAG Demo", layout="wide")

st.title("🔍 VulnRAG Vulnerability Detector & Patcher")

code_input = st.text_area("Paste C/C++ code here:", height=250)

# Set up in-memory log capture
log_stream = StringIO()
log_handler = logging.StreamHandler(log_stream)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
root_logger = logging.getLogger()
if log_handler not in root_logger.handlers:
    root_logger.addHandler(log_handler)
root_logger.setLevel(logging.INFO)

if st.button("Run Analysis") and code_input.strip():
    with st.spinner("Running detection…"):
        det = detect_vulnerability(code_input)
    st.subheader("Detection Result")
    st.json(det)
    if "timings_ms" in det:
        st.success(f"Total time: {det['timings_ms']['total']:.1f} ms")

    # Display runtime logs
    st.subheader("Logs")
    log_handler.flush()
    st.code(log_stream.getvalue())

    if det.get("is_vulnerable"):
        with st.spinner("Generating patch…"):
            patch = generate_patch(code_input, detection_result=det)
        st.subheader("Suggested Patch")
        st.code(patch, language="c")

        # Show additional logs after patch
        log_handler.flush()
        st.code(log_stream.getvalue())
else:
    st.info("Enter code and press 'Run Analysis'.")
