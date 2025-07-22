"""Streamlit frontend for VulnRAG pipeline"""
import streamlit as st
import json
import dataclasses
import logging
import time
from io import StringIO
from pathlib import Path
from rag.core.pipeline import detect_vulnerability, generate_patch, _get_default_pipeline

st.set_page_config(page_title="VulnRAG Demo", layout="wide")

# Warm-up pipeline at app launch (indexes, searchers, models) with animation
if "pipeline_ready" not in st.session_state:
    placeholder = st.empty()
    with st.spinner("🚀 Bootstrapping VulnRAG …"):
        prog = st.progress(0, text="Indexing KBs…")
        steps = [
            (20, "Loading Whoosh index…"),
            (40, "Loading FAISS indexes…"),
            (60, "Initializing encoders…"),
            (80, "Compiling models…"),
        ]
        for pct, msg in steps:
            time.sleep(0.5)
            prog.progress(pct, text=msg)
        pipeline = _get_default_pipeline()
        prog.progress(90, text="Warming up Quick LLM …")
        # Suppression du warm-up quick_llm_gate
        prog.progress(100, text="Finalizing…")
    placeholder.empty()
    st.session_state["pipeline_ready"] = True
    st.success("✨ Warm-up completed! Ready to analyse code.")

st.title("🔍 VulnRAG Vulnerability Detector & Patcher")

# ✅ CORRECTION: Add reset button if results are displayed
if "detection_result" in st.session_state:
    if st.button("🔄 New Analysis"):
        # Clear session state
        for key in ["detection_result", "code_input"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

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
    # Créer des placeholders pour l'affichage en temps réel
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    logs_placeholder = st.empty()
    
    # Fonction callback pour mettre à jour l'affichage
    def update_display(message, progress=None):
        status_placeholder.info(f"🔄 {message}")
        if progress is not None:
            progress_placeholder.progress(progress, text=message)
        else:
            progress_placeholder.empty()
        
        # Afficher les logs en temps réel
        log_handler.flush()
        logs_content = log_stream.getvalue()
        if logs_content:
            logs_placeholder.code(logs_content, language="log")
    
    # Exécuter la détection avec callback de progression
    det = detect_vulnerability(code_input, progress_callback=update_display)
    
    # Sauvegarder le résultat dans session_state
    st.session_state["detection_result"] = det
    st.session_state["code_input"] = code_input
    
    # Nettoyer les placeholders
    status_placeholder.empty()
    progress_placeholder.empty()
    
    # Sauvegarder les logs pour téléchargement
    log_handler.flush()
    final_logs = log_stream.getvalue()
    
    st.subheader("Detection Summary")
    
    # Affichage du verdict principal
    verdict = det.get("decision", "UNKNOWN")
    is_vulnerable = det.get("is_vulnerable", False)
    
    if verdict == "VULNERABLE":
        st.error(f"🚨 **Verdict:** {verdict}")
    elif verdict == "SAFE":
        st.success(f"✅ **Verdict:** {verdict}")
    else:
        st.warning(f"⚠️ **Verdict:** {verdict}")
    
    # Affichage des votes
    if det.get("votes"):
        st.write("**Voting Results:**")
        votes_cols = st.columns(len(det["votes"]))
        for i, (voter, vote) in enumerate(det["votes"].items()):
            with votes_cols[i]:
                if vote == "VULN":
                    st.error(f"{voter.upper()}: {vote}")
                elif vote == "SAFE":
                    st.success(f"{voter.upper()}: {vote}")
                else:
                    st.warning(f"{voter.upper()}: {vote}")
    
    # Affichage confiance et CWE si disponibles
    if det.get("confidence") is not None:
        st.write(f"**Confidence:** {det['confidence']:.2f}")
    if det.get("cwe"):
        st.write(f"**CWE:** {det['cwe']}")
    if det.get("explanation"):
        st.write(f"**Explanation:** {det['explanation']}")

    # ✅ CORRECTION: Patch generation with session state persistence
    if is_vulnerable and not det.get("patch"):
        if st.button("🔧 Generate Patch"):
            patch_status = st.empty()
            patch_progress = st.empty()
            
            def patch_progress_callback(message, progress=None):
                patch_status.info(f"🔄 {message}")
                if progress is not None:
                    patch_progress.progress(progress, text=message)
                else:
                    patch_progress.empty()
            
            # Generate patch and store in session state
            patch = generate_patch(code_input, detection_result=det, progress_callback=patch_progress_callback)
            st.session_state["detection_result"]["patch"] = patch
            
            patch_status.success("✅ Patch generated successfully!")
            patch_progress.empty()
            time.sleep(1)  # Montrer le message de succès
            st.rerun()  # Relancer pour afficher le patch

# ✅ CORRECTION: Display results from session state if available
elif "detection_result" in st.session_state:
    det = st.session_state["detection_result"]
    code_input = st.session_state.get("code_input", "")
    
    st.subheader("Detection Summary")
    
    # Affichage du verdict principal
    verdict = det.get("decision", "UNKNOWN")
    is_vulnerable = det.get("is_vulnerable", False)
    
    if verdict == "VULNERABLE":
        st.error(f"🚨 **Verdict:** {verdict}")
    elif verdict == "SAFE":
        st.success(f"✅ **Verdict:** {verdict}")
    else:
        st.warning(f"⚠️ **Verdict:** {verdict}")
    
    # Affichage des votes
    if det.get("votes"):
        st.write("**Voting Results:**")
        votes_cols = st.columns(len(det["votes"]))
        for i, (voter, vote) in enumerate(det["votes"].items()):
            with votes_cols[i]:
                if vote == "VULN":
                    st.error(f"{voter.upper()}: {vote}")
                elif vote == "SAFE":
                    st.success(f"{voter.upper()}: {vote}")
                else:
                    st.warning(f"{voter.upper()}: {vote}")
    
    # Affichage confiance et CWE si disponibles
    if det.get("confidence") is not None:
        st.write(f"**Confidence:** {det['confidence']:.2f}")
    if det.get("cwe"):
        st.write(f"**CWE:** {det['cwe']}")
    if det.get("explanation"):
        st.write(f"**Explanation:** {det['explanation']}")

    # ✅ CORRECTION: Patch generation button for vulnerable code without patch
    if is_vulnerable and not det.get("patch"):
        if st.button("🔧 Generate Patch"):
            patch_status = st.empty()
            patch_progress = st.empty()
            
            def patch_progress_callback(message, progress=None):
                patch_status.info(f"🔄 {message}")
                if progress is not None:
                    patch_progress.progress(progress, text=message)
                else:
                    patch_progress.empty()
            
            # Generate patch and store in session state
            patch = generate_patch(code_input, detection_result=det, progress_callback=patch_progress_callback)
            st.session_state["detection_result"]["patch"] = patch
            
            patch_status.success("✅ Patch generated successfully!")
            patch_progress.empty()
            time.sleep(1)  # Montrer le message de succès
            st.rerun()  # Relancer pour afficher le patch
    
    # Display patch if available
    if det.get("patch"):
        st.subheader("🔧 Suggested Patch")
        st.code(det["patch"], language="c")
        st.download_button(
            "Download patch", 
            det["patch"], 
            file_name="vulnerability_patch.c",
            mime="text/plain"
        )

    # ✅ CORRECTION: Utiliser timings_s uniquement
    if det.get("timings_s"):
        st.subheader("⏱️ Performance Metrics")
        timing_cols = st.columns(len(det["timings_s"]))
        for i, (phase, duration) in enumerate(det["timings_s"].items()):
            with timing_cols[i]:
                st.metric(phase.title(), f"{duration:.2f}s")

    # Section détails dans un expander
    with st.expander("🔍 Detailed Analysis", expanded=False):
        
        # ✅ CORRECTION: Utiliser static_summary (nom cohérent)
        if det.get("static"):
            st.write("#### Static Analysis Results")
            static_data = det["static"]
            
            # Structured display of static results
            if static_data.get("cppcheck_issues"):
                st.write("**Cppcheck Issues:**")
                for issue in static_data["cppcheck_issues"][:5]:
                    severity = issue.get("severity", "info")
                    msg = issue.get("msg", "No message")
                    line = issue.get("line", "?")
                    if severity in ["error", "critical"]:
                        st.error(f"Line {line}: {msg}")
                    elif severity == "warning":
                        st.warning(f"Line {line}: {msg}")
                    else:
                        st.info(f"Line {line}: {msg}")
            
            if static_data.get("flawfinder_issues"):
                st.write("**Flawfinder Issues:**")
                for issue in static_data["flawfinder_issues"][:5]:
                    severity = issue.get("severity", "info")
                    msg = issue.get("msg", "No message")
                    line = issue.get("line", "?")
                    if severity in ["error", "critical"]:
                        st.error(f"Line {line}: {msg}")
                    elif severity == "warning":
                        st.warning(f"Line {line}: {msg}")
                    else:
                        st.info(f"Line {line}: {msg}")

            if static_data.get("message"):
                st.write(f"**Message:** {static_data['message']}")
            
            # Issues détaillées
            for tool in ["cppcheck_issues", "flawfinder_issues"]:
                issues = static_data.get(tool, [])
                if issues:
                    st.write(f"**{tool.replace('_', ' ').title()}:**")
                    for issue in issues[:5]:  # Limiter à 5 issues
                        severity = issue.get("severity", "info")
                        msg = issue.get("msg", "No message")
                        line = issue.get("line", "?")
                        if severity in ["error", "critical"]:
                            st.error(f"Line {line}: {msg}")
                        elif severity == "warning":
                            st.warning(f"Line {line}: {msg}")
                        else:
                            st.info(f"Line {line}: {msg}")
            else:
                st.json(static_data)

        # ✅ CORRECTION: Utiliser heuristic au lieu de heuristic_details
        if det.get("heuristic"):
            st.write("#### Heuristic Analysis")
            heuristic_data = det["heuristic"]
            if isinstance(heuristic_data, dict):
                if heuristic_data.get("risk_score") is not None:
                    risk_score = heuristic_data["risk_score"]
                    st.metric("Risk Score", f"{risk_score:.2f}")
                    
                    # Barre de progression pour le risque
                    risk_color = "red" if risk_score > 0.6 else "orange" if risk_score > 0.3 else "green"
                    st.progress(risk_score, text=f"Risk Level: {risk_score:.1%}")
                
                if heuristic_data.get("dangerous_patterns"):
                    st.write("**Dangerous Patterns Detected:**")
                    for pattern in heuristic_data["dangerous_patterns"][:3]:
                        st.write(f"• {pattern}")
            else:
                st.json(heuristic_data)

        # Documents similaires trouvés
        if det.get("enriched_docs"):
            st.write(f"#### Similar Vulnerabilities Found ({len(det['enriched_docs'])})")
            
            for i, doc in enumerate(det["enriched_docs"][:3], 1):
                with st.container():
                    st.write(f"**Document {i}: {doc.key if hasattr(doc, 'key') else 'Unknown'}**")
                    
                    # Convertir dataclass en dict si nécessaire
                    if hasattr(doc, '__dict__'):
                        doc_dict = dataclasses.asdict(doc) if dataclasses.is_dataclass(doc) else doc.__dict__
                    else:
                        doc_dict = doc
                    
                    cols = st.columns(2)
                    with cols[0]:
                        if doc_dict.get("gpt_purpose"):
                            st.write(f"**Purpose:** {doc_dict['gpt_purpose'][:100]}...")
                        if doc_dict.get("cwe"):
                            st.write(f"**CWE:** {doc_dict['cwe']}")
                    
                    with cols[1]:
                        if doc_dict.get("final_score"):
                            st.metric("Similarity", f"{doc_dict['final_score']:.3f}")
                        if doc_dict.get("dangerous_functions_count"):
                            st.metric("Dangerous Functions", doc_dict["dangerous_functions_count"])
                    
                    if doc_dict.get("code_before_change"):
                        st.write("**Similar Vulnerable Code:**")
                        st.code(doc_dict["code_before_change"][:200] + "..." if len(doc_dict["code_before_change"]) > 200 else doc_dict["code_before_change"], language="c")
                    
                    st.divider()
            
            # Téléchargement des documents
            docs_serializable = []
            for doc in det["enriched_docs"]:
                if dataclasses.is_dataclass(doc):
                    docs_serializable.append(dataclasses.asdict(doc))
                elif hasattr(doc, '__dict__'):
                    docs_serializable.append(doc.__dict__)
                else:
                    docs_serializable.append(doc)
            
            st.download_button(
                "📥 Download similar documents (JSON)",
                data=json.dumps(docs_serializable, indent=2, default=str),
                file_name="similar_vulnerabilities.json",
                mime="application/json",
            )

        # Prompt LLM si disponible
        if det.get("prompt"):
            st.write("#### LLM Detection Prompt")
            with st.container():
                st.text_area("Prompt used for detection:", det["prompt"], height=200, disabled=True)
                st.download_button(
                    "📄 Download prompt", 
                    det["prompt"], 
                    file_name="detection_prompt.txt",
                    mime="text/plain"
                )

        # Réponse brute LLM
        if det.get("llm_raw"):
            st.write("#### Raw LLM Response")
            st.text_area("Raw response:", det["llm_raw"], height=150, disabled=True)

        # Téléchargement résultat complet
        def _default_serializer(o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            if hasattr(o, '__dict__'):
                return o.__dict__
            return str(o)
        
        st.download_button(
            "📊 Download complete analysis (JSON)",
            json.dumps(det, indent=2, default=_default_serializer),
            file_name="vulnerability_analysis.json",
            mime="application/json",
        )

    # Affichage des logs finaux
    st.subheader("📋 Analysis Logs")
    log_handler.flush()
    logs_content = log_stream.getvalue()
    if logs_content:
        # Afficher les logs dans un expander pour économiser l'espace
        with st.expander("🔍 View Analysis Logs", expanded=False):
            st.code(logs_content, language="log")
        
        # Bouton de téléchargement des logs
        st.download_button(
            "📥 Download Analysis Logs",
            logs_content,
            file_name=f"vulnrag_analysis_{time.strftime('%Y%m%d_%H%M%S')}.log",
            mime="text/plain"
        )
    else:
        st.info("No logs captured during analysis.")

else:
    st.info("👆 Enter C/C++ code above and press 'Run Analysis' to start vulnerability detection.")
    
    
    
   