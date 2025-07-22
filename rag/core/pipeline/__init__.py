"""rag.core.pipeline package

This package exposes a high-level API :

    from rag.core.pipeline import detect_vulnerability, generate_patch

The internal logic relies on :
    • StaticAnalysisGate    – 
    • QuickHeuristicGate    – 
    • QuickLLMGate          – make a quick LLM call to arbitrate between static and heuristic

    • EnhancedPreprocessing – extraction LLM + CPG
    • Vuln_RAGRetrievalController – hybrid search + RRF
    • DocumentAssembler     – enriched context
    • ContextBuilder + Qwen – prompts and generation


"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

from rag.core.analysis.static_analysis_gate import StaticAnalysisGate
from rag.core.preprocessing import (
    create_pipeline,
    PreprocessingPipeline,
)
from rag.core.retrieval.fusion_controller import (
    Vuln_RAGRetrievalController,
    FusionCandidate,
)
from rag.core.retrieval.document_assembler import DocumentAssembler, EnrichedDocument
from rag.core.generation.context_builder import ContextBuilder
from rag.core.generation.ollama_qwen import generate as ollama_generate

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

__all__ = [
    "VulnRAGPipeline",
    "detect_vulnerability",
    "generate_patch",
]


class VulnRAGPipeline:
    """End-to-end pipeline relying on all RAG steps."""
    
    # CWE supportés par le système
    SUPPORTED_CWES = [
        "CWE-119", "CWE-120", "CWE-125", "CWE-476", "CWE-362",
        "CWE-787", "CWE-20", "CWE-200", "CWE-264", "CWE-401"
    ]

    def __init__(
        self,
        *,
        llm_interface=None,
        cpg_generator=None,
        llm_model: str = "qwen2.5-coder:latest",
        kb1_index_path: Optional[str] = os.getenv("KB1_INDEX_PATH"),
        kb2_index_path: Optional[str] = os.getenv("KB2_INDEX_PATH"),
        kb2_metadata_path: Optional[str] = os.getenv("KB2_METADATA_PATH"),
        kb3_index_path: Optional[str] = os.getenv("KB3_INDEX_PATH"),
        kb3_metadata_path: Optional[str] = os.getenv("KB3_METADATA_PATH"),
        rrf_k: int = 60,
        rrf_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        # 0. Static analysis gate
        self.static_gate = StaticAnalysisGate()

        # 1. Preprocessing pipeline (LLM + CPG)
        # The new minimal interface only needs the model names; advanced injection
        # of custom interfaces can be added later.
        self.preprocessor: PreprocessingPipeline = create_pipeline(llm_model=llm_model)

        # 2. Retrieval controller RRF
        self.retrieval: Vuln_RAGRetrievalController = Vuln_RAGRetrievalController(
            kb1_index_path ,
            kb2_index_path ,
            kb2_metadata_path ,
            kb3_index_path ,
            kb3_metadata_path ,
            rrf_k=rrf_k,
            rrf_weights=rrf_weights,
        )

        # 2. Heuristic gate uniquement
        from rag.core.analysis.quick_heuristic_gate import QuickHeuristicGate
        self.heuristic_gate = QuickHeuristicGate()
        # (Suppression de QuickLLMGate)

        # 3. Document assembler
        self.assembler = DocumentAssembler(kb1_index_path)

        # ------------------------------------------------------------------
        # Warm-up heavy resources (indexes, searchers)
        # ------------------------------------------------------------------
        try:
            # Force Whoosh index to load
            _ = self.assembler.index
            # Force FAISS/Whoosh searchers to initialize
            _ = self.retrieval._get_kb1_searcher()
            _ = self.retrieval._get_kb2_searcher()
            _ = self.retrieval._get_kb3_searcher()
            # Dummy preprocess to warm LLM & CPG models
            try:
                _ = self.preprocess("int main() { return 0; }")
            except Exception as _ignore:
                pass
            # Warm-up quick LLM gate
            try:
                _ = self.quick_llm_gate.analyse(
                    "int main(){return 0;}",
                    static_verdict="SAFE",
                    heuristic_verdict="SAFE",
                    heuristic_score=0.0,
                )
            except Exception as _ignore:
                pass
            logger.info("VulnRAGPipeline warm-up completed (indexes, searchers, preprocessors ready)")
        except Exception as warm_err:
            logger.warning("Warm-up encountered an error: %s", warm_err)

    # ------------------------------------------------------------------
    # Building blocks ---------------------------------------------------
    # ------------------------------------------------------------------
    def preprocess(self, code: str):
        return self.preprocessor.process(code)

    def search(self, preproc_result) -> List[FusionCandidate]:
        return self.retrieval.search_from_preprocessed_query(preproc_result.to_query_dict())

    def assemble(self, candidates: List[FusionCandidate], top_k: int = 5) -> List[EnrichedDocument]:
        return self.assembler.assemble_documents(candidates, top_k=top_k)

    def _build_static_llm_prompt(self, code: str, static_res: dict, heuristic_res=None) -> str:
        """Builds a LLM prompt from code and static/heuristic analysis results."""
        import json
        prompt = [
            "# Multi-tool Static and Heuristic Analysis",
            "\n## Source Code to Analyze",
            "```c",
            code.strip(),
            "```\n",
            "## Static Analysis Results",
            "```json",
            json.dumps(static_res, indent=2),
            "```\n"
        ]
        
        # Add heuristic results if available
        if heuristic_res:
            prompt.extend([
                "## Heuristic Analysis Results",
                "```json",
                json.dumps(heuristic_res.to_dict() if hasattr(heuristic_res, "to_dict") else heuristic_res, indent=2),
                "```\n"
            ])
        
        prompt.extend([
            f"## Supported CWEs: {', '.join(self.SUPPORTED_CWES)}",
            "## Step-by-Step Analysis Instructions",
            (
                "You are a cybersecurity expert analyzing C code for vulnerabilities. Follow these steps:\n\n"
                "STEP 1: Examine the code structure and identify potential security issues\n"
                "STEP 2: Check for common vulnerability patterns (buffer overflows, format strings, etc.)\n"
                "STEP 3: Consider static/heuristic results as additional context\n"
                "STEP 4: Assess the severity and likelihood of exploitation\n"
                "STEP 5: Make your independent decision\n\n"
                "**CRITICAL: Be independent!** Don't just agree with static analysis. Think for yourself!\n\n"
                "**Decision Criteria:**\n"
                "• Use 'VULNERABLE' if you identify a security vulnerability (even if tools missed it)\n"
                "• Use 'SAFE' if the code appears secure (even if tools flagged it)\n"
                "• Use 'NEED MORE CONTEXT' if you need more information to decide\n\n"
                "**Remember**: Static tools can miss subtle vulnerabilities. Be thorough!\n\n"
                "Respond STRICTLY in this JSON format:\n"
                "{\n  \"verdict\": \"VULNERABLE\"|\"SAFE\"|\"NEED MORE CONTEXT\",\n  \"cwe\": \"CWE-XXX\",\n  \"confidence\": 0.0-1.0,\n  \"explanation\": \"Brief explanation of your findings\",\n  \"vulnerability_type\": \"Specific type if vulnerable (e.g., buffer_overflow, format_string)\",\n  \"affected_lines\": \"Line numbers or code sections of concern\",\n  \"reasoning\": \"Your independent analysis reasoning\"\n}"
            )
        ])
        return "\n".join(prompt)

    # ------------------------------------------------------------------
    # Public tasks ------------------------------------------------------
    # ------------------------------------------------------------------
    def detect(
        self,
        code: str,
        *,
        top_k: int = 5,
        llm_model: str = "qwen2.5-coder:latest",
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Detection : static → (optional) RAG → Qwen."""
        
        # Structure de base unifiée pour tous les paths
        def create_base_result():
            return {
                "decision": None,
                "is_vulnerable": None,
                "confidence": None,
                "cwe": None,
                "explanation": None,
                "votes": {},
                "static": {},
                "static_summary": {},  # Pour compatibilité Streamlit
                "heuristic": {},
                "llm_raw": None,
                "prompt": None,
                "enriched_docs": [],
                "patch": None,  # Initialisé à None
                "timings_s": {}  # Structure unique
            }
        
        # Helper function to update progress
        def update_progress(message, progress=None):
            if progress_callback:
                progress_callback(message, progress)
            logger.info(message)
        
        # 0-A. Static analysis rapide
        overall_start = time.time()
        update_progress("🔍 Starting static analysis with Cppcheck, Clang-Tidy, and Flawfinder...", 0.05)
        static_res = self.static_gate.analyze(code)
        static_time = (time.time() - overall_start) * 1000  # milliseconds
        static_time_s = static_time / 1000.0  # seconds
        update_progress(f"✅ Static analysis completed in {static_time_s:.2f}s", 0.15)

        # 0-B. Quick heuristic gate (runs regardless of static verdict)
        update_progress("🧠 Running heuristic analysis with Semgrep patterns...", 0.2)
        heuristic_res = self.heuristic_gate.analyse(code)
        logger.info("[1/5] Static+Heuristic done (static=%s, risk=%.2f) (%.2f s)", 
                   static_res["security_assessment"], heuristic_res.risk_score, static_time_s)
        update_progress(f"✅ Heuristic analysis completed (risk score: {heuristic_res.risk_score:.2f})", 0.25)

        # ---------------- Quick LLM arbitration ---------------------------
        # Build votes dictionary from results
        votes = {
            "static": "VULN" if static_res["security_assessment"] == "POTENTIALLY_VULNERABLE" else "SAFE",
            "heuristic": "VULN" if heuristic_res.security_assessment == "UNCERTAIN_RISK" and heuristic_res.risk_score > 0.5 else "SAFE"
        }
        
        update_progress(f"🗳️ Voting results - Static: {votes['static']}, Heuristic: {votes['heuristic']}", 0.3)
        
        # VALUE-ADDED MODE: LLM for complex cases where it adds real value
        static_issues = len(static_res.get("cppcheck_issues", [])) + len(static_res.get("clang_tidy_issues", [])) + len(static_res.get("flawfinder_issues", []))
        heuristic_risk = heuristic_res.risk_score
        code_lines = len(code.split('\n'))
        
        # Fast path: Clear agreement between tools (no LLM needed)
        # EVALUATION MODE: Very strict to catch subtle vulnerabilities
        if votes["static"] == "SAFE" and votes["heuristic"] == "SAFE" and heuristic_risk < 0.01 and code_lines < 10:
            decision = "SAFE"
            update_progress("✅ Both analyses agree: code appears safe (fast path)", 0.35)
        # Fast path: Clear vulnerability detected (no LLM needed)
        elif votes["static"] == "VULN" and votes["heuristic"] == "VULN" and heuristic_risk > 0.7:
            decision = "VULNERABLE"
            update_progress("⚠️ Both analyses agree: vulnerability detected (fast path)", 0.35)
        # Fast path: Simple code with low risk (no LLM needed)
        elif code_lines < 20 and static_issues == 0 and heuristic_risk < 0.2:
            decision = "SAFE"
            update_progress("✅ Simple code with low risk: safe (fast path)", 0.35)
        # Fast path: Obvious vulnerability patterns (no LLM needed)
        elif static_issues > 3 and heuristic_risk > 0.8:
            decision = "VULNERABLE"
            update_progress("⚠️ Multiple high-risk indicators: vulnerable (fast path)", 0.35)
        # LLM for complex cases where it adds value
        # More sensitive to potential subtle vulnerabilities
        # Force LLM for dangerous patterns even with low risk
        elif (static_issues > 1 and heuristic_risk > 0.3) or (code_lines > 30) or (votes["static"] != votes["heuristic"]) or any(pattern in code.lower() for pattern in ["strcpy", "strlen", "strcat", "sprintf", "gets"]):
            decision = "ACTIVATE_LLM_ARBITRATION"
            update_progress("🤖 Activating LLM for complex case analysis", 0.35)
        else:
            # Default to static+heuristic for edge cases
            decision = "SAFE" if votes["static"] == "SAFE" else "VULNERABLE"
            update_progress("⚡ Using static+heuristic for edge case", 0.35)

        if decision == "SAFE":
            # Calculate confidence for static+heuristic agreement
            static_confidence = 0.6 + (static_issues * 0.05)  # More issues = higher confidence in SAFE
            heuristic_confidence = 1.0 - heuristic_risk  # Low risk = high confidence in SAFE
            final_confidence = (static_confidence * 0.6) + (heuristic_confidence * 0.4)
            
            result = create_base_result()
            result.update({
                "decision": decision,
                "is_vulnerable": decision == "VULNERABLE",
                "confidence": final_confidence,
                "decision_analysis": "STATIC_HEURISTIC_AGREEMENT",
                "votes": votes,
                "static": static_res,
                "static_summary": static_res,  # Pour Streamlit
                "heuristic": heuristic_res.to_dict() if hasattr(heuristic_res, "to_dict") else {},
                "timings_s": {
                    "static": static_time_s,
                    "total": static_time_s,
                },
            })
            update_progress(f"🎯 Analysis completed: {decision} (confidence: {final_confidence:.2f})", 1.0)
            return result

        elif decision == "VULNERABLE":
            # Calculate confidence for static+heuristic agreement
            static_confidence = min(0.3 + (static_issues * 0.1), 0.8) if static_res["security_assessment"] == "POTENTIALLY_VULNERABLE" else 0.6
            heuristic_confidence = heuristic_risk
            final_confidence = (static_confidence * 0.6) + (heuristic_confidence * 0.4)
            
            result = create_base_result()
            result.update({
                "decision": decision,
                "is_vulnerable": decision == "VULNERABLE",
                "confidence": final_confidence,
                "decision_analysis": "STATIC_HEURISTIC_AGREEMENT",
                "votes": votes,
                "static": static_res,
                "static_summary": static_res,  # Pour Streamlit
                "heuristic": heuristic_res.to_dict() if hasattr(heuristic_res, "to_dict") else {},
                "timings_s": {
                    "static": static_time_s,
                    "total": static_time_s,
                },
            })
            update_progress(f"🎯 Analysis completed: {decision} (confidence: {final_confidence:.2f})", 1.0)
            return result

        # --- LLM ARBITRATION ONLY WHEN NECESSARY ---
        update_progress("🤖 Running LLM arbitration for robust decision...", 0.4)
        static_prompt = self._build_static_llm_prompt(code, static_res, heuristic_res)
        llm_start = time.time()
        update_progress("🤖 Querying LLM for arbitration...", 0.45)
        llm_response = ollama_generate(static_prompt, model=llm_model)
        llm_time = (time.time() - llm_start)
        import json, re
        match = re.search(r"\{[\s\S]*\}", llm_response)
        parsed = {"raw": llm_response}
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                pass
        
        # Normalisation du verdict avec critères objectifs
        if "verdict" in parsed:
            verdict = parsed["verdict"].upper()
        else:
            # Fallback intelligent basé sur les signaux
            static_issues = len(static_res.get("cppcheck_issues", [])) + len(static_res.get("clang_tidy_issues", [])) + len(static_res.get("flawfinder_issues", []))
            heuristic_risk = heuristic_res.risk_score
            
            # Critères objectifs pour NEED MORE CONTEXT
            needs_more_context = (
                (static_issues > 2 and heuristic_risk > 0.3) or  # Conflit d'outils
                (heuristic_risk > 0.7 and static_res["security_assessment"] == "LIKELY_SAFE") or  # Heuristic très inquiet, static rassuré
                (static_issues > 5) or  # Beaucoup d'issues statiques
                (len(code.split('\n')) > 50)  # Code très long
            )
            
            verdict = "NEED MORE CONTEXT" if needs_more_context else "SAFE"
        
        # Ensure confidence is always present
        if "confidence" not in parsed or parsed["confidence"] is None:
            parsed["confidence"] = 0.7  # Default confidence
        
        if verdict in ["VULNERABLE", "SAFE"]:
            # Calculate confidence based on actual tool results
            llm_confidence = parsed.get("confidence", 0.7)  # From LLM response
            
            # Static confidence: based on number and severity of issues
            static_issues = (
                len(static_res.get("cppcheck_issues", [])) +
                len(static_res.get("clang_tidy_issues", [])) +
                len(static_res.get("flawfinder_issues", []))
            )
            static_confidence = min(0.3 + (static_issues * 0.1), 0.8) if static_res["security_assessment"] == "POTENTIALLY_VULNERABLE" else 0.6
            
            # Heuristic confidence: use actual Semgrep risk score
            heuristic_confidence = heuristic_res.risk_score
            
            # Weighted confidence: LLM (50%) + Static (30%) + Heuristic (20%)
            weighted_confidence = (llm_confidence * 0.5) + (static_confidence * 0.3) + (heuristic_confidence * 0.2)
            
            # Boost confidence if tools agree with LLM verdict
            agreement_boost = 0.0
            if votes["static"] == votes["heuristic"]:
                if (verdict == "VULNERABLE" and votes["static"] == "VULN") or (verdict == "SAFE" and votes["static"] == "SAFE"):
                    agreement_boost = 0.1
            
            final_confidence = min(weighted_confidence + agreement_boost, 0.95)
            
            result = create_base_result()
            result.update({
                "decision": verdict,
                "is_vulnerable": verdict == "VULNERABLE",
                "confidence": final_confidence,
                "cwe": parsed.get("cwe"),
                "explanation": parsed.get("explanation"),
                "decision_analysis": "LLM_ARBITRATION",
                "static": static_res,
                "static_summary": static_res,
                "llm_raw": llm_response,
                "prompt": static_prompt,
                "votes": votes,
                "heuristic": heuristic_res.to_dict() if hasattr(heuristic_res, "to_dict") else {},
                "timings_s": {
                    "static": static_time_s,
                    "llm_arbitration": llm_time,
                    "total": static_time_s + llm_time,
                },
            })
            update_progress(f"🎯 LLM arbitration completed: {verdict} (confidence: {final_confidence:.2f})", 1.0)
            return result
        
        # If LLM arbitration is inconclusive, escalate to full RAG
        logger.info(f"🔍 NEED MORE CONTEXT triggered - Static issues: {len(static_res.get('cppcheck_issues', [])) + len(static_res.get('clang_tidy_issues', [])) + len(static_res.get('flawfinder_issues', []))}, Heuristic risk: {heuristic_res.risk_score:.2f}, Code lines: {len(code.split(chr(10)))}")
        update_progress("🔍 LLM arbitration inconclusive, activating full RAG pipeline...", 0.5)

        # 2. Full RAG escalated
        pre_start = time.time()
        update_progress("🔧 Preprocessing code with LLM and CPG extraction...", 0.55)
        preproc = self.preprocess(code)
        pre_time = (time.time() - pre_start) * 1000
        logger.info("[2/5] Preprocessing finished (purpose='%s', function='%s' ; %.2f s)", 
                   preproc.purpose, preproc.function, pre_time/1000.0)
        update_progress(f"✅ Preprocessing completed (purpose: {preproc.purpose})", 0.6)
        
        ret_start = time.time()
        update_progress("🔍 Searching knowledge bases for similar vulnerabilities...", 0.65)
        candidates = self.search(preproc)
        ret_time = (time.time() - ret_start) * 1000
        logger.info("[3/5] Retrieval finished: %d candidates (%.2f s)", len(candidates), ret_time/1000.0)
        update_progress(f"✅ Found {len(candidates)} similar vulnerability candidates", 0.7)
        
        asm_start = time.time()
        update_progress("📚 Assembling and enriching documents...", 0.75)
        docs = self.assemble(candidates, top_k=min(3, top_k))
        asm_time = (time.time() - asm_start) * 1000
        logger.info("[4/5] Document assembly finished: %d docs (%.2f s)", len(docs), asm_time/1000.0)
        update_progress(f"✅ Assembled {len(docs)} enriched documents", 0.8)
        
        import json
        static_summary = json.dumps(static_res, indent=2)
        prompt_body = ContextBuilder.build_detection_context(code, docs)
        # Add supported CWEs list to RAG prompt
        prompt = (
            f"### Static Analysis Findings\n{static_summary}\n\n" +
            prompt_body +
            f"\n\n## Supported CWEs: {', '.join(self.SUPPORTED_CWES)}"
        )
        logger.info("[4/5] Detection prompt built")
        update_progress("📝 Building comprehensive detection prompt...", 0.85)
        
        llm_start = time.time()
        update_progress("🤖 Running LLM analysis with enriched context...", 0.9)
        llm_response = ollama_generate(prompt, model=llm_model)  # noqa: E501
        llm_time = (time.time() - llm_start) * 1000
        logger.info("[5/5] LLM detection completed (%.2f s)", llm_time/1000.0)
        update_progress("✅ LLM analysis completed", 1.0)

        # Parsing JSON best-effort
        import json, re

        match = re.search(r"\{[\s\S]*\}", llm_response)
        parsed: Dict[str, Any]
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = {"raw": llm_response}
        else:
            parsed = {"raw": llm_response}
        
        # Normalisation des champs de réponse
        # Le LLM peut retourner soit "verdict" soit "is_vulnerable"
        if "is_vulnerable" in parsed:
            is_vulnerable = parsed["is_vulnerable"]
            verdict = "VULNERABLE" if is_vulnerable else "SAFE"
        elif "verdict" in parsed:
            verdict = parsed["verdict"].upper()
            is_vulnerable = verdict == "VULNERABLE"
        else:
            # Fallback: analyze raw text
            response_lower = llm_response.lower()
            if "vulnerable" in response_lower or "vulnerability" in response_lower:
                verdict = "VULNERABLE"
                is_vulnerable = True
            elif "safe" in response_lower or "no vulnerability" in response_lower:
                verdict = "SAFE"
                is_vulnerable = False
            else:
                verdict = "NEED MORE CONTEXT"
                is_vulnerable = None
        
        # Ensure confidence is always present for full RAG
        if "confidence" not in parsed or parsed["confidence"] is None:
            parsed["confidence"] = 0.7  # Default confidence
        
        # CWE validation (keep CWE even if not in supported list)
        cwe = parsed.get("cwe")
        if cwe and cwe not in self.SUPPORTED_CWES:
            # Keep the CWE but mark as unsupported
            logger.info(f"Unsupported CWE detected: {cwe}")
        
        # Créer résultat final avec structure unifiée
        result = create_base_result()
        result.update(parsed)
        
        # Calculate confidence for full RAG pipeline
        llm_confidence = parsed.get("confidence", 0.7)  # From LLM response
        
        # Static confidence: based on actual issues found
        static_issues = (
            len(static_res.get("cppcheck_issues", [])) +
            len(static_res.get("clang_tidy_issues", [])) +
            len(static_res.get("flawfinder_issues", []))
        )
        static_confidence = min(0.3 + (static_issues * 0.1), 0.8) if static_res["security_assessment"] == "POTENTIALLY_VULNERABLE" else 0.6
        
        # Heuristic confidence: use actual Semgrep risk score
        heuristic_confidence = heuristic_res.risk_score
        
        # Full RAG gets higher weight for LLM (70%) since it has enriched context
        weighted_confidence = (llm_confidence * 0.7) + (static_confidence * 0.2) + (heuristic_confidence * 0.1)
        
        # Boost confidence if we found relevant documents
        doc_boost = min(len(docs) * 0.05, 0.15)  # Up to 0.15 boost for 3+ docs
        
        final_confidence = min(weighted_confidence + doc_boost, 0.95)
        
        # Ensure unified fields
        result["decision"] = verdict
        result["is_vulnerable"] = is_vulnerable
        result["confidence"] = final_confidence
        result["decision_analysis"] = "FULL_RAG_PIPELINE"
        
        # Données garanties
        result["votes"] = votes
        result["static"] = static_res
        result["static_summary"] = static_res  # Pour Streamlit
        result["llm_raw"] = llm_response
        result["prompt"] = prompt
        result["enriched_docs"] = docs
        result["timings_s"] = {
            "static": static_time_s,
            "preprocessing": pre_time / 1000.0,
            "retrieval": ret_time / 1000.0,
            "assembly": asm_time / 1000.0,
            "llm": llm_time / 1000.0,
            "total": (time.time() - overall_start),
        }
        
        update_progress(f"🎯 Full analysis completed: {result['decision']} (confidence: {final_confidence:.2f})", 1.0)
        return result

    def patch(
        self,
        code: str,
        detection_result: Optional[Dict[str, Any]] = None,
        *,
        top_k: int = 5,
        llm_model: str = "qwen2.5-coder:latest",
        progress_callback=None,
    ) -> str:
        """Patch generation (Qwen) based on detection_result."""
        
        # Helper function to update progress
        def update_progress(message, progress=None):
            if progress_callback:
                progress_callback(message, progress)
            logger.info(message)
        
        if detection_result is None:
            update_progress("🔍 No detection result provided, running detection first...", 0.1)
            detection_result = self.detect(code, top_k=top_k, llm_model=llm_model, progress_callback=progress_callback)
        
        docs: List[EnrichedDocument] = detection_result.get("enriched_docs", [])
        update_progress("🔧 Starting patch generation...", 0.3)
        logger.info("[Patch] Building patch context")
        import json
        # If detection was quick-circuited (no RAG docs), use only static context
        if (not docs) and detection_result.get("static"):
            update_progress("📋 Using static analysis results for patch generation...", 0.4)
            static_summary = json.dumps(detection_result["static"], indent=2)
            prompt = (
                "# Patch Generation Based on Static Analysis Only\n"
                "## Static Analysis Findings\n"
                f"```json\n{static_summary}\n```\n"
                "## Vulnerable Code\n"
                f"```c\n{code.strip()}\n```\n"
                "## Instructions\n"
                "Generate a secure patch for the code above based solely on static analysis results. "
                "Respond **only** with the complete corrected code, no additional commentary."
            )
        else:
            update_progress(f"📚 Using {len(docs)} similar vulnerabilities for patch generation...", 0.4)
            prompt_body = ContextBuilder.build_patch_context(code, detection_result, docs)
            prompt = prompt_body
        logger.info("[Patch] Patch prompt built")
        update_progress("📝 Building patch generation prompt...", 0.5)
        
        # Generate patch
        update_progress("🤖 Generating secure patch with LLM...", 0.7)
        patch_response = ollama_generate(
            prompt=prompt,
            model=llm_model,
            temperature=0.1,
            max_tokens=1000,
        )
        update_progress("✅ Patch generated successfully!", 1.0)
        
        return patch_response


# ----------------------------------------------------------------------
# Helpers module-level --------------------------------------------------
# ----------------------------------------------------------------------
_default_pipeline: Optional[VulnRAGPipeline] = None


def _get_default_pipeline() -> VulnRAGPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = VulnRAGPipeline()
    return _default_pipeline


def detect_vulnerability(code: str, **kwargs) -> Dict[str, Any]:
    """Raccourci : `_get_default_pipeline().detect`"""
    return _get_default_pipeline().detect(code, **kwargs)


def generate_patch(code: str, **kwargs) -> str:
    """Shortcut: `_get_default_pipeline().patch`"""
    return _get_default_pipeline().patch(code, **kwargs)