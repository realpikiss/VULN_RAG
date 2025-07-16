"""rag.core.pipeline package

This package exposes a high-level API :

    from rag.core.pipeline import detect_vulnerability, generate_patch

The internal logic relies on :
    • StaticAnalysisGate    – quick exit if clear vulnerability
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

    def __init__(
        self,
        *,
        llm_interface=None,
        cpg_generator=None,
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
        self.preprocessor: PreprocessingPipeline = create_pipeline()

        # 2. Retrieval controller RRF
        self.retrieval: Vuln_RAGRetrievalController = Vuln_RAGRetrievalController(
            kb1_index_path or Path("data/KBs/kb1_index"),
            kb2_index_path or Path("data/KBs/kb2_index/kb2_code.index"),
            kb2_metadata_path or Path("data/KBs/kb2_index/kb2_metadata.json"),
            kb3_index_path or Path("data/KBs/kb3_index/kb3_code.index"),
            kb3_metadata_path or Path("data/KBs/kb3_index/kb3_metadata.json"),
            rrf_k=rrf_k,
            rrf_weights=rrf_weights,
        )

        # 3. Document assembler
        self.assembler = DocumentAssembler(kb1_index_path)

    # ------------------------------------------------------------------
    # Building blocks ---------------------------------------------------
    # ------------------------------------------------------------------
    def preprocess(self, code: str):
        return self.preprocessor.process(code)

    def search(self, preproc_result) -> List[FusionCandidate]:
        return self.retrieval.search_from_preprocessed_query(preproc_result.to_query_dict())

    def assemble(self, candidates: List[FusionCandidate], top_k: int = 5) -> List[EnrichedDocument]:
        return self.assembler.assemble_documents(candidates, top_k=top_k)

    # ------------------------------------------------------------------
    # Public tasks ------------------------------------------------------
    # ------------------------------------------------------------------
    def detect(
        self,
        code: str,
        *,
        top_k: int = 5,
        llm_model: str = "ovftank/unisast:latest",
    ) -> Dict[str, Any]:
        """Detection : static → (optional) RAG → Qwen."""       
        # 0-A. Static analysis rapide
        overall_start = time.time()
        logger.info("[1/5] Static analysis started")
        static_res = self.static_gate.analyze(code)
        logger.info("[1/5] Static analysis done: %s", static_res["security_assessment"])
        static_time = (time.time() - overall_start) * 1000
        if static_res["security_assessment"] == "POTENTIALLY_VULNERABLE":
            static_res["timing_ms"] = static_time
            logger.info("Static analysis flagged code as vulnerable – short-circuiting RAG and returning early result")
            return {
                "is_vulnerable": True,
                "confidence": 0.95,
                "cwe": "CWE-Unknown",
                "explanation": static_res["message"],
                "static_issues": static_res,
                "enriched_docs": [],
            }

        # 1. Voie RAG si pas d’alerte critique
        pre_start = time.time()
        logger.info("[2/5] Preprocessing started")
        preproc = self.preprocess(code)
        pre_time = (time.time() - pre_start) * 1000
        logger.info("[2/5] Preprocessing finished (purpose='%s', function='%s' ; %.1f ms)", preproc.purpose, preproc.function, pre_time)
        ret_start = time.time()
        logger.info("[3/5] Retrieval started")
        candidates = self.search(preproc)
        ret_time = (time.time() - ret_start) * 1000
        logger.info("[3/5] Retrieval finished: %d candidates (%.1f ms)", len(candidates), ret_time)
        asm_start = time.time()
        logger.info("[4/5] Document assembly started")
        docs = self.assemble(candidates, top_k=top_k)
        asm_time = (time.time() - asm_start) * 1000
        logger.info("[4/5] Document assembly finished: %d docs (%.1f ms)", len(docs), asm_time)
        import json
        static_summary = json.dumps(static_res, indent=2)
        prompt_body = ContextBuilder.build_detection_context(code, docs)
        prompt = f"### Static Analysis Findings\n{static_summary}\n\n" + prompt_body
        logger.debug("[4/5] Detection prompt built:\n%s", prompt)
        logger.debug("[4/5] Detection prompt built:\n%s", prompt)
        llm_start = time.time()
        logger.info("[5/5] LLM detection started (model=%s)", llm_model)
        llm_response = ollama_generate(prompt, model=llm_model)
        llm_time = (time.time() - llm_start) * 1000
        logger.info("[5/5] LLM detection completed (%.1f ms)", llm_time)

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
        parsed["enriched_docs"] = docs
        parsed["timings_ms"] = {
            "static": static_time,
            "preprocessing": pre_time,
            "retrieval": ret_time,
            "assembly": asm_time,
            "llm": llm_time,
            "total": (time.time() - overall_start) * 1000,
        }
        return parsed

    def patch(
        self,
        code: str,
        detection_result: Optional[Dict[str, Any]] = None,
        *,
        top_k: int = 5,
        llm_model: str = "ovftank/unisast:latest",
    ) -> str:
        """Génération du patch (Qwen) en se basant sur detection_result."""
        if detection_result is None:
            detection_result = self.detect(code, top_k=top_k, llm_model=llm_model)
        docs: List[EnrichedDocument] = detection_result.pop("enriched_docs", [])
        logger.info("[Patch] Building patch context")
        import json
        prompt_body = ContextBuilder.build_patch_context(code, detection_result, docs)
        if detection_result.get("static_issues") and not docs:
            static_summary = json.dumps(detection_result["static_issues"], indent=2)
            prompt = f"### Static Analysis Findings\n{static_summary}\n\n" + prompt_body
        else:
            prompt = prompt_body
        logger.debug("[Patch] Patch prompt built:\n%s", prompt)
        logger.info("[Patch] Calling LLM for patch generation (model=%s)", llm_model)
        return ollama_generate(prompt, model=llm_model)


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
    """Raccourci : `_get_default_pipeline().patch`"""
    return _get_default_pipeline().patch(code, **kwargs)