"""
Query Builder
Constructs textual and structural queries for retrieval
"""

from typing import Dict, List, Tuple, Optional

from rag.data.preprocessing.function_processor import FunctionProcessor

_TEXT_SEP = " "


class QueryBuilder:
    """Construct textual and structural queries consumed by the hybrid retriever."""

    def __init__(self) -> None:
        self._processor = FunctionProcessor()

    # ------------------------------------------------------------------
    # Textual query (KB-1)
    # ------------------------------------------------------------------
    def build_textual_query(self, function_code: str) -> str:
        """Return a space-separated query string for KB-1.

        We include:
        • function call names
        • string literals (potentially vulnerable) – quick regex
        • dangerous keywords (e.g., strcpy, memcpy…)
        """
        calls = self._processor.extract_function_calls(function_code)

        # crude literal extraction for additional context
        import re

        literals: List[str] = re.findall(r"\"([^\"]{1,30})\"", function_code)
        tokens = list({*calls, *literals})
        if not tokens:
            tokens = ["c function"]
        return _TEXT_SEP.join(tokens)

    # ------------------------------------------------------------------
    # Structural query (KB-2)
    # ------------------------------------------------------------------
    def build_structural_query(self, cpg_features: Dict) -> Dict:
        """Build dict consumed by `KB2StructuralSearcher`.

        Returns a flattened subset plus some heuristic fields for feature similarity.
        Adapted to work with real KB2 feature structure.
        """
        # Extract features from the real KB2 structure as seen in kb2_loader.py
        security_features = cpg_features.get("security_features", {})
        complexity_metrics = cpg_features.get("complexity_metrics", {})
        code_patterns = cpg_features.get("code_patterns", {})
        function_detection = cpg_features.get("function_detection", {})
        spatial_mapping = cpg_features.get("spatial_mapping", {})
        
        # Build the flattened query structure expected by KB2StructuralSearcher
        flattened = {
            # Signatures from code patterns
            "call_signature": code_patterns.get("call_signature", []),
            "structure_signature": code_patterns.get("structure_signature", []),
            
            # Complexity metrics
            "complexity": complexity_metrics,
            "complexity_score": complexity_metrics.get("mccabe_complexity_approx", 0),
            "control_structure_count": complexity_metrics.get("control_structure_count", 0),
            "nesting_indicator": complexity_metrics.get("nesting_indicator", 0),
            
            # Security features
            "monitored_calls_detected": security_features.get("monitored_calls_detected", 0),
            "dangerous_calls": security_features.get("dangerous_calls", []),
            "total_function_calls": security_features.get("total_function_calls", 0),
            "inherently_dangerous_count": security_features.get("inherently_dangerous_count", 0),
            
            # Code patterns
            "vertex_type_distribution": code_patterns.get("vertex_type_distribution", {}),
            "edge_type_distribution": code_patterns.get("edge_type_distribution", {}),
            "all_calls": code_patterns.get("all_calls", {}),
            "graph_density": code_patterns.get("graph_density", 0),
            
            # Function detection
            "function_detection": function_detection,
            
            # Spatial mapping
            "sloc_confirmed": spatial_mapping.get("sloc_confirmed", 0),
            "line_range": spatial_mapping.get("line_range", []),
        }
        return flattened

    # ------------------------------------------------------------------
    # Hybrid
    # ------------------------------------------------------------------
    def create_hybrid_query(
        self, function_code: str, *, preprocessed: Optional[Dict] = None
    ) -> Dict:
        """Build the 5-dimension query dict expected by VulRAG hybrid retriever.

        Returns a dictionary with keys:
            code      : raw function code (or token-based textual query)
            purpose   : high-level natural-language purpose extracted by VulRAGPreprocessor
            behavior  : behaviour description extracted by VulRAGPreprocessor
            embedding : vector embedding produced by CPG processing
            features  : structural feature dict produced by CPG processing
        """
        if preprocessed is None:
            # Lazy import to avoid heavy deps if unused
            from rag.data.preprocessing.vulrag_preprocessor import VulRAGPreprocessor
            preproc = VulRAGPreprocessor()
            proc_res = preproc.process_input_function(function_code)
        else:
            proc_res = preprocessed

        # Ensure embedding / features present (may require CPG)
        if "embedding" not in proc_res or "features" not in proc_res:
            from rag.data.preprocessing.function_processor import FunctionProcessor
            cpg_proc = FunctionProcessor()
            cpg_data = cpg_proc.process_input_function(function_code)
            proc_res.setdefault("embedding", cpg_data.get("embedding"))
            proc_res.setdefault("features", cpg_data.get("features", {}))

    
        return {
            "code": proc_res["raw_code"],
            "purpose": proc_res["purpose"], 
            "behavior": proc_res["behavior"],
            "embedding": proc_res["embedding"],
            "features": proc_res["features"]
    }
