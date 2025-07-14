"""
Function Processor
Analyzes input C/C++ function and extracts features
"""

"""Utility to prepare user-supplied C functions for the KB-2 pipeline.

The processor performs three tasks:
1. Adds minimal headers / wrapper so that Joern can parse a standalone snippet.
2. Extracts a quick list of function calls via regex (lightweight, no AST).
3. Generates the CPG + structural features / embedding using `CPGGenerator`.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from rag.data.preprocessing.cpg_generator import CPGGenerator

_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")


class FunctionProcessor:
    """High-level helper to transform an input C function into structured data."""

    def __init__(self) -> None:
        self._cpg_gen = CPGGenerator()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def process_input_function(self, c_code: str) -> Dict[str, object]:
        """Return a dict with calls, features and embedding for the snippet.

        Keys:
            calls: List[str]
            features: Dict (see `extract_kb2_features`)
            embedding: np.ndarray (float32, 128-d)
        """
        wrapped = self.wrap_function_for_joern(c_code)
        cpg_path = self._cpg_gen.generate_single_cpg(wrapped)

        embedding: np.ndarray = self._cpg_gen.compute_function_embedding(cpg_path)
        features: Dict = self._cpg_gen.extract_structural_features(cpg_path)
        calls = self.extract_function_calls(c_code)

        return {
            "calls": calls,
            "features": features,
            "embedding": embedding,
        }

    def extract_function_calls(self, c_code: str) -> List[str]:
        """Rudimentary extraction of function call identifiers using regex.

        This avoids heavy dependencies; it works well enough for heuristics.
        """
        matches = _CALL_RE.findall(c_code)
        # Filter out control keywords and type casts that appear similar
        blacklist = {
            "if",
            "for",
            "while",
            "switch",
            "return",
            "sizeof",
        }
        calls = [m for m in matches if m not in blacklist]
        # Deduplicate while preserving order
        seen: set[str] = set()
        ordered: List[str] = []
        for name in calls:
            if name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered

    def wrap_function_for_joern(self, function_code: str) -> str:
        """Ensure the snippet is self-contained before sending to Joern."""
        headers = ["#include <stdio.h>", "#include <string.h>"]
        has_include = re.search(r"#include", function_code) is not None
        snippet = function_code.strip()
        if not has_include:
            snippet = "\n".join(headers) + "\n\n" + snippet

        # Ensure there is at least one translation unit (main) for Joern.
        if "main(" not in snippet:
            snippet += "\n\nint __dummy_main__() { return 0; }\n"
        return snippet

