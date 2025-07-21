"""Quick heuristic gate for fast low-cost vulnerability risk estimation.

This gate performs extremely cheap, deterministic checks on the source code to
quickly decide whether it is OBVIOUSLY safe.  It is *not* authoritative for
vulnerable verdicts – it only attempts to identify code that is almost
certainly safe so that the heavy RAG pipeline can be skipped.

Returned structure mirrors StaticAnalysisGate output but is intentionally
simpler.
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Dict, List, Union
import subprocess
import tempfile
import json

# ---------- heuristic configuration -------------------------------------------------

# Dangerous C / C++ API calls – incomplete list, pick high-risk ones only.
DANGEROUS_FUNCTIONS: List[str] = [
    "strcpy",
    "gets",
    "scanf",
    "sprintf",
    "strcat",
    "memcpy",
    "strncpy",  # context-dependent but keep
    "malloc",
    "free",
    "calloc",
    "realloc",
    "system",
    "exec",
    "popen",
]

# C++ specific dangerous patterns
CPP_DANGEROUS: List[str] = [
    "new",
    "delete",
    "std::cin",
]

# Pre-compiled regexes for speed.
DANGEROUS_PATTERNS = [re.compile(rf"\b{func}\b") for func in DANGEROUS_FUNCTIONS]
CPP_PATTERNS = [re.compile(rf"\b{func}\b") for func in CPP_DANGEROUS]

# Pointer arithmetic patterns (more specific - exclude function parameters)
POINTER_PATTERNS = [
    re.compile(r"\*\s*\(\s*\w+\s*\+\+\s*\)"),  # *(ptr++)
    re.compile(r"\*\s*\(\s*\w+\s*\-\-\s*\)"),  # *(ptr--) 
    re.compile(r"\*\s*\(\s*\w+"),  # *(ptr) - dereferencing expressions
    re.compile(r"\w+\s*(\+\+|--)"),  # ptr++, ptr-- - increment/decrement
    re.compile(r"\w+\s*\[\s*\w+\s*[\+\-]"),  # array[i+n], array[i-n] - arithmetic indexing
    re.compile(r"\*\s*\(\s*\w+\s*[\+\-]"),  # *(ptr + offset) - pointer arithmetic
]

# Buffer/array declarations (exclude function parameters)
BUFFER_PATTERNS = [
    re.compile(r"\bchar\s+\w+\s*\[\s*\d*\s*\]"),  # char buf[N] - only local arrays
    re.compile(r"=\s*malloc\s*\("),  # = malloc( - dynamic allocation
    re.compile(r"=\s*calloc\s*\("),  # = calloc( - dynamic allocation
]

# Thresholds – tweakable.
MAX_LOC_SAFE = 25  # if code shorter than this and no patterns -> safe
HIGH_RISK_THRESHOLD = 3  # number of dangerous hits for high risk


@dataclass
class HeuristicResult:
    security_assessment: str  # "LIKELY_SAFE" | "UNCERTAIN_RISK"
    risk_score: float  # 0..1
    loc: int
    dangerous_hits: Dict[str, int]
    pointer_arithmetic: bool
    buffer_usage: bool
    is_cpp: bool
    message: str

    def to_dict(self) -> Dict:
        return {
            "security_assessment": self.security_assessment,
            "risk_score": self.risk_score,
            "loc": self.loc,
            "dangerous_hits": self.dangerous_hits,
            "pointer_arithmetic": self.pointer_arithmetic,
            "buffer_usage": self.buffer_usage,
            "is_cpp": self.is_cpp,
            "message": self.message,
        }

    def explain(self) -> str:
        """Detailed explanation of the assessment."""
        if self.security_assessment == "LIKELY_SAFE":
            return f"[SAFE] LOC={self.loc}, no dangerous patterns found."
        
        details = []
        if self.dangerous_hits:
            funcs = ", ".join(self.dangerous_hits.keys())
            details.append(f"dangerous functions: {funcs}")
        if self.pointer_arithmetic:
            details.append("pointer arithmetic detected")
        if self.buffer_usage:
            details.append("buffer/array usage detected")
        
        detail_str = "; ".join(details) if details else "complex code structure"
        return f"[RISK {self.risk_score:.2f}] LOC={self.loc}, {detail_str}"


class SemgrepHeuristicGate:
    """Advanced heuristic based on Semgrep for rapid vulnerability analysis."""
    def __init__(self, semgrep_config: str = "auto"):
        self.semgrep_config = semgrep_config  # Can be a rules.yml path or "auto"

    def analyse(self, code: str) -> HeuristicResult:
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            cmd = [
                "semgrep",
                "--quiet",
                "--json",
                "--config", self.semgrep_config,
                tmp_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            findings = []
            if result.returncode == 0 and result.stdout:
                try:
                    data = json.loads(result.stdout)
                    findings = data.get("results", [])
                except Exception:
                    findings = []
            loc = code.count("\n") + 1
            if not findings and loc <= 25:
                return HeuristicResult(
                    security_assessment="LIKELY_SAFE",
                    risk_score=0.05,
                    loc=loc,
                    dangerous_hits={},
                    pointer_arithmetic=False,
                    buffer_usage=False,
                    is_cpp="std::" in code or "#include <iostream>" in code,
                    message="Semgrep: no findings and short code."
                )
            else:
                msg = f"Semgrep findings: {len(findings)}" if findings else "Complex code or potential findings."
                return HeuristicResult(
                    security_assessment="UNCERTAIN_RISK",
                    risk_score=0.7 if findings else 0.3,
                    loc=loc,
                    dangerous_hits={f.get('check_id', 'unknown'): 1 for f in findings},
                    pointer_arithmetic=False,
                    buffer_usage=False,
                    is_cpp="std::" in code or "#include <iostream>" in code,
                    message=msg
                )
        finally:
            import os
            os.unlink(tmp_path)

# Pour compatibilité pipeline
QuickHeuristicGate = SemgrepHeuristicGate