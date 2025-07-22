"""Static Analysis Gate

This module orchestrates multiple static-analysis tools (cppcheck, flawfinder)
so that the pipeline can quickly decide whether to mark a C/C++ function as
POTENTIALLY_VULNERABLE or LIKELY_SAFE before invoking expensive RAG stages.

Strategy (waterfall):
1. Run cppcheck (fast, AST-based, low FP) – critical findings => vulnerable.
2. If clean, run flawfinder (pattern-based) – high-risk levels (>=4) => vulnerable.
3. If both clean, return safe.

"""

from __future__ import annotations

import logging
from typing import Dict, List

try:
    # When used as package
    from .cppcheck_wrapper import has_critical_vulnerability, scan_code as scan_cppcheck
except ImportError:  # Fallback when running as standalone script
    from rag.analysis.cppcheck_wrapper import has_critical_vulnerability, scan_code as scan_cppcheck

try:
    from .flawfinder_wrapper import scan_code as scan_flawfinder  # type: ignore
except ImportError:  # wrapper might not exist yet
    def scan_flawfinder(code: str) -> List[Dict]:  # type: ignore
        return []

try:
    from .clang_tidy_wrapper import has_critical_vulnerability as has_critical_vulnerability_clang, scan_code as scan_clang_tidy
except ImportError:
    def scan_clang_tidy(code: str):
        return []
    def has_critical_vulnerability_clang(issues):
        return False

logger = logging.getLogger(__name__)


class StaticAnalysisGate:
    """Combine multiple static analyzers with early exit on critical issues."""

    def __init__(self, enable_flawfinder: bool = True):
        self.enable_flawfinder = enable_flawfinder

    def analyze(self, code: str) -> Dict:
        """Run analyzers and return decision + collected issues.

        Returns a dict of:
            {
                "security_assessment": "LIKELY_SAFE" | "POTENTIALLY_VULNERABLE",
                "cppcheck_issues": [...],
                "clang_tidy_issues": [...],
                "flawfinder_issues": [...],
                "message": str,
            }
        """
        # 1. Cppcheck
        cppcheck_issues = scan_cppcheck(code)
        if has_critical_vulnerability(cppcheck_issues):
            return {
                "security_assessment": "POTENTIALLY_VULNERABLE",
                "cppcheck_issues": cppcheck_issues,
                "clang_tidy_issues": [],
                "flawfinder_issues": [],
                "message": "Critical vulnerabilities detected by cppcheck",
            }

        # 2. Clang-Tidy
        clang_tidy_issues = scan_clang_tidy(code)
        if has_critical_vulnerability_clang(clang_tidy_issues):
            return {
                "security_assessment": "POTENTIALLY_VULNERABLE",
                "cppcheck_issues": cppcheck_issues,
                "clang_tidy_issues": clang_tidy_issues,
                "flawfinder_issues": [],
                "message": "Critical vulnerabilities detected by clang-tidy",
            }

        # 3. Flawfinder (if enabled)
        flaw_issues: List[Dict] = []
        if self.enable_flawfinder:
            flaw_issues = scan_flawfinder(code)
            if any(int(issue.get("level", 0)) >= 4 for issue in flaw_issues):
                return {
                    "security_assessment": "POTENTIALLY_VULNERABLE",
                    "cppcheck_issues": cppcheck_issues,
                    "clang_tidy_issues": clang_tidy_issues,
                    "flawfinder_issues": flaw_issues,
                    "message": "High-risk findings detected by flawfinder",
                }

        # Safe path
        return {
            "security_assessment": "LIKELY_SAFE",
            "cppcheck_issues": cppcheck_issues,
            "clang_tidy_issues": clang_tidy_issues,
            "flawfinder_issues": flaw_issues,
            "message": "No high-severity findings from static analyzers",
        }


