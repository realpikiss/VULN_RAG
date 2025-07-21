"""Clang-Tidy wrapper"""

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def scan_code(
    code: str,
    checks: Optional[str] = None,
    timeout: int = 30,
) -> List[Dict]:
    """Run clang-tidy on C/C++ code and parse output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_path = Path(tmp_dir) / "code.c"
        src_path.write_text(code, encoding="utf-8")

        cmd = [
            "clang-tidy",
            str(src_path),
            "--",
        ]
        if checks:
            cmd.insert(1, f"-checks={checks}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp_dir,
            )
            return _parse_clang_tidy_text(result.stdout)
        except subprocess.TimeoutExpired:
            logger.warning("Clang-Tidy analysis timed out")
            return []
        except Exception as e:
            logger.error(f"Clang-Tidy failed: {e}")
            return []

def _parse_clang_tidy_text(output: str) -> List[Dict]:
    """Parse clang-tidy text output."""
    issues = []
    # Example: code.c:3:5: warning: ... [check-name]
    pattern = r'([^:]+):(\d+):(\d+): (warning|error|note): ([^[]+)(?: \[([^\]]+)\])?'
    for line in output.splitlines():
        match = re.search(pattern, line)
        if match:
            file_name, line_num, col_num, severity, message, check_name = match.groups()
            issues.append({
                "severity": severity.lower(),
                "msg": message.strip(),
                "line": int(line_num),
                "col": int(col_num),
                "id": check_name.strip() if check_name else None,
                "tool": "clang-tidy",
                "file": file_name
            })
    return issues

def has_critical_vulnerability(issues: List[Dict]) -> bool:
    """Check for critical vulnerabilities (bugprone, security, etc)."""
    critical_prefixes = ["security", "bugprone", "cert", "cppcoreguidelines"]
    critical_severities = ["error", "warning"]
    for issue in issues:
        check_id = issue.get("id") or ""
        if any(check_id.startswith(prefix) for prefix in critical_prefixes):
            return True
        if issue.get("severity") in critical_severities and check_id:
            return True
    return False 