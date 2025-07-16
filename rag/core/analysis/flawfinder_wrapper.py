"""Flawfinder wrapper - Version CORRIGÉE basée sur la vraie sortie."""

import subprocess
import tempfile
import re
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

def scan_code(code: str, timeout: int = 10) -> List[Dict]:
    """Run flawfinder on C/C++ code."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_path = Path(tmp_dir) / "code.c"
        enhanced_code = _add_headers(code)
        src_path.write_text(enhanced_code, encoding="utf-8")
        
        try:
            # Flawfinder with --dataonly for simple format
            result = subprocess.run([
                "flawfinder", 
                "--dataonly", 
                "--minlevel=1", 
                str(src_path)
            ], capture_output=True, text=True, timeout=timeout)
            
            return _parse_flawfinder_output(result.stdout)
            
        except FileNotFoundError:
            logger.warning("Flawfinder is not installed or not found in PATH.")
            return []
        except Exception as e:
            logger.error(f"Flawfinder error: {e}")
            return []

def _add_headers(code: str) -> str:
    """Add headers and main function."""
    if "#include" not in code:
        return f"""#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {{
    {code}
    return 0;
}}"""
    return code

def _parse_flawfinder_output(output: str) -> List[Dict]:
    """Parse flawfinder output avec le VRAI format."""
    issues = []
    
    # Pattern basé sur la vraie sortie : "filename:line:  [level] (category) function:"
    pattern = r'([^:]+):(\d+):\s*\[(\d+)\]\s*\([^)]+\)\s*([^:]+):'
    
    lines = output.split('\n')
    
    for i, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            file_name, line_num, level, function = match.groups()
            
            # La description est sur les lignes suivantes
            description = ""
            j = i + 1
            while j < len(lines) and not re.search(pattern, lines[j]) and lines[j].strip():
                description += lines[j].strip() + " "
                j += 1
            
            level_int = int(level)
            severity = "error" if level_int >= 4 else "warning" if level_int >= 2 else "info"
            
            issues.append({
                "severity": severity,
                "msg": f"{function.strip()}: {description.strip()}",
                "line": int(line_num),
                "id": f"flawfinder_{function.strip()}",
                "tool": "flawfinder",
                "function": function.strip(),
                "level": level_int
            })
    
    return issues

def has_critical_vulnerability(issues: List[Dict]) -> bool:
    """Return True if any issue is high risk (level >= 4)."""
    return any(issue.get('level', 0) >= 4 for issue in issues)



