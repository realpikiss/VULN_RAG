"""Cppcheck wrapper"""

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def scan_code(
    code: str,
    includes: Optional[List[str]] = None,
    timeout: int = 30,
) -> List[Dict]:
    """Run cppcheck on C/C++ code with text parsing."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write code with basic headers
        src_path = Path(tmp_dir) / "code.c"
        enhanced_code = _add_basic_headers(code)
        src_path.write_text(enhanced_code, encoding="utf-8")

        # Simple cppcheck command (no JSON template)
        cmd = [
            "cppcheck", 
            "--enable=all",
            "--inconclusive", 
            "--error-exitcode=0",
            str(src_path)
        ]
        
        # Add includes if provided
        if includes:
            for include in includes:
                cmd.extend(["-I", include])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp_dir,
            )
            
            # Parse the text output (cppcheck outputs to stderr)
            return _parse_cppcheck_text(result.stderr)
            
        except subprocess.TimeoutExpired:
            logger.warning("Cppcheck analysis timed out")
            return []
        except Exception as e:
            logger.error(f"Cppcheck failed: {e}")
            return []

def _add_basic_headers(code: str) -> str:
    """Add essential headers if missing."""
    if "#include" not in code:
        headers = [
            "#include <stdio.h>",
            "#include <stdlib.h>", 
            "#include <string.h>"
        ]
        return '\n'.join(headers) + '\n\n' + code
    return code

def _parse_cppcheck_text(output: str) -> List[Dict]:
    """Parse cppcheck text output."""
    issues = []
    
    # Pattern based on your test: "test.c:1:36: error: message [id]"
    pattern = r'([^:]+):(\d+):(?:\d+:)?\s*(\w+):\s*([^[]+)\s*\[([^\]]+)\]'
    
    for line in output.splitlines():
        match = re.search(pattern, line)
        if match:
            file_name, line_num, severity, message, issue_id = match.groups()
            
            # Skip non-error/warning items
            if severity.lower() not in ['error', 'warning']:
                continue
                
            issues.append({
                "severity": severity.lower(),
                "msg": message.strip(),
                "line": int(line_num),
                "id": issue_id.strip(),
                "tool": "cppcheck",
                "file": file_name
            })
    
    return issues

def has_critical_vulnerability(issues: List[Dict]) -> bool:
    """Check for critical vulnerabilities."""
    critical_ids = [
        'bufferAccessOutOfBounds',
        'nullPointer', 
        'useAfterFree',
        'memleak',
        'uninitvar'
    ]
    
    critical_severities = ['error']
    
    return any(
        issue.get('id') in critical_ids or 
        issue.get('severity') in critical_severities
        for issue in issues
    )