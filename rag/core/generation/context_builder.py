"""Context Builder for Vuln_RAG 

Generates structured prompts (detection / patch) from the original code and a
list of `EnrichedDocument` objects returned by the `retrieval.DocumentAssembler`.
"""
from __future__ import annotations

import logging
from typing import List, Dict

from ..retrieval.document_assembler import EnrichedDocument

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build structured LLM contexts for detection and patch generation."""

    @staticmethod
    def build_detection_context(
        original_code: str,
        enriched_docs: List[EnrichedDocument],
        max_context_length: int = 4000,
    ) -> str:
        """Return a Markdown prompt guiding the LLM to detect vulnerabilities."""
        context_parts: List[str] = []
        context_parts.append("# C/C++ Vulnerability Analysis")
        context_parts.append("\n## Source Code to Analyze")
        context_parts.append("```c")
        context_parts.append(original_code.strip())
        context_parts.append("```\n")

        if enriched_docs:
            context_parts.append("## Detected Similar Examples")
            for i, doc in enumerate(enriched_docs[:3], 1):
                context_parts.append(
                    f"\n### Example {i} (Score: {doc.final_score:.3f}, CWE: {doc.cwe})"
                )
                if doc.gpt_purpose:
                    context_parts.append(f"**Purpose**: {doc.gpt_purpose}")
                if doc.gpt_function:
                    context_parts.append(f"**Function**: {doc.gpt_function}")
                if doc.dangerous_functions:
                    funcs = ", ".join(doc.dangerous_functions[:5])
                    context_parts.append(
                        f"**Dangerous functions ({doc.dangerous_functions_count})**: {funcs}"
                    )
                if doc.risk_class != "unknown":
                    context_parts.append(f"**Risk class**: {doc.risk_class}")
                if doc.embedding_text:
                    txt = doc.embedding_text[:200] + ("..." if len(doc.embedding_text) > 200 else "")
                    context_parts.append(f"**Context summary**: {txt}")
                if doc.code_before_change:
                    context_parts.append("**Similar vulnerable code:**")
                    code_prev = doc.code_before_change[:300] + (
                        "..." if len(doc.code_before_change) > 300 else ""
                    )
                    context_parts.append("```c")
                    context_parts.append(code_prev)
                    context_parts.append("```")
                if doc.cpg_vulnerability_pattern:
                    pat = doc.cpg_vulnerability_pattern[:150] + (
                        "..." if len(doc.cpg_vulnerability_pattern) > 150 else ""
                    )
                    context_parts.append(f"**Vulnerability pattern**: {pat}")
        else:
            context_parts.append("## No Similar Examples Found")
            context_parts.append("Analyze the code autonomously.")

        context_parts.append("\n## Step-by-Step Analysis Instructions")
        context_parts.append(
            "You are a cybersecurity expert analyzing C code for vulnerabilities. Follow these steps:"
        )
        context_parts.append(
            "STEP 1: Examine the code structure and identify potential security issues"
        )
        context_parts.append(
            "STEP 2: Check for common vulnerability patterns (buffer overflows, format strings, etc.)"
        )
        context_parts.append(
            "STEP 3: Consider the similar examples as additional context"
        )
        context_parts.append(
            "STEP 4: Assess the severity and likelihood of exploitation"
        )
        context_parts.append(
            "STEP 5: Provide a structured analysis"
        )
        context_parts.append(
            "Base your reasoning on the similar examples if available. Respond using *only* the following JSON format:"
        )
        context_parts.append("```json")
        context_parts.append("{\n  \"verdict\": \"VULNERABLE\"|\"SAFE\"|\"NEED MORE CONTEXT\",\n  \"confidence\": 0.0-1.0,\n  \"cwe\": \"CWE-XXX\",\n  \"explanation\": \"Brief explanation of your findings\",\n  \"vulnerability_type\": \"Specific type if vulnerable (e.g., buffer_overflow, format_string)\",\n  \"affected_lines\": \"Line numbers or code sections of concern\"\n}")
        context_parts.append("```")
        context_parts.append("\n**IMPORTANT**: Respond with ONLY the JSON object, no additional text or commentary.")
        context_parts.append("\n**CRITICAL**: Always include a confidence score between 0.0 and 1.0 based on your certainty.")

        return "\n".join(context_parts)

    @staticmethod
    def build_patch_context(
        original_code: str,
        detection_result: Dict[str, str | bool | float],
        enriched_docs: List[EnrichedDocument],
        max_context_length: int = 4000,
    ) -> str:
        """Return a Markdown prompt guiding the LLM to generate a secure patch."""
        context_parts: List[str] = []
        context_parts.append("# C/C++ Patch Generation Task")
        context_parts.append("\n## Vulnerability Report")
        context_parts.append("```json")
        import json as _json
        context_parts.append(_json.dumps(detection_result, indent=2))
        context_parts.append("```\n")

        context_parts.append("## Vulnerable Code")
        context_parts.append("```c")
        context_parts.append(original_code.strip())
        context_parts.append("```\n")

        has_examples = any(doc.code_after_change for doc in enriched_docs)
        if has_examples:
            context_parts.append("## Patch Examples from Similar Vulnerabilities")
            for i, doc in enumerate(enriched_docs[:3], 1):
                if not doc.code_after_change:
                    continue
                context_parts.append(f"\n### Example {i} (CWE: {doc.cwe}) - Before")
                context_parts.append("```c")
                context_parts.append(doc.code_before_change[:300])
                context_parts.append("```")
                context_parts.append("\n### After")
                context_parts.append("```c")
                context_parts.append(doc.code_after_change[:300])
                context_parts.append("```")
        else:
            context_parts.append("## No Patch Examples Available")

        context_parts.append("\n## Instructions")
        context_parts.append("Generate a secure patch for the vulnerable code above.")
        context_parts.append("Respond **only** with the complete corrected code, no additional commentary.")

        return "\n".join(context_parts)
