"""Ollama wrapper for running chat locally."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import logging

import ollama

logger = logging.getLogger(__name__)

__all__ = ["generate", "to_messages"]


SYSTEM_PROMPT = "You are a C/C++ security expert able to detect vulnerabilities and craft patches."


def to_messages(context: str) -> List[Dict[str, str]]:
    """Wrap the given context into chat style message list."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]


def generate(
    prompt: str,
    model: str = "kirito1/qwen3-coder:latest",
    temperature: float = 0.1,
    max_tokens: int = 500,
    **options,
) -> str:
    """Generate text using Ollama Qwen model."""
    
    # Prepare Ollama options
    ollama_opts = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "top_p": 0.9,
        "top_k": 40,
    }
    ollama_opts.update(options)
    
    # Call Ollama
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options=ollama_opts,
        )
        content = response.get("response", "").strip()
        return content
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        return ""
