"""Ollama wrapper for running Qwen-chat locally."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import logging

import ollama

logger = logging.getLogger(__name__)

__all__ = ["generate", "to_messages"]


SYSTEM_PROMPT = "You are a C/C++ security expert able to detect vulnerabilities and craft patches."


def to_messages(context: str) -> List[Dict[str, str]]:
    """Wrap the given context into Qwen-chat style message list."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]


def generate(
    context: str,
    *,
    model: str = "qwen:7b-chat",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **options: Any,
) -> str:
    """Run the Qwen model via Ollama and return the assistant text.

    Parameters
    ----------
    context: str
        The prompt/context for the model (already structured).
    model: str, default ``"qwen:7b-chat"``
        Name of the Ollama model to use.
    temperature: float
        Sampling temperature.
    max_tokens: int | None
        Maximum tokens to generate (`num_predict` in Ollama terminology).
    options: Any
        Additional Ollama generation options (see `ollama.chat`).
    """
    messages = to_messages(context)

    # Map our high-level kwargs to Ollama options dict
    ollama_opts: Dict[str, Any] = {
        "temperature": temperature,
    }
    if max_tokens is not None:
        ollama_opts["num_predict"] = max_tokens

    # Merge caller-provided options last so they override defaults.
    ollama_opts.update(options)

    logger.debug("Sending chat completion to Ollama: model=%s, opts=%s", model, ollama_opts)

    response = ollama.chat(model=model, messages=messages, options=ollama_opts)
    content = response.get("message", {}).get("content", "")

    logger.debug("Received response (%d chars)", len(content))
    return content
