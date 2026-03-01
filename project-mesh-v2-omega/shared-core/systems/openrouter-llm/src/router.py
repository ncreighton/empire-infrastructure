"""
openrouter-llm — Multi-model LLM routing via OpenRouter.
Used by VideoForge for script generation. Supports model selection, retry, streaming.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Model presets
MODELS = {
    "fast": "anthropic/claude-haiku-4-5-20251001",
    "default": "anthropic/claude-sonnet-4-20250514",
    "smart": "anthropic/claude-opus-4-20250514",
    "cheap": "google/gemini-2.0-flash-001",
}


def complete(
    prompt: str,
    system: str = "",
    model: str = "default",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> str:
    """Send a completion request to OpenRouter. Returns text response."""
    import requests

    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set")

    model_id = MODELS.get(model, model)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough cost estimate in USD."""
    pricing = {
        "anthropic/claude-haiku-4-5-20251001": (0.80, 4.00),
        "anthropic/claude-sonnet-4-20250514": (3.00, 15.00),
        "anthropic/claude-opus-4-20250514": (15.00, 75.00),
        "google/gemini-2.0-flash-001": (0.10, 0.40),
    }
    model_id = MODELS.get(model, model)
    rates = pricing.get(model_id, (3.00, 15.00))
    return (input_tokens / 1_000_000 * rates[0]) + (output_tokens / 1_000_000 * rates[1])
