"""
openrouter-llm -- Multi-model LLM routing via OpenRouter API.
Extracted from videoforge-engine/videoforge/assembly/script_engine.py.

Provides:
- complete(): single-shot text completion
- chat(): multi-turn chat completion
- complete_with_fallback(): try OpenRouter, fall back to Anthropic direct
- estimate_cost(): cost estimation for model/token counts
- MODELS: model presets with pricing information
- parse_json_response(): extract JSON from LLM output

Key patterns from VideoForge ScriptEngine:
- OpenRouter as primary, Anthropic Haiku as cheap fallback
- DeepSeek V3 for bulk/cheap tasks, Claude Sonnet for quality
- Always set temperature for creative vs analytical tasks
- Parse JSON from markdown code blocks in responses
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
ANTHROPIC_BASE = "https://api.anthropic.com/v1/messages"

# Model presets with pricing (per 1M tokens)
MODELS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "id": "anthropic/claude-haiku-4-5-20251001",
        "name": "Claude Haiku",
        "input_cost": 0.80,
        "output_cost": 4.00,
    },
    "default": {
        "id": "anthropic/claude-sonnet-4-20250514",
        "name": "Claude Sonnet",
        "input_cost": 3.00,
        "output_cost": 15.00,
    },
    "smart": {
        "id": "anthropic/claude-opus-4-20250514",
        "name": "Claude Opus",
        "input_cost": 15.00,
        "output_cost": 75.00,
    },
    "cheap": {
        "id": "deepseek/deepseek-chat",
        "name": "DeepSeek V3",
        "input_cost": 0.27,
        "output_cost": 1.10,
    },
    "flash": {
        "id": "google/gemini-2.0-flash-001",
        "name": "Gemini Flash",
        "input_cost": 0.10,
        "output_cost": 0.40,
    },
}


def _get_openrouter_key(api_key: Optional[str] = None) -> str:
    """Resolve OpenRouter API key."""
    return api_key or os.environ.get("OPENROUTER_API_KEY", "")


def _get_anthropic_key() -> str:
    """Resolve Anthropic API key for direct fallback."""
    return os.environ.get("ANTHROPIC_API_KEY", "")


def complete(
    prompt: str,
    system: str = "",
    model: str = "default",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> str:
    """Send a completion request to OpenRouter.

    Args:
        prompt: User message
        system: Optional system prompt
        model: Model preset key or full model ID
        max_tokens: Maximum output tokens
        temperature: Sampling temperature (0=deterministic, 1=creative)
        api_key: OpenRouter API key

    Returns:
        Text response string.

    Raises:
        ValueError: If no API key is available
        requests.HTTPError: On API error
    """
    import requests

    key = _get_openrouter_key(api_key)
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Pass api_key or set env var."
        )

    model_info = MODELS.get(model, {})
    model_id = model_info.get("id", model) if model_info else model

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


def chat(
    messages: List[Dict[str, str]],
    model: str = "default",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Multi-turn chat completion via OpenRouter.

    Args:
        messages: List of message dicts with role and content
        model: Model preset key or full model ID
        max_tokens: Maximum output tokens
        temperature: Sampling temperature

    Returns:
        Dict with content, model, usage, and cost fields.
    """
    import requests

    key = _get_openrouter_key(api_key)
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set")

    model_info = MODELS.get(model, {})
    model_id = model_info.get("id", model) if model_info else model

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

    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    return {
        "content": data["choices"][0]["message"]["content"],
        "model": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": estimate_cost(model, input_tokens, output_tokens),
    }


def complete_with_fallback(
    prompt: str,
    system: str = "",
    model: str = "cheap",
    max_tokens: int = 1500,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Try OpenRouter first, fall back to Anthropic Haiku on failure.

    This is the pattern used by VideoForge ScriptEngine -- ensures
    script generation always works even if OpenRouter is down.

    Returns:
        Dict with content, model, cost fields.
    """
    # Try OpenRouter
    or_key = _get_openrouter_key()
    if or_key:
        try:
            result = chat(
                messages=[
                    *([{"role": "system", "content": system}] if system else []),
                    {"role": "user", "content": prompt},
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result
        except Exception as e:
            log.warning("OpenRouter failed, trying Anthropic fallback: %s", e)

    # Fallback to Anthropic direct API
    anthropic_key = _get_anthropic_key()
    if anthropic_key:
        try:
            return _call_anthropic_direct(
                anthropic_key, system, prompt, max_tokens
            )
        except Exception as e:
            log.error("Anthropic fallback also failed: %s", e)

    return {"content": "", "model": "none", "cost": 0.0,
            "error": "No API keys available"}


def _call_anthropic_direct(
    api_key: str, system: str, prompt: str, max_tokens: int
) -> Dict[str, Any]:
    """Direct Anthropic API call using Haiku (cheapest)."""
    import requests

    resp = requests.post(
        ANTHROPIC_BASE,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = (input_tokens / 1_000_000 * 0.80
            + output_tokens / 1_000_000 * 4.00)

    return {
        "content": data["content"][0]["text"],
        "model": "claude-haiku-4-5-20251001",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def estimate_cost(model: str, input_tokens: int,
                  output_tokens: int) -> float:
    """Estimate cost in USD for a model and token counts."""
    model_info = MODELS.get(model, {})
    if not model_info:
        # Try matching by model ID
        for info in MODELS.values():
            if info["id"] == model:
                model_info = info
                break
    if not model_info:
        model_info = MODELS["default"]

    input_cost = model_info.get("input_cost", 3.00)
    output_cost = model_info.get("output_cost", 15.00)

    return (input_tokens / 1_000_000 * input_cost
            + output_tokens / 1_000_000 * output_cost)


def parse_json_response(text: str) -> Any:
    """Extract and parse JSON from an LLM response.

    Handles responses wrapped in markdown code blocks.
    """
    cleaned = text.strip()
    # Strip markdown code block wrapper
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("Failed to parse JSON from LLM response")
        return None
