"""ScriptEngine — AI-powered narration script generation via OpenRouter.

Uses DeepSeek V3 for cheap bulk ($0.002/script) and Claude Sonnet for quality ($0.02/script).
"""

import os
import json
import logging
import requests
from ..models import VideoScript, Storyboard

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"

# Model routing
MODELS = {
    "cheap": {
        "id": "deepseek/deepseek-chat",
        "name": "DeepSeek V3",
        "cost_per_1k_input": 0.00027,
        "cost_per_1k_output": 0.0011,
    },
    "quality": {
        "id": "anthropic/claude-sonnet-4-20250514",
        "name": "Claude Sonnet",
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
    },
}


def _get_api_key() -> str:
    """Get OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        # Try loading from config
        env_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "api_keys.env"
        )
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY="):
                        key = line.strip().split("=", 1)[1]
    return key


class ScriptEngine:
    """Generates video narration scripts via OpenRouter API."""

    def __init__(self, model_tier: str = "cheap"):
        self.model_tier = model_tier
        self.model = MODELS.get(model_tier, MODELS["cheap"])

    def generate_script(self, storyboard: Storyboard,
                        model_tier: str = None) -> VideoScript:
        """Generate a narration script from a storyboard.

        Falls back to storyboard narration if API is unavailable.
        """
        model = MODELS.get(model_tier or self.model_tier, self.model)
        api_key = _get_api_key()

        if not api_key:
            logger.info("No OpenRouter API key — using storyboard narration as script")
            return self._fallback_script(storyboard)

        prompt = self._build_prompt(storyboard)

        try:
            response = requests.post(
                OPENROUTER_BASE,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model["id"],
                    "messages": [
                        {"role": "system", "content": self._system_prompt(storyboard)},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            # Calculate cost
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cost = (input_tokens / 1000 * model["cost_per_1k_input"] +
                    output_tokens / 1000 * model["cost_per_1k_output"])

            return self._parse_script(content, storyboard, model["name"], cost)

        except Exception as e:
            logger.warning(f"Script generation failed: {e} — using fallback")
            return self._fallback_script(storyboard)

    def generate_topics(self, niche: str, count: int = 10,
                        content_type: str = "educational") -> list:
        """Generate topic ideas for a niche. Returns list of topic strings."""
        api_key = _get_api_key()
        if not api_key:
            return [f"{niche} topic idea {i+1}" for i in range(count)]

        prompt = (
            f"Generate {count} viral video topic ideas for the '{niche}' niche.\n"
            f"Content type: {content_type}\n"
            f"Format: Return ONLY a JSON array of topic strings, no other text.\n"
            f"Example: [\"topic 1\", \"topic 2\"]\n"
            f"Make them specific, hook-worthy, and optimized for short-form video."
        )

        try:
            response = requests.post(
                OPENROUTER_BASE,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODELS["cheap"]["id"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.9,
                },
                timeout=20,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            # Parse JSON array from response
            # Handle potential markdown code blocks
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(content)

        except Exception as e:
            logger.warning(f"Topic generation failed: {e}")
            return [f"{niche} topic idea {i+1}" for i in range(count)]

    def _system_prompt(self, storyboard: Storyboard) -> str:
        """Build system prompt for script generation."""
        return (
            f"You are a viral video scriptwriter for the '{storyboard.niche}' niche. "
            f"Platform: {storyboard.platform}. Format: {storyboard.format}. "
            f"Target duration: {storyboard.total_duration:.0f} seconds. "
            f"Write conversational, engaging narration that hooks in the first 2 seconds. "
            f"Each segment should be 1-3 sentences. Use simple, punchy language. "
            f"NEVER start with 'Welcome to' or 'In this video'. Jump straight into the hook."
        )

    def _build_prompt(self, storyboard: Storyboard) -> str:
        """Build the generation prompt from storyboard."""
        scenes_desc = []
        for scene in storyboard.scenes:
            scenes_desc.append(
                f"Scene {scene.scene_number} ({scene.duration_seconds}s, {scene.shot_type}): "
                f"Role: {scene.narration[:50] if scene.narration else 'visual only'}"
            )

        return (
            f"Topic: {storyboard.title}\n"
            f"Hook formula: {storyboard.hook_formula}\n"
            f"CTA: {storyboard.cta_text}\n\n"
            f"Storyboard scenes:\n" + "\n".join(scenes_desc) + "\n\n"
            f"Write a complete narration script. Return each scene's narration "
            f"on a separate line, prefixed with 'Scene N: '. "
            f"Total word count should be approximately "
            f"{int(storyboard.total_duration * 2.5)} words."
        )

    def _parse_script(self, content: str, storyboard: Storyboard,
                      model_name: str, cost: float) -> VideoScript:
        """Parse AI response into a VideoScript."""
        lines = content.strip().split("\n")
        segments = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Remove "Scene N: " prefix if present
                if ":" in line[:20]:
                    line = line.split(":", 1)[1].strip()
                if line:
                    segments.append(line)

        full_text = " ".join(segments)
        word_count = len(full_text.split())

        # Extract hook (first segment)
        hook = segments[0] if segments else ""

        return VideoScript(
            title=storyboard.title,
            hook=hook,
            body_segments=segments[1:-1] if len(segments) > 2 else segments[1:],
            cta=segments[-1] if len(segments) > 1 else storyboard.cta_text,
            full_text=full_text,
            word_count=word_count,
            estimated_duration=word_count / 2.5,  # ~2.5 words/sec
            model_used=model_name,
            cost=cost,
        )

    def _fallback_script(self, storyboard: Storyboard) -> VideoScript:
        """Build script from storyboard narration (no API needed)."""
        segments = [s.narration for s in storyboard.scenes if s.narration]
        full_text = " ".join(segments)
        word_count = len(full_text.split())

        return VideoScript(
            title=storyboard.title,
            hook=segments[0] if segments else "",
            body_segments=segments[1:-1] if len(segments) > 2 else segments[1:],
            cta=segments[-1] if len(segments) > 1 else storyboard.cta_text,
            full_text=full_text,
            word_count=word_count,
            estimated_duration=word_count / 2.5,
            model_used="fallback_storyboard",
            cost=0.0,
        )
