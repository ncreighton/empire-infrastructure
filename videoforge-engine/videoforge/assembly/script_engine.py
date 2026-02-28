"""ScriptEngine — AI-powered narration script generation.

Supports OpenRouter (DeepSeek/Claude) and direct Anthropic API (Haiku fallback).
Structured around HOOK-COMMITMENT-VALUE-CTA with retention anchors.
"""

import os
import json
import logging
import re
import requests
from ..models import VideoScript, Storyboard
from ..knowledge.niche_profiles import get_niche_profile
from ..knowledge.domain_expertise import get_domain_expertise

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
ANTHROPIC_BASE = "https://api.anthropic.com/v1/messages"

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


def _load_key_from_file(filepath: str, key_name: str) -> str:
    """Load a key from an env-style file."""
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key_name}="):
                    val = line.split("=", 1)[1].strip()
                    if val:
                        return val
    return ""


def _get_api_key() -> str:
    """Get OpenRouter API key from env → configs/api_keys.env → empire config/.env."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    # Check project config
    key = _load_key_from_file(
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "api_keys.env"),
        "OPENROUTER_API_KEY",
    )
    if key:
        return key
    # Check empire-wide config
    key = _load_key_from_file(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", ".env"),
        "OPENROUTER_API_KEY",
    )
    return key


def _get_anthropic_key() -> str:
    """Get Anthropic API key from env → configs/api_keys.env → empire config/.env."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    key = _load_key_from_file(
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "api_keys.env"),
        "ANTHROPIC_API_KEY",
    )
    if key:
        return key
    key = _load_key_from_file(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", ".env"),
        "ANTHROPIC_API_KEY",
    )
    return key


class ScriptEngine:
    """Generates video narration scripts via OpenRouter API."""

    def __init__(self, model_tier: str = "cheap"):
        self.model_tier = model_tier
        self.model = MODELS.get(model_tier, MODELS["cheap"])

    def generate_script(self, storyboard: Storyboard,
                        model_tier: str = None) -> VideoScript:
        """Generate a narration script from a storyboard.

        Priority: OpenRouter → Anthropic Haiku → storyboard fallback.
        """
        model = MODELS.get(model_tier or self.model_tier, self.model)
        prompt = self._build_prompt(storyboard)
        system = self._system_prompt(storyboard)

        # Try OpenRouter first
        api_key = _get_api_key()
        if api_key:
            result = self._call_openrouter(api_key, model, system, prompt, storyboard)
            if result:
                return result

        # Try Anthropic Haiku as fallback
        anthropic_key = _get_anthropic_key()
        if anthropic_key:
            result = self._call_anthropic(anthropic_key, system, prompt, storyboard)
            if result:
                return result

        logger.info("No API keys available — using storyboard narration as script")
        return self._fallback_script(storyboard)

    def _call_openrouter(self, api_key, model, system, prompt, storyboard):
        """Call OpenRouter API for script generation."""
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
                        {"role": "system", "content": system},
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

            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cost = (input_tokens / 1000 * model["cost_per_1k_input"] +
                    output_tokens / 1000 * model["cost_per_1k_output"])

            return self._parse_script(content, storyboard, model["name"], cost)

        except Exception as e:
            logger.warning(f"OpenRouter script generation failed: {e}")
            return None

    def _call_anthropic(self, api_key, system, prompt, storyboard):
        """Call Anthropic API directly using Haiku for cheap script generation."""
        try:
            response = requests.post(
                ANTHROPIC_BASE,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1000,
                    "system": system,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            content = data["content"][0]["text"]
            usage = data.get("usage", {})

            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            # Haiku pricing: $0.80/M input, $4.00/M output
            cost = (input_tokens / 1_000_000 * 0.80 +
                    output_tokens / 1_000_000 * 4.00)

            logger.info(f"Anthropic Haiku script: {output_tokens} tokens, ${cost:.4f}")
            return self._parse_script(content, storyboard, "Claude Haiku", cost)

        except Exception as e:
            logger.warning(f"Anthropic script generation failed: {e}")
            return None

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
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(content)

        except Exception as e:
            logger.warning(f"Topic generation failed: {e}")
            return [f"{niche} topic idea {i+1}" for i in range(count)]

    def _system_prompt(self, storyboard: Storyboard) -> str:
        """Build system prompt for script generation with HOOK-COMMITMENT-VALUE-CTA structure."""
        profile = get_niche_profile(storyboard.niche)
        voice = profile.get("voice", {})
        tone = voice.get("tone", "engaging, clear")
        vocab = voice.get("vocab", [])
        avoid = voice.get("avoid", [])
        vocab_str = ", ".join(vocab[:5]) if vocab else ""
        avoid_str = ", ".join(avoid[:3]) if avoid else ""

        # Inject domain expertise
        expertise = get_domain_expertise(storyboard.niche, storyboard.title)
        expertise_block = ""
        if expertise:
            products = expertise.get("key_products", [])
            tips = expertise.get("expert_tips", [])
            products_str = ", ".join(products[:8]) if products else "none available"
            tips_str = "\n".join(f"- {t}" for t in tips[:5]) if tips else "- Use your expertise"

            expertise_block = (
                f"\nDOMAIN EXPERTISE — use these real facts in your script:\n"
                f"Key products/tools: {products_str}\n"
                f"Expert tips:\n{tips_str}\n"
            )

            # Add topic-matched talking point if available
            matched = expertise.get("matched_talking_point")
            if matched:
                expertise_block += (
                    f'\nTalking points for "{matched["topic"]}":\n'
                    f'- {matched["content"]}\n'
                )

        return (
            f"You are an expert {storyboard.niche} content creator making viral short-form videos.\n"
            f"Platform: {storyboard.platform}. Format: {storyboard.format}. "
            f"Target duration: {storyboard.total_duration:.0f} seconds.\n\n"
            f"VOICE: {tone}\n"
            f"KEY VOCABULARY: {vocab_str}\n"
            f"NEVER USE: {avoid_str}\n"
            f"{expertise_block}\n"
            f"STRUCTURE: HOOK → COMMITMENT → VALUE → CTA\n"
            f"- HOOK (first 2 seconds): Pattern interrupt using a specific fact or product name.\n"
            f"- COMMITMENT (next 3 seconds): Tell them exactly what specific things they'll learn.\n"
            f"- VALUE (body): 3 insights with REAL product names, numbers, or techniques. "
            f"Each one lands in 1-2 short sentences.\n"
            f"- CTA (last 3 seconds): Clear, direct call to action.\n\n"
            f"CRITICAL RULES:\n"
            f"- Every insight MUST mention a specific product, technique, or fact\n"
            f"- NEVER be vague — 'this one tool' is banned, name the actual tool\n"
            f"- Include at least 2 specific product/tool names from the domain expertise\n"
            f"- NEVER start with 'Welcome to' or 'In this video'\n"
            f"- NEVER say 'Let me explain' or 'As you can see'\n"
            f"- Short punchy sentences. 8-12 words max per sentence.\n"
            f"- Add a retention anchor every 5 seconds ('But here's the thing...', "
            f"'And it gets better...', 'Watch this...')\n"
            f"- Conversational tone — like talking to a friend\n"
            f"- Every sentence must earn the next second of watch time"
        )

    def _build_prompt(self, storyboard: Storyboard) -> str:
        """Build the generation prompt from storyboard."""
        scenes_desc = []
        for scene in storyboard.scenes:
            scenes_desc.append(
                f"Scene {scene.scene_number} ({scene.duration_seconds}s, {scene.shot_type}): "
                f"{scene.narration[:80] if scene.narration else 'visual only'}"
            )

        target_words = int(storyboard.total_duration * 2.5)

        return (
            f"Topic: {storyboard.title}\n"
            f"Hook formula: {storyboard.hook_formula}\n"
            f"CTA: {storyboard.cta_text}\n\n"
            f"Storyboard scenes:\n" + "\n".join(scenes_desc) + "\n\n"
            f"Write the complete narration with visual directions. Rules:\n"
            f"- One line per scene, format: 'Scene N: [narration] | VISUAL: [image description]'\n"
            f"- The VISUAL must describe what should be shown during this narration\n"
            f"- VISUAL descriptions should be specific subjects/objects, not camera directions\n"
            f"- Example: 'Scene 3: The Echo Dot costs just thirty dollars. | VISUAL: Amazon Echo Dot smart speaker on a white countertop, blue LED ring glowing'\n"
            f"- Target ~{target_words} words total (narration only, not counting VISUAL descriptions)\n"
            f"- Scene 1 must be an immediate hook — no setup, no intro\n"
            f"- Use short, punchy sentences (8-12 words)\n"
            f"- Include at least 2 retention anchors ('But here's the thing...', etc.)\n"
            f"- End with a specific, actionable CTA\n"
            f"- Make every word count. Cut any filler."
        )

    def _parse_script(self, content: str, storyboard: Storyboard,
                      model_name: str, cost: float) -> VideoScript:
        """Parse AI response into a VideoScript.

        Handles lines in format: 'Scene N: narration | VISUAL: description'
        Splits narration from visual directions.
        """
        lines = content.strip().split("\n")
        segments = []
        visual_directions = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Remove "Scene N: " prefix if present
                if ":" in line[:20]:
                    line = line.split(":", 1)[1].strip()
                # Strip markdown formatting (bold, italic markers)
                line = line.strip("*_").strip()
                if not line:
                    continue

                # Split narration from visual direction
                visual = ""
                # Try several delimiter patterns tolerantly
                for delim in [" | VISUAL: ", " |VISUAL: ", " | VISUAL:", "|VISUAL:"]:
                    if delim in line:
                        narration_part, visual = line.split(delim, 1)
                        line = narration_part.strip()
                        visual = visual.strip()
                        break

                segments.append(line)
                visual_directions.append(visual)

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
            visual_directions=visual_directions,
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
            visual_directions=[],
        )
