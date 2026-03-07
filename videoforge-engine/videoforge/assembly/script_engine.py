"""ScriptEngine — AI-powered narration script generation with anti-slop pipeline.

Supports OpenRouter (DeepSeek/Claude) and direct Anthropic API (Haiku fallback).
Framework-aware prompting with deep voice cards, TTS optimization, and post-processing.
"""

import os
from pathlib import Path
import json
import logging
import re
import requests
from ..models import VideoScript, Storyboard
from ..knowledge.niche_profiles import get_niche_profile
from ..knowledge.domain_expertise import get_domain_expertise
from ..knowledge.script_frameworks import (
    SCRIPT_FRAMEWORKS, get_framework, get_framework_key,
)

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

# ── Anti-Slop: Banned Vocabulary ─────────────────────────────────────────────

BANNED_WORDS = [
    "furthermore", "moreover", "additionally", "consequently", "nevertheless",
    "notwithstanding", "henceforth", "whereby", "wherein", "thereof",
    "delve", "delves", "delving",
    "crucial", "pivotal", "paramount", "indispensable",
    "leverage", "leveraging", "leveraged",
    "utilize", "utilizing", "utilized", "utilization",
    "facilitate", "facilitating",
    "comprehensive", "robust", "seamless", "streamline",
    "paradigm", "synergy", "synergistic",
    "groundbreaking", "cutting-edge", "state-of-the-art",
    "revolutionize", "revolutionizing", "revolutionary",
    "transformative", "disruptive",
    "empower", "empowering", "empowerment",
    "navigate", "navigating",
    "landscape", "realm", "sphere",
    "embark", "embarking",
    "bolster", "bolstering",
    "spearhead", "spearheading",
    "underpin", "underpinning",
    "multifaceted", "nuanced",
    "testament",
    "myriad",
    "plethora",
    "aforementioned",
    "subsequently",
    "endeavor", "endeavors",
]

BANNED_PHRASES = [
    "it's important to note",
    "it is important to note",
    "it's worth noting",
    "it is worth noting",
    "it's worth mentioning",
    "it is worth mentioning",
    "let me explain",
    "let me break down",
    "let me walk you through",
    "as we delve into",
    "without further ado",
    "in today's video",
    "in this video",
    "welcome to",
    "hey guys",
    "what's up guys",
    "in the world of",
    "in the realm of",
    "in today's digital landscape",
    "in today's fast-paced world",
    "in today's busy world",
    "as technology evolves",
    "as we navigate",
    "the key to success",
    "unlock your potential",
    "unlock your true potential",
    "embrace the journey",
    "on this journey",
    "it remains to be seen",
    "only time will tell",
    "at the end of the day",
    "having said that",
    "with that being said",
    "it goes without saying",
    "needless to say",
    "last but not least",
    "first and foremost",
    "each and every",
    "when it comes to",
    "in terms of",
    "the fact of the matter is",
    "in conclusion",
    "to sum up",
    "in summary",
    "as a matter of fact",
    "it's no secret that",
    "the bottom line is",
    "at this point in time",
    "for all intents and purposes",
    "a testament to",
]

# Contraction enforcement map (formal → contraction)
CONTRACTION_MAP = {
    "it is": "it's",
    "it has": "it's",
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    "have not": "haven't",
    "has not": "hasn't",
    "had not": "hadn't",
    "will not": "won't",
    "would not": "wouldn't",
    "could not": "couldn't",
    "should not": "shouldn't",
    "can not": "can't",
    "cannot": "can't",
    "that is": "that's",
    "there is": "there's",
    "here is": "here's",
    "what is": "what's",
    "who is": "who's",
    "you are": "you're",
    "they are": "they're",
    "we are": "we're",
    "I am": "I'm",
    "I have": "I've",
    "I will": "I'll",
    "you will": "you'll",
    "they will": "they'll",
    "we will": "we'll",
    "you have": "you've",
    "they have": "they've",
    "we have": "we've",
    "let us": "let's",
    "I would": "I'd",
    "you would": "you'd",
    "we would": "we'd",
    "they would": "they'd",
}

# Content type detection keywords
_CONTENT_TYPE_KEYWORDS = {
    "tutorial": ["how to", "guide", "setup", "install", "step by step", "beginner", "learn", "diy"],
    "review": ["review", "worth it", "honest", "vs", "versus", "comparison", "tested", "best"],
    "story": ["story", "tale", "legend", "myth", "origin", "history of", "ancient", "folklore", "who was"],
    "listicle": ["top", "best", "worst", "most", "reasons", "ways", "things", "tips", "hacks", "mistakes"],
    "news": ["just announced", "breaking", "new release", "update", "launched", "dropped", "announced"],
    "motivation": ["transform", "changed my life", "secret to", "manifest", "attract", "mindset", "affirmation"],
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
    """Get OpenRouter API key from env -> configs/api_keys.env -> empire config/.env."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    key = _load_key_from_file(
        Path(os.path.dirname(__file__)) / ".." / ".." / "configs" / "api_keys.env",
        "OPENROUTER_API_KEY",
    )
    if key:
        return key
    key = _load_key_from_file(
        Path(os.path.dirname(__file__)) / ".." / ".." / ".." / "config" / ".env",
        "OPENROUTER_API_KEY",
    )
    return key


def _get_anthropic_key() -> str:
    """Get Anthropic API key from env -> configs/api_keys.env -> empire config/.env."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    key = _load_key_from_file(
        Path(os.path.dirname(__file__)) / ".." / ".." / "configs" / "api_keys.env",
        "ANTHROPIC_API_KEY",
    )
    if key:
        return key
    key = _load_key_from_file(
        Path(os.path.dirname(__file__)) / ".." / ".." / ".." / "config" / ".env",
        "ANTHROPIC_API_KEY",
    )
    return key


class ScriptEngine:
    """Generates video narration scripts via OpenRouter API with anti-slop pipeline."""

    def __init__(self, model_tier: str = "cheap"):
        self.model_tier = model_tier
        self.model = MODELS.get(model_tier, MODELS["cheap"])

    def generate_script(self, storyboard: Storyboard,
                        model_tier: str = None) -> VideoScript:
        """Generate a narration script from a storyboard.

        Priority: OpenRouter -> Anthropic Haiku -> storyboard fallback.
        Post-processes all AI output to strip slop.
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

        logger.info("No API keys available -- using storyboard narration as script")
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
                    "max_tokens": 1500,
                    "temperature": 0.85,
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

            script = self._parse_script(content, storyboard, model["name"], cost)
            return self._post_process(script)

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
                    "max_tokens": 1500,
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
            script = self._parse_script(content, storyboard, "Claude Haiku", cost)
            return self._post_process(script)

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
            f'Example: ["topic 1", "topic 2"]\n'
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

    # ── Content Type Detection ────────────────────────────────────────────

    def _detect_content_type(self, title: str) -> str:
        """Detect content type from title keywords."""
        title_lower = title.lower()
        scores = {}
        for content_type, keywords in _CONTENT_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in title_lower)
            if score > 0:
                scores[content_type] = score
        if scores:
            return max(scores, key=scores.get)
        return "educational"

    # ── System Prompt (the main quality lever) ────────────────────────────

    def _system_prompt(self, storyboard: Storyboard) -> str:
        """Build the system prompt with voice cards, frameworks, TTS rules, and anti-slop."""
        profile = get_niche_profile(storyboard.niche)
        voice = profile.get("voice", {})
        voice_card = profile.get("voice_card", {})
        category = profile.get("category", "tech")

        # Framework selection
        content_type = self._detect_content_type(storyboard.title)
        framework_key = get_framework_key(content_type=content_type, category=category)
        framework = SCRIPT_FRAMEWORKS.get(framework_key, SCRIPT_FRAMEWORKS["hook_problem_solution_cta"])

        # Domain expertise
        expertise = get_domain_expertise(storyboard.niche, storyboard.title)
        expertise_block = ""
        if expertise:
            products = expertise.get("key_products", [])
            tips = expertise.get("expert_tips", [])
            products_str = ", ".join(products[:8]) if products else "none available"
            tips_str = "\n".join(f"- {t}" for t in tips[:5]) if tips else "- Use your expertise"

            expertise_block = (
                f"\nDOMAIN EXPERTISE -- use these real facts in your script:\n"
                f"Key products/tools: {products_str}\n"
                f"Expert tips:\n{tips_str}\n"
            )

            matched = expertise.get("matched_talking_point")
            if matched:
                expertise_block += (
                    f'\nTalking points for "{matched["topic"]}":\n'
                    f'- {matched["content"]}\n'
                )

        # Voice card blocks
        identity = voice_card.get("identity", f"an expert {storyboard.niche} creator")
        emotional_register = voice_card.get("emotional_register", voice.get("tone", "engaging"))
        viewer_rel = voice_card.get("viewer_relationship", "sharing knowledge with a friend")
        speaking_style = voice_card.get("speaking_style", "short punchy sentences, conversational")
        forbidden_tones = voice_card.get("forbidden_tones", [])
        signature_phrases = voice_card.get("signature_phrases", [])
        niche_never_say = voice_card.get("never_say", [])
        forbidden_str = ", ".join(forbidden_tones) if forbidden_tones else "generic, boring, robotic"
        signature_str = " / ".join(f'"{p}"' for p in signature_phrases[:4]) if signature_phrases else ""

        # Framework instruction
        framework_instruction = framework.get("prompt_instruction", "")

        # Build the banned vocabulary block for the prompt
        banned_words_sample = ", ".join(BANNED_WORDS[:30])
        banned_phrases_sample = " / ".join(f'"{p}"' for p in BANNED_PHRASES[:15])
        niche_banned = " / ".join(f'"{p}"' for p in niche_never_say[:5]) if niche_never_say else ""

        sections = []

        # IDENTITY
        sections.append(
            f"You are {identity}.\n"
            f"EMOTIONAL REGISTER: {emotional_register}\n"
            f"VIEWER RELATIONSHIP: {viewer_rel}\n"
            f"SPEAKING STYLE: {speaking_style}\n"
            f"NEVER sound like: {forbidden_str}"
        )

        # SIGNATURE (if available)
        if signature_str:
            sections.append(f"USE phrases like: {signature_str}")

        # FRAMEWORK
        sections.append(
            f"SCRIPT FRAMEWORK: {framework.get('name', 'Hook-Problem-Solution-CTA')}\n"
            f"{framework_instruction}"
        )

        # DOMAIN EXPERTISE
        if expertise_block:
            sections.append(expertise_block.strip())

        # TTS WRITING RULES
        sections.append(
            "TTS WRITING RULES (your script will be read aloud by a text-to-speech voice):\n"
            "- Max 20 words per sentence. Shorter is better. Mix lengths: 5, 12, 8, 15, 6.\n"
            "- Use contractions ALWAYS: 'don't' not 'do not', 'it's' not 'it is', 'you're' not 'you are'.\n"
            "- Spell out numbers under one hundred: 'thirty' not '30', 'five thousand' not '5,000'.\n"
            "- Punctuation IS timing: Period = full stop. Dash = dramatic pause. Ellipsis = suspense.\n"
            "- Start sentences with different words. Never start two consecutive sentences the same way.\n"
            "- Rhythm pattern: short punch. medium detail. short punch. longer context sentence. short punch."
        )

        # BANNED VOCABULARY
        sections.append(
            f"BANNED VOCABULARY -- never use these words: {banned_words_sample}\n"
            f"BANNED PHRASES -- never use: {banned_phrases_sample}"
        )
        if niche_banned:
            sections.append(f"NICHE-SPECIFIC BANS: {niche_banned}")

        # RETENTION MECHANICS
        sections.append(
            "RETENTION MECHANICS:\n"
            "- Open a loop in scene one. Don't close it until scene three or four.\n"
            "- Pattern interrupt every two to three sentences: a question, a surprising fact, a contradiction.\n"
            "- Power words by category -- curiosity: 'secret, hidden, unknown, bizarre, shocking' / "
            "urgency: 'now, immediately, before, deadline, running out' / "
            "fear: 'mistake, avoid, warning, never, danger' / "
            "authority: 'proven, research, data, expert, tested'.\n"
            "- Compound hooks: stack two hooks in the first sentence. 'This thirty dollar device replaced my entire security system.'"
        )

        # CRITICAL RULES
        sections.append(
            "CRITICAL RULES:\n"
            "- SPECIFICITY: Every claim needs a name, number, or concrete detail. No vague language.\n"
            "- 'This tool' is BANNED. Name the actual tool.\n"
            "- 'This changes everything' is BANNED. Say what specifically changes.\n"
            "- No throat-clearing: jump straight into the content. No 'so' or 'well' to start.\n"
            "- No meta-commentary: never say 'in this video' or 'let me tell you about'.\n"
            "- Every sentence must earn the next second of watch time.\n"
            f"- Target duration: {storyboard.total_duration:.0f} seconds. "
            f"Platform: {storyboard.platform}."
        )

        return "\n\n".join(sections)

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
            f"Write the complete narration with visual directions.\n"
            f"- One line per scene, format: 'Scene N: [narration] | VISUAL: [image description]'\n"
            f"- The VISUAL must describe what should be shown during this narration\n"
            f"- VISUAL descriptions should be specific subjects/objects, not camera directions\n"
            f"- VISUAL descriptions MUST be bright and vivid -- well-lit scenes, vibrant colors, clear subjects against clean backgrounds. NEVER dark, shadowy, or dimly-lit.\n"
            f"- Target ~{target_words} words total (narration only)\n"
            f"- Scene 1: immediate hook, no setup\n"
            f"- Make every word count. Cut any filler."
        )

    # ── Post-Processing Pipeline ──────────────────────────────────────────

    def _post_process(self, script: VideoScript) -> VideoScript:
        """Run the anti-slop post-processing pipeline on a VideoScript."""
        script.full_text = self._clean_text(script.full_text)
        script.hook = self._clean_text(script.hook)
        script.cta = self._clean_text(script.cta)
        script.body_segments = [self._clean_text(s) for s in script.body_segments]
        script.word_count = len(script.full_text.split())
        script.estimated_duration = script.word_count / 2.5
        return script

    def _clean_text(self, text: str) -> str:
        """Full cleaning pipeline for a text string."""
        if not text:
            return text
        text = self._strip_markdown(text)
        text = self._strip_banned_phrases(text)
        text = self._strip_banned_words(text)
        text = self._enforce_contractions(text)
        # Clean up extra whitespace
        text = re.sub(r"  +", " ", text).strip()
        return text

    @staticmethod
    def _strip_banned_phrases(text: str) -> str:
        """Remove banned phrases (case-insensitive)."""
        for phrase in BANNED_PHRASES:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            text = pattern.sub("", text)
        return text

    @staticmethod
    def _strip_banned_words(text: str) -> str:
        """Remove banned words at word boundaries (case-insensitive)."""
        for word in BANNED_WORDS:
            pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
            text = pattern.sub("", text)
        return text

    @staticmethod
    def _enforce_contractions(text: str) -> str:
        """Replace formal forms with contractions."""
        for formal, contraction in CONTRACTION_MAP.items():
            pattern = re.compile(r"\b" + re.escape(formal) + r"\b", re.IGNORECASE)
            # Preserve original case for first char
            def _replace(m, c=contraction):
                matched = m.group(0)
                if matched[0].isupper():
                    return c[0].upper() + c[1:]
                return c
            text = pattern.sub(_replace, text)
        return text

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove markdown bold/italic markers."""
        # **bold** -> bold
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        # *italic* -> italic
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        # __bold__ -> bold
        text = re.sub(r"__(.+?)__", r"\1", text)
        # _italic_ -> italic
        text = re.sub(r"_(.+?)_", r"\1", text)
        return text

    # ── Parsing ───────────────────────────────────────────────────────────

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
