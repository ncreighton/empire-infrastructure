"""Central Anthropic API wrapper with cost optimization.

Uses Sonnet for writing, Haiku for classification/tags.
All system prompts >2048 tokens get prompt caching.
"""

import os
import time
from dataclasses import dataclass, field

import anthropic


@dataclass
class UsageStats:
    """Track API costs across calls."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    calls: int = 0
    total_cost_usd: float = 0.0

    def record(self, usage, model: str):
        self.calls += 1
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_read_tokens += cache_read
        self.total_cache_creation_tokens += cache_creation

        # Calculate cost
        if "haiku" in model:
            input_price = 1.00 / 1_000_000
            output_price = 5.00 / 1_000_000
        elif "sonnet" in model:
            input_price = 3.00 / 1_000_000
            output_price = 15.00 / 1_000_000
        else:
            input_price = 5.00 / 1_000_000
            output_price = 25.00 / 1_000_000

        # Cache reads are 90% cheaper
        cache_read_price = input_price * 0.1
        non_cached = input_tokens - cache_read - cache_creation
        cost = (
            max(0, non_cached) * input_price
            + cache_read * cache_read_price
            + cache_creation * input_price * 1.25
            + output_tokens * output_price
        )
        self.total_cost_usd += cost
        return cost


# Model constants matching CLAUDE.md cost rules
MODEL_WRITER = "claude-sonnet-4-20250514"
MODEL_CLASSIFIER = "claude-haiku-4-5-20251001"
CACHE_THRESHOLD = 2048


class ContentWriter:
    """Anthropic API wrapper optimized for content generation."""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.stats = UsageStats()

    def _build_system(self, text: str) -> list[dict] | str:
        """Build system prompt, adding cache_control if over threshold."""
        if len(text) > CACHE_THRESHOLD:
            return [{
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }]
        return text

    def _call(self, model: str, max_tokens: int, system: str, user_message: str) -> str:
        """Make an API call and track usage."""
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=self._build_system(system),
            messages=[{"role": "user", "content": user_message}],
        )
        self.stats.record(response.usage, model)
        return response.content[0].text

    def classify(self, topic: str, categories: list[str]) -> str:
        """Classify a topic into one of the given categories. Uses Haiku."""
        cats = ", ".join(categories)
        return self._call(
            model=MODEL_CLASSIFIER,
            max_tokens=100,
            system=f"Classify the given topic into exactly one category. Reply with ONLY the category name, nothing else. Categories: {cats}",
            user_message=topic,
        ).strip().lower()

    def generate_tags(self, title: str, content_type: str, keywords: list[str] | None = None) -> list[str]:
        """Generate Printables tags. Uses Haiku. Returns up to 10 tags."""
        kw_hint = f" Related keywords: {', '.join(keywords)}" if keywords else ""
        result = self._call(
            model=MODEL_CLASSIFIER,
            max_tokens=200,
            system="Generate tags for a Printables.com listing. Return ONLY a comma-separated list of lowercase tags. Max 10 tags. Focus on 3D printing, the specific topic, and searchability.",
            user_message=f"Content type: {content_type}\nTitle: {title}{kw_hint}",
        )
        tags = [t.strip().lower() for t in result.split(",") if t.strip()]
        return tags[:10]

    def generate_title(self, topic: str, content_type: str) -> str:
        """Generate an optimized title. Uses Haiku. Max 70 chars."""
        result = self._call(
            model=MODEL_CLASSIFIER,
            max_tokens=100,
            system="Generate a title for Printables.com content. Max 70 characters. Make it specific, searchable, and engaging. Reply with ONLY the title.",
            user_message=f"Content type: {content_type}\nTopic: {topic}",
        )
        return result.strip().strip('"')[:70]

    def write_article(self, topic: str, keywords: list[str], difficulty: str, voice_prompt: str) -> str:
        """Write a full how-to article. Uses Sonnet with cached system prompt."""
        system = self._article_system_prompt(voice_prompt)
        user_msg = (
            f"Write a how-to article about: {topic}\n"
            f"Difficulty level: {difficulty}\n"
            f"Target keywords: {', '.join(keywords)}\n\n"
            "Structure:\n"
            "1. Brief intro (2-3 sentences, hook the reader)\n"
            "2. 5-8 sections with ## headings\n"
            "3. Include a 'Common Mistakes' section\n"
            "4. Include a 'Pro Tips' section\n"
            "5. Short conclusion with next steps\n\n"
            "Target: 1500-2500 words. Use markdown formatting."
        )
        return self._call(MODEL_WRITER, 4096, system, user_msg)

    def write_review(self, product_name: str, specs: dict, voice_prompt: str) -> str:
        """Write a product review. Uses Sonnet with cached system prompt."""
        system = self._review_system_prompt(voice_prompt)
        specs_text = "\n".join(f"- {k}: {v}" for k, v in specs.items())
        user_msg = (
            f"Write a review of the {product_name}.\n\n"
            f"Specs:\n{specs_text}\n\n"
            "Structure:\n"
            "1. Overview (personal experience opening)\n"
            "2. Specs at a Glance (formatted table or list)\n"
            "3. Print Quality (with specific examples)\n"
            "4. Ease of Use\n"
            "5. Value for Money\n"
            "6. Pros & Cons (bullet lists)\n"
            "7. Best For / Skip If\n"
            "8. The Verdict (with rating X/10)\n\n"
            "Target: 1200-2000 words. Use markdown. Be honest about flaws."
        )
        return self._call(MODEL_WRITER, 3000, system, user_msg)

    def write_listing(self, product_name: str, metadata: dict, voice_prompt: str) -> str:
        """Write a model listing description. Uses Sonnet."""
        system = self._listing_system_prompt(voice_prompt)
        meta_text = "\n".join(f"- {k}: {v}" for k, v in metadata.items())
        user_msg = (
            f"Write a Printables listing description for: {product_name}\n\n"
            f"Metadata:\n{meta_text}\n\n"
            "Include: what it is, print settings, dimensions, tested printers.\n"
            "Target: 200-500 words. Natural, helpful tone. Markdown formatted."
        )
        return self._call(MODEL_WRITER, 1000, system, user_msg)

    def write_post(self, topic: str, voice_prompt: str) -> str:
        """Write a community post. Uses Sonnet."""
        system = self._post_system_prompt(voice_prompt)
        user_msg = (
            f"Write a community post about: {topic}\n\n"
            "Keep it under 300 words. Conversational tone.\n"
            "End with an engagement question to spark discussion."
        )
        return self._call(MODEL_WRITER, 500, system, user_msg)

    def _article_system_prompt(self, voice_prompt: str) -> str:
        return (
            "You are a knowledgeable 3D printing content writer for Printables.com. "
            "You write practical, helpful how-to articles that makers actually want to read.\n\n"
            f"VOICE AND STYLE:\n{voice_prompt}\n\n"
            "CONTENT RULES:\n"
            "- Every claim must be specific (temperatures, speeds, settings)\n"
            "- Include real printer names and slicer settings\n"
            "- Use markdown with ## headings, bullet lists, and bold for key terms\n"
            "- Write for the specified difficulty level\n"
            "- Include actionable steps, not vague advice\n"
            "- Mention specific filament brands and types when relevant\n"
            "- Add print setting recommendations where appropriate\n"
        )

    def _review_system_prompt(self, voice_prompt: str) -> str:
        return (
            "You are an experienced 3D printer reviewer for Printables.com. "
            "You write honest, balanced reviews based on real testing.\n\n"
            f"VOICE AND STYLE:\n{voice_prompt}\n\n"
            "REVIEW RULES:\n"
            "- Lead with hands-on experience, not marketing specs\n"
            "- Include specific test results (benchy times, calibration results)\n"
            "- Be honest about flaws — credibility over hype\n"
            "- Compare to alternatives the reader might consider\n"
            "- Include a clear Best For / Skip If verdict\n"
            "- Rate on a 1-10 scale with justification\n"
            "- Mention price-to-value ratio\n"
        )

    def _listing_system_prompt(self, voice_prompt: str) -> str:
        return (
            "You are writing a product listing description for Printables.com. "
            "Make it natural, informative, and helpful for someone deciding to download and print.\n\n"
            f"VOICE AND STYLE:\n{voice_prompt}\n\n"
            "LISTING RULES:\n"
            "- Start with what the model is and who it's for\n"
            "- Include recommended print settings\n"
            "- Mention tested printers and materials\n"
            "- Add dimensions and any assembly notes\n"
            "- Keep it concise but complete\n"
        )

    def _post_system_prompt(self, voice_prompt: str) -> str:
        return (
            "You are a 3D printing enthusiast writing a community post on Printables.com. "
            "Write like you're talking to friends at a maker meetup.\n\n"
            f"VOICE AND STYLE:\n{voice_prompt}\n\n"
            "POST RULES:\n"
            "- Keep it casual and conversational\n"
            "- Share one specific tip, experience, or question\n"
            "- Under 300 words\n"
            "- End with a question to encourage replies\n"
        )

    def get_cost_summary(self) -> dict:
        """Return cost summary for the session."""
        return {
            "total_cost_usd": round(self.stats.total_cost_usd, 4),
            "total_calls": self.stats.calls,
            "total_input_tokens": self.stats.total_input_tokens,
            "total_output_tokens": self.stats.total_output_tokens,
            "cache_read_tokens": self.stats.total_cache_read_tokens,
        }
