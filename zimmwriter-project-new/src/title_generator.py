"""
AI Title Generation with Diversity Controls.

Uses Claude Sonnet API to generate unique article titles for each site,
validates against existing content, and ensures article type diversity.

Usage:
    from src.title_generator import TitleGenerator

    gen = TitleGenerator()
    titles = gen.generate_for_site("smarthomewizards.com", existing_titles, count=20)
    all_titles = gen.generate_for_all_sites(all_existing)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .article_types import classify_title, ARTICLE_TYPES
from .site_presets import SITE_PRESETS
from .title_checker import TitleChecker
from .utils import setup_logger

logger = setup_logger("title_generator")

# Site config for niche/tone/special_requirements
_SITE_CONFIGS_PATH = Path(__file__).parent.parent / "configs" / "site-configs.json"

# Review-focused domains (get "review" type allocation instead of extra "informational")
_REVIEW_DOMAINS = {
    "smarthomegearreviews.com",
    "wearablegearreviews.com",
    "pulsegearreviews.com",
}


def _load_site_configs() -> Dict[str, dict]:
    """Load site configs from configs/site-configs.json."""
    if not _SITE_CONFIGS_PATH.exists():
        logger.warning("site-configs.json not found at %s", _SITE_CONFIGS_PATH)
        return {}
    with open(_SITE_CONFIGS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sites", {})


def _get_type_distribution(count: int, is_review_site: bool) -> Dict[str, int]:
    """Calculate target article type distribution.

    For 20 titles (default):
        4 how_to, 4 listicle, 3 guide, 3 review/informational, 4 informational, 2 news

    For review sites, the middle slot is "review" instead of "informational".
    Scales proportionally for other counts.
    """
    if count <= 0:
        return {}

    # Base ratios for 20 articles
    ratios = {
        "how_to": 4 / 20,
        "listicle": 4 / 20,
        "guide": 3 / 20,
        "review" if is_review_site else "informational": 3 / 20,
        "informational": 4 / 20,
        "news": 2 / 20,
    }

    # If not a review site, merge the two informational slots
    if not is_review_site:
        ratios["informational"] = 7 / 20

    dist = {}
    allocated = 0
    for type_name, ratio in ratios.items():
        n = max(1, round(ratio * count))
        dist[type_name] = n
        allocated += n

    # Adjust to hit exact count
    while allocated > count:
        # Remove from the largest bucket
        largest = max(dist, key=lambda k: dist[k])
        dist[largest] -= 1
        allocated -= 1
    while allocated < count:
        # Add to the smallest bucket
        smallest = min(dist, key=lambda k: dist[k])
        dist[smallest] += 1
        allocated += 1

    return dist


def _build_system_prompt(
    domain: str,
    site_config: dict,
    preset: dict,
    existing_titles: Set[str],
    count: int,
    is_review_site: bool,
) -> str:
    """Build the system prompt for title generation."""
    niche = site_config.get("niche", preset.get("niche", "General"))
    tone = site_config.get("tone", "professional, helpful")
    audience = preset.get("audience_personality", "Explorer")
    special_reqs = site_config.get("special_requirements", [])

    distribution = _get_type_distribution(count, is_review_site)
    dist_lines = []
    for type_name, n in distribution.items():
        type_obj = ARTICLE_TYPES.get(type_name)
        if type_obj:
            pattern_examples = [p.pattern for p in type_obj.patterns[:3]]
            dist_lines.append(f"- {type_name}: {n} titles (patterns: {', '.join(pattern_examples)})")
        else:
            dist_lines.append(f"- {type_name}: {n} titles")

    # Truncate existing titles to most recent 500
    existing_list = sorted(existing_titles)[:500]
    existing_block = "\n".join(f"  - {t}" for t in existing_list) if existing_list else "  (No existing content yet)"

    special_block = "\n".join(f"- {r}" for r in special_reqs) if special_reqs else "- None"

    return f"""You are an SEO content strategist for {preset.get('domain', domain)} ({domain}), a {niche} website.
Brand voice: {tone}
Target audience personality: {audience}

EXISTING CONTENT (do NOT duplicate these topics):
{existing_block}

Generate exactly {count} article titles with this type distribution:
{chr(10).join(dist_lines)}

ARTICLE TYPE PATTERNS (titles MUST match these regex patterns for classification):
- how_to: Start with "How to" or include "Step-by-Step", "Tutorial", "Setting Up", "DIY"
- listicle: Start with a number like "10 Best...", "7 Top...", "5 Essential..." or include "Ways to", "Tips for"
- guide: Include "Complete Guide", "Ultimate Guide", "Beginner's Guide", "Everything You Need to Know"
- review: Include "Review", "vs", "Comparison", "Alternatives", "Tested", "Worth It"
- informational: Start with "What Is", "Why", "Understanding", include "Explained", "Difference Between"
- news: Include "2026", "Launches", "Announces", "New:", "Now Available", "First Look"

REQUIREMENTS:
1. Target unique long-tail keywords (50-65 characters per title)
2. ZERO overlap with existing content listed above
3. Mix evergreen + trending 2026 topics
4. Each title must be distinct in topic (not just rephrased variations)
5. Titles must be compelling and click-worthy for real readers
6. Use proper capitalization (Title Case)

NICHE-SPECIFIC REQUIREMENTS:
{special_block}

Return ONLY valid JSON array with NO extra text. Each item:
{{"title": "...", "type": "how_to|listicle|guide|review|informational|news", "keyword": "primary target keyword"}}"""


class TitleGenerator:
    """Generates unique, diverse article titles using Claude Sonnet API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Set it in environment or pass api_key."
            )
        self._site_configs = _load_site_configs()
        self._checker = TitleChecker()

    def _call_claude(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
        """Call Claude Sonnet API with prompt caching.

        Uses requests directly to call the Anthropic Messages API.
        System prompt gets cache_control since it exceeds 2048 tokens.
        """
        import requests

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "anthropic-beta": "prompt-caching-2024-07-31",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=120,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Claude API error {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        content = data.get("content", [])
        if content and content[0].get("type") == "text":
            return content[0]["text"]
        raise RuntimeError(f"Unexpected Claude response format: {data}")

    def _parse_titles(self, raw_text: str) -> List[Dict[str, str]]:
        """Parse JSON title list from Claude response.

        Handles common issues: markdown code fences, trailing commas.
        """
        text = raw_text.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            titles = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON array from text
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                text = text[start:end + 1]
                try:
                    titles = json.loads(text)
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse Claude response as JSON: %s", e)
                    logger.debug("Raw text: %s", raw_text[:500])
                    return []
            else:
                logger.error("No JSON array found in Claude response")
                return []

        if not isinstance(titles, list):
            logger.error("Claude response is not a list: %s", type(titles))
            return []

        # Validate each item
        valid = []
        for item in titles:
            if isinstance(item, dict) and "title" in item:
                valid.append({
                    "title": item["title"].strip(),
                    "type": item.get("type", "informational"),
                    "keyword": item.get("keyword", ""),
                })
        return valid

    def generate_for_site(
        self,
        domain: str,
        existing: Optional[Dict] = None,
        count: int = 20,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate unique titles for a single site.

        Args:
            domain: Site domain (e.g., "smarthomewizards.com").
            existing: Existing titles dict from TitleChecker.check_site().
                     If None, fetches automatically.
            count: Number of titles to generate.
            max_retries: Maximum API retry attempts.

        Returns:
            Dict with "domain", "titles" (list of dicts), "duplicates_removed",
            "type_distribution", "generated_at".
        """
        preset = SITE_PRESETS.get(domain)
        if not preset:
            raise ValueError(f"No preset for domain: {domain}")

        if existing is None:
            existing = self._checker.check_site(domain)

        existing_titles = existing.get("titles", set())
        is_review = domain in _REVIEW_DOMAINS

        # Look up site config (try both domain and short key)
        site_config = self._site_configs.get(domain, {})

        system_prompt = _build_system_prompt(
            domain, site_config, preset, existing_titles, count + 5, is_review
        )

        user_prompt = (
            f"Generate {count + 5} unique article titles for {domain}. "
            f"Return JSON array only, no other text."
        )

        # Request extra titles (count + 5) to have room for dedup filtering
        all_candidates = []
        retry_delay = 30

        for attempt in range(max_retries):
            try:
                logger.info(
                    "Generating titles for %s (attempt %d/%d)",
                    domain, attempt + 1, max_retries,
                )
                raw = self._call_claude(system_prompt, user_prompt, max_tokens=4096)
                parsed = self._parse_titles(raw)
                all_candidates.extend(parsed)
                logger.info("  Got %d candidates from Claude", len(parsed))
                break
            except Exception as e:
                logger.warning("Claude API attempt %d failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        # Request additional batch if we don't have enough after dedup
        candidate_titles = [c["title"] for c in all_candidates]
        unique, duplicates = self._checker.filter_duplicates(candidate_titles, existing)

        if len(unique) < count and len(all_candidates) > 0:
            logger.info(
                "Only %d unique after dedup (need %d), requesting more",
                len(unique), count,
            )
            extra_prompt = (
                f"Generate {count - len(unique) + 5} MORE unique article titles for {domain}. "
                f"These must be DIFFERENT from all previous titles. "
                f"Avoid these topics: {', '.join(unique[:10])}. "
                f"Return JSON array only."
            )
            try:
                raw = self._call_claude(system_prompt, extra_prompt, max_tokens=4096)
                extra = self._parse_titles(raw)
                for item in extra:
                    all_candidates.append(item)
                candidate_titles = [c["title"] for c in all_candidates]
                unique, duplicates = self._checker.filter_duplicates(candidate_titles, existing)
            except Exception as e:
                logger.warning("Extra title batch failed: %s", e)

        # Map unique titles back to their full candidate dicts
        unique_set = set(unique)
        selected = []
        seen = set()
        for candidate in all_candidates:
            title = candidate["title"]
            if title in unique_set and title not in seen:
                # Validate the type via our classifier
                actual_type = classify_title(title)
                candidate["type"] = actual_type
                selected.append(candidate)
                seen.add(title)
            if len(selected) >= count:
                break

        # Calculate type distribution of selected titles
        type_dist = {}
        for item in selected:
            t = item["type"]
            type_dist[t] = type_dist.get(t, 0) + 1

        result = {
            "domain": domain,
            "titles": selected,
            "count": len(selected),
            "duplicates_removed": len(duplicates),
            "type_distribution": type_dist,
            "generated_at": datetime.now().isoformat(),
        }

        logger.info(
            "%s: generated %d titles (%d dupes removed), types: %s",
            domain, len(selected), len(duplicates), type_dist,
        )

        return result

    def generate_for_all_sites(
        self,
        all_existing: Optional[Dict[str, Dict]] = None,
        count: int = 20,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate titles for all 14 sites.

        Args:
            all_existing: Dict of {domain: check_site_result}. If None, fetches all.
            count: Number of titles per site.
            output_dir: Directory to save results. Defaults to output/batches/batch_{timestamp}/.

        Returns:
            Dict with "sites" (per-site results), "total_titles", "saved_to".
        """
        if all_existing is None:
            checker = TitleChecker()
            all_existing = checker.check_all_sites()

        if output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(
                Path(__file__).parent.parent / "output" / "batches" / f"batch_{ts}"
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = {}
        total = 0

        for domain in SITE_PRESETS:
            existing = all_existing.get(domain, {"titles": set(), "slugs": set()})
            try:
                site_result = self.generate_for_site(domain, existing, count=count)
                results[domain] = site_result
                total += site_result["count"]
            except Exception as e:
                logger.error("Title generation failed for %s: %s", domain, e)
                results[domain] = {
                    "domain": domain,
                    "titles": [],
                    "count": 0,
                    "error": str(e),
                    "generated_at": datetime.now().isoformat(),
                }

            # Brief pause between sites to avoid rate limits
            time.sleep(2)

        # Save results
        output_path = Path(output_dir) / "generated_titles.json"
        serializable = {}
        for domain, result in results.items():
            serializable[domain] = {
                k: v for k, v in result.items()
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(
            "Generated %d total titles across %d sites. Saved to %s",
            total, len(results), output_path,
        )

        return {
            "sites": results,
            "total_titles": total,
            "saved_to": output_path,
            "output_dir": output_dir,
        }
