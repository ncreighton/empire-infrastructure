"""
Dynamic Campaign Planning Engine for ZimmWriter.

Analyzes article titles, detects article types, selects optimal ZimmWriter
settings, and generates SEO CSV files ready for bulk writing.

Usage:
    from src.campaign_engine import CampaignEngine

    engine = CampaignEngine()
    plan = engine.plan_campaign("smarthomewizards.com", [
        "How to Set Up Your First Smart Home Hub",
        "10 Best Smart Thermostats of 2026",
        "Zigbee vs Z-Wave: Which Protocol is Better?",
    ])
    csv_path = engine.generate_seo_csv(plan)

    # Or in one step:
    plan, csv_path = engine.plan_and_generate("smarthomewizards.com", titles)
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .article_types import (
    classify_title,
    classify_titles,
    get_dominant_type,
    get_settings_overrides,
)
from .csv_generator import generate_bulk_csv
from .outline_templates import get_template, rotate_template
from .site_presets import SITE_PRESETS, get_preset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default output directory
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = r"D:\Claude Code Projects\zimmwriter-project-new\output\campaigns"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CampaignPlan:
    """Holds a fully resolved campaign plan ready for CSV generation.

    Attributes:
        domain: The target WordPress site domain.
        titles: Ordered list of article titles in this campaign.
        title_types: Mapping of each title to its detected article type name.
        dominant_type: The single most common article type across all titles.
        settings_overrides: Merged ZimmWriter settings for the dominant type.
        per_title_config: Per-title configuration dicts containing
            ``outline_focus``, ``category``, ``section_length``, and ``outline``.
        outline_template: The selected outline template text for the
            dominant article type.
        created_at: ISO-8601 timestamp of when the plan was created.
    """

    domain: str
    titles: List[str]
    title_types: Dict[str, str]
    dominant_type: str
    settings_overrides: Dict[str, Any]
    per_title_config: List[Dict[str, Any]]
    outline_template: str
    created_at: str


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


_FOCUS_HINTS: Dict[str, str] = {
    "how_to": "Focus on clear step-by-step instructions with practical details",
    "listicle": "Focus on variety and concise descriptions for each item",
    "review": "Focus on objective evaluation, features, and comparisons",
    "guide": "Provide comprehensive coverage with expert depth",
    "news": "Focus on key facts, timeline, and implications",
    "informational": "Explain clearly with examples and practical takeaways",
}


def _outline_focus_for_type(article_type: str) -> str:
    """Return a brief focus hint for the given article type.

    Args:
        article_type: One of the recognized type names (``how_to``,
            ``listicle``, ``review``, ``guide``, ``news``, ``informational``).

    Returns:
        A short instructional string. Falls back to the ``informational``
        hint for unrecognized types.
    """
    return _FOCUS_HINTS.get(article_type, _FOCUS_HINTS["informational"])


def _section_length_for_title(title: str) -> str:
    """Deterministically assign a section length to a title.

    Uses a hash-based pseudo-random distribution:
        - 30% Short  (hash % 100 in  0..29)
        - 50% Medium (hash % 100 in 30..79)
        - 20% Long   (hash % 100 in 80..99)

    The same title always produces the same result.

    Args:
        title: The article title string.

    Returns:
        One of ``"Short"``, ``"Medium"``, or ``"Long"``.
    """
    h = hash(title) % 100
    if h < 30:
        return "Short"
    elif h < 80:
        return "Medium"
    else:
        return "Long"


# ---------------------------------------------------------------------------
# Campaign Engine
# ---------------------------------------------------------------------------


class CampaignEngine:
    """Plans and generates ZimmWriter campaigns from article titles.

    Workflow:
        1. Classify each title into an article type.
        2. Determine the dominant type for the batch.
        3. Select ZimmWriter settings overrides for the dominant type.
        4. Pick an outline template variant (rotated by domain + date hash).
        5. Build per-title configs with focus hints, categories, and lengths.
        6. Optionally export the plan as a ZimmWriter SEO CSV.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Core planning
    # ------------------------------------------------------------------

    def plan_campaign(self, domain: str, titles: List[str]) -> CampaignPlan:
        """Build a complete campaign plan from a domain and title list.

        Args:
            domain: Target site domain (must exist in ``SITE_PRESETS``).
            titles: List of article title strings.

        Returns:
            A :class:`CampaignPlan` with all fields populated.

        Raises:
            ValueError: If *domain* is not found in ``SITE_PRESETS``.
        """
        if not titles:
            raise ValueError("titles list must not be empty")

        preset = get_preset(domain)
        if preset is None:
            raise ValueError(
                f"Domain '{domain}' not found in SITE_PRESETS. "
                f"Available: {', '.join(sorted(SITE_PRESETS.keys()))}"
            )

        self.logger.info("Planning campaign for %s with %d titles", domain, len(titles))

        # 1. Classify titles
        title_types = classify_titles(titles)
        self.logger.debug("Title classifications: %s", title_types)

        # 2. Dominant type
        dominant_type = get_dominant_type(titles)
        self.logger.info("Dominant article type: %s", dominant_type)

        # 3. Settings overrides
        settings_overrides = get_settings_overrides(dominant_type)

        # 4. Select outline template — rotate based on hash(domain + today's date)
        rotation_key = f"{domain}:{datetime.now().strftime('%Y-%m-%d')}"
        rotation_index = hash(rotation_key) % 1_000_000  # large positive-ish value
        outline_template = rotate_template(dominant_type, rotation_index)

        # 5. Per-title config
        wordpress_category = preset.get("wordpress_category", "")
        per_title_config: List[Dict[str, Any]] = []
        for title in titles:
            detected_type = title_types[title]
            per_title_config.append({
                "title": title,
                "article_type": detected_type,
                "outline_focus": _outline_focus_for_type(detected_type),
                "category": wordpress_category,
                "section_length": _section_length_for_title(title),
                "outline": outline_template,
            })

        created_at = datetime.now().isoformat()

        plan = CampaignPlan(
            domain=domain,
            titles=titles,
            title_types=title_types,
            dominant_type=dominant_type,
            settings_overrides=settings_overrides,
            per_title_config=per_title_config,
            outline_template=outline_template,
            created_at=created_at,
        )

        self.logger.info(
            "Campaign plan created: %d titles, dominant=%s, template variant selected",
            len(titles),
            dominant_type,
        )
        return plan

    # ------------------------------------------------------------------
    # CSV generation
    # ------------------------------------------------------------------

    def generate_seo_csv(
        self,
        plan: CampaignPlan,
        output_dir: Optional[str] = None,
    ) -> str:
        """Export a campaign plan as a ZimmWriter SEO CSV.

        Converts the per-title configs into the format expected by
        :func:`csv_generator.generate_bulk_csv`.

        Args:
            plan: A :class:`CampaignPlan` (from :meth:`plan_campaign`).
            output_dir: Directory for the output CSV. Defaults to
                ``D:\\Claude Code Projects\\zimmwriter-project-new\\output\\campaigns``.

        Returns:
            Absolute path to the generated CSV file.
        """
        if output_dir is None:
            output_dir = _DEFAULT_OUTPUT_DIR

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Build articles list for csv_generator
        articles: List[Dict[str, Any]] = []
        for cfg in plan.per_title_config:
            articles.append({
                "title": cfg["title"],
                "outline_focus": cfg["outline_focus"],
                "wordpress_category": cfg["category"],
                "outline": cfg.get("outline", ""),
            })

        # Generate filename from domain and timestamp
        safe_domain = plan.domain.replace(".", "-")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"campaign-{safe_domain}-{timestamp}.csv"
        output_path = os.path.join(output_dir, filename)

        csv_path = generate_bulk_csv(
            articles=articles,
            output_path=output_path,
            site_domain=plan.domain,
        )

        self.logger.info("SEO CSV generated: %s (%d articles)", csv_path, len(articles))
        return csv_path

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def plan_and_generate(
        self,
        domain: str,
        titles: List[str],
        output_dir: Optional[str] = None,
    ) -> Tuple[CampaignPlan, str]:
        """Plan a campaign and generate its SEO CSV in one call.

        Args:
            domain: Target site domain.
            titles: List of article title strings.
            output_dir: Optional override for the CSV output directory.

        Returns:
            A tuple of ``(CampaignPlan, csv_filepath)``.
        """
        plan = self.plan_campaign(domain, titles)
        csv_path = self.generate_seo_csv(plan, output_dir=output_dir)
        return plan, csv_path

    # ------------------------------------------------------------------
    # Summary / introspection
    # ------------------------------------------------------------------

    def get_campaign_summary(self, plan: CampaignPlan) -> Dict[str, Any]:
        """Return a human-readable summary of a campaign plan.

        Args:
            plan: A :class:`CampaignPlan`.

        Returns:
            Dict with keys: ``title_count``, ``type_distribution``,
            ``dominant_type``, ``selected_template_variant``,
            ``settings_overrides``.
        """
        # Type distribution
        type_counts: Dict[str, int] = {}
        for article_type in plan.title_types.values():
            type_counts[article_type] = type_counts.get(article_type, 0) + 1

        # Identify which template variant was selected (first 80 chars)
        template_preview = plan.outline_template[:80].replace("\n", " | ")
        if len(plan.outline_template) > 80:
            template_preview += "..."

        return {
            "title_count": len(plan.titles),
            "type_distribution": type_counts,
            "dominant_type": plan.dominant_type,
            "selected_template_variant": template_preview,
            "settings_overrides": plan.settings_overrides,
        }
