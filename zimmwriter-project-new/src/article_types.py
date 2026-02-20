"""
Article type detection from titles.

Regex-based classifier that identifies what kind of article a title describes.
Used by the campaign engine to select appropriate outlines, section lengths,
and feature settings per article.

Article types (checked in priority order):
  how_to        - Step-by-step instructions ("How to...", "DIY", "Tutorial")
  guide         - Comprehensive reference ("Complete Guide", "Ultimate Guide")
  review        - Product evaluation ("Review", "vs", "Tested", "Comparison")
  listicle      - Numbered lists ("10 Best...", "Top 5...", "Ways to...")
  news          - Timely coverage ("Launches", "Announces", "Update", year refs)
  informational - Explanatory ("What is", "Explained", "Why") — also the fallback

Priority order matters: "Ultimate Guide to 10 Best..." matches GUIDE, not LISTICLE.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple


@dataclass
class ArticleType:
    """An article type with compiled regex patterns and ZimmWriter settings overrides."""

    name: str
    patterns: List[Pattern[str]] = field(default_factory=list)
    settings_overrides: Dict = field(default_factory=dict)


def _compile(patterns: List[str]) -> List[Pattern[str]]:
    """Compile a list of raw regex strings into case-insensitive patterns."""
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# ═══════════════════════════════════════════
# ARTICLE TYPE REGISTRY (priority order)
# ═══════════════════════════════════════════
# The dict is ordered — first match wins during classification.

ARTICLE_TYPES: Dict[str, ArticleType] = {
    "how_to": ArticleType(
        name="how_to",
        patterns=_compile([
            r"^How to\b",
            r"\bStep[\s-]by[\s-]Step\b",
            r"^DIY\b",
            r"\bTutorial\b",
            r"\bSetup Guide\b",
            r"\bInstall(?:ing|ation)?\b.*\b(?:Guide|Steps?)\b",
            r"\bSet(?:ting)?\s+Up\b",
            r"\bBuild(?:ing)?\s+(?:a|your|the)\b",
            r"\bMake\s+(?:a|your)\b",
            r"\bCreate\s+(?:a|your)\b.*\b(?:in|with|using)\b",
            r"\bConfigure\s+(?:a|your)\b",
        ]),
        settings_overrides={
            "h2_lower_limit": 6,
            "h2_upper_limit": 8,
            "section_length": "Medium",
            "faq": "FAQ + Long Answers",
            "outline_hint": "step-by-step structure with materials/tools list",
        },
    ),

    "guide": ArticleType(
        name="guide",
        patterns=_compile([
            r"\b(?:Complete|Ultimate|Definitive|Comprehensive|Essential)\s+Guide\b",
            r"\bGuide\s+to\b",
            r"\bEverything\s+You\s+Need\s+to\s+Know\b",
            r"\bBeginner'?s?\s+Guide\b",
            r"\bMaster(?:ing)?\s+(?:the\s+)?(?:Art|Basics|Fundamentals)\b",
            r"\bA[\s-]to[\s-]Z\b",
            r"\b(?:In[\s-]Depth|Deep[\s-]Dive)\b.*\bGuide\b",
            r"\bPillar\s+(?:Content|Post|Article)\b",
        ]),
        settings_overrides={
            "h2_lower_limit": 8,
            "h2_upper_limit": 12,
            "section_length": "Long",
            "faq": "FAQ + Long Answers",
            "outline_hint": "deep coverage with subtopics, examples, and resources",
        },
    ),

    "review": ArticleType(
        name="review",
        patterns=_compile([
            r"\bReviews?\b",
            r"\bvs\.?\s",
            r"\bVersus\b",
            r"\bCompar(?:ison|e|ed|ing)\b",
            r"\bAlternatives?\b",
            r"\bPros?\s+(?:and|&)\s+Cons?\b",
            r"\bTested\b",
            r"\bHands[\s-]On\b",
            r"\bUnboxing\b",
            r"\bWorth\s+(?:It|Buying|the\s+(?:Money|Price|Hype))\b",
            r"\bBetter\s+Than\b",
            r"\bBuying\s+Guide\b",
        ]),
        settings_overrides={
            "h2_lower_limit": 6,
            "h2_upper_limit": 9,
            "section_length": "Medium",
            "faq": "FAQ + Long Answers",
            "outline_hint": "spec table, testing results, pros/cons, alternatives",
        },
    ),

    "listicle": ArticleType(
        name="listicle",
        patterns=_compile([
            r"^\d+\s+(?:Best|Top|Essential|Must|Amazing|Great|Easy|Simple|Quick|Creative|Fun|Unique|Cool|Cheap|Free|Affordable|Popular)\b",
            r"^(?:Best|Top)\s+\d+\b",
            r"\b\d+\s+(?:Ways|Tips|Tricks|Ideas|Reasons|Things|Steps|Hacks|Tools|Apps|Products|Gadgets|Devices|Picks|Options|Examples|Mistakes|Myths|Facts|Secrets|Signs|Benefits|Strategies|Features|Recipes|Methods|Techniques|Exercises)\b",
            r"\b(?:Ways|Tips|Tricks|Ideas|Reasons)\s+to\b",
            r"^The\s+\d+\b",
        ]),
        settings_overrides={
            "h2_lower_limit": 5,
            "h2_upper_limit": 7,
            "section_length": "Short",
            "faq": "FAQ + Short Answers",
            "outline_hint": "numbered items with brief descriptions",
        },
    ),

    "news": ArticleType(
        name="news",
        patterns=_compile([
            r"\b(?:Launches|Launched|Announces|Announced|Reveals|Revealed|Introduces|Introduced|Releases|Released|Unveils|Unveiled)\b",
            r"\b(?:Breaking|Latest)\b.*\b(?:Feature|Update|Release|Version|Model)\b",
            r"\b20(?:2[4-9]|3[0-9])\b.*\b(?:Update|Release|Launch|Announcement)\b",
            r"\b(?:Update|Release|Version)\s+\d",
            r"\bFirst\s+Look\b",
            r"\bNow\s+Available\b",
            r"^New:\s",
        ]),
        settings_overrides={
            "h2_lower_limit": 4,
            "h2_upper_limit": 6,
            "section_length": "Short",
            "faq": "No FAQ",
            "outline_hint": "timely coverage: what, why it matters, what's next",
        },
    ),

    "informational": ArticleType(
        name="informational",
        patterns=_compile([
            r"^What\s+(?:Is|Are)\b",
            r"\bExplained\b",
            r"^Why\s+",
            r"^When\s+(?:to|Should)\b",
            r"\bUnderstanding\b",
            r"^(?:The\s+)?(?:History|Science|Psychology|Meaning)\s+(?:of|Behind)\b",
            r"\b(?:Difference|Differences)\s+Between\b",
            r"\bDefined\b",
            r"^Is\s+.*\?$",
            r"^(?:Can|Should|Does|Do|Will)\s+",
        ]),
        settings_overrides={
            "h2_lower_limit": 5,
            "h2_upper_limit": 8,
            "section_length": "Medium",
            "faq": "FAQ + Long Answers",
            "outline_hint": "clear explanation with examples and practical applications",
        },
    ),
}

_DEFAULT_TYPE = "informational"


# ═══════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════


def classify_title(title: str) -> str:
    """Classify a single article title into an article type.

    Returns the first matching type from priority-ordered patterns.
    Falls back to "informational" if no pattern matches.

    >>> classify_title("How to Set Up a Smart Home Hub")
    'how_to'
    >>> classify_title("10 Best AI Writing Tools for 2025")
    'listicle'
    >>> classify_title("Samsung Galaxy Ring Review: 30 Days Later")
    'review'
    """
    title = title.strip()
    if not title:
        return _DEFAULT_TYPE

    for type_name, article_type in ARTICLE_TYPES.items():
        for pattern in article_type.patterns:
            if pattern.search(title):
                return type_name
    return _DEFAULT_TYPE


def classify_titles(titles: List[str]) -> Dict[str, str]:
    """Classify multiple titles. Returns dict mapping title -> type name."""
    return {title: classify_title(title) for title in titles}


def get_settings_overrides(article_type: str) -> Dict:
    """Return ZimmWriter settings overrides for an article type."""
    if article_type in ARTICLE_TYPES:
        return dict(ARTICLE_TYPES[article_type].settings_overrides)
    return dict(ARTICLE_TYPES[_DEFAULT_TYPE].settings_overrides)


def get_dominant_type(titles: List[str]) -> str:
    """Return the most common article type across a list of titles.

    Useful for choosing a single outline template for a mixed batch.
    Ties broken by priority order (earlier type wins).
    """
    if not titles:
        return _DEFAULT_TYPE
    classifications = [classify_title(title) for title in titles]
    counter = Counter(classifications)

    # Priority: earlier in ARTICLE_TYPES wins ties
    priority = {name: i for i, name in enumerate(ARTICLE_TYPES)}
    return max(
        counter.keys(),
        key=lambda t: (counter[t], -priority.get(t, 99))
    )


def classify_with_settings(title: str) -> Dict:
    """Classify a title and return type + settings overrides combined."""
    article_type = classify_title(title)
    settings = get_settings_overrides(article_type)
    return {"title": title, "type": article_type, **settings}


def batch_analysis(titles: List[str]) -> Dict:
    """Full batch analysis for campaign engine.

    Returns:
        {
            "titles": [{"title": str, "type": str}, ...],
            "dominant_type": str,
            "type_counts": {"how_to": 3, "listicle": 5, ...},
            "settings": {...}
        }
    """
    classified = [(t, classify_title(t)) for t in titles]
    counts = dict(Counter(t for _, t in classified))
    dominant = get_dominant_type(titles)

    return {
        "titles": [{"title": t, "type": at} for t, at in classified],
        "dominant_type": dominant,
        "type_counts": counts,
        "settings": get_settings_overrides(dominant),
    }
