"""
OCR Data Extraction Pipeline — OpenClaw Empire Edition

Structured data extraction from app dashboards and analytics screens.
Turns screenshots of Google AdSense, Analytics, Search Console, Amazon
Associates, Etsy, KDP, WordPress, and social media dashboards into clean
JSON data suitable for revenue_tracker.py and downstream analytics.

Pipeline:
    Screenshot -> Crop (optional) -> Claude Haiku OCR -> Parse -> Validate
    -> Store -> (optional) Push to revenue_tracker

All data persisted to: data/ocr_extractions/

Usage:
    from src.ocr_extractor import get_extractor

    extractor = get_extractor()
    result = await extractor.extract_from_screenshot(
        image_path="/tmp/adsense-dashboard.png",
        app_name="Google AdSense",
        extraction_type="adsense",
    )
    print(result.structured_data)

Sync usage:
    result = extractor.extract_from_screenshot_sync(
        image_path="/tmp/adsense-dashboard.png",
        app_name="Google AdSense",
        extraction_type="adsense",
    )
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("ocr_extractor")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

EXTRACTION_DATA_DIR = BASE_DIR / "data" / "ocr_extractions"
RESULTS_DIR = EXTRACTION_DATA_DIR / "results"
SCHEDULES_FILE = EXTRACTION_DATA_DIR / "schedules.json"
CORRECTIONS_FILE = EXTRACTION_DATA_DIR / "corrections.json"
CONFIG_FILE = EXTRACTION_DATA_DIR / "config.json"

# Ensure directories exist on import
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAIKU_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 2000
CONFIDENCE_THRESHOLD = 0.7
ANOMALY_DEVIATION_PCT = 0.50  # 50% deviation from historical average

# Anthropic API base URL
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# All known extraction types
EXTRACTION_TYPES = [
    "adsense", "analytics", "search_console", "amazon_associates",
    "etsy", "kdp", "wordpress", "instagram", "tiktok", "pinterest",
    "twitter", "generic",
]

MAX_RESULT_HISTORY = 5000
MAX_SCHEDULE_HISTORY = 200

# ---------------------------------------------------------------------------
# JSON helpers (atomic writes)
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when the file is missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(str(tmp), str(path))


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _today_iso() -> str:
    return _now_utc().strftime("%Y-%m-%d")


def _gen_id() -> str:
    """Generate a short unique extraction ID."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Number / text parsing utilities
# ---------------------------------------------------------------------------


def parse_number(text: str) -> Optional[float]:
    """Parse human-readable numbers: '1.2K', '45.3M', '$1,234.56', '12%', etc.

    Returns the numeric value as a float, or None if unparseable.
    Strips currency symbols, commas, whitespace, and percentage signs.
    Handles suffixes K (thousand), M (million), B (billion), T (trillion).
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()
    # Remove currency symbols
    cleaned = re.sub(r'[$\u00a3\u20ac\u00a5]', '', cleaned)
    # Remove commas
    cleaned = cleaned.replace(',', '')
    # Remove percentage sign (caller decides if it's a percentage)
    cleaned = cleaned.replace('%', '')
    # Remove whitespace
    cleaned = cleaned.strip()

    if not cleaned:
        return None

    # Check for suffix multipliers
    multiplier = 1.0
    suffix_map = {
        'k': 1_000, 'K': 1_000,
        'm': 1_000_000, 'M': 1_000_000,
        'b': 1_000_000_000, 'B': 1_000_000_000,
        't': 1_000_000_000_000, 'T': 1_000_000_000_000,
    }

    if cleaned and cleaned[-1] in suffix_map:
        multiplier = suffix_map[cleaned[-1]]
        cleaned = cleaned[:-1]

    # Handle parenthetical negatives: (123.45) -> -123.45
    neg = False
    if cleaned.startswith('(') and cleaned.endswith(')'):
        neg = True
        cleaned = cleaned[1:-1]
    elif cleaned.startswith('-'):
        neg = True
        cleaned = cleaned[1:]

    try:
        value = float(cleaned) * multiplier
        return -value if neg else value
    except (ValueError, TypeError):
        return None


def parse_currency(text: str) -> Optional[float]:
    """Extract a currency amount from text.

    Handles: $1,234.56, USD 1234.56, 1,234.56 USD, EUR 50.00, etc.
    Returns the numeric value or None.
    """
    if not text or not isinstance(text, str):
        return None

    # Try to find a currency pattern
    patterns = [
        r'[\$\u00a3\u20ac\u00a5]\s*([\d,]+\.?\d*)',  # $1,234.56
        r'([\d,]+\.?\d*)\s*(?:USD|EUR|GBP|JPY)',       # 1234.56 USD
        r'(?:USD|EUR|GBP|JPY)\s*([\d,]+\.?\d*)',       # USD 1234.56
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_number(match.group(1))

    # Fallback: try parsing the whole thing
    return parse_number(text)


def parse_percentage(text: str) -> Optional[float]:
    """Extract a percentage value from text.

    Handles: '12.5%', '12.5 %', '12.5 percent', etc.
    Returns the numeric percentage value (e.g. 12.5, not 0.125).
    """
    if not text or not isinstance(text, str):
        return None

    match = re.search(r'([\d.]+)\s*(?:%|percent)', text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    # If text itself looks like just a number with %, try parse_number
    if '%' in text:
        return parse_number(text)

    return None


def parse_date_range(text: str) -> Optional[Dict[str, str]]:
    """Extract date range from dashboard header text.

    Handles formats like:
        'Jan 1 - Jan 31, 2026'
        '2026-01-01 to 2026-01-31'
        'Last 7 days'
        'Last 28 days'
        'Last 3 months'

    Returns dict with 'start' and 'end' ISO date strings, or None.
    """
    if not text or not isinstance(text, str):
        return None

    today = _now_utc().date()

    # 'Last N days' pattern
    match = re.search(r'[Ll]ast\s+(\d+)\s+days?', text)
    if match:
        n = int(match.group(1))
        end = today
        start = today - timedelta(days=n)
        return {"start": start.isoformat(), "end": end.isoformat()}

    # 'Last N months' pattern
    match = re.search(r'[Ll]ast\s+(\d+)\s+months?', text)
    if match:
        n = int(match.group(1))
        end = today
        start_month = today.month - n
        start_year = today.year
        while start_month <= 0:
            start_month += 12
            start_year -= 1
        start = today.replace(year=start_year, month=start_month, day=1)
        return {"start": start.isoformat(), "end": end.isoformat()}

    # ISO date range: '2026-01-01 to 2026-01-31' or '2026-01-01 - 2026-01-31'
    match = re.search(
        r'(\d{4}-\d{2}-\d{2})\s*(?:to|-|through)\s*(\d{4}-\d{2}-\d{2})',
        text,
    )
    if match:
        return {"start": match.group(1), "end": match.group(2)}

    # Month name ranges: 'Jan 1 - Jan 31, 2026' or 'January 1, 2026 - January 31, 2026'
    month_names = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'june': 6, 'july': 7, 'august': 8, 'september': 9,
        'october': 10, 'november': 11, 'december': 12,
    }
    match = re.search(
        r'(\w+)\s+(\d{1,2})(?:,?\s*(\d{4}))?\s*[-\u2013]\s*(\w+)\s+(\d{1,2})(?:,?\s*(\d{4}))?',
        text,
    )
    if match:
        m1 = month_names.get(match.group(1).lower())
        d1 = int(match.group(2))
        y1 = int(match.group(3)) if match.group(3) else today.year
        m2 = month_names.get(match.group(4).lower())
        d2 = int(match.group(5))
        y2 = int(match.group(6)) if match.group(6) else y1
        if m1 and m2:
            try:
                start = date(y1, m1, d1)
                end = date(y2, m2, d2)
                return {"start": start.isoformat(), "end": end.isoformat()}
            except ValueError:
                pass

    return None


def parse_table(raw_text: str, columns: List[str]) -> List[Dict[str, str]]:
    """Parse tabular data from OCR text into a list of row dicts.

    Attempts to align values with the provided column names.
    Handles both tab-separated and space-aligned layouts.

    Args:
        raw_text: The raw OCR text containing a table.
        columns: Expected column headers.

    Returns:
        List of dicts, one per row, keyed by column name.
    """
    if not raw_text or not columns:
        return []

    rows: List[Dict[str, str]] = []
    lines = raw_text.strip().split('\n')

    # Skip header-like lines (lines matching column names)
    data_lines: List[str] = []
    for line in lines:
        line_lower = line.lower().strip()
        # Skip empty lines and lines that look like headers
        if not line_lower:
            continue
        is_header = sum(1 for col in columns if col.lower() in line_lower) >= len(columns) // 2
        if is_header:
            continue
        # Skip separator lines
        if set(line_lower) <= {'-', '=', '+', '|', ' '}:
            continue
        data_lines.append(line)

    for line in data_lines:
        # Try tab-separated first
        if '\t' in line:
            values = [v.strip() for v in line.split('\t')]
        else:
            # Fall back to splitting by 2+ spaces
            values = [v.strip() for v in re.split(r'\s{2,}', line.strip())]

        if not values:
            continue

        row: Dict[str, str] = {}
        for i, col in enumerate(columns):
            if i < len(values):
                row[col] = values[i]
            else:
                row[col] = ""
        rows.append(row)

    return rows


def clean_ocr_text(text: str) -> str:
    """Fix common OCR misreadings.

    Corrects typical character confusion: 0/O, 1/l/I, S/5, etc.
    Normalizes whitespace and removes stray artifacts.
    """
    if not text:
        return ""

    result = text

    # Normalize various dash and space characters
    result = result.replace('\u2013', '-')   # en-dash
    result = result.replace('\u2014', '-')   # em-dash
    result = result.replace('\u00a0', ' ')   # non-breaking space
    result = result.replace('\u200b', '')    # zero-width space

    # Fix common OCR artifacts
    result = re.sub(r'\|(?=\d)', '1', result)  # pipe before digit -> 1
    result = re.sub(r'(?<=\d)O(?=\d)', '0', result)  # O between digits -> 0
    result = re.sub(r'(?<=\d)o(?=\d)', '0', result)  # o between digits -> 0
    result = re.sub(r'(?<=\$)O', '0', result)  # $O -> $0
    result = re.sub(r'(?<=\$)l', '1', result)  # $l -> $1
    result = re.sub(r'(?<=\d)l(?=\d)', '1', result)  # l between digits -> 1
    result = re.sub(r'(?<=\d)I(?=\d)', '1', result)  # I between digits -> 1
    result = re.sub(r'(?<=\d)S(?=\d)', '5', result)  # S between digits -> 5

    # Clean up multiple spaces
    result = re.sub(r' {2,}', ' ', result)
    # Clean up multiple newlines
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


# ===================================================================
# Extraction Templates (system prompts for Claude Haiku)
# ===================================================================

class ExtractionTemplate:
    """Pre-built extraction prompt templates for various dashboard types.

    Each template includes:
        - A system prompt for Claude Haiku describing what to extract
        - The expected output schema
        - A validator for the structured output
    """

    ADSENSE = {
        "name": "Google AdSense",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of a Google AdSense dashboard. Extract the following metrics as JSON:\n\n"
            "1. Earnings: today, yesterday, last_7_days, last_28_days, this_month\n"
            "2. Metrics: page_views, clicks, page_rpm, cpc, impressions\n"
            "3. Date range visible on screen\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "earnings": {\n'
            '    "today": <float>,\n'
            '    "yesterday": <float>,\n'
            '    "last_7_days": <float>,\n'
            '    "last_28_days": <float>,\n'
            '    "this_month": <float>\n'
            "  },\n"
            '  "metrics": {\n'
            '    "page_views": <int>,\n'
            '    "clicks": <int>,\n'
            '    "page_rpm": <float>,\n'
            '    "cpc": <float>,\n'
            '    "impressions": <int>\n'
            "  },\n"
            '  "date_range": <string or null>,\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "If a metric is not visible, use null. Use raw numbers, not formatted strings."
        ),
        "expected_fields": [
            "earnings.today", "earnings.yesterday", "earnings.last_7_days",
            "earnings.last_28_days", "earnings.this_month",
            "metrics.page_views", "metrics.clicks", "metrics.page_rpm",
            "metrics.cpc", "metrics.impressions",
        ],
    }

    ANALYTICS = {
        "name": "Google Analytics",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of a Google Analytics dashboard. Extract the following metrics as JSON:\n\n"
            "1. Active users (real-time or period)\n"
            "2. Sessions (total for visible period)\n"
            "3. Bounce rate\n"
            "4. Average session duration\n"
            "5. Top pages (up to 10, with pageviews)\n"
            "6. Traffic sources breakdown: organic, direct, social, referral\n"
            "7. Date range visible on screen\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "active_users": <int>,\n'
            '  "sessions": <int>,\n'
            '  "bounce_rate": <float as percentage>,\n'
            '  "avg_session_duration": <string "M:SS" or seconds as float>,\n'
            '  "top_pages": [{"page": <string>, "pageviews": <int>}],\n'
            '  "traffic_sources": {\n'
            '    "organic": <int or float>,\n'
            '    "direct": <int or float>,\n'
            '    "social": <int or float>,\n'
            '    "referral": <int or float>\n'
            "  },\n"
            '  "date_range": <string or null>,\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "If a metric is not visible, use null. Use raw numbers, not formatted strings."
        ),
        "expected_fields": [
            "active_users", "sessions", "bounce_rate", "avg_session_duration",
            "top_pages", "traffic_sources",
        ],
    }

    SEARCH_CONSOLE = {
        "name": "Google Search Console",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of a Google Search Console dashboard. Extract the following metrics as JSON:\n\n"
            "1. Total clicks\n"
            "2. Total impressions\n"
            "3. Average CTR (click-through rate)\n"
            "4. Average position\n"
            "5. Top queries (up to 10, with clicks and impressions)\n"
            "6. Top pages (up to 10, with clicks and impressions)\n"
            "7. Date range visible on screen\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "total_clicks": <int>,\n'
            '  "total_impressions": <int>,\n'
            '  "avg_ctr": <float as percentage>,\n'
            '  "avg_position": <float>,\n'
            '  "top_queries": [{"query": <string>, "clicks": <int>, "impressions": <int>}],\n'
            '  "top_pages": [{"page": <string>, "clicks": <int>, "impressions": <int>}],\n'
            '  "date_range": <string or null>,\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "If a metric is not visible, use null."
        ),
        "expected_fields": [
            "total_clicks", "total_impressions", "avg_ctr", "avg_position",
            "top_queries", "top_pages",
        ],
    }

    AMAZON_ASSOCIATES = {
        "name": "Amazon Associates",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of an Amazon Associates dashboard. Extract the following metrics as JSON:\n\n"
            "1. Clicks\n"
            "2. Ordered items\n"
            "3. Shipped items\n"
            "4. Conversion rate\n"
            "5. Total earnings\n"
            "6. By-product breakdown (up to 10 items with product name, clicks, orders, earnings)\n"
            "7. Date range visible on screen\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "clicks": <int>,\n'
            '  "ordered_items": <int>,\n'
            '  "shipped_items": <int>,\n'
            '  "conversion_rate": <float as percentage>,\n'
            '  "total_earnings": <float>,\n'
            '  "by_product": [{"product": <string>, "clicks": <int>, "orders": <int>, "earnings": <float>}],\n'
            '  "date_range": <string or null>,\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "If a metric is not visible, use null."
        ),
        "expected_fields": [
            "clicks", "ordered_items", "shipped_items", "conversion_rate",
            "total_earnings",
        ],
    }

    ETSY = {
        "name": "Etsy Dashboard",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of an Etsy seller dashboard. Extract the following metrics as JSON:\n\n"
            "1. Views\n"
            "2. Visits\n"
            "3. Orders\n"
            "4. Revenue\n"
            "5. Conversion rate\n"
            "6. Favorites\n"
            "7. By-listing breakdown (up to 10 items with title, views, orders, revenue)\n"
            "8. Date range visible on screen\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "views": <int>,\n'
            '  "visits": <int>,\n'
            '  "orders": <int>,\n'
            '  "revenue": <float>,\n'
            '  "conversion_rate": <float as percentage>,\n'
            '  "favorites": <int>,\n'
            '  "by_listing": [{"title": <string>, "views": <int>, "orders": <int>, "revenue": <float>}],\n'
            '  "date_range": <string or null>,\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "If a metric is not visible, use null."
        ),
        "expected_fields": [
            "views", "visits", "orders", "revenue", "conversion_rate", "favorites",
        ],
    }

    KDP = {
        "name": "KDP Reports",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of a Kindle Direct Publishing (KDP) reports dashboard. Extract the following:\n\n"
            "1. Units sold (ebook, paperback separately if visible)\n"
            "2. Royalties earned\n"
            "3. Pages read (Kindle Unlimited / KENP)\n"
            "4. By-book breakdown (up to 10 books with title, units, royalties, pages_read)\n"
            "5. By-marketplace breakdown if visible (US, UK, DE, etc.)\n"
            "6. Date range visible on screen\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "units_sold": {"ebook": <int>, "paperback": <int>, "total": <int>},\n'
            '  "royalties": <float>,\n'
            '  "pages_read": <int>,\n'
            '  "by_book": [{"title": <string>, "units": <int>, "royalties": <float>, "pages_read": <int>}],\n'
            '  "by_marketplace": [{"marketplace": <string>, "units": <int>, "royalties": <float>}],\n'
            '  "date_range": <string or null>,\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "If a metric is not visible, use null."
        ),
        "expected_fields": [
            "units_sold", "royalties", "pages_read",
        ],
    }

    WORDPRESS = {
        "name": "WordPress Dashboard",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of a WordPress admin dashboard. Extract the following metrics as JSON:\n\n"
            "1. Posts count (total published)\n"
            "2. Pages count\n"
            "3. Comments pending moderation\n"
            "4. Recent drafts (up to 5 with title)\n"
            "5. Popular posts if visible (up to 5 with title, views)\n"
            "6. RankMath SEO scores if visible\n"
            "7. Any plugin notifications or warnings\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "posts_count": <int>,\n'
            '  "pages_count": <int>,\n'
            '  "comments_pending": <int>,\n'
            '  "recent_drafts": [{"title": <string>, "date": <string or null>}],\n'
            '  "popular_posts": [{"title": <string>, "views": <int>}],\n'
            '  "seo_scores": [{"post": <string>, "score": <int>}],\n'
            '  "notifications": [<string>],\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "If a metric is not visible, use null."
        ),
        "expected_fields": [
            "posts_count", "pages_count", "comments_pending",
        ],
    }

    INSTAGRAM = {
        "name": "Instagram Insights",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of Instagram Insights or profile. Extract the following metrics as JSON:\n\n"
            "1. Followers count\n"
            "2. Following count\n"
            "3. Posts count\n"
            "4. Reach (if visible in insights)\n"
            "5. Impressions\n"
            "6. Profile visits\n"
            "7. Date range if visible\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "followers": <int>,\n'
            '  "following": <int>,\n'
            '  "posts_count": <int>,\n'
            '  "reach": <int or null>,\n'
            '  "impressions": <int or null>,\n'
            '  "profile_visits": <int or null>,\n'
            '  "date_range": <string or null>,\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "If a metric is not visible, use null."
        ),
        "expected_fields": ["followers", "following", "posts_count"],
    }

    TIKTOK = {
        "name": "TikTok Analytics",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of a TikTok analytics or profile page. Extract the following metrics as JSON:\n\n"
            "1. Followers count\n"
            "2. Total likes\n"
            "3. Total views (video views)\n"
            "4. Profile views\n"
            "5. Date range if visible\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "followers": <int>,\n'
            '  "likes": <int>,\n'
            '  "views": <int>,\n'
            '  "profile_views": <int or null>,\n'
            '  "date_range": <string or null>,\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "If a metric is not visible, use null."
        ),
        "expected_fields": ["followers", "likes", "views"],
    }

    PINTEREST = {
        "name": "Pinterest Analytics",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of a Pinterest Analytics dashboard. Extract the following metrics as JSON:\n\n"
            "1. Monthly views\n"
            "2. Followers\n"
            "3. Impressions\n"
            "4. Saves (pin saves)\n"
            "5. Outbound clicks\n"
            "6. Date range if visible\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "monthly_views": <int>,\n'
            '  "followers": <int>,\n'
            '  "impressions": <int>,\n'
            '  "saves": <int>,\n'
            '  "outbound_clicks": <int>,\n'
            '  "date_range": <string or null>,\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "If a metric is not visible, use null."
        ),
        "expected_fields": ["monthly_views", "followers", "impressions", "saves", "outbound_clicks"],
    }

    TWITTER = {
        "name": "Twitter/X Analytics",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of a Twitter/X analytics dashboard or profile. Extract the following:\n\n"
            "1. Followers count\n"
            "2. Impressions (if in analytics view)\n"
            "3. Profile visits\n"
            "4. Mentions\n"
            "5. Date range if visible\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "followers": <int>,\n'
            '  "impressions": <int or null>,\n'
            '  "profile_visits": <int or null>,\n'
            '  "mentions": <int or null>,\n'
            '  "date_range": <string or null>,\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "If a metric is not visible, use null."
        ),
        "expected_fields": ["followers"],
    }

    GENERIC = {
        "name": "Generic Dashboard",
        "system_prompt": (
            "You are a precise data extraction agent. You are looking at a screenshot "
            "of some kind of analytics dashboard or app screen. Extract ALL visible "
            "numeric metrics, labels, and data points.\n\n"
            "Return ONLY valid JSON with this structure:\n"
            "{\n"
            '  "metrics": {<label>: <value>, ...},\n'
            '  "tables": [{<column>: <value>, ...}, ...],\n'
            '  "text_content": [<visible text strings>],\n'
            '  "app_name": <detected app name or null>,\n'
            '  "date_range": <string or null>,\n'
            '  "confidence": <float 0-1>\n'
            "}\n\n"
            "Extract all numbers with their labels. Be thorough."
        ),
        "expected_fields": ["metrics"],
    }

    _REGISTRY: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get(cls, extraction_type: str) -> Dict[str, Any]:
        """Get the extraction template for a given type."""
        # Initialize registry on first access
        if not cls._REGISTRY:
            cls._REGISTRY = {
                "adsense": cls.ADSENSE,
                "analytics": cls.ANALYTICS,
                "search_console": cls.SEARCH_CONSOLE,
                "amazon_associates": cls.AMAZON_ASSOCIATES,
                "etsy": cls.ETSY,
                "kdp": cls.KDP,
                "wordpress": cls.WORDPRESS,
                "instagram": cls.INSTAGRAM,
                "tiktok": cls.TIKTOK,
                "pinterest": cls.PINTEREST,
                "twitter": cls.TWITTER,
                "generic": cls.GENERIC,
            }

        template = cls._REGISTRY.get(extraction_type.lower())
        if template is None:
            logger.warning("Unknown extraction type '%s', falling back to generic", extraction_type)
            return cls.GENERIC
        return template

    @classmethod
    def list_types(cls) -> List[Dict[str, str]]:
        """List all available extraction types with their names."""
        # Force initialization
        cls.get("generic")
        return [
            {"type": k, "name": v["name"]}
            for k, v in cls._REGISTRY.items()
        ]


# ===================================================================
# Data Classes
# ===================================================================


@dataclass
class ExtractionResult:
    """Result of a single OCR extraction."""
    extraction_id: str
    source_app: str
    extraction_type: str
    raw_text: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    screenshot_path: str = ""
    timestamp: str = ""
    duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.extraction_id:
            self.extraction_id = _gen_id()
        if not self.timestamp:
            self.timestamp = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractionResult:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExtractionSchedule:
    """A scheduled recurring extraction."""
    schedule_id: str
    app_name: str
    extraction_type: str
    device_id: str = ""
    cron_expr: str = ""
    last_run: str = ""
    next_run: str = ""
    enabled: bool = True
    results_count: int = 0

    def __post_init__(self) -> None:
        if not self.schedule_id:
            self.schedule_id = _gen_id()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractionSchedule:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AnomalyFlag:
    """A flagged value that deviates significantly from historical average."""
    metric_path: str
    current_value: float
    historical_avg: float
    deviation_pct: float
    severity: str = "warning"  # "info", "warning", "critical"
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrendPoint:
    """A single data point in a trend analysis."""
    date: str
    value: float
    extraction_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrendAnalysis:
    """Trend analysis result for a specific metric over time."""
    app_name: str
    metric_path: str
    period_days: int
    data_points: List[TrendPoint] = field(default_factory=list)
    current_value: float = 0.0
    average: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    trend_direction: str = "stable"  # "up", "down", "stable"
    change_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["data_points"] = [p if isinstance(p, dict) else asdict(p) for p in self.data_points]
        return d


@dataclass
class PeriodComparison:
    """Period-over-period comparison result."""
    app_name: str
    extraction_type: str
    period1: Dict[str, str] = field(default_factory=dict)  # {"start": ..., "end": ...}
    period2: Dict[str, str] = field(default_factory=dict)
    period1_data: Dict[str, Any] = field(default_factory=dict)
    period2_data: Dict[str, Any] = field(default_factory=dict)
    changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===================================================================
# Anthropic API Client (lightweight, Haiku-only)
# ===================================================================


class _HaikuClient:
    """Minimal Anthropic API client optimized for OCR extraction via Claude Haiku.

    Uses prompt caching on system prompts exceeding 2048 tokens.
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set. OCR extraction will fail. "
                "Set the environment variable before calling extraction methods."
            )

    async def extract(
        self,
        system_prompt: str,
        image_b64: str,
        user_prompt: str = "Extract the data from this screenshot.",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Dict[str, Any]:
        """Send an image to Claude Haiku and get structured JSON back.

        Args:
            system_prompt: The extraction template system prompt.
            image_b64: Base64-encoded screenshot.
            user_prompt: Additional user-level instruction.
            max_tokens: Maximum response tokens.

        Returns:
            Parsed JSON dict from Haiku's response.

        Raises:
            RuntimeError: On API errors or unparseable responses.
        """
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Cannot call Claude Haiku for OCR extraction."
            )

        # Build system message with cache control for large prompts
        system_message: Any
        if len(system_prompt) > 2048:
            system_message = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_message = system_prompt

        payload = {
            "model": HAIKU_MODEL,
            "max_tokens": max_tokens,
            "system": system_message,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                }
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ANTHROPIC_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(
                            f"Anthropic API {resp.status}: {body[:500]}"
                        )
                    data = await resp.json()
        except ImportError:
            # Fallback to httpx if aiohttp not available
            import httpx

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    ANTHROPIC_API_URL,
                    json=payload,
                    headers=headers,
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Anthropic API {resp.status_code}: {resp.text[:500]}"
                    )
                data = resp.json()

        # Extract text from response
        content_blocks = data.get("content", [])
        text_response = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text_response += block.get("text", "")

        if not text_response:
            raise RuntimeError("Empty response from Claude Haiku")

        # Parse JSON from response (handle markdown code blocks)
        json_text = text_response.strip()
        if json_text.startswith("```"):
            # Strip markdown code fences
            lines = json_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            json_text = "\n".join(lines)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as exc:
            # Try to find JSON within the response
            json_match = re.search(r'\{[\s\S]*\}', json_text)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            logger.error("Failed to parse Haiku response as JSON: %s", json_text[:300])
            raise RuntimeError(f"Unparseable JSON from Haiku: {exc}") from exc


# ===================================================================
# OCRExtractor — Main Class
# ===================================================================


class OCRExtractor:
    """
    Structured OCR data extraction pipeline for analytics dashboards.

    Sends screenshots to Claude Haiku with extraction-type-specific prompts,
    parses the structured JSON response, validates against expected schemas,
    persists results, and provides trend analysis and anomaly detection.

    Features:
        - 12 extraction templates (AdSense, Analytics, GSC, Amazon, Etsy,
          KDP, WordPress, Instagram, TikTok, Pinterest, Twitter, Generic)
        - Scheduled recurring extractions
        - Historical data storage and trend analysis
        - Period-over-period comparisons
        - Anomaly detection
        - CSV export
        - Revenue tracker integration
    """

    def __init__(self) -> None:
        self._client = _HaikuClient()
        self._config = self._load_config()
        self._schedules: Dict[str, ExtractionSchedule] = self._load_schedules()
        logger.info(
            "OCRExtractor initialized — data dir: %s, schedules: %d",
            EXTRACTION_DATA_DIR, len(self._schedules),
        )

    # ------------------------------------------------------------------
    # Config / persistence helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        defaults = {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "anomaly_deviation_pct": ANOMALY_DEVIATION_PCT,
            "max_result_history": MAX_RESULT_HISTORY,
            "default_max_tokens": DEFAULT_MAX_TOKENS,
        }
        config = _load_json(CONFIG_FILE, defaults.copy())
        for k, v in defaults.items():
            if k not in config:
                config[k] = v
        return config

    def _save_config(self) -> None:
        _save_json(CONFIG_FILE, self._config)

    def _load_schedules(self) -> Dict[str, ExtractionSchedule]:
        raw = _load_json(SCHEDULES_FILE, {})
        schedules: Dict[str, ExtractionSchedule] = {}
        for sid, data in raw.items():
            try:
                schedules[sid] = ExtractionSchedule.from_dict(data)
            except (TypeError, KeyError) as exc:
                logger.warning("Skipping malformed schedule %s: %s", sid, exc)
        return schedules

    def _save_schedules(self) -> None:
        _save_json(
            SCHEDULES_FILE,
            {k: v.to_dict() for k, v in self._schedules.items()},
        )

    def _result_file(self, extraction_type: str, iso_date: str) -> Path:
        """Return path to a daily results file for a given type."""
        type_dir = RESULTS_DIR / extraction_type
        type_dir.mkdir(parents=True, exist_ok=True)
        return type_dir / f"{iso_date}.json"

    def _load_day_results(self, extraction_type: str, iso_date: str) -> List[Dict[str, Any]]:
        """Load all extraction results for a given type and date."""
        path = self._result_file(extraction_type, iso_date)
        data = _load_json(path, [])
        if isinstance(data, list):
            return data
        return []

    def _save_day_results(self, extraction_type: str, iso_date: str, results: List[Dict[str, Any]]) -> None:
        """Save extraction results for a given type and date."""
        # Trim to max
        max_results = self._config.get("max_result_history", MAX_RESULT_HISTORY)
        if len(results) > max_results:
            results = results[-max_results:]
        path = self._result_file(extraction_type, iso_date)
        _save_json(path, results)

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image_b64(image_path: str) -> str:
        """Load an image file and return its base64 encoding."""
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Screenshot not found: {image_path}")
        with open(p, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    @staticmethod
    def _crop_image_b64(image_b64: str, region: Dict[str, int]) -> str:
        """Crop a base64-encoded image to a region and return new base64.

        Args:
            image_b64: Base64-encoded source image.
            region: Dict with keys x, y, w, h.

        Returns:
            Base64-encoded cropped image.
        """
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Pillow is required for region cropping. Install: pip install Pillow")

        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes))

        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", img.width - x)
        h = region.get("h", img.height - y)

        cropped = img.crop((x, y, x + w, y + h))
        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ==================================================================
    # CORE EXTRACTION ENGINE
    # ==================================================================

    async def extract_from_screenshot(
        self,
        image_path: str,
        app_name: str,
        extraction_type: str,
        user_prompt: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract structured data from a dashboard screenshot.

        Sends the screenshot to Claude Haiku with the appropriate extraction
        template prompt, parses the JSON response, validates it, and stores
        the result.

        Args:
            image_path: Path to the screenshot file.
            app_name: Name of the application (e.g. "Google AdSense").
            extraction_type: Template key (e.g. "adsense", "analytics").
            user_prompt: Optional override for the user-level prompt.

        Returns:
            ExtractionResult with structured_data populated.
        """
        start_ms = time.monotonic() * 1000
        result = ExtractionResult(
            extraction_id=_gen_id(),
            source_app=app_name,
            extraction_type=extraction_type,
            screenshot_path=image_path,
        )

        try:
            image_b64 = self._load_image_b64(image_path)
            template = ExtractionTemplate.get(extraction_type)
            system_prompt = template["system_prompt"]
            prompt = user_prompt or "Extract the data from this screenshot. Return ONLY valid JSON."

            raw_data = await self._client.extract(
                system_prompt=system_prompt,
                image_b64=image_b64,
                user_prompt=prompt,
            )

            result.structured_data = raw_data
            result.confidence = float(raw_data.get("confidence", 0.0))
            result.raw_text = json.dumps(raw_data, indent=2)

            # Validate against expected fields
            validation_errors = self._validate_fields(raw_data, template.get("expected_fields", []))
            if validation_errors:
                result.errors.extend(validation_errors)
                logger.warning(
                    "Extraction %s has validation issues: %s",
                    result.extraction_id, validation_errors,
                )

        except Exception as exc:
            result.errors.append(str(exc))
            logger.error("Extraction failed for %s/%s: %s", app_name, extraction_type, exc)

        elapsed_ms = time.monotonic() * 1000 - start_ms
        result.duration_ms = round(elapsed_ms, 1)

        # Persist result
        self.store_result(result)

        logger.info(
            "Extraction %s complete: app=%s type=%s confidence=%.2f duration=%.0fms errors=%d",
            result.extraction_id, app_name, extraction_type,
            result.confidence, result.duration_ms, len(result.errors),
        )
        return result

    async def extract_from_screen(
        self,
        app_name: str,
        extraction_type: str,
        device_id: str = "local",
        capture_fn: Optional[Callable[[], str]] = None,
    ) -> ExtractionResult:
        """Take a screenshot and then extract data from it.

        If a capture_fn is provided, it is called to produce a screenshot path.
        Otherwise, attempts to use Screenpipe's latest frame for the app.

        Args:
            app_name: Application name.
            extraction_type: Template key.
            device_id: Device identifier (for scheduled extractions).
            capture_fn: Optional callable that returns a screenshot file path.

        Returns:
            ExtractionResult with structured data.
        """
        if capture_fn:
            screenshot_path = capture_fn()
        else:
            # Fall back to screenpipe frame export
            screenshot_path = await self._capture_screenpipe_frame(app_name)

        if not screenshot_path or not Path(screenshot_path).exists():
            result = ExtractionResult(
                extraction_id=_gen_id(),
                source_app=app_name,
                extraction_type=extraction_type,
                errors=["Failed to capture screenshot"],
            )
            self.store_result(result)
            return result

        return await self.extract_from_screenshot(
            image_path=screenshot_path,
            app_name=app_name,
            extraction_type=extraction_type,
        )

    async def extract_from_region(
        self,
        image_path: str,
        region: Dict[str, int],
        extraction_type: str,
        app_name: str = "cropped_region",
    ) -> ExtractionResult:
        """Crop a region from a screenshot and extract data from it.

        Args:
            image_path: Path to the full screenshot.
            region: Dict with keys x, y, w, h defining the crop area.
            extraction_type: Template key.
            app_name: Source app name.

        Returns:
            ExtractionResult from the cropped region.
        """
        start_ms = time.monotonic() * 1000
        result = ExtractionResult(
            extraction_id=_gen_id(),
            source_app=app_name,
            extraction_type=extraction_type,
            screenshot_path=image_path,
        )

        try:
            full_b64 = self._load_image_b64(image_path)
            cropped_b64 = self._crop_image_b64(full_b64, region)

            template = ExtractionTemplate.get(extraction_type)
            system_prompt = template["system_prompt"]

            raw_data = await self._client.extract(
                system_prompt=system_prompt,
                image_b64=cropped_b64,
                user_prompt=(
                    f"Extract data from this cropped region "
                    f"(x={region.get('x')}, y={region.get('y')}, "
                    f"w={region.get('w')}, h={region.get('h')}). "
                    f"Return ONLY valid JSON."
                ),
            )

            result.structured_data = raw_data
            result.confidence = float(raw_data.get("confidence", 0.0))
            result.raw_text = json.dumps(raw_data, indent=2)

        except Exception as exc:
            result.errors.append(str(exc))
            logger.error("Region extraction failed: %s", exc)

        result.duration_ms = round(time.monotonic() * 1000 - start_ms, 1)
        self.store_result(result)
        return result

    async def batch_extract(
        self,
        screenshots: List[Dict[str, str]],
        max_concurrent: int = 3,
    ) -> List[ExtractionResult]:
        """Extract data from multiple screenshots in parallel.

        Args:
            screenshots: List of dicts with keys:
                - image_path (required)
                - app_name (required)
                - extraction_type (required)
                - user_prompt (optional)
            max_concurrent: Maximum concurrent extractions.

        Returns:
            List of ExtractionResult objects.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _extract_one(spec: Dict[str, str]) -> ExtractionResult:
            async with semaphore:
                return await self.extract_from_screenshot(
                    image_path=spec["image_path"],
                    app_name=spec["app_name"],
                    extraction_type=spec["extraction_type"],
                    user_prompt=spec.get("user_prompt"),
                )

        tasks = [_extract_one(spec) for spec in screenshots]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final: List[ExtractionResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                error_result = ExtractionResult(
                    extraction_id=_gen_id(),
                    source_app=screenshots[i].get("app_name", "unknown"),
                    extraction_type=screenshots[i].get("extraction_type", "generic"),
                    screenshot_path=screenshots[i].get("image_path", ""),
                    errors=[str(r)],
                )
                self.store_result(error_result)
                final.append(error_result)
            else:
                final.append(r)

        logger.info("Batch extraction complete: %d/%d successful",
                     sum(1 for r in final if not r.errors), len(final))
        return final

    # ------------------------------------------------------------------
    # Screenpipe frame capture helper
    # ------------------------------------------------------------------

    async def _capture_screenpipe_frame(self, app_name: str) -> Optional[str]:
        """Attempt to get the latest screenshot frame from Screenpipe for an app.

        Returns the path to a saved screenshot, or None.
        """
        try:
            import aiohttp

            params = {
                "app_name": app_name,
                "content_type": "ocr",
                "limit": 1,
                "include_frames": "true",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:3030/search",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()

            items = data.get("data", [])
            if not items:
                return None

            frame = items[0].get("content", {}).get("frame")
            if not frame:
                return None

            # Save frame to temp file
            screenshot_dir = BASE_DIR / "data" / "screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            ts = _now_utc().strftime("%Y%m%d-%H%M%S")
            path = screenshot_dir / f"screenpipe-{app_name.lower().replace(' ', '-')}-{ts}.png"

            img_bytes = base64.b64decode(frame)
            with open(path, "wb") as fh:
                fh.write(img_bytes)

            return str(path)
        except Exception as exc:
            logger.debug("Screenpipe frame capture failed for %s: %s", app_name, exc)
            return None

    # ------------------------------------------------------------------
    # Field validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_fields(data: Dict[str, Any], expected_fields: List[str]) -> List[str]:
        """Check that expected fields exist in the extracted data.

        Supports dotted paths (e.g. 'earnings.today').
        Returns a list of error messages for missing fields.
        """
        errors: List[str] = []
        for field_path in expected_fields:
            parts = field_path.split(".")
            current = data
            found = True
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    found = False
                    break
            if not found:
                errors.append(f"Missing expected field: {field_path}")
        return errors

    def validate_extraction(
        self,
        result: ExtractionResult,
        expected_schema: Optional[Dict[str, type]] = None,
    ) -> List[str]:
        """Validate an extraction result against an expected schema.

        Args:
            result: The ExtractionResult to validate.
            expected_schema: Optional dict of {field_path: expected_type}.
                If not provided, validates against the template's expected_fields.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        # Check confidence threshold
        threshold = self._config.get("confidence_threshold", CONFIDENCE_THRESHOLD)
        if result.confidence < threshold:
            errors.append(
                f"Low confidence: {result.confidence:.2f} < threshold {threshold:.2f}"
            )

        if expected_schema:
            for field_path, expected_type in expected_schema.items():
                parts = field_path.split(".")
                current: Any = result.structured_data
                found = True
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        found = False
                        break

                if not found:
                    errors.append(f"Missing field: {field_path}")
                elif current is not None and not isinstance(current, expected_type):
                    errors.append(
                        f"Type mismatch for {field_path}: "
                        f"expected {expected_type.__name__}, got {type(current).__name__}"
                    )
        else:
            template = ExtractionTemplate.get(result.extraction_type)
            field_errors = self._validate_fields(
                result.structured_data, template.get("expected_fields", [])
            )
            errors.extend(field_errors)

        return errors

    # ==================================================================
    # DATA STORAGE & HISTORY
    # ==================================================================

    def store_result(self, result: ExtractionResult) -> None:
        """Persist an extraction result to the daily results file."""
        iso_date = result.timestamp[:10] if result.timestamp else _today_iso()
        day_results = self._load_day_results(result.extraction_type, iso_date)
        day_results.append(result.to_dict())
        self._save_day_results(result.extraction_type, iso_date, day_results)
        logger.debug(
            "Stored result %s for %s/%s on %s",
            result.extraction_id, result.source_app, result.extraction_type, iso_date,
        )

    def get_history(
        self,
        app_name: Optional[str] = None,
        extraction_type: Optional[str] = None,
        days: int = 7,
    ) -> List[ExtractionResult]:
        """Get historical extraction results.

        Args:
            app_name: Filter by source app (optional).
            extraction_type: Filter by extraction type (optional).
            days: How many days back to look.

        Returns:
            List of ExtractionResult objects, newest first.
        """
        today = _now_utc().date()
        results: List[ExtractionResult] = []

        # Determine which extraction types to scan
        types_to_scan = [extraction_type] if extraction_type else EXTRACTION_TYPES

        for etype in types_to_scan:
            for day_offset in range(days):
                d = (today - timedelta(days=day_offset)).isoformat()
                day_data = self._load_day_results(etype, d)
                for raw in day_data:
                    try:
                        r = ExtractionResult.from_dict(raw)
                        if app_name and r.source_app.lower() != app_name.lower():
                            continue
                        results.append(r)
                    except (TypeError, KeyError) as exc:
                        logger.debug("Skipping malformed result: %s", exc)

        # Sort newest first
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results

    def get_latest(
        self,
        app_name: Optional[str] = None,
        extraction_type: Optional[str] = None,
    ) -> Optional[ExtractionResult]:
        """Get the most recent extraction result for an app/type.

        Args:
            app_name: Filter by source app.
            extraction_type: Filter by extraction type.

        Returns:
            The most recent ExtractionResult, or None.
        """
        history = self.get_history(
            app_name=app_name,
            extraction_type=extraction_type,
            days=30,
        )
        return history[0] if history else None

    def trend_analysis(
        self,
        app_name: str,
        metric_path: str,
        days: int = 30,
        extraction_type: Optional[str] = None,
    ) -> TrendAnalysis:
        """Analyze trends for a specific metric over time.

        Args:
            app_name: Source app name.
            metric_path: Dotted path to the metric (e.g. 'earnings.today').
            days: Number of days to analyze.
            extraction_type: Extraction type to filter by.

        Returns:
            TrendAnalysis with data points, average, min, max, and direction.
        """
        history = self.get_history(
            app_name=app_name,
            extraction_type=extraction_type,
            days=days,
        )

        points: List[TrendPoint] = []
        for result in reversed(history):  # oldest first
            value = self._get_nested_value(result.structured_data, metric_path)
            if value is not None:
                numeric = parse_number(str(value)) if isinstance(value, str) else value
                if numeric is not None:
                    points.append(TrendPoint(
                        date=result.timestamp[:10],
                        value=float(numeric),
                        extraction_id=result.extraction_id,
                    ))

        if not points:
            return TrendAnalysis(
                app_name=app_name,
                metric_path=metric_path,
                period_days=days,
            )

        values = [p.value for p in points]
        avg = sum(values) / len(values)
        current = values[-1]
        first = values[0]

        # Determine trend direction
        if len(values) >= 2:
            change_pct = ((current - first) / first * 100) if first != 0 else 0.0
            if change_pct > 5:
                direction = "up"
            elif change_pct < -5:
                direction = "down"
            else:
                direction = "stable"
        else:
            change_pct = 0.0
            direction = "stable"

        return TrendAnalysis(
            app_name=app_name,
            metric_path=metric_path,
            period_days=days,
            data_points=points,
            current_value=current,
            average=round(avg, 2),
            min_value=min(values),
            max_value=max(values),
            trend_direction=direction,
            change_pct=round(change_pct, 2),
        )

    def compare_periods(
        self,
        app_name: str,
        extraction_type: str,
        period1_start: str,
        period1_end: str,
        period2_start: str,
        period2_end: str,
    ) -> PeriodComparison:
        """Compare extraction data between two date ranges.

        Aggregates and averages metrics from each period and computes
        percentage changes.

        Args:
            app_name: Source app name.
            extraction_type: Extraction type.
            period1_start, period1_end: First period (ISO dates).
            period2_start, period2_end: Second period (ISO dates).

        Returns:
            PeriodComparison with aggregated data and change metrics.
        """
        p1_data = self._aggregate_period(app_name, extraction_type, period1_start, period1_end)
        p2_data = self._aggregate_period(app_name, extraction_type, period2_start, period2_end)

        # Compute changes for all numeric fields
        changes: Dict[str, Dict[str, Any]] = {}
        all_keys = set(list(p1_data.keys()) + list(p2_data.keys()))
        for key in all_keys:
            v1 = p1_data.get(key)
            v2 = p2_data.get(key)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                diff = v2 - v1
                pct = ((v2 - v1) / v1 * 100) if v1 != 0 else (100.0 if v2 != 0 else 0.0)
                changes[key] = {
                    "period1": v1,
                    "period2": v2,
                    "diff": round(diff, 2),
                    "change_pct": round(pct, 2),
                }

        return PeriodComparison(
            app_name=app_name,
            extraction_type=extraction_type,
            period1={"start": period1_start, "end": period1_end},
            period2={"start": period2_start, "end": period2_end},
            period1_data=p1_data,
            period2_data=p2_data,
            changes=changes,
        )

    def _aggregate_period(
        self,
        app_name: str,
        extraction_type: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """Aggregate extraction data for a date range.

        Averages numeric values and takes the most recent non-numeric ones.
        """
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        total_days = (end - start).days + 1

        all_results: List[ExtractionResult] = []
        for day_offset in range(total_days):
            d = (start + timedelta(days=day_offset)).isoformat()
            day_data = self._load_day_results(extraction_type, d)
            for raw in day_data:
                try:
                    r = ExtractionResult.from_dict(raw)
                    if r.source_app.lower() == app_name.lower():
                        all_results.append(r)
                except (TypeError, KeyError):
                    pass

        if not all_results:
            return {}

        # Flatten and average numeric fields from structured_data
        aggregated: Dict[str, Any] = {}
        numeric_sums: Dict[str, float] = {}
        numeric_counts: Dict[str, int] = {}

        for r in all_results:
            flat = self._flatten_dict(r.structured_data)
            for key, value in flat.items():
                if isinstance(value, (int, float)) and key != "confidence":
                    numeric_sums[key] = numeric_sums.get(key, 0.0) + value
                    numeric_counts[key] = numeric_counts.get(key, 0) + 1
                elif value is not None:
                    aggregated[key] = value  # take latest

        for key in numeric_sums:
            count = numeric_counts.get(key, 1)
            aggregated[key] = round(numeric_sums[key] / count, 2)

        return aggregated

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten a nested dict into dotted-key format."""
        items: Dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(OCRExtractor._flatten_dict(v, new_key))
            elif isinstance(v, list):
                items[new_key] = v  # keep lists as-is
            else:
                items[new_key] = v
        return items

    @staticmethod
    def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
        """Get a value from a nested dict using a dotted path."""
        parts = path.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    # ==================================================================
    # EXPORT
    # ==================================================================

    def export_csv(
        self,
        app_name: Optional[str] = None,
        extraction_type: Optional[str] = None,
        days: int = 30,
        output_path: Optional[str] = None,
    ) -> str:
        """Export extraction history to CSV.

        Args:
            app_name: Filter by source app.
            extraction_type: Filter by extraction type.
            days: How many days back.
            output_path: Output file path. If None, auto-generates.

        Returns:
            Path to the written CSV file.
        """
        history = self.get_history(
            app_name=app_name,
            extraction_type=extraction_type,
            days=days,
        )

        if not output_path:
            ts = _now_utc().strftime("%Y%m%d-%H%M%S")
            app_slug = (app_name or "all").lower().replace(" ", "-")
            type_slug = extraction_type or "all"
            output_path = str(
                EXTRACTION_DATA_DIR / f"export-{app_slug}-{type_slug}-{ts}.csv"
            )

        # Collect all unique metric keys across results
        all_keys: set = set()
        rows: List[Dict[str, Any]] = []
        for r in history:
            flat = self._flatten_dict(r.structured_data)
            # Filter out non-scalar values for CSV
            scalar_flat = {
                k: v for k, v in flat.items()
                if isinstance(v, (int, float, str, bool)) or v is None
            }
            all_keys.update(scalar_flat.keys())
            row = {
                "extraction_id": r.extraction_id,
                "timestamp": r.timestamp,
                "source_app": r.source_app,
                "extraction_type": r.extraction_type,
                "confidence": r.confidence,
                "duration_ms": r.duration_ms,
            }
            row.update(scalar_flat)
            rows.append(row)

        # Define column order
        base_cols = ["extraction_id", "timestamp", "source_app", "extraction_type",
                     "confidence", "duration_ms"]
        metric_cols = sorted(all_keys - set(base_cols))
        fieldnames = base_cols + metric_cols

        with open(output_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        logger.info("Exported %d results to %s", len(rows), output_path)
        return output_path

    def export_to_revenue_tracker(self, result: ExtractionResult) -> Optional[Dict[str, Any]]:
        """Push extraction data into revenue_tracker.py format.

        Maps extracted metrics to RevenueEntry fields and records them
        via the revenue tracker.

        Args:
            result: The ExtractionResult to export.

        Returns:
            Dict with recorded entries summary, or None if not applicable.
        """
        try:
            from src.revenue_tracker import get_tracker, RevenueStream
        except ImportError:
            logger.warning("revenue_tracker not available for export")
            return None

        tracker = get_tracker()
        data = result.structured_data
        today = _today_iso()
        recorded: List[Dict[str, Any]] = []

        etype = result.extraction_type.lower()

        if etype == "adsense":
            earnings = data.get("earnings", {})
            today_earnings = earnings.get("today")
            if today_earnings is not None:
                entry = tracker.record_revenue(
                    date=today,
                    stream=RevenueStream.ADS,
                    source="adsense",
                    amount=float(today_earnings),
                    metadata={"ocr_extraction_id": result.extraction_id},
                )
                recorded.append(entry.to_dict())

        elif etype == "amazon_associates":
            total = data.get("total_earnings")
            if total is not None:
                entry = tracker.record_revenue(
                    date=today,
                    stream=RevenueStream.AFFILIATE,
                    source="amazon",
                    amount=float(total),
                    metadata={"ocr_extraction_id": result.extraction_id},
                )
                recorded.append(entry.to_dict())

        elif etype == "etsy":
            revenue = data.get("revenue")
            if revenue is not None:
                entry = tracker.record_revenue(
                    date=today,
                    stream=RevenueStream.ETSY,
                    source="etsy",
                    amount=float(revenue),
                    metadata={"ocr_extraction_id": result.extraction_id},
                )
                recorded.append(entry.to_dict())

        elif etype == "kdp":
            royalties = data.get("royalties")
            if royalties is not None:
                entry = tracker.record_revenue(
                    date=today,
                    stream=RevenueStream.KDP,
                    source="kdp",
                    amount=float(royalties),
                    metadata={"ocr_extraction_id": result.extraction_id},
                )
                recorded.append(entry.to_dict())

        if recorded:
            logger.info(
                "Exported %d entries from extraction %s to revenue tracker",
                len(recorded), result.extraction_id,
            )
            return {"entries": recorded, "count": len(recorded)}

        return None

    # ==================================================================
    # ANOMALY DETECTION
    # ==================================================================

    def flag_anomalies(
        self,
        result: ExtractionResult,
        days_for_avg: int = 14,
    ) -> List[AnomalyFlag]:
        """Flag values that deviate significantly from historical average.

        Compares each numeric metric in the result against its historical
        average over the specified period. Flags deviations exceeding the
        configured threshold.

        Args:
            result: The ExtractionResult to check.
            days_for_avg: Number of days for computing historical averages.

        Returns:
            List of AnomalyFlag objects for anomalous metrics.
        """
        deviation_threshold = self._config.get("anomaly_deviation_pct", ANOMALY_DEVIATION_PCT)
        history = self.get_history(
            app_name=result.source_app,
            extraction_type=result.extraction_type,
            days=days_for_avg,
        )

        # Exclude the current result from history
        history = [r for r in history if r.extraction_id != result.extraction_id]
        if not history:
            return []

        # Compute historical averages for all numeric fields
        flat_current = self._flatten_dict(result.structured_data)
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        for h in history:
            flat_h = self._flatten_dict(h.structured_data)
            for key, value in flat_h.items():
                if isinstance(value, (int, float)) and key != "confidence":
                    metric_sums[key] = metric_sums.get(key, 0.0) + value
                    metric_counts[key] = metric_counts.get(key, 0) + 1

        flags: List[AnomalyFlag] = []
        for key, current_value in flat_current.items():
            if not isinstance(current_value, (int, float)) or key == "confidence":
                continue
            if key not in metric_sums or metric_counts.get(key, 0) < 2:
                continue

            avg = metric_sums[key] / metric_counts[key]
            if avg == 0:
                continue

            deviation = abs(current_value - avg) / abs(avg)
            if deviation > deviation_threshold:
                direction = "higher" if current_value > avg else "lower"
                severity = "critical" if deviation > deviation_threshold * 2 else "warning"
                flags.append(AnomalyFlag(
                    metric_path=key,
                    current_value=float(current_value),
                    historical_avg=round(avg, 2),
                    deviation_pct=round(deviation * 100, 1),
                    severity=severity,
                    message=(
                        f"{key} is {deviation * 100:.0f}% {direction} than "
                        f"the {days_for_avg}-day average "
                        f"({current_value} vs avg {avg:.2f})"
                    ),
                ))

        if flags:
            logger.info(
                "Flagged %d anomalies in extraction %s",
                len(flags), result.extraction_id,
            )

        return flags

    # ==================================================================
    # MANUAL CORRECTIONS
    # ==================================================================

    def manual_correction(
        self,
        extraction_id: str,
        corrections: Dict[str, Any],
    ) -> Optional[ExtractionResult]:
        """Apply manual corrections to an extraction result.

        Finds the result by ID, applies corrections to structured_data,
        records the correction, and re-persists.

        Args:
            extraction_id: The ID of the extraction to correct.
            corrections: Dict of {field_path: corrected_value}.

        Returns:
            The corrected ExtractionResult, or None if not found.
        """
        # Find the result across all types and dates
        for etype in EXTRACTION_TYPES:
            today = _now_utc().date()
            for day_offset in range(90):  # Search up to 90 days back
                d = (today - timedelta(days=day_offset)).isoformat()
                day_results = self._load_day_results(etype, d)

                for i, raw in enumerate(day_results):
                    if raw.get("extraction_id") == extraction_id:
                        # Apply corrections
                        structured = raw.get("structured_data", {})
                        for path, value in corrections.items():
                            self._set_nested_value(structured, path, value)

                        raw["structured_data"] = structured
                        raw.setdefault("errors", [])
                        raw["errors"].append(
                            f"Manual correction applied at {_now_iso()}: "
                            f"{json.dumps(corrections)}"
                        )

                        day_results[i] = raw
                        self._save_day_results(etype, d, day_results)

                        # Record correction in corrections log
                        corrections_log = _load_json(CORRECTIONS_FILE, [])
                        corrections_log.append({
                            "extraction_id": extraction_id,
                            "corrections": corrections,
                            "timestamp": _now_iso(),
                        })
                        _save_json(CORRECTIONS_FILE, corrections_log[-500:])

                        logger.info(
                            "Applied manual corrections to %s: %s",
                            extraction_id, corrections,
                        )
                        return ExtractionResult.from_dict(raw)

        logger.warning("Extraction %s not found for correction", extraction_id)
        return None

    @staticmethod
    def _set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
        """Set a value in a nested dict using a dotted path."""
        parts = path.split(".")
        current = data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    # ==================================================================
    # SCHEDULED EXTRACTIONS
    # ==================================================================

    def schedule_extraction(
        self,
        app_name: str,
        extraction_type: str,
        cron_expr: str,
        device_id: str = "local",
    ) -> ExtractionSchedule:
        """Set up a recurring scheduled extraction.

        Args:
            app_name: Application name.
            extraction_type: Template key.
            cron_expr: Cron expression (minute hour day-of-month month day-of-week).
            device_id: Device identifier.

        Returns:
            The created ExtractionSchedule.
        """
        schedule = ExtractionSchedule(
            schedule_id=_gen_id(),
            app_name=app_name,
            extraction_type=extraction_type,
            device_id=device_id,
            cron_expr=cron_expr,
            enabled=True,
        )

        # Calculate next run time
        schedule.next_run = self._next_cron_run(cron_expr)

        self._schedules[schedule.schedule_id] = schedule
        self._save_schedules()

        logger.info(
            "Scheduled extraction %s: %s/%s cron='%s' next_run=%s",
            schedule.schedule_id, app_name, extraction_type, cron_expr, schedule.next_run,
        )
        return schedule

    async def run_scheduled(self) -> List[ExtractionResult]:
        """Execute all due scheduled extractions.

        Checks each enabled schedule's next_run against the current time
        and runs extractions that are due.

        Returns:
            List of ExtractionResult objects from executed extractions.
        """
        now = _now_utc()
        now_iso = now.isoformat()
        results: List[ExtractionResult] = []

        for sid, schedule in self._schedules.items():
            if not schedule.enabled:
                continue
            if not schedule.next_run or schedule.next_run > now_iso:
                continue

            logger.info(
                "Running scheduled extraction %s: %s/%s",
                sid, schedule.app_name, schedule.extraction_type,
            )

            try:
                result = await self.extract_from_screen(
                    app_name=schedule.app_name,
                    extraction_type=schedule.extraction_type,
                    device_id=schedule.device_id,
                )
                results.append(result)
                schedule.results_count += 1
            except Exception as exc:
                logger.error("Scheduled extraction %s failed: %s", sid, exc)
                error_result = ExtractionResult(
                    extraction_id=_gen_id(),
                    source_app=schedule.app_name,
                    extraction_type=schedule.extraction_type,
                    errors=[f"Scheduled execution failed: {exc}"],
                )
                self.store_result(error_result)
                results.append(error_result)

            schedule.last_run = now_iso
            schedule.next_run = self._next_cron_run(schedule.cron_expr)

        self._save_schedules()
        if results:
            logger.info("Executed %d scheduled extractions", len(results))
        return results

    def list_schedules(self) -> List[ExtractionSchedule]:
        """List all extraction schedules."""
        return list(self._schedules.values())

    def enable_schedule(self, schedule_id: str) -> Optional[ExtractionSchedule]:
        """Enable a schedule by ID."""
        schedule = self._schedules.get(schedule_id)
        if schedule:
            schedule.enabled = True
            schedule.next_run = self._next_cron_run(schedule.cron_expr)
            self._save_schedules()
            logger.info("Enabled schedule %s", schedule_id)
        return schedule

    def disable_schedule(self, schedule_id: str) -> Optional[ExtractionSchedule]:
        """Disable a schedule by ID."""
        schedule = self._schedules.get(schedule_id)
        if schedule:
            schedule.enabled = False
            self._save_schedules()
            logger.info("Disabled schedule %s", schedule_id)
        return schedule

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule by ID."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            self._save_schedules()
            logger.info("Deleted schedule %s", schedule_id)
            return True
        return False

    @staticmethod
    def _next_cron_run(cron_expr: str) -> str:
        """Calculate the next run time from a cron expression.

        Supports basic cron format: minute hour day-of-month month day-of-week.
        Uses a simple forward-scan approach for the next matching minute.

        Returns ISO timestamp string of the next run.
        """
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            # Invalid cron, default to 1 hour from now
            return (_now_utc() + timedelta(hours=1)).isoformat()

        def _parse_cron_field(field: str, min_val: int, max_val: int) -> set:
            """Parse a single cron field into a set of valid values."""
            values: set = set()
            for token in field.split(","):
                token = token.strip()
                if token == "*":
                    values.update(range(min_val, max_val + 1))
                elif "/" in token:
                    base, step_str = token.split("/", 1)
                    step = int(step_str)
                    start = min_val if base == "*" else int(base)
                    values.update(range(start, max_val + 1, step))
                elif "-" in token:
                    low, high = token.split("-", 1)
                    values.update(range(int(low), int(high) + 1))
                else:
                    values.add(int(token))
            return values

        try:
            valid_minutes = _parse_cron_field(parts[0], 0, 59)
            valid_hours = _parse_cron_field(parts[1], 0, 23)
            valid_days = _parse_cron_field(parts[2], 1, 31)
            valid_months = _parse_cron_field(parts[3], 1, 12)
            valid_weekdays = _parse_cron_field(parts[4], 0, 6)
        except (ValueError, IndexError):
            return (_now_utc() + timedelta(hours=1)).isoformat()

        # Forward-scan from next minute
        candidate = _now_utc().replace(second=0, microsecond=0) + timedelta(minutes=1)
        max_scan = 60 * 24 * 400  # scan up to ~400 days

        for _ in range(max_scan):
            if (candidate.minute in valid_minutes
                    and candidate.hour in valid_hours
                    and candidate.day in valid_days
                    and candidate.month in valid_months
                    and candidate.weekday() in valid_weekdays):
                return candidate.isoformat()
            candidate += timedelta(minutes=1)

        # Fallback
        return (_now_utc() + timedelta(hours=1)).isoformat()

    # ==================================================================
    # ASYNC INTERFACES
    # ==================================================================

    async def aextract_from_screenshot(self, *args: Any, **kwargs: Any) -> ExtractionResult:
        """Alias for extract_from_screenshot (already async)."""
        return await self.extract_from_screenshot(*args, **kwargs)

    async def aextract_from_screen(self, *args: Any, **kwargs: Any) -> ExtractionResult:
        """Alias for extract_from_screen (already async)."""
        return await self.extract_from_screen(*args, **kwargs)

    async def aextract_from_region(self, *args: Any, **kwargs: Any) -> ExtractionResult:
        """Alias for extract_from_region (already async)."""
        return await self.extract_from_region(*args, **kwargs)

    async def abatch_extract(self, *args: Any, **kwargs: Any) -> List[ExtractionResult]:
        """Alias for batch_extract (already async)."""
        return await self.batch_extract(*args, **kwargs)

    async def aget_history(self, *args: Any, **kwargs: Any) -> List[ExtractionResult]:
        """Async wrapper for get_history."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_history(*args, **kwargs))

    async def aget_latest(self, *args: Any, **kwargs: Any) -> Optional[ExtractionResult]:
        """Async wrapper for get_latest."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_latest(*args, **kwargs))

    async def atrend_analysis(self, *args: Any, **kwargs: Any) -> TrendAnalysis:
        """Async wrapper for trend_analysis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.trend_analysis(*args, **kwargs))

    async def acompare_periods(self, *args: Any, **kwargs: Any) -> PeriodComparison:
        """Async wrapper for compare_periods."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.compare_periods(*args, **kwargs))

    async def aexport_csv(self, *args: Any, **kwargs: Any) -> str:
        """Async wrapper for export_csv."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.export_csv(*args, **kwargs))

    async def aflag_anomalies(self, *args: Any, **kwargs: Any) -> List[AnomalyFlag]:
        """Async wrapper for flag_anomalies."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.flag_anomalies(*args, **kwargs))

    async def avalidate_extraction(self, *args: Any, **kwargs: Any) -> List[str]:
        """Async wrapper for validate_extraction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.validate_extraction(*args, **kwargs))

    # ==================================================================
    # SYNC WRAPPERS
    # ==================================================================

    def extract_from_screenshot_sync(self, *args: Any, **kwargs: Any) -> ExtractionResult:
        """Synchronous wrapper for extract_from_screenshot."""
        return self._run_sync(self.extract_from_screenshot(*args, **kwargs))

    def extract_from_screen_sync(self, *args: Any, **kwargs: Any) -> ExtractionResult:
        """Synchronous wrapper for extract_from_screen."""
        return self._run_sync(self.extract_from_screen(*args, **kwargs))

    def extract_from_region_sync(self, *args: Any, **kwargs: Any) -> ExtractionResult:
        """Synchronous wrapper for extract_from_region."""
        return self._run_sync(self.extract_from_region(*args, **kwargs))

    def batch_extract_sync(self, *args: Any, **kwargs: Any) -> List[ExtractionResult]:
        """Synchronous wrapper for batch_extract."""
        return self._run_sync(self.batch_extract(*args, **kwargs))

    def run_scheduled_sync(self) -> List[ExtractionResult]:
        """Synchronous wrapper for run_scheduled."""
        return self._run_sync(self.run_scheduled())

    @staticmethod
    def _run_sync(coro: Any) -> Any:
        """Run an async coroutine in a sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    # ==================================================================
    # FORMATTING / DISPLAY
    # ==================================================================

    @staticmethod
    def format_result(result: ExtractionResult, style: str = "text") -> str:
        """Format an ExtractionResult for display.

        Styles:
            text     — plain text summary
            json     — raw JSON
            markdown — rich markdown
        """
        if style == "json":
            return json.dumps(result.to_dict(), indent=2, default=str)

        if style == "markdown":
            lines = [
                f"# Extraction: {result.extraction_id}",
                f"**App:** {result.source_app}",
                f"**Type:** {result.extraction_type}",
                f"**Confidence:** {result.confidence:.2f}",
                f"**Duration:** {result.duration_ms:.0f}ms",
                f"**Timestamp:** {result.timestamp}",
                "",
                "## Extracted Data",
                "```json",
                json.dumps(result.structured_data, indent=2, default=str),
                "```",
            ]
            if result.errors:
                lines.extend(["", "## Errors"])
                for e in result.errors:
                    lines.append(f"- {e}")
            return "\n".join(lines)

        # Plain text
        lines = [
            f"EXTRACTION: {result.extraction_id}",
            f"  App: {result.source_app}",
            f"  Type: {result.extraction_type}",
            f"  Confidence: {result.confidence:.2f}",
            f"  Duration: {result.duration_ms:.0f}ms",
            f"  Time: {result.timestamp}",
            "",
            "  Data:",
        ]
        flat = OCRExtractor._flatten_dict(result.structured_data)
        for key, value in sorted(flat.items()):
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value, default=str)[:80]
            else:
                value_str = str(value)
            lines.append(f"    {key}: {value_str}")

        if result.errors:
            lines.extend(["", "  Errors:"])
            for e in result.errors:
                lines.append(f"    - {e}")

        return "\n".join(lines)

    @staticmethod
    def format_trend(trend: TrendAnalysis) -> str:
        """Format a TrendAnalysis as a summary string."""
        arrow = {"up": "+", "down": "-", "stable": "~"}.get(trend.trend_direction, "?")
        lines = [
            f"TREND: {trend.app_name} / {trend.metric_path}",
            f"  Period: {trend.period_days} days ({len(trend.data_points)} data points)",
            f"  Current: {trend.current_value}",
            f"  Average: {trend.average}",
            f"  Range: {trend.min_value} - {trend.max_value}",
            f"  Direction: {trend.trend_direction} ({arrow}{trend.change_pct:.1f}%)",
        ]
        return "\n".join(lines)

    @staticmethod
    def format_comparison(comparison: PeriodComparison) -> str:
        """Format a PeriodComparison as a summary string."""
        lines = [
            f"COMPARISON: {comparison.app_name} / {comparison.extraction_type}",
            f"  Period 1: {comparison.period1.get('start')} to {comparison.period1.get('end')}",
            f"  Period 2: {comparison.period2.get('start')} to {comparison.period2.get('end')}",
            "",
            "  Changes:",
        ]
        for metric, change in sorted(comparison.changes.items()):
            arrow = "+" if change["change_pct"] >= 0 else ""
            lines.append(
                f"    {metric}: {change['period1']} -> {change['period2']} "
                f"({arrow}{change['change_pct']:.1f}%)"
            )
        return "\n".join(lines)


# ===================================================================
# Module-Level Convenience API
# ===================================================================

_extractor_instance: Optional[OCRExtractor] = None


def get_extractor() -> OCRExtractor:
    """Return the singleton OCRExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = OCRExtractor()
    return _extractor_instance


# ===================================================================
# CLI Entry Point
# ===================================================================


def _cli_main() -> None:
    """CLI entry point: python -m src.ocr_extractor <command> [options]."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="ocr_extractor",
        description="OpenClaw Empire OCR Data Extraction Pipeline — CLI Interface",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- extract ---
    p_extract = subparsers.add_parser("extract", help="Extract data from a screenshot")
    p_extract.add_argument("image_path", help="Path to the screenshot file")
    p_extract.add_argument("--app", required=True, help="Source application name")
    p_extract.add_argument(
        "--type", required=True, choices=EXTRACTION_TYPES,
        help="Extraction type / template",
    )
    p_extract.add_argument("--prompt", help="Custom user prompt override")
    p_extract.add_argument(
        "--format", choices=["text", "json", "markdown"], default="text",
        help="Output format (default: text)",
    )
    p_extract.add_argument("--push-revenue", action="store_true",
                           help="Push results to revenue tracker")

    # --- screen ---
    p_screen = subparsers.add_parser("screen", help="Capture screen and extract")
    p_screen.add_argument("--app", required=True, help="Source application name")
    p_screen.add_argument(
        "--type", required=True, choices=EXTRACTION_TYPES,
        help="Extraction type / template",
    )
    p_screen.add_argument("--device", default="local", help="Device ID (default: local)")

    # --- schedule ---
    p_sched = subparsers.add_parser("schedule", help="Manage scheduled extractions")
    p_sched.add_argument("action", choices=["add", "list", "enable", "disable", "delete", "run"],
                         help="Schedule action")
    p_sched.add_argument("--app", help="Source application name (for add)")
    p_sched.add_argument("--type", choices=EXTRACTION_TYPES, help="Extraction type (for add)")
    p_sched.add_argument("--cron", help="Cron expression (for add)")
    p_sched.add_argument("--device", default="local", help="Device ID (for add)")
    p_sched.add_argument("--id", help="Schedule ID (for enable/disable/delete)")

    # --- history ---
    p_hist = subparsers.add_parser("history", help="View extraction history")
    p_hist.add_argument("--app", help="Filter by source app")
    p_hist.add_argument("--type", choices=EXTRACTION_TYPES, help="Filter by extraction type")
    p_hist.add_argument("--days", type=int, default=7, help="Days back (default: 7)")
    p_hist.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    p_hist.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )

    # --- trend ---
    p_trend = subparsers.add_parser("trend", help="Analyze metric trends")
    p_trend.add_argument("--app", required=True, help="Source application name")
    p_trend.add_argument("--metric", required=True,
                         help="Metric path (e.g. 'earnings.today')")
    p_trend.add_argument("--days", type=int, default=30, help="Days to analyze (default: 30)")
    p_trend.add_argument("--type", choices=EXTRACTION_TYPES, help="Extraction type filter")

    # --- compare ---
    p_comp = subparsers.add_parser("compare", help="Compare two time periods")
    p_comp.add_argument("--app", required=True, help="Source application name")
    p_comp.add_argument("--type", required=True, choices=EXTRACTION_TYPES,
                        help="Extraction type")
    p_comp.add_argument("--period1-start", required=True, help="Period 1 start (YYYY-MM-DD)")
    p_comp.add_argument("--period1-end", required=True, help="Period 1 end (YYYY-MM-DD)")
    p_comp.add_argument("--period2-start", required=True, help="Period 2 start (YYYY-MM-DD)")
    p_comp.add_argument("--period2-end", required=True, help="Period 2 end (YYYY-MM-DD)")

    # --- export ---
    p_export = subparsers.add_parser("export", help="Export data to CSV")
    p_export.add_argument("--app", help="Filter by source app")
    p_export.add_argument("--type", choices=EXTRACTION_TYPES, help="Filter by extraction type")
    p_export.add_argument("--days", type=int, default=30, help="Days back (default: 30)")
    p_export.add_argument("--output", help="Output file path (auto-generated if omitted)")

    # --- validate ---
    p_val = subparsers.add_parser("validate", help="Validate an extraction result")
    p_val.add_argument("extraction_id", help="Extraction ID to validate")

    # --- templates ---
    subparsers.add_parser("templates", help="List available extraction templates")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    extractor = get_extractor()

    # --- Execute commands ---

    if args.command == "extract":
        result = extractor.extract_from_screenshot_sync(
            image_path=args.image_path,
            app_name=args.app,
            extraction_type=args.type,
            user_prompt=args.prompt,
        )
        print(extractor.format_result(result, style=args.format))

        if args.push_revenue:
            export = extractor.export_to_revenue_tracker(result)
            if export:
                print(f"\nPushed {export['count']} entries to revenue tracker")
            else:
                print("\nNo revenue data to push for this extraction type")

        # Run anomaly detection
        anomalies = extractor.flag_anomalies(result)
        if anomalies:
            print(f"\nANOMALIES DETECTED ({len(anomalies)}):")
            for a in anomalies:
                print(f"  [{a.severity.upper()}] {a.message}")

    elif args.command == "screen":
        result = extractor.extract_from_screen_sync(
            app_name=args.app,
            extraction_type=args.type,
            device_id=args.device,
        )
        print(extractor.format_result(result))

    elif args.command == "schedule":
        if args.action == "add":
            if not all([args.app, args.type, args.cron]):
                print("Error: --app, --type, and --cron are required for 'add'")
                sys.exit(1)
            schedule = extractor.schedule_extraction(
                app_name=args.app,
                extraction_type=args.type,
                cron_expr=args.cron,
                device_id=args.device,
            )
            print(f"Created schedule {schedule.schedule_id}:")
            print(f"  App: {schedule.app_name}")
            print(f"  Type: {schedule.extraction_type}")
            print(f"  Cron: {schedule.cron_expr}")
            print(f"  Next run: {schedule.next_run}")

        elif args.action == "list":
            schedules = extractor.list_schedules()
            if not schedules:
                print("No schedules configured.")
            else:
                print(f"EXTRACTION SCHEDULES ({len(schedules)})")
                print(f"{'=' * 70}")
                for s in schedules:
                    status = "ENABLED" if s.enabled else "DISABLED"
                    print(f"  {s.schedule_id} [{status}]")
                    print(f"    App: {s.app_name} | Type: {s.extraction_type}")
                    print(f"    Cron: {s.cron_expr} | Runs: {s.results_count}")
                    print(f"    Last: {s.last_run or 'never'} | Next: {s.next_run or 'N/A'}")
                    print()

        elif args.action == "enable":
            if not args.id:
                print("Error: --id is required for 'enable'")
                sys.exit(1)
            s = extractor.enable_schedule(args.id)
            if s:
                print(f"Enabled schedule {s.schedule_id}, next run: {s.next_run}")
            else:
                print(f"Schedule {args.id} not found")

        elif args.action == "disable":
            if not args.id:
                print("Error: --id is required for 'disable'")
                sys.exit(1)
            s = extractor.disable_schedule(args.id)
            if s:
                print(f"Disabled schedule {s.schedule_id}")
            else:
                print(f"Schedule {args.id} not found")

        elif args.action == "delete":
            if not args.id:
                print("Error: --id is required for 'delete'")
                sys.exit(1)
            deleted = extractor.delete_schedule(args.id)
            if deleted:
                print(f"Deleted schedule {args.id}")
            else:
                print(f"Schedule {args.id} not found")

        elif args.action == "run":
            results = extractor.run_scheduled_sync()
            if results:
                print(f"Executed {len(results)} scheduled extractions:")
                for r in results:
                    status = "OK" if not r.errors else "ERROR"
                    print(f"  [{status}] {r.source_app}/{r.extraction_type} "
                          f"(conf={r.confidence:.2f}, {r.duration_ms:.0f}ms)")
            else:
                print("No scheduled extractions are due.")

    elif args.command == "history":
        history = extractor.get_history(
            app_name=args.app,
            extraction_type=getattr(args, "type", None),
            days=args.days,
        )
        history = history[:args.limit]

        if args.format == "json":
            print(json.dumps([r.to_dict() for r in history], indent=2, default=str))
        else:
            if not history:
                print("No extraction history found.")
            else:
                print(f"EXTRACTION HISTORY ({len(history)} results)")
                print(f"{'=' * 70}")
                for r in history:
                    status = "OK" if not r.errors else "ERR"
                    errors_str = f" [{len(r.errors)} errors]" if r.errors else ""
                    print(
                        f"  [{status}] {r.extraction_id} | "
                        f"{r.timestamp[:16]} | {r.source_app} / {r.extraction_type} | "
                        f"conf={r.confidence:.2f}{errors_str}"
                    )

    elif args.command == "trend":
        trend = extractor.trend_analysis(
            app_name=args.app,
            metric_path=args.metric,
            days=args.days,
            extraction_type=getattr(args, "type", None),
        )
        print(extractor.format_trend(trend))
        if trend.data_points:
            print("\n  Data points:")
            for p in trend.data_points[-10:]:
                print(f"    {p.date}: {p.value}")

    elif args.command == "compare":
        comparison = extractor.compare_periods(
            app_name=args.app,
            extraction_type=args.type,
            period1_start=args.period1_start,
            period1_end=args.period1_end,
            period2_start=args.period2_start,
            period2_end=args.period2_end,
        )
        print(extractor.format_comparison(comparison))

    elif args.command == "export":
        output = extractor.export_csv(
            app_name=args.app,
            extraction_type=getattr(args, "type", None),
            days=args.days,
            output_path=args.output,
        )
        print(f"Exported to: {output}")

    elif args.command == "validate":
        # Find the result and validate it
        result = None
        for etype in EXTRACTION_TYPES:
            today = _now_utc().date()
            for day_offset in range(90):
                d = (today - timedelta(days=day_offset)).isoformat()
                day_data = extractor._load_day_results(etype, d)
                for raw in day_data:
                    if raw.get("extraction_id") == args.extraction_id:
                        result = ExtractionResult.from_dict(raw)
                        break
                if result:
                    break
            if result:
                break

        if not result:
            print(f"Extraction {args.extraction_id} not found")
            sys.exit(1)

        errors = extractor.validate_extraction(result)
        if errors:
            print(f"VALIDATION ISSUES ({len(errors)}):")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"Extraction {args.extraction_id} is valid.")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Type: {result.extraction_type}")
            print(f"  Fields extracted: {len(extractor._flatten_dict(result.structured_data))}")

    elif args.command == "templates":
        templates = ExtractionTemplate.list_types()
        print(f"EXTRACTION TEMPLATES ({len(templates)})")
        print(f"{'=' * 50}")
        for t in templates:
            template_data = ExtractionTemplate.get(t["type"])
            field_count = len(template_data.get("expected_fields", []))
            print(f"  {t['type']:<22} {t['name']:<25} ({field_count} expected fields)")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli_main()
