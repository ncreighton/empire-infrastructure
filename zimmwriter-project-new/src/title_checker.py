"""
WordPress Title Deduplication Checker.

Queries each site's published, draft, and scheduled posts via the WordPress
REST API to collect all existing titles and slugs. Provides duplicate
detection via exact title match, slug match, and token overlap ratio.

Usage:
    from src.title_checker import TitleChecker

    checker = TitleChecker()
    existing = checker.check_site("smarthomewizards.com")
    # existing = {"titles": {"How to Set Up Alexa", ...}, "slugs": {"how-to-set-up-alexa", ...}, "count": 42}

    is_dupe = checker.is_duplicate("Setting Up Alexa: A Guide", existing)
    # True (token overlap > 0.7)

    all_existing = checker.check_all_sites()
    # {domain: {titles, slugs, count}, ...}
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

from .link_pack_builder import strip_html
from .site_presets import SITE_PRESETS

logger = logging.getLogger(__name__)

# Path to sites.json with WordPress app passwords
_SITES_JSON = Path(r"D:\Claude Code Projects\config\sites.json")

# Domain -> sites.json key mapping (sites.json uses short keys without TLD)
_DOMAIN_TO_KEY = {
    "aiinactionhub.com": "aiinactionhub",
    "aidiscoverydigest.com": "aidiscoverydigest",
    "clearainews.com": "clearainews",
    "wealthfromai.com": "wealthfromai",
    "smarthomewizards.com": "smarthomewizards",
    "smarthomegearreviews.com": "smarthomegearreviews",
    "theconnectedhaven.com": "theconnectedhaven",
    "witchcraftforbeginners.com": "witchcraftforbeginners",
    "manifestandalign.com": "manifestandalign",
    "family-flourish.com": "familyflourish",
    "mythicalarchives.com": "mythicalarchives",
    "wearablegearreviews.com": "wearablegearreviews",
    "pulsegearreviews.com": "pulsegearreviews",
    "bulletjournals.net": "bulletjournals",
}


def _load_site_credentials() -> Dict[str, dict]:
    """Load WordPress credentials from sites.json.

    Returns:
        Dict mapping domain -> {"user": str, "app_password": str}.
    """
    if not _SITES_JSON.exists():
        logger.warning("sites.json not found at %s", _SITES_JSON)
        return {}

    with open(_SITES_JSON, encoding="utf-8") as f:
        data = json.load(f)

    sites = data.get("sites", {})
    creds = {}
    for domain, key in _DOMAIN_TO_KEY.items():
        site_cfg = sites.get(key, {})
        wp = site_cfg.get("wordpress", {})
        user = wp.get("user", "")
        password = wp.get("app_password", "") or site_cfg.get("wp_app_password", "")
        if user and password:
            creds[domain] = {"user": user, "app_password": password}
    return creds


def _title_to_slug(title: str) -> str:
    """Convert a title to a WordPress-style slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _normalize_for_comparison(text: str) -> str:
    """Normalize a title for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> Set[str]:
    """Tokenize a normalized title into a set of words, removing stopwords."""
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "between",
        "through", "during", "before", "after", "and", "but", "or", "nor",
        "not", "so", "yet", "both", "either", "neither", "each", "every",
        "this", "that", "these", "those", "it", "its", "your", "you",
    }
    words = set(_normalize_for_comparison(text).split())
    return words - stopwords


class TitleChecker:
    """Checks WordPress sites for existing titles to prevent duplicates."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ZimmWriter-TitleChecker/1.0",
            "Accept": "application/json",
        })
        self._credentials = _load_site_credentials()

    def _fetch_posts(
        self,
        site_url: str,
        status: str = "publish",
        auth: Optional[Tuple[str, str]] = None,
        max_items: int = 500,
    ) -> List[dict]:
        """Fetch posts from WordPress REST API with pagination.

        Args:
            site_url: Base URL (e.g., "https://smarthomewizards.com").
            status: Post status to fetch ("publish", "draft", "future").
            auth: Optional (user, app_password) tuple for authenticated requests.
            max_items: Maximum number of posts to fetch.

        Returns:
            List of {"id": int, "title": str, "slug": str, "link": str, "date": str}.
        """
        results = []
        per_page = 100
        total_pages_needed = math.ceil(max_items / per_page)
        page = 1

        while page <= total_pages_needed and len(results) < max_items:
            url = (
                f"{site_url.rstrip('/')}/wp-json/wp/v2/posts"
                f"?per_page={per_page}&page={page}"
                f"&_fields=id,title,slug,link,date&status={status}"
            )

            try:
                kwargs = {"timeout": 30}
                if auth:
                    kwargs["auth"] = auth

                resp = self.session.get(url, **kwargs)

                if resp.status_code == 400:
                    break
                if resp.status_code == 401:
                    logger.debug(
                        "Auth required for status=%s on %s (skipping)",
                        status, site_url,
                    )
                    break
                resp.raise_for_status()

            except requests.exceptions.HTTPError as exc:
                logger.warning(
                    "HTTP %s fetching %s (status=%s) page %d: %s",
                    getattr(exc.response, "status_code", "?"),
                    site_url, status, page, exc,
                )
                break
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Request error fetching %s (status=%s) page %d: %s",
                    site_url, status, page, exc,
                )
                break

            data = resp.json()
            if not data:
                break

            for item in data:
                title = strip_html(item.get("title", {}).get("rendered", ""))
                slug = item.get("slug", "")
                if title:
                    results.append({
                        "id": item.get("id"),
                        "title": title,
                        "slug": slug,
                        "link": item.get("link", ""),
                        "date": item.get("date", ""),
                    })
                if len(results) >= max_items:
                    break

            wp_total_pages = int(resp.headers.get("X-WP-TotalPages", total_pages_needed))
            if page >= wp_total_pages:
                break

            page += 1

        return results[:max_items]

    def check_site(self, domain: str) -> Dict:
        """Check a single site for all existing titles and slugs.

        Fetches published posts (unauthenticated) plus draft/scheduled posts
        (authenticated, if credentials are available).

        Args:
            domain: Site domain (e.g., "smarthomewizards.com").

        Returns:
            {"titles": set of str, "slugs": set of str, "count": int,
             "posts": list of post dicts}
        """
        preset = SITE_PRESETS.get(domain)
        if not preset:
            logger.warning("No preset for domain: %s", domain)
            return {"titles": set(), "slugs": set(), "count": 0, "posts": []}

        site_url = preset.get("wordpress_settings", {}).get("site_url", "")
        if not site_url:
            logger.warning("No site_url for domain: %s", domain)
            return {"titles": set(), "slugs": set(), "count": 0, "posts": []}

        logger.info("Checking existing titles on %s", domain)

        # Fetch published posts (no auth needed)
        all_posts = self._fetch_posts(site_url, status="publish")
        logger.info("  %s: %d published posts", domain, len(all_posts))

        # Fetch draft and scheduled posts (auth required)
        creds = self._credentials.get(domain)
        if creds:
            auth = (creds["user"], creds["app_password"])
            for status in ("draft", "future"):
                posts = self._fetch_posts(site_url, status=status, auth=auth, max_items=200)
                logger.info("  %s: %d %s posts", domain, len(posts), status)
                all_posts.extend(posts)
        else:
            logger.info("  %s: no credentials, skipping draft/scheduled", domain)

        titles = set()
        slugs = set()
        for post in all_posts:
            titles.add(post["title"])
            if post["slug"]:
                slugs.add(post["slug"])

        logger.info("  %s: %d unique titles, %d unique slugs", domain, len(titles), len(slugs))

        return {
            "titles": titles,
            "slugs": slugs,
            "count": len(titles),
            "posts": all_posts,
        }

    def check_all_sites(self) -> Dict[str, Dict]:
        """Check all 14 sites for existing titles.

        Returns:
            Dict mapping domain -> {"titles": set, "slugs": set, "count": int, "posts": list}.
        """
        results = {}
        for domain in SITE_PRESETS:
            try:
                results[domain] = self.check_site(domain)
            except Exception as exc:
                logger.error("Failed to check %s: %s", domain, exc)
                results[domain] = {"titles": set(), "slugs": set(), "count": 0, "posts": [], "error": str(exc)}
        return results

    @staticmethod
    def is_duplicate(
        candidate: str,
        existing: Dict,
        overlap_threshold: float = 0.7,
    ) -> bool:
        """Check if a candidate title is a duplicate of any existing title.

        Detection methods (any match = duplicate):
        1. Exact match on lowercased, stripped title
        2. Exact match on generated slug
        3. Token overlap ratio > threshold (default 0.7)

        Args:
            candidate: The candidate title to check.
            existing: Dict with "titles" (set) and "slugs" (set) keys.
            overlap_threshold: Minimum token overlap ratio to flag as duplicate.

        Returns:
            True if the candidate is a duplicate.
        """
        existing_titles = existing.get("titles", set())
        existing_slugs = existing.get("slugs", set())

        candidate_normalized = _normalize_for_comparison(candidate)
        candidate_slug = _title_to_slug(candidate)

        # 1. Exact title match (normalized)
        for title in existing_titles:
            if _normalize_for_comparison(title) == candidate_normalized:
                return True

        # 2. Slug match
        if candidate_slug in existing_slugs:
            return True

        # 3. Token overlap ratio
        candidate_tokens = _tokenize(candidate)
        if not candidate_tokens:
            return False

        for title in existing_titles:
            existing_tokens = _tokenize(title)
            if not existing_tokens:
                continue

            shared = candidate_tokens & existing_tokens
            max_len = max(len(candidate_tokens), len(existing_tokens))
            if max_len > 0 and len(shared) / max_len > overlap_threshold:
                return True

        return False

    def filter_duplicates(
        self,
        candidates: List[str],
        existing: Dict,
        overlap_threshold: float = 0.7,
    ) -> Tuple[List[str], List[str]]:
        """Filter a list of candidate titles, removing duplicates.

        Also checks for duplicates within the candidate list itself.

        Args:
            candidates: List of candidate titles.
            existing: Dict with "titles" and "slugs" keys.
            overlap_threshold: Token overlap threshold.

        Returns:
            Tuple of (unique_titles, duplicate_titles).
        """
        unique = []
        duplicates = []

        # Build an augmented existing set that grows as we accept candidates
        augmented = {
            "titles": set(existing.get("titles", set())),
            "slugs": set(existing.get("slugs", set())),
        }

        for candidate in candidates:
            if self.is_duplicate(candidate, augmented, overlap_threshold):
                duplicates.append(candidate)
            else:
                unique.append(candidate)
                # Add accepted candidate to augmented set for intra-list dedup
                augmented["titles"].add(candidate)
                augmented["slugs"].add(_title_to_slug(candidate))

        return unique, duplicates

    def to_serializable(self, check_result: Dict) -> Dict:
        """Convert a check_site result to JSON-serializable format.

        Converts sets to sorted lists for JSON serialization.
        """
        return {
            "titles": sorted(check_result.get("titles", set())),
            "slugs": sorted(check_result.get("slugs", set())),
            "count": check_result.get("count", 0),
            "error": check_result.get("error"),
        }
