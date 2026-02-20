"""
Link Pack Builder for ZimmWriter.

Scrapes published post/page URLs from WordPress sites via the REST API
and generates ZimmWriter-compatible link pack files.

ZimmWriter link pack format: one line per link, pipe-separated:
    URL|summary text

Usage:
    python -m src.link_pack_builder                 # Build all 14 sites
    python -m src.link_pack_builder --site smarthomewizards.com  # Single site
"""

import argparse
import logging
import math
import os
import re
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = r"D:\Claude Code Projects\zimmwriter-project-new\data\link_packs"


def strip_html(text: str) -> str:
    """Remove HTML tags from a string and collapse whitespace."""
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = cleaned.replace("&nbsp;", " ")
    cleaned = cleaned.replace("&amp;", "&")
    cleaned = cleaned.replace("&lt;", "<")
    cleaned = cleaned.replace("&gt;", ">")
    cleaned = cleaned.replace("&#8217;", "'")
    cleaned = cleaned.replace("&#8216;", "'")
    cleaned = cleaned.replace("&#8220;", '"')
    cleaned = cleaned.replace("&#8221;", '"')
    cleaned = cleaned.replace("&hellip;", "...")
    cleaned = re.sub(r"&[#\w]+;", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class WordPressLinkScraper:
    """Scrapes published post and page URLs from a WordPress site via REST API."""

    def __init__(self, site_url: str, per_page: int = 100):
        self.site_url = site_url.rstrip("/")
        self.per_page = min(per_page, 100)  # WP REST API caps at 100
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ZimmWriter-LinkPackBuilder/1.0",
            "Accept": "application/json",
        })

    def _fetch_collection(self, endpoint: str, max_items: int) -> List[dict]:
        """Generic paginated fetch for posts or pages endpoint."""
        results = []
        total_pages_needed = math.ceil(max_items / self.per_page)
        page = 1

        while page <= total_pages_needed and len(results) < max_items:
            url = (
                f"{self.site_url}/wp-json/wp/v2/{endpoint}"
                f"?per_page={self.per_page}&page={page}"
                f"&_fields=id,title,link,excerpt&status=publish"
            )
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 400:
                    # Page beyond range
                    logger.debug(
                        "%s page %d returned 400 — no more results",
                        endpoint, page,
                    )
                    break
                resp.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                logger.warning(
                    "HTTP %s fetching %s page %d from %s: %s",
                    getattr(exc.response, "status_code", "?"),
                    endpoint, page, self.site_url, exc,
                )
                break
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Request error fetching %s page %d from %s: %s",
                    endpoint, page, self.site_url, exc,
                )
                break

            data = resp.json()
            if not data:
                break

            for item in data:
                title = strip_html(item.get("title", {}).get("rendered", ""))
                excerpt = strip_html(item.get("excerpt", {}).get("rendered", ""))
                link = item.get("link", "")
                if link and title:
                    results.append({
                        "url": link,
                        "title": title,
                        "excerpt": excerpt,
                    })
                if len(results) >= max_items:
                    break

            # Check WP total pages header to avoid unnecessary requests
            wp_total_pages = int(resp.headers.get("X-WP-TotalPages", total_pages_needed))
            if page >= wp_total_pages:
                break

            page += 1

        return results[:max_items]

    def fetch_posts(self, max_posts: int = 200) -> List[dict]:
        """Fetch published posts. Returns list of {url, title, excerpt}."""
        logger.info("Fetching up to %d posts from %s", max_posts, self.site_url)
        return self._fetch_collection("posts", max_posts)

    def fetch_pages(self, max_pages: int = 50) -> List[dict]:
        """Fetch published pages. Returns list of {url, title, excerpt}."""
        logger.info("Fetching up to %d pages from %s", max_pages, self.site_url)
        return self._fetch_collection("pages", max_pages)


class LinkPackBuilder:
    """Builds ZimmWriter link pack text files from WordPress site content."""

    def __init__(self):
        self.packs: Dict[str, str] = {}

    @staticmethod
    def _format_line(item: dict) -> str:
        """Format a single link pack line: URL|title - excerpt_snippet."""
        url = item["url"]
        title = item["title"]
        excerpt = item.get("excerpt", "")

        # Truncate excerpt to ~100 chars at a word boundary
        if len(excerpt) > 100:
            excerpt = excerpt[:100].rsplit(" ", 1)[0].rstrip(".,;:!?") + "..."

        if excerpt:
            summary = f"{title} - {excerpt}"
        else:
            summary = title

        return f"{url}|{summary}"

    def build_pack(
        self,
        domain: str,
        site_url: str,
        max_posts: int = 200,
    ) -> str:
        """
        Scrape a WordPress site and build a link pack string.

        Args:
            domain: The site domain (e.g. "smarthomewizards.com").
            site_url: Full URL (e.g. "https://smarthomewizards.com").
            max_posts: Maximum number of posts to fetch.

        Returns:
            Link pack text with one URL|summary per line.
        """
        scraper = WordPressLinkScraper(site_url)
        posts = scraper.fetch_posts(max_posts=max_posts)
        pages = scraper.fetch_pages(max_pages=50)

        all_items = posts + pages
        lines = [self._format_line(item) for item in all_items]
        pack_text = "\n".join(lines)

        self.packs[domain] = pack_text
        logger.info(
            "Built link pack for %s: %d posts + %d pages = %d links",
            domain, len(posts), len(pages), len(all_items),
        )
        return pack_text

    def build_all_packs(self, site_configs: dict) -> Dict[str, str]:
        """
        Build link packs for multiple sites.

        Args:
            site_configs: Dict of {domain: {"site_url": "https://..."}} or
                          {domain: preset_dict_with_wordpress_settings}.

        Returns:
            Dict of {domain: pack_text}.
        """
        packs = {}
        for domain, config in site_configs.items():
            # Support both flat {"site_url": ...} and nested wordpress_settings
            if "site_url" in config:
                site_url = config["site_url"]
            elif "wordpress_settings" in config:
                site_url = config["wordpress_settings"].get("site_url", "")
            else:
                logger.warning("No site_url found for %s, skipping", domain)
                continue

            if not site_url:
                logger.warning("Empty site_url for %s, skipping", domain)
                continue

            try:
                pack_text = self.build_pack(domain, site_url)
                packs[domain] = pack_text
            except Exception as exc:
                logger.error("Failed to build pack for %s: %s", domain, exc)

        return packs

    def save_pack(
        self,
        domain: str,
        pack_text: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Save a link pack to a text file.

        Args:
            domain: Site domain used for the filename.
            pack_text: The link pack content.
            output_dir: Directory to save into. Defaults to data/link_packs.

        Returns:
            Absolute file path of the saved pack.
        """
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR

        os.makedirs(output_dir, exist_ok=True)

        # Filename: domain_internal.txt (replace dots/hyphens for cleanliness)
        safe_name = domain.replace(".", "_").replace("-", "_")
        filename = f"{safe_name}_internal.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(pack_text)

        logger.info("Saved link pack: %s", filepath)
        return filepath

    def save_all_packs(
        self,
        packs: Dict[str, str],
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save all link packs to disk.

        Args:
            packs: Dict of {domain: pack_text}.
            output_dir: Directory to save into.

        Returns:
            Dict of {domain: filepath}.
        """
        filepaths = {}
        for domain, pack_text in packs.items():
            if pack_text.strip():
                filepath = self.save_pack(domain, pack_text, output_dir)
                filepaths[domain] = filepath
            else:
                logger.warning("Empty pack for %s, not saving", domain)
        return filepaths


def build_packs_from_presets() -> dict:
    """
    Build link packs for all sites defined in SITE_PRESETS.

    Imports site_presets.SITE_PRESETS and iterates over all domains,
    extracting wordpress_settings.site_url for each.

    Returns:
        Dict of {domain: pack_text}.
    """
    from .site_presets import SITE_PRESETS

    builder = LinkPackBuilder()
    packs = builder.build_all_packs(SITE_PRESETS)
    filepaths = builder.save_all_packs(packs)

    print(f"\nLink Pack Summary")
    print(f"{'=' * 55}")
    total_links = 0
    for domain in sorted(filepaths.keys()):
        pack_text = packs[domain]
        link_count = len([line for line in pack_text.splitlines() if line.strip()])
        total_links += link_count
        print(f"  {domain:<35} {link_count:>4} links")
    print(f"{'=' * 55}")
    print(f"  {'TOTAL':<35} {total_links:>4} links")
    print(f"\nFiles saved to: {DEFAULT_OUTPUT_DIR}")

    return packs


def _build_single_site(domain: str) -> dict:
    """Build link pack for a single site by domain lookup in SITE_PRESETS."""
    from .site_presets import SITE_PRESETS

    preset = SITE_PRESETS.get(domain)
    if not preset:
        print(f"Error: Domain '{domain}' not found in SITE_PRESETS.")
        print(f"Available domains: {', '.join(sorted(SITE_PRESETS.keys()))}")
        return {}

    site_url = preset.get("wordpress_settings", {}).get("site_url", "")
    if not site_url:
        print(f"Error: No site_url configured for '{domain}'.")
        return {}

    builder = LinkPackBuilder()
    pack_text = builder.build_pack(domain, site_url)
    filepath = builder.save_pack(domain, pack_text)

    link_count = len([line for line in pack_text.splitlines() if line.strip()])
    print(f"\nLink Pack: {domain}")
    print(f"  URLs scraped: {link_count}")
    print(f"  Saved to: {filepath}")

    return {domain: pack_text}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build ZimmWriter link packs from WordPress sites.",
    )
    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help="Single domain to build (e.g. smarthomewizards.com). Omit for all sites.",
    )
    args = parser.parse_args()

    if args.site:
        _build_single_site(args.site)
    else:
        build_packs_from_presets()
