"""
seo-toolkit -- GSC and Bing Webmaster API wrappers with keyword analysis.
Extracted from scripts/gsc_bing_checkup.py.

Provides:
- GSCClient: Google Search Console API wrapper
- BingClient: Bing Webmaster Tools API wrapper
- analyze_queries(): keyword analysis with striking distance detection
- compare_periods(): delta comparison between time periods
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)


class GSCClient:
    """Google Search Console API wrapper.

    Requires google-auth + google-api-python-client packages and a
    service account JSON key file.

    Usage:
        client = GSCClient("/path/to/service-account.json")
        data = client.fetch_performance("https://example.com/", days=28)
    """

    def __init__(self, service_account_path: str):
        self.service_account_path = service_account_path
        self._service = None

    def _get_service(self):
        """Lazy-load the GSC API service."""
        if self._service is None:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_path,
                scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
            )
            self._service = build("searchconsole", "v1",
                                  credentials=credentials)
        return self._service

    def fetch_performance(
        self,
        site_url: str,
        days: int = 28,
        row_limit: int = 100,
        dimensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch search performance data for a site.

        Args:
            site_url: GSC property URL (https://... or sc-domain:...)
            days: Number of days to query (ends 3 days ago for data freshness)
            row_limit: Max rows returned
            dimensions: Query dimensions (default: ["query"])

        Returns:
            Dict with clicks, impressions, avg_position, avg_ctr, queries list.
        """
        service = self._get_service()
        end_date = datetime.now() - timedelta(days=3)
        start_date = end_date - timedelta(days=days)

        body = {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "dimensions": dimensions or ["query"],
            "rowLimit": row_limit,
            "dataState": "final",
        }

        try:
            resp = (
                service.searchanalytics()
                .query(siteUrl=site_url, body=body)
                .execute()
            )
            rows = resp.get("rows", [])
        except Exception as e:
            log.error("GSC fetch failed for %s: %s", site_url, e)
            return {"clicks": 0, "impressions": 0, "error": str(e)}

        clicks = sum(r.get("clicks", 0) for r in rows)
        impressions = sum(r.get("impressions", 0) for r in rows)
        avg_pos = (
            sum(r.get("position", 0) * r.get("impressions", 0) for r in rows)
            / impressions
            if impressions > 0 else 0
        )
        avg_ctr = clicks / impressions if impressions > 0 else 0

        queries = []
        for r in rows:
            queries.append({
                "query": r.get("keys", [""])[0],
                "clicks": r.get("clicks", 0),
                "impressions": r.get("impressions", 0),
                "position": round(r.get("position", 0), 1),
                "ctr": round(r.get("ctr", 0), 4),
            })

        return {
            "clicks": clicks,
            "impressions": impressions,
            "avg_position": round(avg_pos, 1),
            "avg_ctr": round(avg_ctr, 4),
            "query_count": len(rows),
            "queries": queries,
            "date_range": f"{start_date.strftime('%Y-%m-%d')} to "
                          f"{end_date.strftime('%Y-%m-%d')}",
        }


class BingClient:
    """Bing Webmaster Tools API wrapper.

    Usage:
        client = BingClient("your-api-key")
        data = client.fetch_query_stats("https://example.com/")
    """

    BASE_URL = "https://ssl.bing.com/webmaster/api.svc/json"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_verified_sites(self) -> List[str]:
        """Get list of verified site URLs."""
        import requests

        try:
            resp = requests.get(
                f"{self.BASE_URL}/GetUserSites",
                params={"apikey": self.api_key},
                timeout=30,
            )
            sites = resp.json().get("d", [])
            return [s.get("Url", "").rstrip("/").lower() for s in sites]
        except Exception as e:
            log.error("Bing GetUserSites failed: %s", e)
            return []

    def fetch_query_stats(self, site_url: str) -> Dict[str, Any]:
        """Fetch query statistics for a site.

        Returns dict with clicks, impressions, query_count.
        """
        import requests

        try:
            resp = requests.get(
                f"{self.BASE_URL}/GetQueryStats",
                params={"apikey": self.api_key, "siteUrl": site_url},
                timeout=30,
            )
            stats = resp.json().get("d", [])
            clicks = sum(s.get("Clicks", 0) for s in stats) if stats else 0
            impressions = sum(s.get("Impressions", 0)
                              for s in stats) if stats else 0

            return {
                "clicks": clicks,
                "impressions": impressions,
                "query_count": len(stats) if stats else 0,
            }
        except Exception as e:
            log.error("Bing query stats failed for %s: %s", site_url, e)
            return {"clicks": 0, "impressions": 0, "error": str(e)}


def find_striking_distance(
    queries: List[Dict],
    min_position: float = 5.0,
    max_position: float = 15.0,
    min_impressions: int = 10,
) -> List[Dict]:
    """Find queries in striking distance (near page 1).

    These are high-opportunity keywords that are close to ranking
    on page 1 and could benefit from content optimization.

    Args:
        queries: List of query dicts with position and impressions keys
        min_position: Minimum position to include
        max_position: Maximum position to include
        min_impressions: Minimum impressions threshold

    Returns:
        Filtered and sorted list of striking distance queries.
    """
    striking = [
        q for q in queries
        if min_position <= q.get("position", 0) <= max_position
        and q.get("impressions", 0) >= min_impressions
    ]
    striking.sort(key=lambda x: x.get("impressions", 0), reverse=True)
    return striking


def compare_metrics(
    current: Dict[str, Any],
    previous: Dict[str, Any],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Compare two sets of metrics and compute deltas.

    Args:
        current: Current period metrics
        previous: Previous period metrics
        metric_keys: Keys to compare (default: clicks, impressions)

    Returns:
        Dict mapping each key to {current, previous, delta, direction}.
    """
    keys = metric_keys or ["clicks", "impressions"]
    result = {}
    for key in keys:
        cur = current.get(key, 0)
        prev = previous.get(key, 0)
        delta = cur - prev
        if delta > 0:
            direction = "up"
        elif delta < 0:
            direction = "down"
        else:
            direction = "flat"
        result[key] = {
            "current": cur, "previous": prev,
            "delta": delta, "direction": direction,
        }
    return result


def save_report(data: Dict, output_dir: str,
                prefix: str = "seo_report") -> str:
    """Save an SEO report as JSON. Returns the filepath."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = out / f"{prefix}_{ts}.json"
    filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.info("Report saved: %s", filepath)
    return str(filepath)
