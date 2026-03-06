"""
Search Analytics — Google Search Console + Bing Webmaster API integration.

Pulls search performance data, identifies declining pages, rising keywords,
indexing issues. Feeds data into the Site Auditor for SEO scoring.

Requirements:
    pip install google-api-python-client google-auth-oauthlib requests

GSC Auth: OAuth2 credentials at D:\\Claude Code Projects\\credentials\\gsc_credentials.json
Bing Auth: API key in D:\\Claude Code Projects\\credentials\\bing_webmaster_key.txt
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from systems.site_evolution.utils import load_site_config

log = logging.getLogger(__name__)

CREDENTIALS_DIR = Path(r"D:\Claude Code Projects\credentials")
GSC_CREDENTIALS = CREDENTIALS_DIR / "gsc_credentials.json"
GSC_TOKEN = CREDENTIALS_DIR / "gsc_token.json"
BING_KEY_FILE = CREDENTIALS_DIR / "bing_webmaster_key.txt"


class SearchAnalytics:
    """Unified GSC + Bing Webmaster search analytics for all 14 sites."""

    def __init__(self):
        self._gsc_service = None
        self._bing_key = None

    # -- Google Search Console --

    def _get_gsc_service(self):
        """Lazy-load authenticated GSC API service."""
        if self._gsc_service:
            return self._gsc_service

        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            SCOPES = [
                "https://www.googleapis.com/auth/webmasters.readonly",
                "https://www.googleapis.com/auth/webmasters",
            ]

            creds = None
            if GSC_TOKEN.exists():
                creds = Credentials.from_authorized_user_file(str(GSC_TOKEN), SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif GSC_CREDENTIALS.exists():
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(GSC_CREDENTIALS), SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                else:
                    log.warning("GSC credentials not found at %s", GSC_CREDENTIALS)
                    return None

                GSC_TOKEN.parent.mkdir(parents=True, exist_ok=True)
                GSC_TOKEN.write_text(creds.to_json(), "utf-8")

            self._gsc_service = build("searchconsole", "v1", credentials=creds)
            return self._gsc_service

        except ImportError:
            log.warning("google-api-python-client not installed. GSC features disabled.")
            return None
        except Exception as e:
            log.error("GSC auth failed: %s", e)
            return None

    def gsc_get_performance(self, site_slug: str, days: int = 28,
                            dimensions: List[str] = None,
                            row_limit: int = 100) -> Dict:
        """Get search performance data from GSC.

        Returns clicks, impressions, CTR, position by query/page.
        """
        service = self._get_gsc_service()
        if not service:
            return {"error": "GSC not available", "rows": []}

        config = load_site_config(site_slug)
        domain = config.get("domain", "")
        if not domain:
            return {"error": f"No domain for {site_slug}", "rows": []}

        site_url = f"sc-domain:{domain}"
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        request_body = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "dimensions": dimensions or ["query", "page"],
            "rowLimit": row_limit,
        }

        try:
            response = service.searchanalytics().query(
                siteUrl=site_url, body=request_body
            ).execute()

            rows = []
            for row in response.get("rows", []):
                keys = row.get("keys", [])
                rows.append({
                    "query": keys[0] if len(keys) > 0 else "",
                    "page": keys[1] if len(keys) > 1 else "",
                    "clicks": row.get("clicks", 0),
                    "impressions": row.get("impressions", 0),
                    "ctr": round(row.get("ctr", 0) * 100, 2),
                    "position": round(row.get("position", 0), 1),
                })

            return {
                "site": site_slug,
                "domain": domain,
                "period": f"{start_date} to {end_date}",
                "total_rows": len(rows),
                "rows": rows,
            }

        except Exception as e:
            log.error("GSC query failed for %s: %s", site_slug, e)
            return {"error": str(e), "rows": []}

    def gsc_get_top_queries(self, site_slug: str, days: int = 28,
                            limit: int = 50) -> List[Dict]:
        """Get top search queries by clicks."""
        result = self.gsc_get_performance(
            site_slug, days, dimensions=["query"], row_limit=limit
        )
        return sorted(result.get("rows", []), key=lambda r: r.get("clicks", 0), reverse=True)

    def gsc_get_declining_pages(self, site_slug: str) -> List[Dict]:
        """Identify pages losing traffic (compare last 28d vs previous 28d)."""
        recent = self.gsc_get_performance(
            site_slug, days=28, dimensions=["page"], row_limit=500
        )
        previous = self.gsc_get_performance(
            site_slug, days=56, dimensions=["page"], row_limit=500
        )

        recent_map = {r["page"]: r for r in recent.get("rows", []) if r.get("page")}
        previous_map = {r["page"]: r for r in previous.get("rows", []) if r.get("page")}

        declining = []
        for page, prev_data in previous_map.items():
            curr_data = recent_map.get(page, {"clicks": 0, "impressions": 0})
            click_change = curr_data.get("clicks", 0) - prev_data.get("clicks", 0)
            if click_change < -5:
                declining.append({
                    "page": page,
                    "clicks_current": curr_data.get("clicks", 0),
                    "clicks_previous": prev_data.get("clicks", 0),
                    "click_change": click_change,
                    "position_current": curr_data.get("position", 0),
                    "position_previous": prev_data.get("position", 0),
                })

        return sorted(declining, key=lambda d: d["click_change"])

    def gsc_get_rising_keywords(self, site_slug: str) -> List[Dict]:
        """Identify keywords gaining impressions/clicks."""
        recent = self.gsc_get_performance(
            site_slug, days=14, dimensions=["query"], row_limit=500
        )
        previous = self.gsc_get_performance(
            site_slug, days=28, dimensions=["query"], row_limit=500
        )

        recent_map = {r["query"]: r for r in recent.get("rows", []) if r.get("query")}
        previous_map = {r["query"]: r for r in previous.get("rows", []) if r.get("query")}

        rising = []
        for query, curr in recent_map.items():
            prev = previous_map.get(query, {"impressions": 0, "clicks": 0})
            imp_change = curr.get("impressions", 0) - prev.get("impressions", 0)
            if imp_change > 10:
                rising.append({
                    "query": query,
                    "impressions_current": curr.get("impressions", 0),
                    "impressions_previous": prev.get("impressions", 0),
                    "impression_change": imp_change,
                    "clicks": curr.get("clicks", 0),
                    "position": curr.get("position", 0),
                })

        return sorted(rising, key=lambda r: r["impression_change"], reverse=True)

    def gsc_inspect_url(self, site_slug: str, url: str) -> Dict:
        """Inspect a URL's indexing status via GSC URL Inspection API."""
        service = self._get_gsc_service()
        if not service:
            return {"error": "GSC not available"}

        config = load_site_config(site_slug)
        domain = config.get("domain", "")
        site_url = f"sc-domain:{domain}"

        try:
            result = service.urlInspection().index().inspect(
                body={
                    "inspectionUrl": url,
                    "siteUrl": site_url,
                }
            ).execute()

            inspection = result.get("inspectionResult", {})
            index_status = inspection.get("indexStatusResult", {})

            return {
                "url": url,
                "verdict": index_status.get("verdict", "UNKNOWN"),
                "coverage_state": index_status.get("coverageState", ""),
                "indexing_state": index_status.get("indexingState", ""),
                "page_fetch_state": index_status.get("pageFetchState", ""),
                "robots_txt_state": index_status.get("robotsTxtState", ""),
                "last_crawl_time": index_status.get("lastCrawlTime", ""),
            }
        except Exception as e:
            log.error("URL inspection failed for %s: %s", url, e)
            return {"error": str(e)}

    def gsc_submit_sitemap(self, site_slug: str,
                           sitemap_path: str = "/sitemap_index.xml") -> Dict:
        """Submit a sitemap to GSC."""
        service = self._get_gsc_service()
        if not service:
            return {"error": "GSC not available"}

        config = load_site_config(site_slug)
        domain = config.get("domain", "")
        site_url = f"sc-domain:{domain}"
        sitemap_url = f"https://{domain}{sitemap_path}"

        try:
            service.sitemaps().submit(
                siteUrl=site_url, feedpath=sitemap_url
            ).execute()
            return {"status": "submitted", "sitemap": sitemap_url}
        except Exception as e:
            return {"error": str(e)}

    def gsc_list_sitemaps(self, site_slug: str) -> List[Dict]:
        """List all sitemaps registered in GSC."""
        service = self._get_gsc_service()
        if not service:
            return []

        config = load_site_config(site_slug)
        domain = config.get("domain", "")
        site_url = f"sc-domain:{domain}"

        try:
            result = service.sitemaps().list(siteUrl=site_url).execute()
            return [
                {
                    "path": sm.get("path", ""),
                    "last_submitted": sm.get("lastSubmitted", ""),
                    "last_downloaded": sm.get("lastDownloaded", ""),
                    "warnings": sm.get("warnings", 0),
                    "errors": sm.get("errors", 0),
                }
                for sm in result.get("sitemap", [])
            ]
        except Exception as e:
            log.error("Sitemap list failed for %s: %s", site_slug, e)
            return []

    # -- Bing Webmaster --

    def _get_bing_key(self) -> Optional[str]:
        """Load Bing Webmaster API key."""
        if self._bing_key:
            return self._bing_key
        if BING_KEY_FILE.exists():
            self._bing_key = BING_KEY_FILE.read_text("utf-8").strip()
            return self._bing_key
        log.warning("Bing API key not found at %s", BING_KEY_FILE)
        return None

    def _bing_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated Bing Webmaster API request."""
        import requests

        key = self._get_bing_key()
        if not key:
            return {"error": "Bing API key not configured"}

        url = f"https://ssl.bing.com/webmaster/api.svc/json/{endpoint}"
        params = params or {}
        params["apikey"] = key

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.Timeout:
            log.warning("Bing API timeout on %s", endpoint)
            return {"error": f"Bing API timeout on {endpoint}"}
        except requests.ConnectionError:
            log.warning("Bing API connection failed on %s", endpoint)
            return {"error": "Bing API connection failed"}
        except requests.HTTPError as e:
            log.warning("Bing API HTTP error on %s: %s", endpoint, e)
            return {"error": f"Bing API error: {resp.status_code}"}

        try:
            return resp.json()
        except ValueError:
            log.warning("Bing API non-JSON response on %s", endpoint)
            return {"error": "Non-JSON response from Bing API"}

    def bing_get_traffic(self, site_slug: str) -> Dict:
        """Get URL traffic info from Bing Webmaster."""
        config = load_site_config(site_slug)
        domain = config.get("domain", "")
        if not domain:
            return {"error": f"No domain for {site_slug}"}

        try:
            result = self._bing_request("GetUrlTrafficInfo", {
                "siteUrl": f"https://{domain}/",
            })
            return {
                "site": site_slug,
                "domain": domain,
                "data": result,
            }
        except Exception as e:
            return {"error": str(e)}

    def bing_get_query_stats(self, site_slug: str) -> Dict:
        """Get search query statistics from Bing."""
        config = load_site_config(site_slug)
        domain = config.get("domain", "")
        if not domain:
            return {"error": f"No domain for {site_slug}"}

        try:
            result = self._bing_request("GetQueryStats", {
                "siteUrl": f"https://{domain}/",
            })
            return {
                "site": site_slug,
                "domain": domain,
                "queries": result,
            }
        except Exception as e:
            return {"error": str(e)}

    def bing_submit_url(self, site_slug: str, url: str) -> Dict:
        """Submit a URL to Bing for indexing."""
        import requests

        key = self._get_bing_key()
        if not key:
            return {"error": "Bing API key not configured"}

        config = load_site_config(site_slug)
        domain = config.get("domain", "")

        try:
            resp = requests.post(
                "https://ssl.bing.com/webmaster/api.svc/json/SubmitUrl",
                params={"apikey": key},
                json={
                    "siteUrl": f"https://{domain}/",
                    "url": url,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return {"status": "submitted", "url": url}
        except Exception as e:
            return {"error": str(e)}

    def bing_submit_sitemap(self, site_slug: str,
                            sitemap_path: str = "/sitemap_index.xml") -> Dict:
        """Submit sitemap to Bing."""
        import requests

        key = self._get_bing_key()
        if not key:
            return {"error": "Bing API key not configured"}

        config = load_site_config(site_slug)
        domain = config.get("domain", "")
        sitemap_url = f"https://{domain}{sitemap_path}"

        try:
            resp = requests.post(
                "https://ssl.bing.com/webmaster/api.svc/json/SubmitFeed",
                params={"apikey": key},
                json={
                    "siteUrl": f"https://{domain}/",
                    "feedUrl": sitemap_url,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return {"status": "submitted", "sitemap": sitemap_url}
        except Exception as e:
            return {"error": str(e)}

    def bing_get_crawl_stats(self, site_slug: str) -> Dict:
        """Get crawl statistics from Bing."""
        config = load_site_config(site_slug)
        domain = config.get("domain", "")
        if not domain:
            return {"error": f"No domain for {site_slug}"}

        try:
            result = self._bing_request("GetCrawlStats", {
                "siteUrl": f"https://{domain}/",
            })
            return {"site": site_slug, "crawl_stats": result}
        except Exception as e:
            return {"error": str(e)}

    # -- Combined Analytics --

    def get_full_analytics(self, site_slug: str) -> Dict:
        """Get combined GSC + Bing analytics for a site with summary metrics."""
        top_queries = self.gsc_get_top_queries(site_slug, limit=20)
        declining = self.gsc_get_declining_pages(site_slug)[:10]
        rising = self.gsc_get_rising_keywords(site_slug)[:10]
        sitemaps = self.gsc_list_sitemaps(site_slug)
        health = self.get_seo_health_score(site_slug)

        # Calculate summary metrics
        total_clicks = sum(q.get("clicks", 0) for q in top_queries)
        total_impressions = sum(q.get("impressions", 0) for q in top_queries)
        avg_position = (
            sum(q.get("position", 0) for q in top_queries) / len(top_queries)
            if top_queries else 0
        )
        avg_ctr = (
            sum(q.get("ctr", 0) for q in top_queries) / len(top_queries)
            if top_queries else 0
        )

        return {
            "site": site_slug,
            "summary": {
                "seo_health_score": health.get("score", 0),
                "total_clicks_28d": total_clicks,
                "total_impressions_28d": total_impressions,
                "avg_position": round(avg_position, 1),
                "avg_ctr": round(avg_ctr, 2),
                "declining_pages_count": len(declining),
                "rising_keywords_count": len(rising),
                "issues": health.get("issues", []),
                "data_sources": health.get("data_available", {}),
            },
            "gsc": {
                "top_queries": top_queries,
                "declining_pages": declining,
                "rising_keywords": rising,
                "sitemaps": sitemaps,
            },
            "bing": {
                "traffic": self.bing_get_traffic(site_slug),
                "query_stats": self.bing_get_query_stats(site_slug),
                "crawl_stats": self.bing_get_crawl_stats(site_slug),
            },
        }

    def get_seo_health_score(self, site_slug: str) -> Dict:
        """Calculate an SEO health score from search analytics data."""
        score = 50  # Base score
        issues = []

        # GSC data
        gsc_perf = self.gsc_get_performance(site_slug, days=28, dimensions=["query"])
        rows = gsc_perf.get("rows", [])

        if gsc_perf.get("error"):
            issues.append({"type": "warning", "msg": f"GSC unavailable: {gsc_perf['error']}"})
        elif rows:
            total_clicks = sum(r.get("clicks", 0) for r in rows)
            total_impressions = sum(r.get("impressions", 0) for r in rows)
            avg_position = sum(r.get("position", 50) for r in rows) / len(rows)

            if total_clicks > 100:
                score += 15
            elif total_clicks > 10:
                score += 8

            if total_impressions > 1000:
                score += 10
            elif total_impressions > 100:
                score += 5

            if avg_position < 20:
                score += 15
            elif avg_position < 40:
                score += 8

            # Check for declining pages
            declining = self.gsc_get_declining_pages(site_slug)
            if len(declining) > 10:
                score -= 15
                issues.append({"type": "critical", "msg": f"{len(declining)} pages declining"})
            elif len(declining) > 5:
                score -= 8
                issues.append({"type": "warning", "msg": f"{len(declining)} pages declining"})
        else:
            score -= 10
            issues.append({"type": "info", "msg": "No GSC data available"})

        # Bing data
        bing_traffic = self.bing_get_traffic(site_slug)
        if not bing_traffic.get("error"):
            score += 5  # Bing is configured

        return {
            "score": min(100, max(0, score)),
            "issues": issues,
            "data_available": {
                "gsc": not bool(gsc_perf.get("error")),
                "bing": not bool(bing_traffic.get("error")),
            },
        }
