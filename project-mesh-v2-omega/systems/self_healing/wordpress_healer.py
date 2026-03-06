"""WordPress Healer — Health checks and diagnostics for all WordPress sites."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

log = logging.getLogger(__name__)

SITES_CONFIG = Path(__file__).parent.parent.parent.parent / "config" / "sites.json"


def _load_sites() -> Dict:
    """Load site config."""
    if not SITES_CONFIG.exists():
        return {}
    try:
        data = json.loads(SITES_CONFIG.read_text("utf-8"))
        return data.get("sites", data)
    except Exception:
        return {}


class WordPressHealer:
    """Checks WordPress site health and diagnoses issues."""

    def __init__(self):
        self.sites = _load_sites()

    def check_site(self, site_id: str) -> Dict:
        """Full health check on a single WordPress site."""
        site = self.sites.get(site_id)
        if not site:
            return {"site": site_id, "status": "unknown", "error": "Site not in config"}

        domain = site.get("domain", "")
        if not domain:
            return {"site": site_id, "status": "unknown", "error": "No domain configured"}

        result = {
            "site": site_id,
            "domain": domain,
            "checks": {},
        }

        # Check 1: Homepage loads
        homepage = self._check_url(f"https://{domain}/", timeout=10)
        result["checks"]["homepage"] = homepage

        # Check 2: REST API accessible
        api = self._check_url(f"https://{domain}/wp-json/wp/v2/posts?per_page=1", timeout=10)
        result["checks"]["rest_api"] = api

        # Check 3: SSL certificate
        result["checks"]["ssl"] = {"status": "ok" if homepage.get("status_code") else "unknown"}

        # Check 4: Response time
        resp_time = homepage.get("response_time_ms", 0)
        if resp_time > 5000:
            result["checks"]["speed"] = {"status": "slow", "response_time_ms": resp_time}
        elif resp_time > 2000:
            result["checks"]["speed"] = {"status": "degraded", "response_time_ms": resp_time}
        else:
            result["checks"]["speed"] = {"status": "ok", "response_time_ms": resp_time}

        # Overall status
        if homepage.get("status") == "down":
            result["status"] = "down"
            result["diagnosis"] = self._diagnose_down(homepage)
        elif api.get("status") == "down":
            result["status"] = "degraded"
            result["diagnosis"] = "REST API unreachable — plugins may be blocking it"
        elif resp_time > 5000:
            result["status"] = "slow"
            result["diagnosis"] = "Response time over 5 seconds — check hosting/caching"
        else:
            result["status"] = "healthy"
            result["diagnosis"] = None

        return result

    def check_all(self) -> List[Dict]:
        """Check all configured WordPress sites."""
        results = []
        for site_id in self.sites:
            results.append(self.check_site(site_id))
        return results

    def _check_url(self, url: str, timeout: int = 10) -> Dict:
        """Check a URL and return status info."""
        try:
            req = Request(url, headers={"User-Agent": "EmpireMesh/1.0"})
            start = time.time()
            resp = urlopen(req, timeout=timeout)
            elapsed = time.time() - start
            return {
                "status": "ok",
                "status_code": resp.getcode(),
                "response_time_ms": round(elapsed * 1000, 1),
            }
        except HTTPError as e:
            return {
                "status": "error",
                "status_code": e.code,
                "error": str(e.reason),
                "response_time_ms": 0,
            }
        except URLError as e:
            return {
                "status": "down",
                "error": str(e.reason),
                "response_time_ms": 0,
            }
        except Exception as e:
            return {
                "status": "down",
                "error": str(e),
                "response_time_ms": 0,
            }

    def _diagnose_down(self, check_result: Dict) -> str:
        """Produce a diagnosis string for a down site."""
        error = check_result.get("error", "")
        if "SSL" in error or "certificate" in error.lower():
            return "SSL certificate issue — check cert expiry or renewal"
        if "timed out" in error.lower():
            return "Connection timeout — hosting may be down or DNS misconfigured"
        if "Name or service not known" in error:
            return "DNS resolution failed — domain may have expired"
        if "Connection refused" in error:
            return "Connection refused — server may be down"
        return f"Site unreachable: {error}"

    def get_summary(self) -> Dict:
        """Quick summary of all site health."""
        results = self.check_all()
        healthy = sum(1 for r in results if r["status"] == "healthy")
        return {
            "total": len(results),
            "healthy": healthy,
            "degraded": sum(1 for r in results if r["status"] == "degraded"),
            "slow": sum(1 for r in results if r["status"] == "slow"),
            "down": sum(1 for r in results if r["status"] == "down"),
            "sites": results,
        }
