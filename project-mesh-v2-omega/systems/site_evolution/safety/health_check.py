"""
Post-Deploy Health Check — Verifies a site is still healthy after deployment.

Runs 5 checks: homepage status, response time, PHP errors, robots.txt, sitemap.
Any failure = unhealthy → triggers rollback for PROTECTED/GUARDED sites.
"""

import logging
import re
import time
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Patterns that indicate PHP fatal/parse errors in page output
PHP_ERROR_PATTERNS = [
    r"Fatal error:",
    r"Parse error:",
    r"Warning:.*on line \d+",
    r"<b>Fatal error</b>",
    r"<b>Parse error</b>",
    r"Uncaught Error:",
    r"Uncaught Exception:",
    r"Call to undefined function",
    r"Cannot redeclare",
    r"Allowed memory size.*exhausted",
]

_PHP_ERROR_RE = re.compile("|".join(PHP_ERROR_PATTERNS), re.IGNORECASE)

# Maximum acceptable response time in milliseconds
MAX_RESPONSE_MS = 3000


class PostDeployHealthCheck:
    """Runs health checks against a live WordPress site."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def check_site_health(self, site_slug: str) -> Dict:
        """Run all health checks for a site.

        Returns:
            {
                "healthy": bool,
                "checks": [{"name": str, "passed": bool, "detail": str}, ...],
                "failed": [str, ...],  # names of failed checks
            }
        """
        from systems.site_evolution.utils import get_site_domain
        domain = get_site_domain(site_slug)

        if not domain:
            return {
                "healthy": False,
                "checks": [],
                "failed": ["no_domain"],
                "error": f"No domain found for {site_slug}",
            }

        base_url = f"https://{domain}"
        checks = []
        failed = []

        # 1. Homepage status + response time
        homepage_result = self._check_homepage(base_url)
        checks.append(homepage_result)
        if not homepage_result["passed"]:
            failed.append("homepage_status")

        # 2. Response time (from homepage check)
        time_result = self._check_response_time(homepage_result)
        checks.append(time_result)
        if not time_result["passed"]:
            failed.append("response_time")

        # 3. PHP errors in homepage body
        php_result = self._check_php_errors(homepage_result)
        checks.append(php_result)
        if not php_result["passed"]:
            failed.append("php_errors")

        # 4. robots.txt
        robots_result = self._check_robots(base_url)
        checks.append(robots_result)
        if not robots_result["passed"]:
            failed.append("robots_txt")

        # 5. Sitemap
        sitemap_result = self._check_sitemap(base_url)
        checks.append(sitemap_result)
        if not sitemap_result["passed"]:
            failed.append("sitemap")

        healthy = len(failed) == 0

        log.info(
            "Health check for %s: %s (%d/%d passed)",
            site_slug,
            "HEALTHY" if healthy else f"UNHEALTHY ({failed})",
            len(checks) - len(failed),
            len(checks),
        )

        return {
            "healthy": healthy,
            "checks": checks,
            "failed": failed,
        }

    def _check_homepage(self, base_url: str) -> Dict:
        """GET homepage, verify 200, capture response time and body."""
        try:
            import requests
            start = time.time()
            resp = requests.get(
                base_url, timeout=self.timeout,
                headers={"User-Agent": "EmpireHealthCheck/1.0"},
                allow_redirects=True,
            )
            elapsed_ms = int((time.time() - start) * 1000)

            passed = resp.status_code == 200
            return {
                "name": "homepage_status",
                "passed": passed,
                "detail": f"HTTP {resp.status_code} in {elapsed_ms}ms",
                "_status_code": resp.status_code,
                "_elapsed_ms": elapsed_ms,
                "_body": resp.text[:10000],  # keep first 10K for PHP error scanning
            }
        except Exception as e:
            return {
                "name": "homepage_status",
                "passed": False,
                "detail": f"Connection failed: {e}",
                "_status_code": 0,
                "_elapsed_ms": 0,
                "_body": "",
            }

    def _check_response_time(self, homepage_result: Dict) -> Dict:
        """Check if response time is acceptable."""
        elapsed_ms = homepage_result.get("_elapsed_ms", 0)
        passed = 0 < elapsed_ms <= MAX_RESPONSE_MS
        return {
            "name": "response_time",
            "passed": passed,
            "detail": f"{elapsed_ms}ms (max {MAX_RESPONSE_MS}ms)",
        }

    def _check_php_errors(self, homepage_result: Dict) -> Dict:
        """Scan homepage body for PHP fatal/parse error patterns."""
        body = homepage_result.get("_body", "")
        if not body:
            return {
                "name": "php_errors",
                "passed": True,
                "detail": "No body to scan (homepage failed)",
            }

        match = _PHP_ERROR_RE.search(body)
        if match:
            return {
                "name": "php_errors",
                "passed": False,
                "detail": f"PHP error detected: {match.group()[:100]}",
            }
        return {
            "name": "php_errors",
            "passed": True,
            "detail": "No PHP errors detected",
        }

    def _check_robots(self, base_url: str) -> Dict:
        """Verify robots.txt exists and doesn't block everything."""
        try:
            import requests
            resp = requests.get(
                f"{base_url}/robots.txt", timeout=self.timeout,
                headers={"User-Agent": "EmpireHealthCheck/1.0"},
            )
            if resp.status_code != 200:
                return {
                    "name": "robots_txt",
                    "passed": False,
                    "detail": f"HTTP {resp.status_code} — robots.txt not found",
                }

            body = resp.text.lower()
            # Check for "disallow: /" blocking everything
            if "disallow: /" in body and "disallow: /wp-" not in body:
                # Might be blocking everything — check more carefully
                lines = body.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line == "disallow: /":
                        return {
                            "name": "robots_txt",
                            "passed": False,
                            "detail": "robots.txt blocks all crawlers (Disallow: /)",
                        }

            return {
                "name": "robots_txt",
                "passed": True,
                "detail": f"robots.txt OK ({len(resp.text)} bytes)",
            }
        except Exception as e:
            return {
                "name": "robots_txt",
                "passed": False,
                "detail": f"robots.txt check failed: {e}",
            }

    def _check_sitemap(self, base_url: str) -> Dict:
        """Verify sitemap exists at common locations."""
        try:
            import requests
            sitemap_paths = ["/sitemap_index.xml", "/sitemap.xml"]

            for path in sitemap_paths:
                try:
                    resp = requests.get(
                        f"{base_url}{path}", timeout=self.timeout,
                        headers={"User-Agent": "EmpireHealthCheck/1.0"},
                    )
                    if resp.status_code == 200 and "<?xml" in resp.text[:200]:
                        return {
                            "name": "sitemap",
                            "passed": True,
                            "detail": f"Sitemap found at {path} ({len(resp.text)} bytes)",
                        }
                except Exception:
                    continue

            return {
                "name": "sitemap",
                "passed": False,
                "detail": "No sitemap found at /sitemap_index.xml or /sitemap.xml",
            }
        except Exception as e:
            return {
                "name": "sitemap",
                "passed": False,
                "detail": f"Sitemap check failed: {e}",
            }
