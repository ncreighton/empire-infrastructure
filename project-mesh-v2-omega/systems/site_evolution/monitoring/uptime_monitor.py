"""
Uptime Monitor — Parallel ping all 14 sites, SSL certificate checks,
DNS resolution, response time tracking with codex persistence.
"""

import logging
import ssl
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional

from systems.site_evolution.utils import load_site_config, get_all_site_slugs, get_site_domain

log = logging.getLogger(__name__)


class UptimeMonitor:
    """Monitor site uptime, response times, and SSL certificates."""

    def check_site(self, site_slug: str) -> Dict:
        """Check a single site: HTTP response time, SSL expiry, DNS.

        Returns: {site_slug, status_code, response_ms, ssl_valid, ssl_expiry_days, error}
        """
        domain = get_site_domain(site_slug)
        result = {
            "site_slug": site_slug,
            "domain": domain,
            "status_code": 0,
            "response_ms": 0,
            "ssl_valid": False,
            "ssl_expiry_days": 0,
            "error": None,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

        if not domain:
            result["error"] = "No domain configured"
            return result

        # HTTP check
        try:
            import requests
            start = datetime.now()
            resp = requests.get(f"https://{domain}", timeout=15,
                                headers={"User-Agent": "EvoUptimeMonitor/1.0"},
                                allow_redirects=True)
            elapsed_ms = int((datetime.now() - start).total_seconds() * 1000)

            result["status_code"] = resp.status_code
            result["response_ms"] = elapsed_ms
        except Exception as e:
            result["error"] = str(e)

        # SSL check
        try:
            ssl_info = self._check_ssl(domain)
            result["ssl_valid"] = ssl_info.get("valid", False)
            result["ssl_expiry_days"] = ssl_info.get("days_until_expiry", 0)
        except Exception as e:
            log.debug("SSL check failed for %s: %s", domain, e)

        # Record in codex
        try:
            from systems.site_evolution import codex
            codex.record_uptime_check(
                site_slug=site_slug,
                status_code=result["status_code"],
                response_ms=result["response_ms"],
                ssl_valid=result["ssl_valid"],
                ssl_expiry_days=result["ssl_expiry_days"],
                error=result.get("error"),
            )
        except Exception as e:
            log.debug("Could not record uptime check: %s", e)

        return result

    def check_all_sites(self) -> Dict:
        """Parallel ping all sites. Returns aggregate results."""
        sites = get_all_site_slugs()
        results = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.check_site, slug): slug for slug in sites}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    slug = futures[future]
                    results.append({"site_slug": slug, "error": str(e)})

        # Aggregate stats
        up = [r for r in results if r.get("status_code") in (200, 301, 302)]
        down = [r for r in results if r.get("status_code", 0) == 0 or r.get("status_code", 0) >= 500]
        response_times = [r["response_ms"] for r in results if r.get("response_ms", 0) > 0]

        return {
            "total_sites": len(results),
            "sites_up": len(up),
            "sites_down": len(down),
            "avg_response_ms": sum(response_times) // max(len(response_times), 1),
            "max_response_ms": max(response_times) if response_times else 0,
            "min_response_ms": min(response_times) if response_times else 0,
            "results": sorted(results, key=lambda r: r.get("response_ms", 99999)),
        }

    def check_ssl_certificates(self) -> List[Dict]:
        """Check SSL certificates for all sites. Flag expiring within 30 days."""
        sites = get_all_site_slugs()
        warnings = []

        for slug in sites:
            domain = get_site_domain(slug)
            if not domain:
                continue

            try:
                info = self._check_ssl(domain)
                if info.get("days_until_expiry", 999) <= 30:
                    warnings.append({
                        "site_slug": slug,
                        "domain": domain,
                        "days_until_expiry": info["days_until_expiry"],
                        "expiry_date": info.get("expiry_date", ""),
                        "severity": "critical" if info["days_until_expiry"] <= 7 else "warning",
                    })
            except Exception as e:
                warnings.append({
                    "site_slug": slug,
                    "domain": domain,
                    "error": str(e),
                    "severity": "warning",
                })

        return warnings

    def get_response_time_history(self, site_slug: str, limit: int = 50) -> List[Dict]:
        """Get historical response times from codex uptime_checks table."""
        try:
            from systems.site_evolution import codex
            return codex.get_uptime_history(site_slug, limit)
        except Exception:
            return []

    def _check_ssl(self, domain: str) -> Dict:
        """Check SSL certificate validity and expiry."""
        context = ssl.create_default_context()
        try:
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    expiry_str = cert.get("notAfter", "")
                    if expiry_str:
                        # Parse SSL date format: 'Sep 30 12:00:00 2025 GMT'
                        expiry = datetime.strptime(expiry_str, "%b %d %H:%M:%S %Y %Z")
                        expiry = expiry.replace(tzinfo=timezone.utc)
                        days_left = (expiry - datetime.now(timezone.utc)).days
                        return {
                            "valid": True,
                            "days_until_expiry": days_left,
                            "expiry_date": expiry.isoformat(),
                        }
        except Exception as e:
            return {"valid": False, "days_until_expiry": 0, "error": str(e)}

        return {"valid": False, "days_until_expiry": 0}
