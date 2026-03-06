"""Traffic Investigator — Automated traffic drop investigation using Supabase data."""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def _get_supabase():
    """Get Supabase client if available."""
    try:
        from supabase import create_client
        import os
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if url and key:
            return create_client(url, key)
    except ImportError:
        pass
    return None


class TrafficInvestigator:
    """Investigates traffic drops across the empire using analytics data."""

    def __init__(self):
        self.supabase = _get_supabase()

    def investigate_site(self, site_slug: str) -> Dict:
        """Investigate traffic changes for a specific site."""
        if not self.supabase:
            return {"site": site_slug, "status": "unavailable",
                    "reason": "Supabase not configured"}

        result = {
            "site": site_slug,
            "investigated_at": datetime.now().isoformat(),
            "findings": [],
            "severity": "normal",
        }

        # Check GSC performance trends
        gsc_findings = self._check_gsc_trends(site_slug)
        result["findings"].extend(gsc_findings)

        # Check GA4 for traffic drops
        ga4_findings = self._check_ga4_trends(site_slug)
        result["findings"].extend(ga4_findings)

        # Check for keyword ranking drops
        keyword_findings = self._check_keyword_drops(site_slug)
        result["findings"].extend(keyword_findings)

        # Determine severity
        critical = sum(1 for f in result["findings"] if f.get("severity") == "critical")
        warning = sum(1 for f in result["findings"] if f.get("severity") == "warning")

        if critical > 0:
            result["severity"] = "critical"
        elif warning > 0:
            result["severity"] = "warning"

        return result

    def investigate_all(self) -> List[Dict]:
        """Investigate all sites for traffic anomalies."""
        if not self.supabase:
            return [{"status": "unavailable", "reason": "Supabase not configured"}]

        results = []
        try:
            # Get distinct sites from gsc_performance
            resp = self.supabase.table("gsc_performance").select("site_slug").execute()
            sites = list({r["site_slug"] for r in (resp.data or [])})
            for site in sites:
                results.append(self.investigate_site(site))
        except Exception as e:
            log.error(f"Investigation error: {e}")
            results.append({"status": "error", "error": str(e)})

        return results

    def _check_gsc_trends(self, site_slug: str) -> List[Dict]:
        """Check GSC performance for click/impression drops."""
        findings = []
        try:
            # Get recent vs previous period data
            resp = self.supabase.table("gsc_performance") \
                .select("clicks,impressions,date") \
                .eq("site_slug", site_slug) \
                .order("date", desc=True) \
                .limit(60) \
                .execute()

            rows = resp.data or []
            if len(rows) < 14:
                return findings

            recent = rows[:7]
            previous = rows[7:14]

            recent_clicks = sum(r.get("clicks", 0) for r in recent)
            prev_clicks = sum(r.get("clicks", 0) for r in previous)

            if prev_clicks > 0:
                change_pct = ((recent_clicks - prev_clicks) / prev_clicks) * 100
                if change_pct < -30:
                    findings.append({
                        "type": "gsc_click_drop",
                        "severity": "critical",
                        "message": f"GSC clicks dropped {change_pct:.0f}% week-over-week",
                        "recent_clicks": recent_clicks,
                        "previous_clicks": prev_clicks,
                    })
                elif change_pct < -15:
                    findings.append({
                        "type": "gsc_click_drop",
                        "severity": "warning",
                        "message": f"GSC clicks declined {change_pct:.0f}% week-over-week",
                        "recent_clicks": recent_clicks,
                        "previous_clicks": prev_clicks,
                    })

        except Exception as e:
            log.debug(f"GSC trend check error for {site_slug}: {e}")

        return findings

    def _check_ga4_trends(self, site_slug: str) -> List[Dict]:
        """Check GA4 for session/pageview drops."""
        findings = []
        try:
            resp = self.supabase.table("ga4_performance") \
                .select("sessions,pageviews,date") \
                .eq("site_slug", site_slug) \
                .order("date", desc=True) \
                .limit(60) \
                .execute()

            rows = resp.data or []
            if len(rows) < 14:
                return findings

            recent = rows[:7]
            previous = rows[7:14]

            recent_sessions = sum(r.get("sessions", 0) for r in recent)
            prev_sessions = sum(r.get("sessions", 0) for r in previous)

            if prev_sessions > 0:
                change_pct = ((recent_sessions - prev_sessions) / prev_sessions) * 100
                if change_pct < -30:
                    findings.append({
                        "type": "ga4_session_drop",
                        "severity": "critical",
                        "message": f"GA4 sessions dropped {change_pct:.0f}% week-over-week",
                        "recent_sessions": recent_sessions,
                        "previous_sessions": prev_sessions,
                    })
                elif change_pct < -15:
                    findings.append({
                        "type": "ga4_session_drop",
                        "severity": "warning",
                        "message": f"GA4 sessions declined {change_pct:.0f}% week-over-week",
                    })

        except Exception as e:
            log.debug(f"GA4 trend check error for {site_slug}: {e}")

        return findings

    def _check_keyword_drops(self, site_slug: str) -> List[Dict]:
        """Check for significant keyword ranking losses."""
        findings = []
        try:
            resp = self.supabase.table("keyword_rankings") \
                .select("keyword,position,previous_position,clicks") \
                .eq("site_slug", site_slug) \
                .execute()

            rows = resp.data or []
            big_drops = []
            for r in rows:
                pos = r.get("position", 0)
                prev = r.get("previous_position", 0)
                if prev > 0 and pos > 0:
                    drop = pos - prev
                    if drop > 10 and prev <= 20:
                        big_drops.append({
                            "keyword": r["keyword"],
                            "was": prev,
                            "now": pos,
                            "drop": drop,
                        })

            if len(big_drops) >= 5:
                findings.append({
                    "type": "keyword_ranking_drops",
                    "severity": "critical",
                    "message": f"{len(big_drops)} keywords dropped 10+ positions (possible algo update)",
                    "keywords": big_drops[:10],
                })
            elif big_drops:
                findings.append({
                    "type": "keyword_ranking_drops",
                    "severity": "warning",
                    "message": f"{len(big_drops)} keyword(s) lost significant positions",
                    "keywords": big_drops[:5],
                })

        except Exception as e:
            log.debug(f"Keyword check error for {site_slug}: {e}")

        return findings
