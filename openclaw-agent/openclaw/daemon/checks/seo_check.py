"""SEO health checks — GSC traffic drop detection.

INTEL tier: runs every 6 hours.
"""

from __future__ import annotations

import logging
import os

from openclaw.models import CheckResult, HealthCheck, HeartbeatTier

logger = logging.getLogger(__name__)


async def check_traffic(gsc_drop_threshold: float = 0.20) -> list[HealthCheck]:
    """Check Google Search Console for traffic drops.

    Compares last 7 days vs prior 7 days. Flags domains with drops
    exceeding the threshold.

    Args:
        gsc_drop_threshold: Fraction drop to flag (0.20 = 20%).

    Returns:
        List of HealthCheck results.
    """
    gsc_creds = os.environ.get("GSC_CREDENTIALS_PATH", "")
    if not gsc_creds:
        return [HealthCheck(
            name="seo:gsc",
            tier=HeartbeatTier.INTEL,
            result=CheckResult.UNKNOWN,
            message="GSC_CREDENTIALS_PATH not configured — skipping traffic check",
        )]

    try:
        from pathlib import Path
        if not Path(gsc_creds).exists():
            return [HealthCheck(
                name="seo:gsc",
                tier=HeartbeatTier.INTEL,
                result=CheckResult.UNKNOWN,
                message=f"GSC credentials file not found: {gsc_creds}",
            )]

        # Import Google API client
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        import json

        with open(gsc_creds) as f:
            cred_data = json.load(f)

        credentials = Credentials.from_authorized_user_info(cred_data)
        service = build("searchconsole", "v1", credentials=credentials)

        from datetime import datetime, timedelta
        today = datetime.now().date()
        end_recent = today - timedelta(days=1)
        start_recent = end_recent - timedelta(days=6)
        end_prior = start_recent - timedelta(days=1)
        start_prior = end_prior - timedelta(days=6)

        # Get list of sites
        site_list = service.sites().list().execute()
        sites = [s["siteUrl"] for s in site_list.get("siteEntry", [])]

        checks: list[HealthCheck] = []
        for site_url in sites:
            try:
                # Recent period
                recent = service.searchanalytics().query(
                    siteUrl=site_url,
                    body={
                        "startDate": start_recent.isoformat(),
                        "endDate": end_recent.isoformat(),
                    },
                ).execute()
                recent_clicks = sum(r.get("clicks", 0) for r in recent.get("rows", []))

                # Prior period
                prior = service.searchanalytics().query(
                    siteUrl=site_url,
                    body={
                        "startDate": start_prior.isoformat(),
                        "endDate": end_prior.isoformat(),
                    },
                ).execute()
                prior_clicks = sum(r.get("clicks", 0) for r in prior.get("rows", []))

                if prior_clicks > 0:
                    drop = (prior_clicks - recent_clicks) / prior_clicks
                    if drop > gsc_drop_threshold:
                        checks.append(HealthCheck(
                            name=f"seo:{site_url}",
                            tier=HeartbeatTier.INTEL,
                            result=CheckResult.DEGRADED,
                            message=f"Traffic drop: {drop:.0%} ({prior_clicks} → {recent_clicks} clicks)",
                            details={
                                "site": site_url,
                                "prior_clicks": prior_clicks,
                                "recent_clicks": recent_clicks,
                                "drop_pct": round(drop * 100, 1),
                            },
                        ))
                    else:
                        checks.append(HealthCheck(
                            name=f"seo:{site_url}",
                            tier=HeartbeatTier.INTEL,
                            result=CheckResult.HEALTHY,
                            message=f"Traffic stable: {recent_clicks} clicks (7d)",
                        ))
            except Exception as e:
                logger.debug(f"GSC check failed for {site_url}: {e}")

        if not checks:
            checks.append(HealthCheck(
                name="seo:gsc",
                tier=HeartbeatTier.INTEL,
                result=CheckResult.HEALTHY,
                message="No GSC sites to check",
            ))

        return checks

    except ImportError:
        return [HealthCheck(
            name="seo:gsc",
            tier=HeartbeatTier.INTEL,
            result=CheckResult.UNKNOWN,
            message="google-api-python-client not installed — skipping GSC check",
        )]
    except Exception as e:
        return [HealthCheck(
            name="seo:gsc",
            tier=HeartbeatTier.INTEL,
            result=CheckResult.DOWN,
            message=f"GSC check failed: {str(e)[:80]}",
        )]
