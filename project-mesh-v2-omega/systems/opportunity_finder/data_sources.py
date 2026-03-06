"""Data Source Aggregator — Fetches analytics data from Supabase."""

import logging
import os
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def _get_supabase():
    """Get Supabase client if available."""
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if url and key:
            return create_client(url, key)
    except ImportError:
        pass
    return None


class DataSourceAggregator:
    """Fetches and aggregates data from Supabase tables for opportunity analysis."""

    def __init__(self):
        self.supabase = _get_supabase()

    @property
    def available(self) -> bool:
        return self.supabase is not None

    def get_striking_distance_keywords(self, site_slug: str,
                                        min_position: float = 5,
                                        max_position: float = 20) -> List[Dict]:
        """Get keywords in striking distance (positions 5-20) with high impressions."""
        if not self.supabase:
            return []

        try:
            resp = self.supabase.table("keyword_rankings") \
                .select("keyword,position,impressions,clicks,ctr,url") \
                .eq("site_slug", site_slug) \
                .gte("position", min_position) \
                .lte("position", max_position) \
                .order("impressions", desc=True) \
                .limit(100) \
                .execute()
            return resp.data or []
        except Exception as e:
            log.error(f"Failed to get keywords for {site_slug}: {e}")
            return []

    def get_content_inventory(self, site_slug: str) -> List[Dict]:
        """Get all published articles for a site."""
        if not self.supabase:
            return []

        try:
            resp = self.supabase.table("wordpress_posts") \
                .select("post_id,title,url,status,word_count") \
                .eq("site_slug", site_slug) \
                .eq("status", "publish") \
                .execute()
            return resp.data or []
        except Exception as e:
            log.error(f"Failed to get content for {site_slug}: {e}")
            return []

    def get_gsc_performance(self, site_slug: str, days: int = 30) -> List[Dict]:
        """Get recent GSC performance data."""
        if not self.supabase:
            return []

        try:
            resp = self.supabase.table("gsc_performance") \
                .select("date,clicks,impressions,ctr,position") \
                .eq("site_slug", site_slug) \
                .order("date", desc=True) \
                .limit(days) \
                .execute()
            return resp.data or []
        except Exception as e:
            log.error(f"Failed to get GSC data for {site_slug}: {e}")
            return []

    def get_ga4_performance(self, site_slug: str, days: int = 30) -> List[Dict]:
        """Get recent GA4 performance data."""
        if not self.supabase:
            return []

        try:
            resp = self.supabase.table("ga4_performance") \
                .select("date,sessions,pageviews,bounce_rate,avg_session_duration") \
                .eq("site_slug", site_slug) \
                .order("date", desc=True) \
                .limit(days) \
                .execute()
            return resp.data or []
        except Exception as e:
            log.error(f"Failed to get GA4 data for {site_slug}: {e}")
            return []

    def get_all_site_slugs(self) -> List[str]:
        """Get all unique site slugs from the data."""
        if not self.supabase:
            return []

        try:
            resp = self.supabase.table("gsc_performance") \
                .select("site_slug") \
                .execute()
            return list({r["site_slug"] for r in (resp.data or [])})
        except Exception as e:
            log.error(f"Failed to get site slugs: {e}")
            return []

    def find_keyword_across_sites(self, keyword: str) -> List[Dict]:
        """Find a keyword's performance across all sites."""
        if not self.supabase:
            return []

        try:
            resp = self.supabase.table("keyword_rankings") \
                .select("site_slug,keyword,position,impressions,clicks") \
                .ilike("keyword", f"%{keyword}%") \
                .execute()
            return resp.data or []
        except Exception as e:
            log.error(f"Failed to search keyword '{keyword}': {e}")
            return []
