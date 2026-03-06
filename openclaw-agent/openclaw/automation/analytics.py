"""Analytics — detailed reporting and insights on signup operations."""

from __future__ import annotations

import csv
import io
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from openclaw.forge.platform_codex import PlatformCodex
from openclaw.knowledge.platforms import get_platform, get_all_platform_ids, PLATFORMS
from openclaw.models import AccountStatus, PlatformCategory

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report."""
    generated_at: datetime = field(default_factory=datetime.now)

    # Overview
    total_platforms: int = 0
    platforms_attempted: int = 0
    platforms_active: int = 0
    platforms_failed: int = 0
    platforms_remaining: int = 0
    overall_success_rate: float = 0.0

    # Category breakdown
    category_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Complexity breakdown
    complexity_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Quality
    avg_sentinel_score: float = 0.0
    score_distribution: dict[str, int] = field(default_factory=dict)  # grade -> count

    # Timing
    avg_signup_duration_seconds: float = 0.0
    fastest_signup: dict[str, Any] = field(default_factory=dict)
    slowest_signup: dict[str, Any] = field(default_factory=dict)

    # CAPTCHA stats
    total_captchas: int = 0
    captcha_auto_solve_rate: float = 0.0

    # Errors
    common_errors: list[dict[str, Any]] = field(default_factory=list)

    # Recommendations
    next_targets: list[dict[str, Any]] = field(default_factory=list)


class Analytics:
    """Generate detailed analytics and reports from Codex data."""

    def __init__(self, codex: PlatformCodex | None = None):
        self.codex = codex or PlatformCodex()

    def generate_report(self) -> AnalyticsReport:
        """Generate a comprehensive analytics report."""
        report = AnalyticsReport()
        accounts = self.codex.get_all_accounts()
        all_ids = get_all_platform_ids()

        report.total_platforms = len(all_ids)
        report.platforms_attempted = len(accounts)
        report.platforms_remaining = report.total_platforms - report.platforms_attempted

        # Status breakdown
        active_statuses = {AccountStatus.ACTIVE.value, AccountStatus.PROFILE_COMPLETE.value}
        failed_statuses = {AccountStatus.SIGNUP_FAILED.value}

        report.platforms_active = sum(
            1 for a in accounts if a.get("status") in active_statuses
        )
        report.platforms_failed = sum(
            1 for a in accounts if a.get("status") in failed_statuses
        )
        report.overall_success_rate = (
            report.platforms_active / report.platforms_attempted * 100
            if report.platforms_attempted > 0 else 0
        )

        # Category breakdown
        report.category_stats = self._category_breakdown(accounts)

        # Complexity breakdown
        report.complexity_stats = self._complexity_breakdown(accounts)

        # Quality stats
        profiles = self.codex.get_all_profiles()
        if profiles:
            scores = [p.get("sentinel_score", 0) for p in profiles if p.get("sentinel_score")]
            if scores:
                report.avg_sentinel_score = sum(scores) / len(scores)
            # Grade distribution
            grades = [p.get("grade", "F") for p in profiles if p.get("grade")]
            report.score_distribution = dict(Counter(grades))

        # CAPTCHA stats
        stats = self.codex.get_stats()
        report.total_captchas = stats.get("total_captchas", 0)
        report.captcha_auto_solve_rate = stats.get("captcha_auto_solve_rate", 0)

        # Error analysis
        report.common_errors = self._analyze_errors(accounts)

        # Next targets (top 5 remaining platforms)
        attempted_ids = {a["platform_id"] for a in accounts}
        remaining = [
            pid for pid in all_ids if pid not in attempted_ids
        ]
        report.next_targets = self._rank_remaining(remaining)[:5]

        return report

    def _category_breakdown(self, accounts: list[dict]) -> dict[str, dict[str, Any]]:
        """Break down stats by platform category."""
        by_category: dict[str, list[dict]] = defaultdict(list)
        for a in accounts:
            platform = get_platform(a["platform_id"])
            if platform:
                by_category[platform.category.value].append(a)

        result = {}
        for cat in PlatformCategory:
            cat_accounts = by_category.get(cat.value, [])
            total_in_cat = sum(
                1 for p in PLATFORMS.values() if p.category == cat
            )
            active = sum(
                1 for a in cat_accounts
                if a.get("status") in ("active", "profile_complete")
            )
            result[cat.value] = {
                "total_platforms": total_in_cat,
                "attempted": len(cat_accounts),
                "active": active,
                "coverage_pct": active / total_in_cat * 100 if total_in_cat > 0 else 0,
            }
        return result

    def _complexity_breakdown(self, accounts: list[dict]) -> dict[str, dict[str, Any]]:
        """Break down stats by signup complexity."""
        by_complexity: dict[str, list[dict]] = defaultdict(list)
        for a in accounts:
            platform = get_platform(a["platform_id"])
            if platform:
                by_complexity[platform.complexity.value].append(a)

        result = {}
        for complexity_name, accts in by_complexity.items():
            active = sum(
                1 for a in accts if a.get("status") in ("active", "profile_complete")
            )
            result[complexity_name] = {
                "attempted": len(accts),
                "active": active,
                "success_rate": active / len(accts) * 100 if accts else 0,
            }
        return result

    def _analyze_errors(self, accounts: list[dict]) -> list[dict[str, Any]]:
        """Analyze common errors from signup logs."""
        error_counts: Counter = Counter()
        for a in accounts:
            if a.get("status") == AccountStatus.SIGNUP_FAILED.value:
                log = self.codex.get_signup_log(a["platform_id"])
                for entry in log:
                    if entry.get("error_message"):
                        # Normalize error messages
                        err = entry["error_message"][:100]
                        error_counts[err] += 1

        return [
            {"error": err, "count": count}
            for err, count in error_counts.most_common(10)
        ]

    def _rank_remaining(self, platform_ids: list[str]) -> list[dict[str, Any]]:
        """Rank remaining platforms by value."""
        ranked = []
        for pid in platform_ids:
            p = get_platform(pid)
            if not p:
                continue
            score = (
                p.monetization_potential * 4.0
                + p.audience_size * 2.5
                + p.seo_value * 2.0
            )
            ranked.append({
                "platform_id": pid,
                "name": p.name,
                "category": p.category.value,
                "complexity": p.complexity.value,
                "score": score,
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def export_json(self, filepath: str | None = None) -> str:
        """Export all account data as JSON."""
        accounts = self.codex.get_all_accounts()
        data = {
            "exported_at": datetime.now().isoformat(),
            "total_accounts": len(accounts),
            "accounts": [],
        }
        for a in accounts:
            profile = self.codex.get_profile(a["platform_id"])
            platform = get_platform(a["platform_id"])
            entry = {
                **a,
                "category": platform.category.value if platform else "",
                "complexity": platform.complexity.value if platform else "",
                "sentinel_score": profile.get("sentinel_score") if profile else None,
                "grade": profile.get("grade") if profile else None,
            }
            data["accounts"].append(entry)

        json_str = json.dumps(data, indent=2, default=str)
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
            logger.info(f"Exported {len(accounts)} accounts to {filepath}")
        return json_str

    def export_csv(self, filepath: str | None = None) -> str:
        """Export all account data as CSV."""
        accounts = self.codex.get_all_accounts()
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "platform_id", "platform_name", "category", "complexity",
            "status", "username", "profile_url", "sentinel_score",
            "grade", "created_at", "updated_at",
        ])

        for a in accounts:
            platform = get_platform(a["platform_id"])
            profile = self.codex.get_profile(a["platform_id"])
            writer.writerow([
                a.get("platform_id", ""),
                a.get("platform_name", ""),
                platform.category.value if platform else "",
                platform.complexity.value if platform else "",
                a.get("status", ""),
                a.get("username", ""),
                a.get("profile_url", ""),
                profile.get("sentinel_score", "") if profile else "",
                profile.get("grade", "") if profile else "",
                a.get("created_at", ""),
                a.get("updated_at", ""),
            ])

        csv_str = output.getvalue()
        if filepath:
            with open(filepath, "w", newline="") as f:
                f.write(csv_str)
            logger.info(f"Exported {len(accounts)} accounts to {filepath}")
        return csv_str

    def get_coverage_map(self) -> dict[str, dict[str, str]]:
        """Get a map of all platforms with their status for visual dashboard."""
        accounts = {a["platform_id"]: a for a in self.codex.get_all_accounts()}
        result = {}
        for pid in get_all_platform_ids():
            p = get_platform(pid)
            account = accounts.get(pid)
            result[pid] = {
                "name": p.name if p else pid,
                "category": p.category.value if p else "",
                "status": account.get("status", "not_started") if account else "not_started",
                "complexity": p.complexity.value if p else "",
            }
        return result

    def get_timeline(self, days: int = 30) -> list[dict[str, Any]]:
        """Get signup activity timeline for the last N days."""
        activity = self.codex.get_recent_activity(limit=100)
        cutoff = datetime.now() - timedelta(days=days)

        timeline = []
        for entry in activity:
            try:
                ts = datetime.fromisoformat(entry.get("timestamp", ""))
                if ts >= cutoff:
                    timeline.append(entry)
            except (ValueError, TypeError):
                timeline.append(entry)

        return timeline
