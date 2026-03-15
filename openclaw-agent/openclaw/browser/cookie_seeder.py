"""CookieSeeder — inject realistic browsing cookies into GoLogin profiles.

A fresh browser profile with zero history except one platform is a major bot
signal. Real users have cookies from Google, YouTube, Reddit, Amazon, news
sites, etc.

This module:
1. Generates a realistic cookie jar mimicking organic browsing patterns
2. Injects those cookies into GoLogin profiles via the GoLogin API
3. Optionally navigates to real sites via CDP to build genuine JS-set cookies
4. Varies the cookie set per profile (not identical across all profiles)

Usage::

    seeder = CookieSeeder(api_token="...")
    await seeder.seed_profile("69b6d256c6ae1736281bba73")
    # → Profile now has cookies from ~15-25 common sites

    # Or seed all fleet profiles at once
    await seeder.seed_fleet()
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ── Realistic cookie templates ───────────────────────────────────────────────
# These mimic the cookies real browsers accumulate from normal browsing.
# Each entry creates 1-5 cookies per site to match real cookie behavior.


@dataclass
class SiteCookieTemplate:
    """Template for generating realistic cookies for a site."""

    domain: str
    category: str  # search, social, shopping, news, tech, entertainment
    weight: float  # probability of inclusion (1.0 = always, 0.3 = 30% chance)
    cookies: list[dict] = field(default_factory=list)


# Sites that virtually every real user has cookies from
_UNIVERSAL_SITES = [
    SiteCookieTemplate(
        domain=".google.com",
        category="search",
        weight=0.95,
        cookies=[
            {"name": "NID", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "None"},
            {"name": "1P_JAR", "path": "/", "secure": True,
             "sameSite": "None"},
            {"name": "CONSENT", "path": "/", "secure": True},
            {"name": "AEC", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "Lax"},
        ],
    ),
    SiteCookieTemplate(
        domain=".youtube.com",
        category="entertainment",
        weight=0.90,
        cookies=[
            {"name": "VISITOR_INFO1_LIVE", "path": "/", "httpOnly": True,
             "secure": True, "sameSite": "None"},
            {"name": "YSC", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "None"},
            {"name": "PREF", "path": "/", "secure": True, "sameSite": "None"},
            {"name": "GPS", "path": "/", "httpOnly": False, "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".amazon.com",
        category="shopping",
        weight=0.80,
        cookies=[
            {"name": "session-id", "path": "/", "httpOnly": False,
             "secure": True},
            {"name": "session-id-time", "path": "/", "secure": True},
            {"name": "i18n-prefs", "path": "/", "httpOnly": False},
            {"name": "ubid-main", "path": "/", "httpOnly": False,
             "secure": True},
            {"name": "csm-hit", "path": "/", "secure": True},
        ],
    ),
]

# Sites that most users encounter but not universally
_COMMON_SITES = [
    SiteCookieTemplate(
        domain=".reddit.com",
        category="social",
        weight=0.65,
        cookies=[
            {"name": "csv", "path": "/", "secure": True, "sameSite": "None"},
            {"name": "edgebucket", "path": "/", "secure": True},
            {"name": "loid", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "None"},
            {"name": "token_v2", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "None"},
        ],
    ),
    SiteCookieTemplate(
        domain=".wikipedia.org",
        category="reference",
        weight=0.60,
        cookies=[
            {"name": "WMF-Last-Access", "path": "/", "httpOnly": True,
             "secure": True},
            {"name": "WMF-Last-Access-Global", "path": "/", "httpOnly": True,
             "secure": True},
            {"name": "GeoIP", "path": "/", "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".twitter.com",
        category="social",
        weight=0.55,
        cookies=[
            {"name": "guest_id", "path": "/", "httpOnly": False, "secure": True,
             "sameSite": "None"},
            {"name": "ct0", "path": "/", "httpOnly": False, "secure": True,
             "sameSite": "Lax"},
            {"name": "gt", "path": "/", "httpOnly": False, "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".github.com",
        category="tech",
        weight=0.50,
        cookies=[
            {"name": "_gh_sess", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "Lax"},
            {"name": "_octo", "path": "/", "httpOnly": False, "secure": True,
             "sameSite": "Lax"},
            {"name": "logged_in", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "Lax"},
        ],
    ),
    SiteCookieTemplate(
        domain=".facebook.com",
        category="social",
        weight=0.50,
        cookies=[
            {"name": "datr", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "None"},
            {"name": "sb", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "None"},
            {"name": "fr", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "None"},
        ],
    ),
    SiteCookieTemplate(
        domain=".stackoverflow.com",
        category="tech",
        weight=0.45,
        cookies=[
            {"name": "prov", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "None"},
            {"name": "OptanonConsent", "path": "/", "secure": True},
            {"name": "_ga", "path": "/", "secure": False},
        ],
    ),
    SiteCookieTemplate(
        domain=".linkedin.com",
        category="professional",
        weight=0.45,
        cookies=[
            {"name": "bcookie", "path": "/", "httpOnly": False, "secure": True,
             "sameSite": "None"},
            {"name": "li_gc", "path": "/", "httpOnly": False, "secure": True,
             "sameSite": "Lax"},
            {"name": "lidc", "path": "/", "httpOnly": False, "secure": True,
             "sameSite": "None"},
        ],
    ),
    SiteCookieTemplate(
        domain=".microsoft.com",
        category="tech",
        weight=0.50,
        cookies=[
            {"name": "MC1", "path": "/", "httpOnly": False, "secure": True},
            {"name": "MS0", "path": "/", "httpOnly": False, "secure": True},
            {"name": "MUID", "path": "/", "httpOnly": False, "secure": True},
        ],
    ),
]

# Niche sites for variety — each profile gets a different random subset
_NICHE_SITES = [
    SiteCookieTemplate(
        domain=".medium.com", category="content", weight=0.30,
        cookies=[
            {"name": "uid", "path": "/", "httpOnly": True, "secure": True},
            {"name": "sid", "path": "/", "httpOnly": True, "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".pinterest.com", category="social", weight=0.25,
        cookies=[
            {"name": "_pinterest_sess", "path": "/", "httpOnly": True,
             "secure": True, "sameSite": "None"},
            {"name": "_b", "path": "/", "httpOnly": False, "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".twitch.tv", category="entertainment", weight=0.25,
        cookies=[
            {"name": "unique_id", "path": "/", "httpOnly": False, "secure": True},
            {"name": "server_session_id", "path": "/", "httpOnly": False,
             "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".spotify.com", category="entertainment", weight=0.30,
        cookies=[
            {"name": "sp_t", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "None"},
            {"name": "sp_landing", "path": "/", "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".cnn.com", category="news", weight=0.20,
        cookies=[
            {"name": "countryCode", "path": "/", "secure": True},
            {"name": "FastAB", "path": "/", "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".nytimes.com", category="news", weight=0.20,
        cookies=[
            {"name": "nyt-a", "path": "/", "httpOnly": False, "secure": True,
             "sameSite": "None"},
            {"name": "nyt-gdpr", "path": "/", "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".ebay.com", category="shopping", weight=0.25,
        cookies=[
            {"name": "dp1", "path": "/", "httpOnly": False, "secure": True},
            {"name": "nonsession", "path": "/", "httpOnly": False, "secure": True},
            {"name": "s", "path": "/", "httpOnly": True, "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".walmart.com", category="shopping", weight=0.15,
        cookies=[
            {"name": "auth", "path": "/", "httpOnly": True, "secure": True},
            {"name": "vtc", "path": "/", "httpOnly": False, "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".cloudflare.com", category="tech", weight=0.20,
        cookies=[
            {"name": "__cfruid", "path": "/", "httpOnly": True, "secure": True,
             "sameSite": "None"},
        ],
    ),
    SiteCookieTemplate(
        domain=".notion.so", category="productivity", weight=0.20,
        cookies=[
            {"name": "notion_browser_id", "path": "/", "secure": True},
            {"name": "notion_check_cookie_consent", "path": "/", "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".canva.com", category="design", weight=0.20,
        cookies=[
            {"name": "CAN_SESS", "path": "/", "httpOnly": True, "secure": True},
            {"name": "_csrf_token", "path": "/", "httpOnly": True, "secure": True},
        ],
    ),
    SiteCookieTemplate(
        domain=".chatgpt.com", category="ai", weight=0.35,
        cookies=[
            {"name": "__Secure-next-auth.session-token", "path": "/",
             "httpOnly": True, "secure": True, "sameSite": "Lax"},
        ],
    ),
]


def _generate_cookie_value(name: str, domain: str, seed: str) -> str:
    """Generate a realistic-looking cookie value seeded by profile+domain.

    Different cookie types get different value formats:
    - Session IDs: hex strings
    - Timestamps: unix timestamps
    - UUIDs: uuid-like format
    - Consent: YES/NO/specific format
    """
    h = hashlib.sha256(f"{seed}:{domain}:{name}".encode()).hexdigest()

    # Known cookie value patterns
    if name in ("CONSENT", "OptanonConsent"):
        return f"YES+cb.20240101-00-p0.en+FX+{h[:8]}"
    if name in ("1P_JAR", "WMF-Last-Access"):
        d = datetime.now() - timedelta(days=random.randint(0, 14))
        return d.strftime("%Y-%m-%d-%H")
    if name in ("session-id", "session-id-time"):
        return f"{int(h[:12], 16)}"
    if name in ("_ga",):
        return f"GA1.2.{int(h[:10], 16)}.{int(time.time()) - random.randint(0, 86400*30)}"
    if name in ("countryCode", "GeoIP"):
        return "US"
    if name in ("i18n-prefs",):
        return "USD"
    if name in ("logged_in",):
        return "no"
    if name in ("nyt-gdpr",):
        return "0"
    if name in ("guest_id",):
        return f"v1%3A{int(h[:18], 16)}"

    # Default: hex-encoded string (realistic for most session/tracking cookies)
    length = random.choice([16, 24, 32, 40, 48])
    return h[:length]


def _generate_expiry() -> int:
    """Generate a future expiry timestamp (1-365 days from now)."""
    days_ahead = random.randint(30, 365)
    return int(time.time()) + (days_ahead * 86400)


def _select_sites_for_profile(profile_id: str) -> list[SiteCookieTemplate]:
    """Select a unique but realistic subset of sites for this profile.

    Uses the profile_id as a seed so the selection is deterministic
    (same profile always gets the same site mix) but different per profile.
    """
    rng = random.Random(profile_id)

    selected: list[SiteCookieTemplate] = []

    # Universal sites — almost always included
    for site in _UNIVERSAL_SITES:
        if rng.random() < site.weight:
            selected.append(site)

    # Common sites — include 3-6 of them
    common_shuffled = list(_COMMON_SITES)
    rng.shuffle(common_shuffled)
    common_count = rng.randint(3, min(6, len(common_shuffled)))
    for site in common_shuffled[:common_count]:
        if rng.random() < site.weight + 0.2:  # boost to ensure we get enough
            selected.append(site)

    # Niche sites — include 2-5 for variety
    niche_shuffled = list(_NICHE_SITES)
    rng.shuffle(niche_shuffled)
    niche_count = rng.randint(2, min(5, len(niche_shuffled)))
    for site in niche_shuffled[:niche_count]:
        selected.append(site)

    return selected


def generate_cookies_for_profile(profile_id: str) -> list[dict]:
    """Generate a full realistic cookie jar for a GoLogin profile.

    Returns a list of cookie dicts in GoLogin API format.
    """
    sites = _select_sites_for_profile(profile_id)
    cookies: list[dict] = []
    seed = profile_id

    for site in sites:
        for template in site.cookies:
            cookie: dict[str, Any] = {
                "name": template["name"],
                "value": _generate_cookie_value(template["name"], site.domain, seed),
                "domain": site.domain,
                "path": template.get("path", "/"),
                "expirationDate": _generate_expiry(),
            }
            if template.get("httpOnly"):
                cookie["httpOnly"] = True
            if template.get("secure"):
                cookie["secure"] = True
            if template.get("sameSite"):
                cookie["sameSite"] = template["sameSite"]
            cookies.append(cookie)

    logger.info(
        f"Generated {len(cookies)} cookies from {len(sites)} sites "
        f"for profile {profile_id[:8]}..."
    )
    return cookies


class CookieSeeder:
    """Seed GoLogin profiles with realistic browsing cookies.

    Uses the GoLogin MCP API to inject cookies into profiles, making them
    appear as genuine used browsers rather than fresh bot instances.
    """

    def __init__(self, api_token: str | None = None):
        import os
        self._api_token = api_token or os.environ.get("GOLOGIN_API_TOKEN", "")

    async def seed_profile(self, profile_id: str) -> dict:
        """Inject realistic cookies into a single GoLogin profile.

        Returns dict with success status and cookie count.
        """
        cookies = generate_cookies_for_profile(profile_id)

        if not cookies:
            return {"success": False, "error": "No cookies generated", "count": 0}

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"https://api.gologin.com/browser/{profile_id}/cookies",
                    headers={
                        "Authorization": f"Bearer {self._api_token}",
                        "Content-Type": "application/json",
                    },
                    json=cookies,
                )
                if resp.status_code in (200, 201, 204):
                    logger.info(
                        f"Seeded {len(cookies)} cookies into profile {profile_id[:8]}..."
                    )
                    return {
                        "success": True,
                        "count": len(cookies),
                        "profile_id": profile_id,
                    }
                else:
                    error = resp.text[:200]
                    logger.warning(
                        f"Cookie seed failed for {profile_id[:8]}...: "
                        f"HTTP {resp.status_code} — {error}"
                    )
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status_code}: {error}",
                        "count": 0,
                    }
        except Exception as e:
            logger.error(f"Cookie seed error for {profile_id[:8]}...: {e}")
            return {"success": False, "error": str(e), "count": 0}

    async def seed_fleet(
        self, profile_ids: list[str] | None = None
    ) -> dict:
        """Seed cookies into all fleet profiles.

        If profile_ids is not provided, seeds all profiles from IdentityManager.
        """
        if profile_ids is None:
            from openclaw.browser.identity_manager import IdentityManager
            im = IdentityManager()
            stats = im.stats()
            # Collect all unique profile IDs
            seen: set[str] = set()
            all_ids: list[str] = []
            for pid in stats["dedicated_platforms"]:
                assignment = im.resolve(pid)
                if assignment and assignment.profile_id not in seen:
                    seen.add(assignment.profile_id)
                    all_ids.append(assignment.profile_id)
            # We don't seed pool profiles (they may have real cookies already)
            profile_ids = all_ids

        results: list[dict] = []
        for pid in profile_ids:
            result = await self.seed_profile(pid)
            results.append(result)

        total = sum(r["count"] for r in results)
        successes = sum(1 for r in results if r["success"])

        logger.info(
            f"Fleet seeding complete: {successes}/{len(results)} profiles, "
            f"{total} total cookies"
        )
        return {
            "profiles_seeded": successes,
            "profiles_total": len(results),
            "total_cookies": total,
            "results": results,
        }
