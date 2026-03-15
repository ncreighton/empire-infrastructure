"""BrowsingWarmup — organic navigation patterns before target platform visits.

Anti-fraud systems flag browsers that:
1. Navigate directly to a signup page as their first-ever page visit
2. Have zero browsing history or referrer chain
3. Show robotic navigation timing (instant page transitions)

This module warms up the browser by visiting 2-4 common sites first,
creating a natural referral chain that ends at the target platform.
The browser accumulates real cookies, localStorage entries, and JS
fingerprint data from each visited site.

Patterns:
- search_referral:  Google → search "platform name" → click through
- social_referral:  Reddit/Twitter/HN → browse → navigate to platform
- direct_browse:    YouTube → Wikipedia → target (casual browsing pattern)
- shopping_detour:  Amazon → eBay → browse → target (e-commerce user)

Each warmup adds 30-90 seconds but dramatically reduces bot-detection flags.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WarmupStep:
    """A single navigation step in a warmup chain."""

    url: str
    wait_seconds: tuple[float, float]  # (min, max) random wait
    scroll: bool = False  # scroll the page after load
    description: str = ""


# ── Warmup route templates ───────────────────────────────────────────────────

def _search_referral(platform_name: str, platform_url: str) -> list[WarmupStep]:
    """Google → search for platform → click through to platform."""
    search_query = platform_name.replace(" ", "+").lower()
    return [
        WarmupStep(
            url="https://www.google.com",
            wait_seconds=(2.0, 5.0),
            scroll=False,
            description="Load Google homepage",
        ),
        WarmupStep(
            url=f"https://www.google.com/search?q={search_query}+signup",
            wait_seconds=(3.0, 7.0),
            scroll=True,
            description=f"Search for '{platform_name}'",
        ),
        WarmupStep(
            url=platform_url,
            wait_seconds=(2.0, 4.0),
            scroll=True,
            description=f"Navigate to {platform_name}",
        ),
    ]


def _social_referral(platform_name: str, platform_url: str) -> list[WarmupStep]:
    """Reddit/YouTube → browse → navigate to platform."""
    social_sites = [
        ("https://www.reddit.com", "Browse Reddit"),
        ("https://www.youtube.com", "Browse YouTube"),
        ("https://news.ycombinator.com", "Browse Hacker News"),
    ]
    site = random.choice(social_sites)
    return [
        WarmupStep(
            url=site[0],
            wait_seconds=(3.0, 8.0),
            scroll=True,
            description=site[1],
        ),
        WarmupStep(
            url="https://www.google.com",
            wait_seconds=(1.5, 3.0),
            scroll=False,
            description="Navigate to Google",
        ),
        WarmupStep(
            url=platform_url,
            wait_seconds=(2.0, 5.0),
            scroll=True,
            description=f"Navigate to {platform_name}",
        ),
    ]


def _direct_browse(platform_name: str, platform_url: str) -> list[WarmupStep]:
    """Casual browsing: visit a couple sites then navigate to platform."""
    browse_sites = [
        ("https://www.youtube.com", "Watch YouTube"),
        ("https://en.wikipedia.org", "Browse Wikipedia"),
        ("https://www.amazon.com", "Browse Amazon"),
        ("https://github.com/trending", "Check GitHub trending"),
        ("https://www.reddit.com/r/popular", "Browse Reddit popular"),
    ]
    random.shuffle(browse_sites)
    # Pick 1-2 sites to visit first
    num_sites = random.randint(1, 2)
    steps: list[WarmupStep] = []
    for site_url, desc in browse_sites[:num_sites]:
        steps.append(WarmupStep(
            url=site_url,
            wait_seconds=(3.0, 8.0),
            scroll=True,
            description=desc,
        ))
    steps.append(WarmupStep(
        url=platform_url,
        wait_seconds=(2.0, 4.0),
        scroll=True,
        description=f"Navigate to {platform_name}",
    ))
    return steps


# All available warmup route generators
_WARMUP_ROUTES = [
    _search_referral,
    _social_referral,
    _direct_browse,
]


def select_warmup_route(
    platform_id: str, platform_name: str, platform_url: str,
) -> list[WarmupStep]:
    """Select a warmup route for a platform.

    Uses consistent selection (same platform → same route type) but with
    randomized timing and sub-choices.  Search referral is preferred for
    signup flows (60% weight) since it creates the most natural referrer.
    """
    # Weighted selection: search_referral gets 60%, others split 40%
    rng = random.Random(platform_id)
    roll = rng.random()
    if roll < 0.60:
        route_fn = _search_referral
    elif roll < 0.80:
        route_fn = _social_referral
    else:
        route_fn = _direct_browse

    return route_fn(platform_name, platform_url)


class BrowsingWarmup:
    """Execute pre-navigation warmup to build organic browsing patterns.

    Works with any Playwright Page or browser-use page that has goto() and
    evaluate() methods.

    Usage::

        warmup = BrowsingWarmup()
        await warmup.execute(page, "gumroad", "Gumroad", "https://gumroad.com")
        # Browser now has organic cookies, localStorage, referrer chain
        # → proceed with signup
    """

    def __init__(self, enabled: bool = True, max_warmup_time: float = 90.0):
        """
        Args:
            enabled: Set False to skip warmup entirely (for testing).
            max_warmup_time: Hard cap on total warmup duration in seconds.
        """
        self.enabled = enabled
        self.max_warmup_time = max_warmup_time
        self._total_warmup_time: float = 0.0

    async def execute(
        self,
        page: Any,
        platform_id: str,
        platform_name: str,
        platform_url: str,
    ) -> dict[str, Any]:
        """Run warmup navigation chain, then return stats.

        The page ends on the platform_url, ready for the agent to take over.

        Args:
            page: Playwright Page or browser-use BrowserContext page
            platform_id: e.g. "gumroad"
            platform_name: e.g. "Gumroad"
            platform_url: e.g. "https://gumroad.com/signup"

        Returns:
            Dict with warmup stats (sites_visited, total_time, route_type).
        """
        if not self.enabled:
            logger.debug(f"[{platform_id}] Warmup disabled, skipping")
            return {"skipped": True, "sites_visited": 0, "total_time": 0}

        route = select_warmup_route(platform_id, platform_name, platform_url)
        route_name = route[0].description if route else "unknown"

        logger.info(
            f"[{platform_id}] Warming up browser: {len(route)} steps "
            f"(starting with: {route_name})"
        )

        sites_visited = 0
        total_time = 0.0

        for i, step in enumerate(route):
            if total_time >= self.max_warmup_time:
                logger.info(
                    f"[{platform_id}] Warmup time limit reached ({total_time:.0f}s), "
                    "proceeding to target"
                )
                break

            try:
                logger.debug(
                    f"[{platform_id}] Warmup step {i+1}/{len(route)}: "
                    f"{step.description} → {step.url}"
                )

                # Navigate
                await self._navigate(page, step.url)
                sites_visited += 1

                # Random human-like wait
                wait = random.uniform(*step.wait_seconds)
                await asyncio.sleep(wait)
                total_time += wait

                # Scroll if specified (builds scroll depth signals)
                if step.scroll:
                    await self._human_scroll(page)
                    scroll_wait = random.uniform(1.0, 3.0)
                    await asyncio.sleep(scroll_wait)
                    total_time += scroll_wait

            except Exception as e:
                # Warmup failures are non-fatal — log and continue
                logger.debug(
                    f"[{platform_id}] Warmup step {i+1} failed: {e}. Continuing."
                )
                # Small wait even on failure to maintain timing pattern
                await asyncio.sleep(1.0)
                total_time += 1.0

        self._total_warmup_time = total_time
        logger.info(
            f"[{platform_id}] Warmup complete: {sites_visited} sites in "
            f"{total_time:.1f}s"
        )

        return {
            "skipped": False,
            "sites_visited": sites_visited,
            "total_time": round(total_time, 1),
            "route": route_name,
        }

    async def _navigate(self, page: Any, url: str) -> None:
        """Navigate page to URL, handling both Playwright and browser-use APIs."""
        try:
            # Playwright Page API
            if hasattr(page, "goto"):
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            # browser-use page API
            elif hasattr(page, "navigate"):
                await page.navigate(url)
            else:
                logger.warning(f"Unknown page type: {type(page)}, cannot navigate")
        except Exception as e:
            # Timeout or navigation errors are OK for warmup
            logger.debug(f"Warmup navigate to {url}: {e}")

    async def _human_scroll(self, page: Any) -> None:
        """Scroll the page like a human (variable speed, not to bottom)."""
        try:
            scroll_js = """
            () => {
                const maxScroll = Math.min(
                    document.body.scrollHeight,
                    window.innerHeight * 3
                );
                const target = Math.random() * maxScroll * 0.6;
                window.scrollTo({
                    top: target,
                    behavior: 'smooth'
                });
                return target;
            }
            """
            if hasattr(page, "evaluate"):
                await page.evaluate(scroll_js)
        except Exception:
            pass  # Scroll failure is completely fine


def should_warmup(platform_id: str, action_type: str) -> bool:
    """Decide if warmup should run for this platform/action.

    Warmup is valuable for:
    - new_signup: First visit to platform — highest detection risk
    - retry_signup: Re-attempting after failure — may be flagged

    Skip warmup for:
    - apply_profile: Already logged in, session cookies restored
    - human_activity: Regular maintenance, low risk
    - publish_content: Already authenticated
    """
    warmup_actions = {"new_signup", "retry_signup"}
    return action_type in warmup_actions
