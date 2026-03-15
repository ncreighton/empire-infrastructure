"""IdentityManager — maps platforms to unique GoLogin browser fingerprints.

With 10 GoLogin profiles (plan limit), we assign dedicated profiles to the
6 highest-revenue platforms and share 4 pooled profiles across the remaining
40 platforms via consistent hashing (same platform always gets the same pool
slot).

The GoLogin MCP API can also be used to rotate fingerprints periodically
(noise, fonts, WebGL hash) without creating new profiles.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProfileAssignment:
    """A GoLogin profile assigned to a platform."""

    profile_id: str
    profile_name: str
    dedicated: bool  # True = 1:1 mapping, False = shared pool


# ── Dedicated profiles (high-revenue platforms) ──────────────────────────────
# These get unique browser identities — never shared.

_DEDICATED: dict[str, ProfileAssignment] = {
    "gumroad": ProfileAssignment(
        profile_id="69b6d256c6ae1736281bba73",
        profile_name="Gumroad Agent",
        dedicated=True,
    ),
    "etsy": ProfileAssignment(
        profile_id="69b6d25626651cd2cf361677",
        profile_name="Etsy Agent",
        dedicated=True,
    ),
    "creative_market": ProfileAssignment(
        profile_id="69b6d257c6ae1736281bbba9",
        profile_name="Creative Market Agent",
        dedicated=True,
    ),
    "envato": ProfileAssignment(
        profile_id="69b6d257c6ae1736281bbc5a",
        profile_name="Envato Agent",
        dedicated=True,
    ),
    "promptbase": ProfileAssignment(
        profile_id="69b6d269fe4e892e100b8faf",
        profile_name="PromptBase Agent",
        dedicated=True,
    ),
    "n8n_creator_hub": ProfileAssignment(
        profile_id="69b6d269b95ced2dd59b3abe",
        profile_name="n8n Creator Hub Agent",
        dedicated=True,
    ),
}

# ── Pool profiles (shared across remaining platforms) ────────────────────────
# Consistent hash distributes platforms evenly across 4 pool slots.

_POOL: list[ProfileAssignment] = [
    ProfileAssignment(
        profile_id=os.environ.get("GOLOGIN_PROFILE_ID", ""),
        profile_name="OpenClaw Primary",
        dedicated=False,
    ),
    ProfileAssignment(
        # n8n Creator Hub Publisher (pre-existing profile)
        profile_id="67e4b1aa8e35e18eea32a9c1",
        profile_name="Pool Slot B",
        dedicated=False,
    ),
    ProfileAssignment(
        # ClawHub Publisher (pre-existing profile)
        profile_id="67e4b1d08e35e18eea32afd1",
        profile_name="Pool Slot C",
        dedicated=False,
    ),
    ProfileAssignment(
        # Cole Shaw - Venture Agent (pre-existing profile)
        profile_id="67f15e81bd5b4a23e3e14c14",
        profile_name="Pool Slot D",
        dedicated=False,
    ),
]


def _hash_slot(platform_id: str, pool_size: int) -> int:
    """Consistent hash: same platform always maps to same pool slot."""
    h = hashlib.md5(platform_id.encode()).hexdigest()
    return int(h, 16) % pool_size


class IdentityManager:
    """Resolve the best GoLogin profile for a given platform.

    Usage::

        im = IdentityManager()
        assignment = im.resolve("gumroad")
        # → ProfileAssignment(profile_id="69b6d2...", dedicated=True)

        assignment = im.resolve("hugging_face")
        # → ProfileAssignment(profile_id="<pool>", dedicated=False)
    """

    def __init__(
        self,
        dedicated: dict[str, ProfileAssignment] | None = None,
        pool: list[ProfileAssignment] | None = None,
    ):
        self._dedicated = dedicated if dedicated is not None else dict(_DEDICATED)
        self._pool = pool if pool is not None else list(_POOL)
        # Filter out empty profile IDs from pool (e.g. unset env var)
        self._pool = [p for p in self._pool if p.profile_id]

    def resolve(self, platform_id: str) -> ProfileAssignment | None:
        """Get the GoLogin profile for a platform.

        Returns None if no profiles are available (GoLogin not configured).
        """
        # Dedicated profile?
        if platform_id in self._dedicated:
            return self._dedicated[platform_id]

        # Pool assignment via consistent hash
        if self._pool:
            slot = _hash_slot(platform_id, len(self._pool))
            assignment = self._pool[slot]
            logger.debug(
                f"[{platform_id}] → pool slot {slot} "
                f"({assignment.profile_name}, {assignment.profile_id[:8]}...)"
            )
            return assignment

        return None

    def get_profile_id(self, platform_id: str) -> str | None:
        """Convenience: return just the profile_id string, or None."""
        assignment = self.resolve(platform_id)
        return assignment.profile_id if assignment else None

    def is_dedicated(self, platform_id: str) -> bool:
        """Check if a platform has a dedicated (not shared) profile."""
        return platform_id in self._dedicated

    def stats(self) -> dict:
        """Return fleet statistics."""
        return {
            "dedicated_profiles": len(self._dedicated),
            "pool_profiles": len(self._pool),
            "total_profiles": len(self._dedicated) + len(self._pool),
            "dedicated_platforms": list(self._dedicated.keys()),
        }
