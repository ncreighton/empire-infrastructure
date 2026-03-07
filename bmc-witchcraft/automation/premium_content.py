"""
Premium Grimoire Content — BMC Membership Integration

Gates Grimoire Intelligence features behind BMC membership tiers:
  - Candlelight ($3): Monthly Full Moon ritual, daily practice archives
  - Moonlit Coven ($7): Monthly spell kit, sabbat workbook, tarot pull
  - High Priestess ($15): Personalized spells, birthday rituals, full knowledge vault

Connects to Grimoire API (port 8080) for content generation.
Zero AI API cost — all content from Grimoire's template engine.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from bmc_config import GRIMOIRE_API, DATA_DIR, TIER_MAP

logger = logging.getLogger("bmc-premium")

# --- Tier Access Levels ---

TIER_LEVELS = {
    "candlelight": 1,
    "moonlit": 2,
    "high_priestess": 3,
}

TIER_CONTENT = {
    "candlelight": {
        "spell_difficulty": "beginner",
        "amplify": False,
        "monthly_rituals": 1,
        "monthly_spells": 0,
        "tarot_pulls": 0,
        "knowledge_access": ["herbs", "crystals", "moon-phases"],
        "personalized": False,
    },
    "moonlit": {
        "spell_difficulty": "intermediate",
        "amplify": True,
        "monthly_rituals": 2,
        "monthly_spells": 2,
        "tarot_pulls": 1,
        "knowledge_access": ["herbs", "crystals", "moon-phases", "sabbats",
                             "tarot", "colors", "elements", "spell-types"],
        "personalized": False,
    },
    "high_priestess": {
        "spell_difficulty": "advanced",
        "amplify": True,
        "monthly_rituals": 4,
        "monthly_spells": 4,
        "tarot_pulls": 4,
        "knowledge_access": ["herbs", "crystals", "moon-phases", "sabbats",
                             "tarot", "colors", "elements", "spell-types",
                             "planetary-hours", "intentions"],
        "personalized": True,
    },
}

# --- Member Registry ---

MEMBERS_FILE = DATA_DIR / "members.json"


def _load_members() -> dict:
    if MEMBERS_FILE.exists():
        with open(MEMBERS_FILE) as f:
            return json.load(f)
    return {}


def _save_members(members: dict):
    with open(MEMBERS_FILE, "w") as f:
        json.dump(members, f, indent=2)


def register_member(email: str, name: str, tier_name: str):
    """Called when membership.started webhook fires."""
    tier_id = TIER_MAP.get(tier_name, "candlelight")
    members = _load_members()
    members[email] = {
        "name": name,
        "tier": tier_id,
        "joined_at": datetime.now(timezone.utc).isoformat(),
        "content_generated": [],
    }
    _save_members(members)
    logger.info(f"Registered member: {name} ({email}) → {tier_id}")
    return tier_id


def cancel_member(email: str):
    """Called when membership.cancelled webhook fires."""
    members = _load_members()
    if email in members:
        members[email]["cancelled_at"] = datetime.now(timezone.utc).isoformat()
        members[email]["tier"] = "cancelled"
        _save_members(members)
        logger.info(f"Cancelled member: {email}")


def get_member(email: str) -> Optional[dict]:
    members = _load_members()
    member = members.get(email)
    if member and member.get("tier") != "cancelled":
        return member
    return None


def get_active_members() -> dict:
    members = _load_members()
    return {k: v for k, v in members.items() if v.get("tier") != "cancelled"}


def check_access(email: str, required_tier: str) -> bool:
    """Check if member has access to content requiring a specific tier."""
    member = get_member(email)
    if not member:
        return False
    member_level = TIER_LEVELS.get(member["tier"], 0)
    required_level = TIER_LEVELS.get(required_tier, 99)
    return member_level >= required_level


# --- Grimoire Content Generator ---

class PremiumGrimoireContent:
    """Generates tier-gated content via Grimoire API."""

    def __init__(self, grimoire_url: str = GRIMOIRE_API):
        self.grimoire_url = grimoire_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)
        self.output_dir = DATA_DIR / "premium_content"
        self.output_dir.mkdir(exist_ok=True)

    def _grimoire_get(self, path: str) -> dict:
        resp = self.client.get(f"{self.grimoire_url}{path}")
        resp.raise_for_status()
        return resp.json()

    def _grimoire_post(self, path: str, data: dict) -> dict:
        resp = self.client.post(f"{self.grimoire_url}{path}", json=data)
        resp.raise_for_status()
        return resp.json()

    # --- Content Generators ---

    def generate_full_moon_ritual(self, tier: str = "candlelight") -> dict:
        """Monthly Full Moon ritual — all tiers."""
        config = TIER_CONTENT[tier]
        result = self._grimoire_post("/craft/ritual", {
            "occasion": "Full Moon",
            "intention": "release and renewal under the Full Moon's light",
            "difficulty": config["spell_difficulty"],
            "amplify": config["amplify"],
        })
        return {
            "type": "full_moon_ritual",
            "tier": tier,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "content": result,
        }

    def generate_spell_kit(self, intention: str = "protection",
                           tier: str = "moonlit") -> dict:
        """Monthly spell kit — Moonlit + High Priestess."""
        if TIER_LEVELS.get(tier, 0) < TIER_LEVELS["moonlit"]:
            return {"error": "Spell kits require Moonlit Coven or higher"}
        config = TIER_CONTENT[tier]
        result = self._grimoire_post("/craft/spell", {
            "intention": intention,
            "spell_type": "candle",
            "difficulty": config["spell_difficulty"],
            "amplify": config["amplify"],
        })
        return {
            "type": "spell_kit",
            "tier": tier,
            "intention": intention,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "content": result,
        }

    def generate_tarot_pull(self, question: str = "general guidance",
                            tier: str = "moonlit") -> dict:
        """Monthly tarot pull — Moonlit + High Priestess."""
        if TIER_LEVELS.get(tier, 0) < TIER_LEVELS["moonlit"]:
            return {"error": "Tarot pulls require Moonlit Coven or higher"}
        result = self._grimoire_post("/tarot/spread", {
            "intention": question,
            "spread_type": "three_card",
        })
        return {
            "type": "tarot_pull",
            "tier": tier,
            "question": question,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "content": result,
        }

    def generate_personalized_spell(self, email: str,
                                    intention: str) -> dict:
        """Personalized spell — High Priestess only."""
        member = get_member(email)
        if not member or member["tier"] != "high_priestess":
            return {"error": "Personalized spells require High Priestess tier"}

        # Use the member's practice history for personalization
        history = member.get("content_generated", [])
        past_intentions = [c.get("intention", "") for c in history
                           if c.get("type") == "spell_kit"]

        # Generate advanced spell with full amplification
        result = self._grimoire_post("/craft/spell", {
            "intention": intention,
            "spell_type": "crystal",
            "difficulty": "advanced",
            "amplify": True,
        })

        content = {
            "type": "personalized_spell",
            "tier": "high_priestess",
            "for_member": member["name"],
            "intention": intention,
            "past_work": past_intentions[-5:] if past_intentions else [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "content": result,
        }

        # Log to member's content history
        self._log_content(email, content)
        return content

    def generate_birthday_ritual(self, email: str) -> dict:
        """Birthday ritual — High Priestess only."""
        member = get_member(email)
        if not member or member["tier"] != "high_priestess":
            return {"error": "Birthday rituals require High Priestess tier"}

        result = self._grimoire_post("/craft/ritual", {
            "occasion": f"Birthday Blessing for {member['name']}",
            "intention": "celebrate your personal power and set intentions for the year ahead",
            "difficulty": "advanced",
            "amplify": True,
        })

        content = {
            "type": "birthday_ritual",
            "tier": "high_priestess",
            "for_member": member["name"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "content": result,
        }
        self._log_content(email, content)
        return content

    def generate_sabbat_workbook(self, tier: str = "moonlit") -> dict:
        """Quarterly sabbat workbook — Moonlit + High Priestess."""
        if TIER_LEVELS.get(tier, 0) < TIER_LEVELS["moonlit"]:
            return {"error": "Sabbat workbooks require Moonlit Coven or higher"}

        sabbat_info = self._grimoire_get("/knowledge/sabbats")
        current = sabbat_info.get("current_sabbat", {})
        next_sabbat = sabbat_info.get("next_sabbat", {})
        target = next_sabbat if next_sabbat else current

        config = TIER_CONTENT[tier]
        ritual = self._grimoire_post("/craft/ritual", {
            "occasion": target.get("name", "Sabbat"),
            "intention": f"honor the energies of {target.get('name', 'the season')}",
            "difficulty": config["spell_difficulty"],
            "amplify": config["amplify"],
        })

        meditation = self._grimoire_post("/craft/meditation", {
            "intention": f"connect with the energies of {target.get('name', 'the season')}",
            "difficulty": config["spell_difficulty"],
        })

        return {
            "type": "sabbat_workbook",
            "tier": tier,
            "sabbat": target.get("name", "Unknown"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sabbat_info": target,
            "ritual": ritual,
            "meditation": meditation,
        }

    def get_knowledge_vault(self, email: str) -> dict:
        """Full knowledge access — tier-gated."""
        member = get_member(email)
        if not member:
            return {"error": "Active membership required"}

        tier = member["tier"]
        config = TIER_CONTENT.get(tier, TIER_CONTENT["candlelight"])
        allowed = config["knowledge_access"]

        vault = {}
        for endpoint in allowed:
            try:
                vault[endpoint] = self._grimoire_get(f"/knowledge/{endpoint}")
            except Exception as e:
                logger.warning(f"Failed to fetch /knowledge/{endpoint}: {e}")
                vault[endpoint] = {"error": str(e)}

        return {
            "tier": tier,
            "member": member["name"],
            "accessible_modules": allowed,
            "vault": vault,
        }

    def generate_monthly_bundle(self, tier: str) -> dict:
        """Generate the full monthly content bundle for a tier."""
        config = TIER_CONTENT[tier]
        bundle = {
            "tier": tier,
            "month": datetime.now(timezone.utc).strftime("%Y-%m"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "items": [],
        }

        # Full Moon ritual (all tiers)
        bundle["items"].append(self.generate_full_moon_ritual(tier))

        # Daily practice (all tiers get current energy)
        try:
            energy = self._grimoire_get("/energy")
            daily = self._grimoire_get("/daily")
            bundle["items"].append({
                "type": "daily_guidance",
                "content": {"energy": energy, "practice": daily},
            })
        except Exception:
            pass

        # Spell kits (Moonlit+)
        if config["monthly_spells"] > 0:
            intentions = ["protection", "abundance", "love", "clarity"]
            for i in range(min(config["monthly_spells"], len(intentions))):
                bundle["items"].append(
                    self.generate_spell_kit(intentions[i], tier))

        # Tarot pulls (Moonlit+)
        if config["tarot_pulls"] > 0:
            bundle["items"].append(self.generate_tarot_pull(tier=tier))

        # Save bundle
        filename = f"bundle_{tier}_{bundle['month']}.json"
        bundle_path = self.output_dir / filename
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)
        logger.info(f"Generated monthly bundle: {filename} ({len(bundle['items'])} items)")

        return bundle

    def _log_content(self, email: str, content: dict):
        """Log generated content to member's history."""
        members = _load_members()
        if email in members:
            if "content_generated" not in members[email]:
                members[email]["content_generated"] = []
            members[email]["content_generated"].append({
                "type": content["type"],
                "intention": content.get("intention", ""),
                "generated_at": content["generated_at"],
            })
            # Keep last 50 items
            members[email]["content_generated"] = \
                members[email]["content_generated"][-50:]
            _save_members(members)

    def get_member_stats(self) -> dict:
        """Stats about premium content members."""
        members = get_active_members()
        tier_counts = {}
        for m in members.values():
            t = m.get("tier", "unknown")
            tier_counts[t] = tier_counts.get(t, 0) + 1

        return {
            "total_active": len(members),
            "by_tier": tier_counts,
            "content_bundles_generated": len(list(self.output_dir.glob("bundle_*.json"))),
        }
