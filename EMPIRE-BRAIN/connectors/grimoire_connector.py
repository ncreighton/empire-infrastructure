"""
Grimoire Knowledge Connector — Cross-pollination bridge.

Imports Grimoire's knowledge base (herbs, crystals, tarot, sabbats, moon phases,
spell templates) and makes it available for witchcraft content enrichment.

This connector enables:
1. BrainSmith to inject Grimoire correspondences into content briefs
2. Content pipelines to auto-enrich witchcraft articles with relevant herbs/crystals
3. Seasonal content alignment via sabbat and moon phase awareness

Usage:
    from connectors.grimoire_connector import GrimoireConnector

    gc = GrimoireConnector()
    herbs = gc.get_herbs_for_intention("protection")
    crystals = gc.get_crystals_for_intention("love")
    energy = gc.get_current_energy()
    enrichment = gc.enrich_topic("full moon ritual for protection")
"""

import logging
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

GRIMOIRE_ROOT = Path(r"D:\Claude Code Projects\grimoire-intelligence")


class GrimoireConnector:
    """Bridge between EMPIRE-BRAIN and Grimoire Intelligence knowledge base."""

    def __init__(self):
        self._loaded = False
        self._herbs = {}
        self._crystals = {}
        self._moon_phases = {}
        self._sabbats = {}
        self._spell_types = {}
        self._intention_map = {}
        self._engine = None
        self._load()

    def _load(self):
        """Import Grimoire knowledge modules."""
        if not GRIMOIRE_ROOT.exists():
            log.warning("Grimoire not found at %s", GRIMOIRE_ROOT)
            return

        grimoire_path = str(GRIMOIRE_ROOT)
        if grimoire_path not in sys.path:
            sys.path.insert(0, grimoire_path)

        try:
            from grimoire.knowledge.correspondences import HERBS, CRYSTALS, INTENTION_MAP
            from grimoire.knowledge.moon_phases import MOON_PHASES
            from grimoire.knowledge.wheel_of_year import SABBATS
            from grimoire.knowledge.spell_templates import SPELL_TYPES

            self._herbs = HERBS
            self._crystals = CRYSTALS
            self._moon_phases = MOON_PHASES
            self._sabbats = SABBATS
            self._spell_types = SPELL_TYPES
            self._intention_map = INTENTION_MAP
            self._loaded = True
            log.info("Grimoire knowledge loaded: %d herbs, %d crystals, %d moon phases, %d sabbats",
                     len(HERBS), len(CRYSTALS), len(MOON_PHASES), len(SABBATS))
        except ImportError as e:
            log.warning("Could not import Grimoire knowledge: %s", e)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_engine(self):
        """Get the full GrimoireEngine (lazy-loaded)."""
        if self._engine is None and self._loaded:
            try:
                from grimoire import GrimoireEngine
                self._engine = GrimoireEngine()
            except Exception as e:
                log.warning("Could not load GrimoireEngine: %s", e)
        return self._engine

    # --- Knowledge Lookups ---

    def get_herbs_for_intention(self, intention: str) -> list[dict]:
        """Find herbs matching an intention (protection, love, prosperity, etc.)."""
        if not self._loaded:
            return []
        intention_lower = intention.lower()
        matches = []
        for key, herb in self._herbs.items():
            props = [p.lower() for p in herb.get("magical_properties", [])]
            if any(intention_lower in p for p in props):
                matches.append({
                    "name": herb["name"],
                    "key": key,
                    "properties": herb.get("magical_properties", []),
                    "element": herb.get("element", ""),
                    "planet": herb.get("planet", ""),
                    "beginner_tip": herb.get("beginner_tip", ""),
                })
        return matches

    def get_crystals_for_intention(self, intention: str) -> list[dict]:
        """Find crystals matching an intention."""
        if not self._loaded:
            return []
        intention_lower = intention.lower()
        matches = []
        for key, crystal in self._crystals.items():
            props = [p.lower() for p in crystal.get("magical_properties", [])]
            if any(intention_lower in p for p in props):
                matches.append({
                    "name": crystal["name"],
                    "key": key,
                    "properties": crystal.get("magical_properties", []),
                    "element": crystal.get("element", ""),
                    "chakra": crystal.get("chakra", ""),
                })
        return matches

    def get_correspondences_for_intention(self, intention: str) -> dict:
        """Get full correspondences (herbs + crystals + colors) for an intention."""
        if not self._loaded:
            return {}
        mapped = self._intention_map.get(intention.lower(), {})
        return {
            "intention": intention,
            "herbs": self.get_herbs_for_intention(intention),
            "crystals": self.get_crystals_for_intention(intention),
            "mapped_correspondences": mapped,
        }

    def get_current_energy(self) -> Optional[dict]:
        """Get current moon phase and magical energy via GrimoireEngine."""
        engine = self.get_engine()
        if engine:
            try:
                return engine.current_energy()
            except Exception as e:
                log.warning("Could not get current energy: %s", e)
        return None

    def get_sabbat_context(self) -> Optional[dict]:
        """Get the current or next sabbat for seasonal content alignment."""
        if not self._loaded:
            return None
        try:
            from grimoire.knowledge.wheel_of_year import get_current_sabbat, get_next_sabbat
            current = get_current_sabbat()
            upcoming = get_next_sabbat()
            return {"current": current, "next": upcoming}
        except Exception:
            return None

    # --- Content Enrichment ---

    def enrich_topic(self, topic: str) -> dict:
        """Auto-enrich a witchcraft topic with relevant knowledge.

        Takes a topic like "full moon ritual for protection" and returns
        relevant herbs, crystals, moon context, and seasonal alignment.
        """
        if not self._loaded:
            return {"topic": topic, "enriched": False}

        # Extract intentions from topic
        intention_keywords = [
            "protection", "love", "prosperity", "healing", "purification",
            "divination", "banishing", "courage", "peace", "wisdom",
            "abundance", "fertility", "creativity", "intuition", "strength",
        ]
        detected_intentions = [k for k in intention_keywords if k in topic.lower()]

        # Gather correspondences for each intention
        all_herbs = []
        all_crystals = []
        for intention in detected_intentions:
            all_herbs.extend(self.get_herbs_for_intention(intention))
            all_crystals.extend(self.get_crystals_for_intention(intention))

        # Deduplicate
        seen_herbs = set()
        unique_herbs = []
        for h in all_herbs:
            if h["key"] not in seen_herbs:
                seen_herbs.add(h["key"])
                unique_herbs.append(h)

        seen_crystals = set()
        unique_crystals = []
        for c in all_crystals:
            if c["key"] not in seen_crystals:
                seen_crystals.add(c["key"])
                unique_crystals.append(c)

        # Moon context
        moon = self.get_current_energy()

        # Sabbat context
        sabbat = self.get_sabbat_context()

        # Detect if topic mentions a moon phase
        moon_phase_match = None
        topic_lower = topic.lower()
        for phase_key, phase_data in self._moon_phases.items():
            if phase_data["name"].lower() in topic_lower:
                moon_phase_match = phase_data
                break

        # Detect if topic mentions a spell type
        spell_type_match = None
        for spell_key, spell_data in self._spell_types.items():
            if spell_key.replace("_", " ") in topic_lower or spell_data["name"].lower() in topic_lower:
                spell_type_match = {
                    "type": spell_key,
                    "name": spell_data["name"],
                    "difficulty": spell_data.get("difficulty", ""),
                    "duration_minutes": spell_data.get("duration_minutes", 0),
                    "best_for": spell_data.get("best_for", []),
                }
                break

        return {
            "topic": topic,
            "enriched": True,
            "detected_intentions": detected_intentions,
            "herbs": unique_herbs[:5],
            "crystals": unique_crystals[:5],
            "moon_phase_match": moon_phase_match,
            "spell_type_match": spell_type_match,
            "current_moon": moon,
            "seasonal_context": sabbat,
        }

    def get_content_brief_enrichment(self, title: str, site_id: str) -> Optional[dict]:
        """Enrich a content brief for witchcraft-related sites.

        Only activates for witchcraft/spiritual sites. Returns None for other niches.
        """
        witchcraft_sites = {
            "witchcraftforbeginners", "manifestandalign", "moonrituallibrary",
        }
        if site_id not in witchcraft_sites:
            return None

        enrichment = self.enrich_topic(title)
        if not enrichment.get("enriched"):
            return None

        # Build content suggestions
        suggestions = []
        if enrichment["herbs"]:
            herb_names = [h["name"] for h in enrichment["herbs"][:3]]
            suggestions.append(f"Mention these herbs: {', '.join(herb_names)}")
        if enrichment["crystals"]:
            crystal_names = [c["name"] for c in enrichment["crystals"][:3]]
            suggestions.append(f"Recommend these crystals: {', '.join(crystal_names)}")
        if enrichment["moon_phase_match"]:
            phase = enrichment["moon_phase_match"]
            suggestions.append(f"Include {phase['name']} correspondences and best practices")
        if enrichment["spell_type_match"]:
            spell = enrichment["spell_type_match"]
            suggestions.append(f"Include a {spell['name']} tutorial (difficulty: {spell['difficulty']})")
        if enrichment["seasonal_context"]:
            sabbat = enrichment["seasonal_context"]
            if sabbat.get("current"):
                suggestions.append(f"Tie into current sabbat: {sabbat['current'].get('name', '')}")

        enrichment["content_suggestions"] = suggestions
        return enrichment
