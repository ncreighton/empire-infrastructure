"""Grimoire Service — connects Luna to the Grimoire Intelligence knowledge base.

Provides zero-cost algorithmic lookups for herbs, crystals, tarot cards,
moon phases, correspondences, spell crafting, and daily practice.
Falls back to inline data if grimoire-intelligence is not available.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

from .persona import get_moon_phase

# Try importing grimoire-intelligence (external project)
_grimoire_available = False
try:
    grimoire_path = Path(__file__).resolve().parents[4] / "grimoire-intelligence"
    if grimoire_path.exists():
        sys.path.insert(0, str(grimoire_path))
        from grimoire.knowledge.herbs import HERBS
        from grimoire.knowledge.crystals import CRYSTALS
        from grimoire.knowledge.tarot_cards import TAROT_CARDS
        from grimoire.knowledge.correspondences import get_correspondences as _grimoire_correspondences
        from grimoire.knowledge.moon_phases import MOON_PHASES
        from grimoire.knowledge.wheel_of_year import SABBATS
        _grimoire_available = True
except Exception:
    pass

# Inline fallback data (subset for when grimoire-intelligence is unavailable)
_FALLBACK_HERBS = {
    "lavender": {"name": "Lavender", "element": "Air", "uses": ["peace", "sleep", "purification", "love"]},
    "rosemary": {"name": "Rosemary", "element": "Fire", "uses": ["protection", "memory", "purification", "healing"]},
    "sage": {"name": "Sage", "element": "Air", "uses": ["cleansing", "wisdom", "protection", "longevity"]},
    "chamomile": {"name": "Chamomile", "element": "Water", "uses": ["peace", "sleep", "money", "love"]},
    "mugwort": {"name": "Mugwort", "element": "Earth", "uses": ["divination", "dreams", "protection", "psychic"]},
    "basil": {"name": "Basil", "element": "Fire", "uses": ["money", "protection", "love", "courage"]},
    "thyme": {"name": "Thyme", "element": "Water", "uses": ["healing", "courage", "purification", "psychic"]},
    "cinnamon": {"name": "Cinnamon", "element": "Fire", "uses": ["money", "power", "protection", "success"]},
}

_FALLBACK_CRYSTALS = {
    "amethyst": {"name": "Amethyst", "element": "Water", "uses": ["intuition", "peace", "protection", "spiritual"]},
    "rose quartz": {"name": "Rose Quartz", "element": "Water", "uses": ["love", "healing", "compassion", "peace"]},
    "black tourmaline": {"name": "Black Tourmaline", "element": "Earth", "uses": ["protection", "grounding", "purification"]},
    "clear quartz": {"name": "Clear Quartz", "element": "Fire", "uses": ["amplification", "clarity", "healing", "energy"]},
    "citrine": {"name": "Citrine", "element": "Fire", "uses": ["money", "success", "confidence", "joy"]},
    "selenite": {"name": "Selenite", "element": "Water", "uses": ["cleansing", "clarity", "peace", "spiritual"]},
    "obsidian": {"name": "Obsidian", "element": "Earth", "uses": ["protection", "grounding", "truth", "shadow work"]},
    "labradorite": {"name": "Labradorite", "element": "Water", "uses": ["intuition", "transformation", "protection", "magic"]},
}

_FALLBACK_CORRESPONDENCES = {
    "protection": {"herbs": ["rosemary", "sage", "basil"], "crystals": ["black tourmaline", "obsidian"], "colors": ["black", "white"], "day": "Tuesday", "moon": "Waning"},
    "love": {"herbs": ["lavender", "chamomile", "basil"], "crystals": ["rose quartz", "amethyst"], "colors": ["pink", "red"], "day": "Friday", "moon": "Waxing"},
    "money": {"herbs": ["basil", "cinnamon", "chamomile"], "crystals": ["citrine", "clear quartz"], "colors": ["green", "gold"], "day": "Thursday", "moon": "Waxing"},
    "healing": {"herbs": ["rosemary", "thyme", "lavender"], "crystals": ["clear quartz", "amethyst"], "colors": ["blue", "green"], "day": "Sunday", "moon": "Full"},
    "divination": {"herbs": ["mugwort", "thyme"], "crystals": ["amethyst", "labradorite"], "colors": ["purple", "silver"], "day": "Monday", "moon": "Full"},
    "cleansing": {"herbs": ["sage", "rosemary", "thyme"], "crystals": ["selenite", "clear quartz"], "colors": ["white", "blue"], "day": "Saturday", "moon": "Waning"},
    "courage": {"herbs": ["basil", "thyme", "cinnamon"], "crystals": ["citrine", "clear quartz"], "colors": ["red", "orange"], "day": "Tuesday", "moon": "Waxing"},
    "peace": {"herbs": ["lavender", "chamomile"], "crystals": ["amethyst", "rose quartz", "selenite"], "colors": ["blue", "white", "lavender"], "day": "Monday", "moon": "Full"},
}

# Sabbat calendar (month, day) tuples
_SABBATS = [
    (2, 1, "Imbolc", "Awakening, purification, new beginnings"),
    (3, 20, "Ostara", "Balance, growth, fertility, renewal"),
    (5, 1, "Beltane", "Passion, creativity, union, fire"),
    (6, 21, "Litha", "Abundance, power, light, celebration"),
    (8, 1, "Lughnasadh", "First harvest, gratitude, sacrifice"),
    (9, 22, "Mabon", "Balance, gratitude, second harvest"),
    (10, 31, "Samhain", "Ancestors, divination, shadow work, death/rebirth"),
    (12, 21, "Yule", "Rebirth, hope, rest, longest night"),
]


class GrimoireService:
    """Adapter connecting Luna to the Grimoire Intelligence knowledge base."""

    def __init__(self):
        self.available = _grimoire_available

    def get_current_energy(self) -> dict:
        """Get current moon phase, nearest sabbat, and elemental energy."""
        moon = get_moon_phase()
        now = datetime.now(timezone.utc)

        # Find nearest sabbat
        nearest_sabbat = self._nearest_sabbat(now)

        # Elemental energy based on moon phase
        phase_elements = {
            "new": "Earth", "waxing_crescent": "Air", "first_quarter": "Fire",
            "waxing_gibbous": "Fire", "full": "Water", "waning_gibbous": "Water",
            "last_quarter": "Earth", "waning_crescent": "Air",
        }
        element = phase_elements.get(moon.get("key", ""), "Spirit")

        return {
            "moon": moon,
            "sabbat": nearest_sabbat,
            "element": element,
            "season": self._current_season(now),
        }

    def lookup_herb(self, name: str) -> dict | None:
        """Look up herb properties by name."""
        key = name.lower().strip()
        if self.available:
            return HERBS.get(key)
        return _FALLBACK_HERBS.get(key)

    def lookup_crystal(self, name: str) -> dict | None:
        """Look up crystal properties by name."""
        key = name.lower().strip()
        if self.available:
            return CRYSTALS.get(key)
        return _FALLBACK_CRYSTALS.get(key)

    def lookup_tarot_card(self, name: str) -> dict | None:
        """Look up tarot card data by name."""
        if self.available:
            key = name.lower().strip()
            return TAROT_CARDS.get(key)
        return None

    def get_correspondences(self, intention: str) -> dict:
        """Get herbs, crystals, colors, timing for an intention."""
        key = intention.lower().strip()

        if self.available:
            try:
                return _grimoire_correspondences(key)
            except Exception:
                pass

        # Fallback: check built-in correspondences
        if key in _FALLBACK_CORRESPONDENCES:
            c = _FALLBACK_CORRESPONDENCES[key]
            return {
                "intention": intention,
                "herbs": [_FALLBACK_HERBS.get(h, {"name": h}) for h in c["herbs"]],
                "crystals": [_FALLBACK_CRYSTALS.get(cr, {"name": cr}) for cr in c["crystals"]],
                "colors": c["colors"],
                "best_day": c["day"],
                "best_moon_phase": c["moon"],
            }

        # Fuzzy match: check if intention keyword appears in any herb/crystal uses
        matching_herbs = []
        matching_crystals = []
        herbs = HERBS if self.available else _FALLBACK_HERBS
        crystals = CRYSTALS if self.available else _FALLBACK_CRYSTALS

        for h_key, h_data in herbs.items():
            uses = h_data.get("uses", [])
            if any(key in u.lower() for u in uses):
                matching_herbs.append(h_data)
        for c_key, c_data in crystals.items():
            uses = c_data.get("uses", [])
            if any(key in u.lower() for u in uses):
                matching_crystals.append(c_data)

        return {
            "intention": intention,
            "herbs": matching_herbs[:3],
            "crystals": matching_crystals[:3],
            "colors": [],
            "best_day": None,
            "best_moon_phase": None,
        }

    def get_daily_practice(self) -> dict:
        """Get a personalized daily spiritual recommendation."""
        energy = self.get_current_energy()
        moon = energy["moon"]
        phase_key = moon.get("key", "new")

        practices = {
            "new": {"practice": "Set an intention for this lunar cycle. Write it down by candlelight.",
                    "herb": "mugwort", "crystal": "labradorite"},
            "waxing_crescent": {"practice": "Take one small step toward your new moon intention today.",
                               "herb": "basil", "crystal": "citrine"},
            "first_quarter": {"practice": "Face a small challenge today. The moon supports courage and action.",
                             "herb": "cinnamon", "crystal": "clear quartz"},
            "waxing_gibbous": {"practice": "Refine your approach. What needs adjusting before the full moon?",
                              "herb": "rosemary", "crystal": "amethyst"},
            "full": {"practice": "Celebrate what you've grown. Release what no longer serves you under the full moon's light.",
                    "herb": "lavender", "crystal": "selenite"},
            "waning_gibbous": {"practice": "Share what you've learned with someone who needs it.",
                              "herb": "sage", "crystal": "rose quartz"},
            "last_quarter": {"practice": "Forgive someone — even yourself. Clear emotional space.",
                            "herb": "chamomile", "crystal": "obsidian"},
            "waning_crescent": {"practice": "Rest and reflect. Journal about this cycle's lessons.",
                               "herb": "lavender", "crystal": "amethyst"},
        }

        rec = practices.get(phase_key, practices["new"])
        return {
            "practice": rec["practice"],
            "recommended_herb": self.lookup_herb(rec["herb"]),
            "recommended_crystal": self.lookup_crystal(rec["crystal"]),
            "moon": moon,
            "sabbat": energy["sabbat"],
        }

    def craft_spell(self, intention: str) -> dict:
        """Generate a spell framework for a given intention."""
        correspondences = self.get_correspondences(intention)
        moon = get_moon_phase()

        herb_names = [h.get("name", str(h)) if isinstance(h, dict) else str(h)
                      for h in correspondences.get("herbs", [])]
        crystal_names = [c.get("name", str(c)) if isinstance(c, dict) else str(c)
                         for c in correspondences.get("crystals", [])]

        return {
            "intention": intention,
            "title": f"Spell for {intention.title()}",
            "ingredients": {
                "herbs": herb_names,
                "crystals": crystal_names,
                "colors": correspondences.get("colors", []),
                "candle_color": correspondences["colors"][0] if correspondences.get("colors") else "white",
            },
            "best_timing": {
                "moon_phase": correspondences.get("best_moon_phase", "Any"),
                "day_of_week": correspondences.get("best_day", "Any"),
                "current_moon": moon["phase"],
            },
            "steps": [
                "Cleanse your space with sage or incense",
                f"Gather your materials: {', '.join(herb_names[:2])} and {crystal_names[0] if crystal_names else 'a white candle'}",
                f"Light a {correspondences['colors'][0] if correspondences.get('colors') else 'white'} candle",
                f"Hold your crystal and state your intention for {intention} clearly",
                "Meditate on your desired outcome for at least 5 minutes",
                "Thank the elements and close your circle",
            ],
            "correspondences": correspondences,
        }

    def craft_ritual(self, intention: str) -> dict:
        """Generate a ritual framework (longer form than a spell)."""
        spell = self.craft_spell(intention)
        energy = self.get_current_energy()

        spell["title"] = f"Ritual for {intention.title()}"
        spell["preparation"] = [
            "Choose a quiet time when you won't be disturbed",
            "Cleanse your ritual space with smoke or sound",
            "Cast a circle by walking clockwise, calling each element",
            "Set up your altar with gathered materials",
        ]
        spell["steps"] = [
            "Ground yourself: feet flat, three deep breaths",
            "Call upon the elements: Earth (North), Air (East), Fire (South), Water (West)",
            f"State your intention for {intention} three times",
            f"Work with your herbs: sprinkle {spell['ingredients']['herbs'][0] if spell['ingredients']['herbs'] else 'dried herbs'} around your candle",
            f"Hold your {spell['ingredients']['crystals'][0] if spell['ingredients']['crystals'] else 'crystal'} to your heart",
            "Visualize your intention manifesting in vivid detail",
            "Chant or speak your desire into the flame",
            "Sit in gratitude for what is already coming to you",
            "Thank the elements and open your circle counter-clockwise",
            "Let your candle burn safely or snuff (never blow) it out",
        ]
        spell["closing"] = "Ground any remaining energy into the earth. Eat or drink something to return fully to the physical."
        spell["current_energy"] = energy

        return spell

    def _nearest_sabbat(self, now: datetime) -> dict:
        """Find the nearest upcoming sabbat."""
        year = now.year
        best = None
        best_days = 999

        for month, day, name, themes in _SABBATS:
            sabbat_date = datetime(year, month, day, tzinfo=timezone.utc)
            delta = (sabbat_date - now).days
            if delta < 0:
                # Try next year
                sabbat_date = datetime(year + 1, month, day, tzinfo=timezone.utc)
                delta = (sabbat_date - now).days

            if 0 <= delta < best_days:
                best_days = delta
                best = {"name": name, "date": sabbat_date.strftime("%B %d"), "days_until": delta, "themes": themes}

        return best or {"name": "Unknown", "date": "", "days_until": 0, "themes": ""}

    def _current_season(self, now: datetime) -> str:
        month = now.month
        if month in (3, 4, 5):
            return "Spring — season of new beginnings"
        elif month in (6, 7, 8):
            return "Summer — season of abundance and fire"
        elif month in (9, 10, 11):
            return "Autumn — season of harvest and release"
        return "Winter — season of rest and reflection"
