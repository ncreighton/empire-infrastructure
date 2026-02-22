"""Grimoire Intelligence System — AI-powered witchcraft practice companion."""

from .grimoire_engine import GrimoireEngine

from .models import (
    MoonPhase, Element, Planet, SpellType, Difficulty, IntentionCategory,
    QueryType, Sabbat, ZodiacSign,
    RitualPlan, GeneratedSpell, GeneratedRitual, GeneratedMeditation,
    AmplifyResult, EnhancedQuery, PracticeEntry, TarotReading,
    DailyPractice, WeeklyForecast, ConsultResult, GrimoireResult,
)

__all__ = [
    "GrimoireEngine",
    "MoonPhase", "Element", "Planet", "SpellType", "Difficulty",
    "IntentionCategory", "QueryType", "Sabbat", "ZodiacSign",
    "RitualPlan", "GeneratedSpell", "GeneratedRitual", "GeneratedMeditation",
    "AmplifyResult", "EnhancedQuery", "PracticeEntry", "TarotReading",
    "DailyPractice", "WeeklyForecast", "ConsultResult", "GrimoireResult",
]
