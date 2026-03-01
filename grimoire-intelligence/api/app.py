"""FastAPI web layer for the Grimoire Intelligence System."""

import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from grimoire import GrimoireEngine
from grimoire.models import PracticeEntry, TarotReading
from grimoire.knowledge.correspondences import (
    HERBS, CRYSTALS, COLORS, ELEMENTS, INTENTION_MAP,
)
from grimoire.knowledge.tarot import MAJOR_ARCANA, MINOR_ARCANA, SPREADS, draw_cards
from grimoire.knowledge.moon_phases import MOON_PHASES, MOON_IN_SIGNS
from grimoire.knowledge.wheel_of_year import (
    SABBATS, get_current_sabbat, get_next_sabbat,
)
from grimoire.knowledge.planetary_hours import (
    PLANET_CORRESPONDENCES, get_current_planetary_hour,
    get_all_hours_for_day, get_day_ruler,
)
from grimoire.knowledge.spell_templates import SPELL_TYPES, RITUAL_STRUCTURE


# ── Pydantic request/response models ──────────────────────────────────────

class ConsultRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Your question or request")


class SpellRequest(BaseModel):
    intention: str = Field(..., min_length=1)
    spell_type: str = Field(default="candle")
    difficulty: str = Field(default="beginner")
    amplify: bool = Field(default=True)


class RitualRequest(BaseModel):
    occasion: str = Field(..., min_length=1)
    intention: str = Field(..., min_length=1)
    difficulty: str = Field(default="beginner")
    amplify: bool = Field(default=True)


class MeditationRequest(BaseModel):
    intention: str = Field(..., min_length=1)
    difficulty: str = Field(default="beginner")


class PracticeLogRequest(BaseModel):
    practice_type: str
    title: str
    intention: str = ""
    moon_phase: str = ""
    correspondences_used: list[str] = []
    notes: str = ""
    mood_before: str = ""
    mood_after: str = ""
    effectiveness_rating: int = Field(default=0, ge=0, le=5)


class TarotLogRequest(BaseModel):
    spread_name: str
    question: str = ""
    cards: list[dict[str, str]] = []
    interpretation: str = ""
    follow_up_actions: list[str] = []


class TarotSpreadRequest(BaseModel):
    intention: str = Field(default="general guidance")
    spread_type: str = Field(default="three_card")


# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Grimoire Intelligence System",
    description="AI-powered witchcraft practice companion",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = GrimoireEngine()


def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses and enums to dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = _to_dict(value)
        return result
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Grimoire Intelligence System",
        "version": "1.0.0",
        "endpoints": [
            "/consult", "/energy", "/forecast",
            "/craft/spell", "/craft/ritual", "/craft/meditation",
            "/daily", "/log", "/journey",
            "/tarot/spread", "/tarot/log",
            "/knowledge/herbs", "/knowledge/crystals", "/knowledge/colors",
            "/knowledge/elements", "/knowledge/intentions", "/knowledge/tarot",
            "/knowledge/moon-phases", "/knowledge/sabbats",
            "/knowledge/planetary-hours", "/knowledge/spell-types",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "grimoire-intelligence"}


@app.post("/consult")
def consult(req: ConsultRequest):
    try:
        result = engine.consult(req.query)
        return _to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/energy")
def energy(lat: float | None = None, lon: float | None = None):
    try:
        if lat is not None or lon is not None:
            from grimoire import GrimoireEngine as _GE
            local_engine = _GE(lat=lat, lon=lon)
            return local_engine.current_energy()
        return engine.current_energy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast")
def forecast():
    try:
        result = engine.weekly_forecast()
        return _to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/craft/spell")
def craft_spell(req: SpellRequest):
    try:
        result = engine.craft_spell(
            intention=req.intention,
            difficulty=req.difficulty,
            spell_type=req.spell_type,
            amplify_result=req.amplify,
        )
        return _to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/craft/ritual")
def craft_ritual(req: RitualRequest):
    try:
        result = engine.craft_ritual(
            occasion=req.occasion,
            intention=req.intention,
            difficulty=req.difficulty,
            amplify_result=req.amplify,
        )
        return _to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/craft/meditation")
def craft_meditation(req: MeditationRequest):
    try:
        result = engine.craft_meditation(
            intention=req.intention,
            difficulty=req.difficulty,
        )
        return _to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/daily")
def daily():
    try:
        result = engine.daily_practice()
        return _to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/log")
def log_practice(req: PracticeLogRequest):
    try:
        entry = PracticeEntry(
            practice_type=req.practice_type,
            title=req.title,
            intention=req.intention,
            moon_phase=req.moon_phase,
            correspondences_used=req.correspondences_used,
            notes=req.notes,
            mood_before=req.mood_before,
            mood_after=req.mood_after,
            effectiveness_rating=req.effectiveness_rating,
        )
        result = engine.log_practice(entry)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/journey")
def journey():
    try:
        result = engine.my_journey()
        return _to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tarot/spread")
def tarot_spread(req: TarotSpreadRequest):
    try:
        # Look up spread template
        spread_template = SPREADS.get(req.spread_type)
        if not spread_template:
            raise HTTPException(status_code=400, detail=f"Unknown spread type: {req.spread_type}")

        spread_name = spread_template["name"]
        positions = spread_template["positions"]
        card_count = spread_template["card_count"]

        # Draw actual random cards
        drawn = draw_cards(card_count)

        # Pair each drawn card with its position
        cards = []
        for i, card in enumerate(drawn):
            position_name = positions[i] if i < len(positions) else f"Card {i + 1}"
            orientation = card.get("orientation", "upright")
            is_reversed = orientation == "reversed"

            # Pick meaning based on orientation
            if is_reversed:
                meaning = card.get("reversed_meaning", "")
                keywords = card.get("keywords_reversed", [])
            else:
                meaning = card.get("upright_meaning", "")
                keywords = card.get("keywords_upright", [])

            cards.append({
                "position": position_name,
                "card_name": card.get("name", ""),
                "orientation": orientation,
                "element": card.get("element", ""),
                "meaning": meaning,
                "keywords": keywords,
                "advice": card.get("advice", ""),
                "journal_prompt": card.get("journal_prompt", ""),
                "correspondences": card.get("correspondences", {}),
            })

        return {
            "spread_name": spread_name,
            "spread_type": req.spread_type,
            "intention": req.intention,
            "card_count": card_count,
            "cards": cards,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tarot/log")
def tarot_log(req: TarotLogRequest):
    try:
        reading = TarotReading(
            spread_name=req.spread_name,
            question=req.question,
            cards=req.cards,
            interpretation=req.interpretation,
            follow_up_actions=req.follow_up_actions,
        )
        result = engine.log_tarot(reading)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Knowledge Endpoints ───────────────────────────────────────────────────


def _filter_dict(data: dict, query: str | None) -> dict:
    """Filter a dict of items by name or properties matching query."""
    if not query:
        return data
    q = query.lower()
    result = {}
    for key, val in data.items():
        if q in key.lower():
            result[key] = val
        elif isinstance(val, dict):
            name = val.get("name", "")
            props = val.get("magical_properties", [])
            if q in name.lower() or any(q in p.lower() for p in props):
                result[key] = val
    return result


@app.get("/knowledge/herbs")
def knowledge_herbs(q: str | None = Query(default=None)):
    return {"count": len(HERBS) if not q else None, "items": _filter_dict(HERBS, q)}


@app.get("/knowledge/crystals")
def knowledge_crystals(q: str | None = Query(default=None)):
    return {"count": len(CRYSTALS) if not q else None, "items": _filter_dict(CRYSTALS, q)}


@app.get("/knowledge/colors")
def knowledge_colors():
    return {"count": len(COLORS), "items": COLORS}


@app.get("/knowledge/elements")
def knowledge_elements():
    return {"count": len(ELEMENTS), "items": ELEMENTS}


@app.get("/knowledge/intentions")
def knowledge_intentions():
    return {"count": len(INTENTION_MAP), "items": INTENTION_MAP}


@app.get("/knowledge/tarot")
def knowledge_tarot(q: str | None = Query(default=None)):
    major = MAJOR_ARCANA
    minor = MINOR_ARCANA
    if q:
        ql = q.lower()
        major = [c for c in major if ql in c.get("name", "").lower()
                 or any(ql in kw.lower() for kw in c.get("keywords_upright", []))]
        filtered_minor = {}
        for suit, data in minor.items():
            cards = [c for c in data.get("cards", [])
                     if ql in c.get("name", "").lower()
                     or any(ql in kw.lower() for kw in c.get("keywords_upright", []))]
            if cards:
                filtered_minor[suit] = {**data, "cards": cards}
        minor = filtered_minor
    return {
        "major_arcana": major,
        "minor_arcana": minor,
        "spreads": SPREADS,
    }


@app.get("/knowledge/moon-phases")
def knowledge_moon_phases():
    return {
        "phases": MOON_PHASES,
        "moon_in_signs": MOON_IN_SIGNS,
    }


@app.get("/knowledge/sabbats")
def knowledge_sabbats():
    now = datetime.datetime.now()
    current = get_current_sabbat(now.month, now.day)
    next_name, next_data, days_until = get_next_sabbat(now.month, now.day)
    return {
        "sabbats": SABBATS,
        "current": current,
        "next": {"name": next_name, "data": next_data, "days_until": days_until},
    }


@app.get("/knowledge/planetary-hours")
def knowledge_planetary_hours():
    now = datetime.datetime.now()
    weekday = now.weekday()
    current_hour = get_current_planetary_hour(weekday, now.hour)
    all_hours = get_all_hours_for_day(weekday)
    day_ruler = get_day_ruler(weekday)
    return {
        "current_hour": current_hour,
        "day_ruler": day_ruler,
        "all_hours": all_hours,
        "planets": PLANET_CORRESPONDENCES,
    }


@app.get("/knowledge/spell-types")
def knowledge_spell_types():
    return {
        "spell_types": SPELL_TYPES,
        "ritual_structures": RITUAL_STRUCTURE,
    }


# ── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
