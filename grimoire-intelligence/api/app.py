"""FastAPI web layer for the Grimoire Intelligence System."""

import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from grimoire import GrimoireEngine
from grimoire.models import PracticeEntry, TarotReading


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
def energy():
    try:
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
        spread = engine.smith.generate_tarot_spread(req.intention)
        return spread
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


# ── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
