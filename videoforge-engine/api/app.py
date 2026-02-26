"""VideoForge FastAPI — REST API on port 8090."""

import sys
import os
from dataclasses import asdict, fields
from enum import Enum
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videoforge.videoforge_engine import VideoForgeEngine

# ── Pydantic request models ──────────────────────────────────────────

class CreateRequest(BaseModel):
    topic: str
    niche: str
    platform: str = "youtube_shorts"
    format: str = "short"
    render: bool = False  # Default to dry-run for safety
    publish: bool = False

class BatchCreateRequest(BaseModel):
    items: list  # List of CreateRequest-like dicts
    render: bool = False

class AnalyzeRequest(BaseModel):
    topic: str
    niche: str
    platform: str = "youtube_shorts"

class TopicRequest(BaseModel):
    niche: str
    count: int = 10
    content_type: str = "educational"

class CostEstimateRequest(BaseModel):
    topic: str
    niche: str
    platform: str = "youtube_shorts"
    format: str = "short"


# ── App setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="VideoForge Intelligence API",
    description="Self-hosted video creation pipeline with FORGE+AMPLIFY intelligence",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = VideoForgeEngine()


def _to_dict(obj):
    """Recursively convert dataclasses/enums to JSON-serializable dicts."""
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(item) for item in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return str(obj)


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """List all available endpoints."""
    return {
        "service": "VideoForge Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "POST /create": "Full video creation pipeline",
            "POST /batch": "Create multiple videos",
            "POST /analyze": "Analyze topic (no render)",
            "POST /topics": "Generate topic ideas",
            "GET /calendar/{niche}": "Content calendar",
            "GET /insights/{niche}": "Performance insights",
            "POST /cost-estimate": "Pre-estimate cost",
            "GET /health": "Health check",
            "GET /knowledge/niches": "All niche profiles",
            "GET /knowledge/hooks": "Hook formulas",
            "GET /knowledge/platforms": "Platform specs",
            "GET /knowledge/moods": "Music moods",
            "GET /knowledge/subtitle-styles": "Subtitle styles",
            "GET /knowledge/trending": "Trending formats",
        },
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "videoforge",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/create")
def create_video(req: CreateRequest):
    """Full video creation pipeline."""
    try:
        result = engine.create_video(
            topic=req.topic,
            niche=req.niche,
            platform=req.platform,
            format=req.format,
            render=req.render,
            publish=req.publish,
        )
        return _to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
def batch_create(req: BatchCreateRequest):
    """Create multiple videos."""
    results = []
    for item in req.items:
        try:
            result = engine.create_video(
                topic=item.get("topic", ""),
                niche=item.get("niche", ""),
                platform=item.get("platform", "youtube_shorts"),
                format=item.get("format", "short"),
                render=req.render,
            )
            results.append(_to_dict(result))
        except Exception as e:
            results.append({"status": "error", "error": str(e), "topic": item.get("topic", "")})
    return {"results": results, "total": len(results)}


@app.post("/analyze")
def analyze_topic(req: AnalyzeRequest):
    """Analyze a topic without creating anything."""
    try:
        result = engine.analyze_topic(req.topic, req.niche, req.platform)
        return _to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/topics")
def generate_topics(req: TopicRequest):
    """Generate topic ideas for a niche."""
    topics = engine.generate_topics(req.niche, req.count, req.content_type)
    return {"niche": req.niche, "topics": topics}


@app.get("/calendar/{niche}")
def get_calendar(niche: str, platform: str = "youtube_shorts"):
    """Get a 7-day content calendar."""
    return engine.get_calendar(niche, platform)


@app.get("/insights")
@app.get("/insights/{niche}")
def get_insights(niche: str = None):
    """Get performance and cost insights."""
    return engine.get_insights(niche=niche)


@app.post("/cost-estimate")
def cost_estimate(req: CostEstimateRequest):
    """Pre-estimate cost for a video."""
    return engine.estimate_cost(req.topic, req.niche, req.platform, req.format)


# ── Knowledge endpoints ───────────────────────────────────────────────

@app.get("/knowledge/niches")
def knowledge_niches():
    """Get all niche profiles."""
    from videoforge.knowledge.niche_profiles import NICHE_PROFILES
    return {"niches": NICHE_PROFILES}


@app.get("/knowledge/hooks")
def knowledge_hooks():
    """Get all hook formulas."""
    from videoforge.knowledge.hook_formulas import HOOK_FORMULAS
    return {"hooks": HOOK_FORMULAS}


@app.get("/knowledge/platforms")
def knowledge_platforms():
    """Get all platform specs."""
    from videoforge.knowledge.platform_specs import PLATFORM_SPECS
    return {"platforms": PLATFORM_SPECS}


@app.get("/knowledge/moods")
def knowledge_moods():
    """Get all music moods."""
    from videoforge.knowledge.music_moods import MUSIC_MOODS
    return {"moods": MUSIC_MOODS}


@app.get("/knowledge/subtitle-styles")
def knowledge_subtitle_styles():
    """Get all subtitle styles."""
    from videoforge.knowledge.subtitle_styles import SUBTITLE_STYLES
    return {"styles": SUBTITLE_STYLES}


@app.get("/knowledge/trending")
def knowledge_trending(niche: str = None, platform: str = None):
    """Get trending video formats."""
    from videoforge.knowledge.trending_formats import get_trending_formats
    formats = get_trending_formats(niche=niche, platform=platform)
    return {"formats": formats}


@app.get("/knowledge/shots")
def knowledge_shots():
    """Get all shot types."""
    from videoforge.knowledge.shot_types import SHOT_TYPES
    return {"shots": SHOT_TYPES}


@app.get("/knowledge/transitions")
def knowledge_transitions():
    """Get all transitions."""
    from videoforge.knowledge.transitions import TRANSITIONS
    return {"transitions": TRANSITIONS}


@app.get("/knowledge/voices")
def knowledge_voices():
    """Get all voice profiles."""
    from videoforge.voice import VOICE_PROFILES
    return {"voices": VOICE_PROFILES}


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
