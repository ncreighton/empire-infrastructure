"""EMPIRE-BRAIN MCP Server

Gives Claude Code instant access to the entire Brain intelligence system.
Runs on port 8200 as a FastAPI-based MCP-compatible server.

Tools:
- brain_query             — Semantic search across all data
- brain_projects          — Get projects with optional filter
- brain_skills            — Get skills by category
- brain_learn             — Record a new learning
- brain_patterns          — Get detected patterns
- brain_opportunities     — Get open opportunities
- brain_cross_reference   — Find all data related to a topic
- brain_briefing          — Get today's briefing
- brain_health            — Get empire health status
- brain_solution          — Search for existing code solutions
- brain_record_solution   — Save a reusable code solution
- brain_session           — Log a Claude Code session
- brain_amplify           — Run AMPLIFY pipeline on any data
- brain_site_context      — Load full context for a specific site
- brain_stats             — Get brain statistics
- brain_evolution_status  — Recent evolution cycles, pending counts, adoption rates
- brain_discoveries       — List discovered APIs/tools (filterable by relevance)
- brain_discovery_update  — Approve/dismiss discovery
- brain_ideas             — List generated ideas (filterable)
- brain_idea_update       — Approve/reject idea
- brain_enhancements      — List code improvements (filterable by confidence)
- brain_enhancement_update — Approve/reject enhancement
- brain_evolve            — Trigger evolution cycle manually
- brain_adoption_metrics  — Proposal acceptance rates
- brain_invalidate_cycle  — Invalidate all results from a bad evolution cycle
- brain_sync_evolution    — Push evolution tables to remote PostgreSQL
- brain_auto_apply        — Auto-apply safe, high-confidence enhancements
- brain_grimoire_health   — Grimoire API health + energy
- brain_grimoire_recommend — Enhanced recommendations (brain + grimoire)
- brain_grimoire_ideas    — Content ideas from brain + grimoire skills
- brain_grimoire_insights — Cross-system insights
- brain_grimoire_sync     — Sync grimoire practice stats to brain
- brain_witchcraft_topics — Generate witchcraft video topics
- brain_witchcraft_video  — Create witchcraft video via VideoForge
- brain_witchcraft_calendar — Content calendar (sabbats + moon + videos)
- brain_witchcraft_costs  — Estimate video production costs
- brain_article_list      — List articles scored by video potential
- brain_article_to_video  — Convert article to video
- brain_article_batch     — Batch convert top articles to videos
- brain_auto_pins         — Generate Pinterest pins for recent articles
- brain_auto_pin          — Generate pin for specific article
- brain_pin_calendar      — Pinterest pin content calendar

Pages:
- /dashboard              — Evolution approval dashboard (HTML)
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.brain_db import BrainDB
from forge.brain_scout import BrainScout
from forge.brain_sentinel import BrainSentinel
from forge.brain_oracle import BrainOracle
from forge.brain_smith import BrainSmith
from forge.brain_codex import BrainCodex
from amplify.pipeline import AmplifyPipeline
from config.settings import EMPIRE_SITES

app = FastAPI(title="EMPIRE-BRAIN MCP Server", version="3.2.0")
db = BrainDB()


# --- Request Models ---
class QueryRequest(BaseModel):
    query: str
    limit: int = 20

class LearnRequest(BaseModel):
    content: str
    source: str = ""
    category: str = "lesson"
    confidence: float = 0.8

class SolutionRequest(BaseModel):
    problem: str
    solution: str = ""
    language: str = "python"
    project: str = ""
    file_path: str = ""
    tags: list[str] = []

class SessionRequest(BaseModel):
    project_slug: str
    summary: str = ""
    files_modified: list[str] = []
    learnings: list[str] = []
    patterns: list[str] = []

class AmplifyRequest(BaseModel):
    data: dict
    context: str = ""
    quick: bool = True

class CrossRefRequest(BaseModel):
    topic: str

class StatusUpdateRequest(BaseModel):
    id: int
    status: str

class EvolveRequest(BaseModel):
    cycle: str = "quick"  # quick, discover, full
    project: Optional[str] = None  # Optional project scope for targeted cycles

class InvalidateRequest(BaseModel):
    evolution_id: int


# --- Health ---
@app.get("/health")
def health():
    stats = db.stats()
    return {
        "status": "healthy",
        "version": "3.2.0",
        "brain_stats": stats,
        "timestamp": datetime.now().isoformat(),
    }


# --- MCP Tool Endpoints ---

@app.post("/tools/brain_query")
def brain_query(req: QueryRequest):
    """Semantic search across all brain data."""
    results = db.search(req.query, limit=req.limit)
    return {"query": req.query, "results": results}


@app.get("/tools/brain_projects")
def brain_projects(category: Optional[str] = None):
    """Get all projects, optionally filtered by category."""
    projects = db.get_projects(category=category)
    return {"projects": projects, "count": len(projects)}


@app.get("/tools/brain_skills")
def brain_skills(category: Optional[str] = None, project: Optional[str] = None):
    """Get skills by category or project."""
    skills = db.get_skills(category=category, project=project)
    return {"skills": skills, "count": len(skills)}


@app.post("/tools/brain_learn")
def brain_learn(req: LearnRequest):
    """Record a new learning."""
    codex = BrainCodex(db)
    learning_id = codex.learn(req.content, req.source, req.category, req.confidence)
    return {"learning_id": learning_id, "status": "recorded"}


@app.get("/tools/brain_patterns")
def brain_patterns(pattern_type: Optional[str] = None):
    """Get detected patterns."""
    patterns = db.get_patterns(pattern_type=pattern_type)
    return {"patterns": patterns, "count": len(patterns)}


@app.get("/tools/brain_opportunities")
def brain_opportunities(status: str = "open"):
    """Get open opportunities."""
    opportunities = db.get_opportunities(status=status)
    return {"opportunities": opportunities, "count": len(opportunities)}


@app.post("/tools/brain_cross_reference")
def brain_cross_reference(req: CrossRefRequest):
    """Find all data related to a topic."""
    smith = BrainSmith(db)
    results = smith.cross_reference(req.topic)
    return {"topic": req.topic, "results": results}


@app.get("/tools/brain_briefing")
def brain_briefing():
    """Get today's briefing."""
    smith = BrainSmith(db)
    briefing = smith.generate_briefing()
    return briefing


@app.get("/tools/brain_health")
def brain_health():
    """Get empire health status."""
    sentinel = BrainSentinel(db)
    return sentinel.full_health_check()


@app.post("/tools/brain_solution")
def brain_search_solution(req: QueryRequest):
    """Search for existing code solutions."""
    smith = BrainSmith(db)
    solutions = smith.find_solution(req.query)
    return {"query": req.query, "solutions": solutions}


@app.post("/tools/brain_record_solution")
def brain_record_solution(req: SolutionRequest):
    """Save a reusable code solution."""
    smith = BrainSmith(db)
    smith.record_solution(
        problem=req.problem, solution=req.solution,
        language=req.language, project=req.project,
        file_path=req.file_path, tags=req.tags,
    )
    return {"status": "recorded"}


@app.post("/tools/brain_session")
def brain_session(req: SessionRequest):
    """Log a Claude Code session."""
    codex = BrainCodex(db)
    codex.capture_session(
        project_slug=req.project_slug,
        summary=req.summary,
        files_modified=req.files_modified,
        learnings=req.learnings,
        patterns=req.patterns,
    )
    return {"status": "logged"}


@app.post("/tools/brain_amplify")
def brain_amplify(req: AmplifyRequest):
    """Run AMPLIFY pipeline on any data."""
    amplify = AmplifyPipeline(db)
    if req.quick:
        result = amplify.amplify_quick(req.data, req.context)
    else:
        result = amplify.amplify(req.data, req.context)
    return result


@app.get("/tools/brain_site_context")
def brain_site_context(site_id: str):
    """Load full context for a specific site."""
    smith = BrainSmith(db)

    # Get project data
    projects = db.get_projects()
    site_proj = next((p for p in projects if site_id in p.get("slug", "")), None)

    if not site_proj:
        raise HTTPException(404, f"Site '{site_id}' not found in brain")

    # Get DNA profile
    dna = smith.project_dna(site_proj["slug"])

    # Get related learnings
    learnings = db.search_learnings(site_id, limit=10)

    # Get related skills
    skills = db.get_skills(project=site_proj["slug"])

    # Cross-reference
    xref = smith.cross_reference(site_id)

    return {
        "site": site_proj,
        "dna": dna,
        "learnings": [dict(l) for l in learnings],
        "skills": [dict(s) for s in skills],
        "cross_references": xref,
    }


@app.get("/tools/brain_stats")
def brain_stats():
    """Get brain statistics."""
    return db.stats()


@app.get("/tools/brain_forecast")
def brain_forecast():
    """Get weekly forecast from Oracle."""
    oracle = BrainOracle(db)
    return oracle.weekly_forecast()


@app.post("/tools/brain_scan")
def brain_scan(project: Optional[str] = None):
    """Trigger a brain scan."""
    scout = BrainScout(db)
    if project:
        projects = scout.discover_projects()
        target = next((p for p in projects if p["slug"] == project), None)
        if not target:
            raise HTTPException(404, f"Project '{project}' not found")
        scout._scan_project(target)
        return {"status": "scanned", "project": project, "stats": scout.stats}
    else:
        stats = scout.full_scan()
        return {"status": "full_scan_complete", "stats": stats}


# --- Events ---
@app.get("/events")
def get_events(limit: int = 50, event_type: Optional[str] = None):
    """Get recent events."""
    return {"events": db.recent_events(limit=limit, event_type=event_type)}


# --- Evolution Engine Endpoints ---

@app.get("/tools/brain_evolution_status")
def brain_evolution_status():
    """Recent evolution cycles, pending counts, and adoption metrics."""
    recent = db.recent_evolutions(limit=5)
    pending_enhancements = len(db.get_enhancements(status="pending"))
    pending_ideas = len(db.get_ideas(status="proposed"))
    new_discoveries = len(db.get_discoveries(status="discovered"))
    stats = db.stats()
    adoption = db.adoption_metrics()
    return {
        "recent_cycles": recent,
        "pending": {
            "enhancements": pending_enhancements,
            "ideas": pending_ideas,
            "discoveries": new_discoveries,
        },
        "totals": {
            "evolutions": stats.get("evolutions", 0),
            "discoveries": stats.get("discoveries", 0),
            "ideas": stats.get("ideas", 0),
            "enhancements": stats.get("enhancements", 0),
        },
        "adoption": adoption,
    }


@app.get("/tools/brain_discoveries")
def brain_discoveries(status: Optional[str] = None, discovery_type: Optional[str] = None,
                      min_relevance: float = 0, limit: int = 50):
    """List discovered APIs/tools with optional relevance threshold."""
    discoveries = db.get_discoveries(status=status, discovery_type=discovery_type,
                                     limit=limit, min_relevance=min_relevance)
    return {"discoveries": discoveries, "count": len(discoveries)}


@app.post("/tools/brain_discovery_update")
def brain_discovery_update(req: StatusUpdateRequest):
    """Approve or dismiss a discovery."""
    valid = ("discovered", "evaluated", "recommended", "integrated", "dismissed")
    if req.status not in valid:
        raise HTTPException(400, f"Invalid status: {req.status}. Valid: {', '.join(valid)}")
    db.update_discovery_status(req.id, req.status)
    return {"id": req.id, "status": req.status, "updated": True}


@app.get("/tools/brain_ideas")
def brain_ideas(status: Optional[str] = None, idea_type: Optional[str] = None, limit: int = 50):
    """List generated ideas."""
    ideas = db.get_ideas(status=status, idea_type=idea_type, limit=limit)
    return {"ideas": ideas, "count": len(ideas)}


@app.post("/tools/brain_idea_update")
def brain_idea_update(req: StatusUpdateRequest):
    """Approve or reject an idea."""
    valid = ("proposed", "approved", "in_progress", "completed", "rejected")
    if req.status not in valid:
        raise HTTPException(400, f"Invalid status: {req.status}. Valid: {', '.join(valid)}")
    db.update_idea_status(req.id, req.status)
    return {"id": req.id, "status": req.status, "updated": True}


@app.get("/tools/brain_enhancements")
def brain_enhancements(status: Optional[str] = None, project: Optional[str] = None,
                       enhancement_type: Optional[str] = None,
                       min_confidence: float = 0, limit: int = 50):
    """List code improvements with optional confidence threshold."""
    enhancements = db.get_enhancements(status=status, project=project,
                                       enhancement_type=enhancement_type,
                                       limit=limit, min_confidence=min_confidence)
    return {"enhancements": enhancements, "count": len(enhancements)}


@app.post("/tools/brain_enhancement_update")
def brain_enhancement_update(req: StatusUpdateRequest):
    """Approve or reject an enhancement."""
    valid = ("pending", "approved", "applied", "rejected")
    if req.status not in valid:
        raise HTTPException(400, f"Invalid status: {req.status}. Valid: {', '.join(valid)}")
    db.update_enhancement_status(req.id, req.status)
    return {"id": req.id, "status": req.status, "updated": True}


@app.post("/tools/brain_evolve")
def brain_evolve(req: EvolveRequest):
    """Trigger an evolution cycle manually."""
    from agents.evolution_agent import EvolutionEngine
    engine = EvolutionEngine()

    if req.cycle == "quick":
        result = engine.quick_enhance()
    elif req.cycle == "discover":
        result = engine.deep_discover()
    elif req.cycle == "full":
        result = engine.full_evolution()
    else:
        raise HTTPException(400, f"Invalid cycle: {req.cycle}. Use: quick, discover, full")

    return {"cycle": req.cycle, "result": result}


@app.get("/tools/brain_adoption_metrics")
def brain_adoption_metrics():
    """Get proposal acceptance rates across all evolution tables."""
    return db.adoption_metrics()


@app.post("/tools/brain_invalidate_cycle")
def brain_invalidate_cycle(req: InvalidateRequest):
    """Invalidate all results from a bad evolution cycle."""
    db.invalidate_evolution(req.evolution_id)
    return {"evolution_id": req.evolution_id, "status": "invalidated"}


@app.post("/tools/brain_sync_evolution")
def brain_sync_evolution():
    """Push evolution tables (discoveries/ideas/enhancements) to remote PostgreSQL."""
    try:
        from connectors.postgres_connector import PostgresConnector
        pg = PostgresConnector()
        if not pg.connect():
            raise HTTPException(503, "PostgreSQL connection failed")
        pg.init_schema()
        result = pg.sync_evolution_tables(db)
        pg.close()
        return {"status": "synced", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Sync failed: {e}")


class AutoApplyRequest(BaseModel):
    dry_run: bool = True


# --- Dashboard ---
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve the Evolution Engine approval dashboard."""
    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# --- Auto-Apply ---
@app.post("/tools/brain_auto_apply")
def brain_auto_apply(req: AutoApplyRequest):
    """Run auto-apply for safe, high-confidence enhancements."""
    from forge.brain_auto_apply import BrainAutoApply
    auto = BrainAutoApply(db=db, dry_run=req.dry_run)
    return auto.run()


# ── Brain-Grimoire Connector Endpoints ────────────────────────────────

@app.get("/tools/brain_grimoire_health")
def brain_grimoire_health():
    """Check Grimoire system health from Brain perspective."""
    from connectors.brain_grimoire import BrainGrimoireConnector
    return BrainGrimoireConnector().get_grimoire_health()


@app.post("/tools/brain_grimoire_recommend")
def brain_grimoire_recommend(query: str):
    """Get brain-enhanced grimoire recommendations for a practice query."""
    from connectors.brain_grimoire import BrainGrimoireConnector
    return BrainGrimoireConnector().get_enhanced_recommendations(query)


@app.get("/tools/brain_grimoire_ideas")
def brain_grimoire_ideas(count: int = 5):
    """Get content ideas combining brain insights + grimoire knowledge."""
    from connectors.brain_grimoire import BrainGrimoireConnector
    return BrainGrimoireConnector().get_content_ideas(count)


@app.get("/tools/brain_grimoire_insights")
def brain_grimoire_insights():
    """Get cross-system insights relevant to witchcraft practice."""
    from connectors.brain_grimoire import BrainGrimoireConnector
    return BrainGrimoireConnector().get_cross_system_insights()


@app.post("/tools/brain_grimoire_sync")
def brain_grimoire_sync():
    """Sync grimoire practice stats back to brain as learnings."""
    from connectors.brain_grimoire import BrainGrimoireConnector
    return BrainGrimoireConnector().sync_practice_stats()


# ── Witchcraft Video Pipeline Endpoints ───────────────────────────────

@app.get("/tools/brain_witchcraft_topics")
def brain_witchcraft_topics(count: int = 5):
    """Generate witchcraft video topic ideas based on current moon/sabbat energy."""
    from connectors.witchcraft_video_pipeline import WitchcraftVideoPipeline
    return WitchcraftVideoPipeline().generate_topics(count)


@app.post("/tools/brain_witchcraft_video")
def brain_witchcraft_video(topic: str, render: bool = False, publish: bool = False):
    """Create a witchcraft YouTube Short (dry_run if render=False)."""
    from connectors.witchcraft_video_pipeline import WitchcraftVideoPipeline
    return WitchcraftVideoPipeline().create_video(
        topic, dry_run=not render, render=render, publish=publish,
    )


@app.get("/tools/brain_witchcraft_calendar")
def brain_witchcraft_calendar(days: int = 14):
    """Content calendar combining grimoire sabbats + moon phases + VideoForge."""
    from connectors.witchcraft_video_pipeline import WitchcraftVideoPipeline
    return WitchcraftVideoPipeline().get_content_calendar(days)


@app.get("/tools/brain_witchcraft_costs")
def brain_witchcraft_costs(count: int = 1):
    """Estimate costs for producing witchcraft videos."""
    from connectors.witchcraft_video_pipeline import WitchcraftVideoPipeline
    return WitchcraftVideoPipeline().estimate_costs(count)


# ── Article-to-Video Pipeline Endpoints ───────────────────────────────

@app.get("/tools/brain_article_list")
def brain_article_list(site: str, count: int = 10):
    """List recent articles from a WordPress site, scored by video potential."""
    from connectors.article_to_video import ArticleToVideoPipeline
    return ArticleToVideoPipeline().list_articles(site, count)


@app.post("/tools/brain_article_to_video")
def brain_article_to_video(site: str, post_id: int, render: bool = False):
    """Convert a WordPress article into a video via VideoForge."""
    from connectors.article_to_video import ArticleToVideoPipeline
    return ArticleToVideoPipeline().convert_article(site, post_id, render=render)


@app.post("/tools/brain_article_batch")
def brain_article_batch(site: str, count: int = 3, render: bool = False):
    """Batch convert top articles from a site to videos."""
    from connectors.article_to_video import ArticleToVideoPipeline
    return ArticleToVideoPipeline().batch_convert(site, count, render=render)


# ── Auto-Pin Connector Endpoints ─────────────────────────────────────

@app.get("/tools/brain_auto_pins")
def brain_auto_pins(site: str, count: int = 5):
    """Generate Pinterest pin data for recent published articles."""
    from connectors.auto_pin import AutoPinConnector
    return AutoPinConnector().generate_pins(site, count)


@app.post("/tools/brain_auto_pin")
def brain_auto_pin(site: str, post_id: int):
    """Generate a Pinterest pin for a specific article."""
    from connectors.auto_pin import AutoPinConnector
    return AutoPinConnector().generate_pin(site, post_id)


@app.get("/tools/brain_pin_calendar")
def brain_pin_calendar(site: str, days: int = 7):
    """Get a Pinterest pin content calendar with optimal scheduling."""
    from connectors.auto_pin import AutoPinConnector
    return AutoPinConnector().get_pin_calendar(site, days)


# --- Credit Optimizer Endpoints ---

@app.get("/tools/brain_credit_status")
def brain_credit_status():
    """Get current Claude Max credit usage status and optimization recommendations."""
    from forge.credit_optimizer import CreditOptimizer
    optimizer = CreditOptimizer(brain_db=db)
    savings = optimizer.calculate_potential_savings()
    sessions = optimizer.analyze_session_patterns()
    return {
        "potential_savings": savings,
        "session_patterns": sessions,
        "advisory": optimizer.generate_session_start_advisory(),
    }


@app.get("/tools/brain_credit_report")
def brain_credit_report():
    """Get full credit optimization report with recommendations."""
    from scripts.credit_optimizer_hook import generate_report
    return generate_report()


@app.get("/tools/brain_credit_analysis")
def brain_credit_analysis():
    """Run deep credit analysis: CLAUDE.md sizes, compliance, waste patterns."""
    from forge.credit_optimizer import CreditOptimizer
    optimizer = CreditOptimizer(brain_db=db)
    return optimizer.full_credit_analysis()


@app.get("/tools/brain_claude_md_sizes")
def brain_claude_md_sizes(limit: int = 15):
    """List CLAUDE.md files sorted by size — oversized files inflate every conversation."""
    from forge.credit_optimizer import CreditOptimizer
    optimizer = CreditOptimizer(brain_db=db)
    sizes = optimizer.analyze_claude_md_sizes()
    return {
        "total_files": len(sizes),
        "total_tokens": sum(f["est_tokens"] for f in sizes),
        "files": sizes[:limit],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
