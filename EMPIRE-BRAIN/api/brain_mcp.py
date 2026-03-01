"""EMPIRE-BRAIN MCP Server

Gives Claude Code instant access to the entire Brain intelligence system.
Runs on port 8200 as a FastAPI-based MCP-compatible server.

Tools:
- brain_query        — Semantic search across all data
- brain_projects     — Get projects with optional filter
- brain_skills       — Get skills by category
- brain_learn        — Record a new learning
- brain_patterns     — Get detected patterns
- brain_opportunities — Get open opportunities
- brain_cross_reference — Find all data related to a topic
- brain_briefing     — Get today's briefing
- brain_health       — Get empire health status
- brain_solution     — Search for existing code solutions
- brain_record_solution — Save a reusable code solution
- brain_session      — Log a Claude Code session
- brain_amplify      — Run AMPLIFY pipeline on any data
- brain_site_context — Load full context for a specific site
- brain_stats        — Get brain statistics
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="EMPIRE-BRAIN MCP Server", version="3.0.0")
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


# --- Health ---
@app.get("/health")
def health():
    stats = db.stats()
    return {
        "status": "healthy",
        "version": "3.0.0",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
