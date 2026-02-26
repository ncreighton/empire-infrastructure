#!/usr/bin/env python3
"""
ForgeFiles Pipeline API
========================
FastAPI wrapper around the orchestrator for n8n / webhook / cron integration.

Endpoints:
    POST /pipeline/run          — Run pipeline on single STL
    POST /pipeline/batch        — Run pipeline on directory of STLs
    GET  /pipeline/status/{id}  — Check pipeline run status
    GET  /pipeline/history      — List recent runs
    GET  /health                — Health check

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8090
"""

import os
import sys
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add scripts dir to path
PIPELINE_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PIPELINE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from orchestrator import run_full_pipeline, batch_pipeline, load_pipeline_config

app = FastAPI(
    title="ForgeFiles Pipeline API",
    description="3D model content pipeline — STL to social-ready video",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracking
_jobs: dict = {}
_jobs_lock = threading.Lock()


# ============================================================================
# MODELS
# ============================================================================

class PipelineRequest(BaseModel):
    stl: str = Field(..., description="Path to STL file")
    output: Optional[str] = Field(None, description="Output base directory")
    mode: str = Field("turntable", description="Render mode")
    platforms: list[str] = Field(default_factory=lambda: ["tiktok", "reels", "youtube", "pinterest", "reddit"])
    material: Optional[str] = Field(None, description="Material preset")
    music: Optional[str] = Field(None, description="Background music path")
    title: Optional[str] = Field(None, description="Custom title")
    preset: str = Field("portfolio", description="Quality preset: social, portfolio, ultra")
    fast: bool = Field(False, description="Use fast/social preset")
    camera_style: str = Field("standard", description="Camera style")
    color_grade: str = Field("cinematic", description="Color grade")
    variants: int = Field(3, description="A/B caption variants")
    skip_existing: bool = Field(True, description="Skip if output exists")


class BatchRequest(BaseModel):
    directory: str = Field(..., description="Directory containing STL files")
    output: Optional[str] = Field(None, description="Output base directory")
    mode: str = Field("turntable")
    platforms: list[str] = Field(default_factory=lambda: ["tiktok", "reels", "youtube", "pinterest", "reddit"])
    material: Optional[str] = None
    music: Optional[str] = None
    preset: str = Field("portfolio")
    fast: bool = False
    camera_style: str = Field("standard")
    color_grade: str = Field("cinematic")
    variants: int = 3


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued, running, completed, failed
    model: Optional[str] = None
    started: Optional[str] = None
    completed: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


# ============================================================================
# BACKGROUND WORKER
# ============================================================================

def _run_pipeline_job(job_id: str, request: PipelineRequest):
    """Execute pipeline in background thread."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started"] = datetime.now().isoformat()

    try:
        result = run_full_pipeline(
            stl_path=request.stl,
            output_base=request.output,
            mode=request.mode,
            platforms=request.platforms,
            material=request.material,
            music_path=request.music,
            preset=request.preset,
            fast=request.fast,
            title=request.title,
            camera_style=request.camera_style,
            color_grade=request.color_grade,
            skip_existing=request.skip_existing,
            variant_count=request.variants,
        )

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed" if result else "failed"
            _jobs[job_id]["completed"] = datetime.now().isoformat()
            _jobs[job_id]["result"] = result
            if not result:
                _jobs[job_id]["error"] = "Pipeline returned no result (render may have failed)"

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["completed"] = datetime.now().isoformat()
            _jobs[job_id]["error"] = str(e)


def _run_batch_job(job_id: str, request: BatchRequest):
    """Execute batch pipeline in background thread."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started"] = datetime.now().isoformat()

    try:
        results = batch_pipeline(
            input_dir=request.directory,
            output_base=request.output,
            mode=request.mode,
            platforms=request.platforms,
            material=request.material,
            music_path=request.music,
            preset=request.preset,
            fast=request.fast,
            camera_style=request.camera_style,
            color_grade=request.color_grade,
            variant_count=request.variants,
        )

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed"] = datetime.now().isoformat()
            _jobs[job_id]["result"] = {
                "total": len(results),
                "models": [r.get("model") for r in results if isinstance(r, dict)],
            }

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["completed"] = datetime.now().isoformat()
            _jobs[job_id]["error"] = str(e)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
def health():
    """Health check."""
    config = load_pipeline_config()
    blender_path = config.get("blender_path", "blender")
    blender_exists = Path(blender_path).exists() if blender_path != "blender" else False

    return {
        "status": "ok",
        "service": "forgefiles-pipeline",
        "version": "1.0.0",
        "blender_configured": blender_exists,
        "blender_path": blender_path,
        "active_jobs": sum(1 for j in _jobs.values() if j["status"] == "running"),
    }


@app.post("/pipeline/run")
def pipeline_run(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Queue a single STL pipeline run. Returns job ID immediately."""
    stl_path = Path(request.stl)
    if not stl_path.exists():
        raise HTTPException(404, f"STL file not found: {request.stl}")

    job_id = uuid.uuid4().hex[:12]
    model_name = stl_path.stem

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "model": model_name,
            "request": request.model_dump(),
            "started": None,
            "completed": None,
            "result": None,
            "error": None,
        }

    background_tasks.add_task(_run_pipeline_job, job_id, request)

    return {"job_id": job_id, "model": model_name, "status": "queued"}


@app.post("/pipeline/run/sync")
def pipeline_run_sync(request: PipelineRequest):
    """Run pipeline synchronously (blocks until complete). Use for n8n webhook callbacks."""
    stl_path = Path(request.stl)
    if not stl_path.exists():
        raise HTTPException(404, f"STL file not found: {request.stl}")

    try:
        result = run_full_pipeline(
            stl_path=request.stl,
            output_base=request.output,
            mode=request.mode,
            platforms=request.platforms,
            material=request.material,
            music_path=request.music,
            preset=request.preset,
            fast=request.fast,
            title=request.title,
            camera_style=request.camera_style,
            color_grade=request.color_grade,
            skip_existing=request.skip_existing,
            variant_count=request.variants,
        )
        if result:
            return {"status": "completed", "result": result}
        else:
            raise HTTPException(500, "Pipeline failed (no result returned)")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/pipeline/batch")
def pipeline_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """Queue a batch pipeline run on a directory of STLs."""
    dir_path = Path(request.directory)
    if not dir_path.is_dir():
        raise HTTPException(404, f"Directory not found: {request.directory}")

    stl_count = len(list(dir_path.glob("*.stl")) + list(dir_path.glob("*.STL")))
    if stl_count == 0:
        raise HTTPException(400, f"No STL files found in {request.directory}")

    job_id = uuid.uuid4().hex[:12]

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "model": f"batch ({stl_count} files)",
            "request": request.model_dump(),
            "started": None,
            "completed": None,
            "result": None,
            "error": None,
        }

    background_tasks.add_task(_run_batch_job, job_id, request)

    return {"job_id": job_id, "stl_count": stl_count, "status": "queued"}


@app.get("/pipeline/status/{job_id}")
def pipeline_status(job_id: str):
    """Get status of a pipeline job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    return job


@app.get("/pipeline/history")
def pipeline_history(limit: int = 20):
    """List recent pipeline runs."""
    with _jobs_lock:
        jobs = sorted(
            _jobs.values(),
            key=lambda j: j.get("started") or j.get("completed") or "",
            reverse=True,
        )[:limit]

    return {"jobs": jobs, "total": len(_jobs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
