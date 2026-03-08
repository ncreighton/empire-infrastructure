"""OpenClaw FastAPI server — port 8100.

Pattern: videoforge-engine/api/app.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import fields
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openclaw.openclaw_engine import OpenClawEngine
from openclaw.knowledge.platforms import get_all_platform_ids, get_platform, get_platforms_by_category
from openclaw.models import PlatformCategory
from openclaw.automation.analytics import Analytics
from openclaw.automation.scheduler import Scheduler

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenClaw Agent API",
    description="Autonomous web agent for platform profile creation and management",
    version="2.3.0",
)

CORS_ORIGINS = os.environ.get("OPENCLAW_CORS_ORIGINS", "").split(",")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = OpenClawEngine()
analytics = Analytics(codex=engine.codex)
scheduler = Scheduler()

# WebSocket connections for live monitoring
ws_connections: list[WebSocket] = []


# ─── Serialization ────────────────────────────────────────────────────────────


def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses and enums to JSON-serializable dicts."""
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(item) for item in obj]
    return obj


# ─── Request Models ───────────────────────────────────────────────────────────


class SignupRequest(BaseModel):
    platform_id: str
    password: str = ""
    email: str = ""


class BatchSignupRequest(BaseModel):
    platform_ids: list[str]
    password: str = ""
    email: str = ""
    delay_seconds: int = 30


class ProfileGenerateRequest(BaseModel):
    platform_id: str


class CaptchaSolutionRequest(BaseModel):
    task_id: str
    solution: str


class ScheduleBatchRequest(BaseModel):
    platform_ids: list[str]
    password: str = ""
    email: str = ""
    delay_seconds: int = 60
    start_after_minutes: int = 0


class SignupRetryRequest(BaseModel):
    platform_id: str
    password: str = ""
    email: str = ""
    max_retries: int = 3


class SyncRequest(BaseModel):
    changes: dict[str, str]  # field_name -> new_value
    platform_ids: list[str] | None = None
    browser: bool = False


class SyncPreviewRequest(BaseModel):
    changes: dict[str, str]
    platform_ids: list[str] | None = None


class ExportRequest(BaseModel):
    format: str = "json"  # json or csv


# ─── Signup Endpoints ─────────────────────────────────────────────────────────


@app.post("/signup")
async def signup(req: SignupRequest):
    """Full pipeline signup for a single platform."""
    if not get_platform(req.platform_id):
        raise HTTPException(status_code=404, detail=f"Unknown platform: {req.platform_id}")

    credentials = {}
    if req.password:
        credentials["password"] = req.password
    if req.email:
        credentials["email"] = req.email

    try:
        await broadcast_ws({"type": "signup_started", "platform_id": req.platform_id})
        result = await engine.signup_async(req.platform_id, credentials or None)
        await broadcast_ws({
            "type": "signup_completed",
            "platform_id": req.platform_id,
            "success": result.success,
        })
        return _to_dict(result)
    except Exception as e:
        await broadcast_ws({
            "type": "signup_failed",
            "platform_id": req.platform_id,
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/signup/batch")
async def signup_batch(req: BatchSignupRequest):
    """Sequential signup across multiple platforms."""
    invalid = [pid for pid in req.platform_ids if not get_platform(pid)]
    if invalid:
        raise HTTPException(status_code=404, detail=f"Unknown platforms: {invalid}")

    credentials = {}
    if req.password:
        credentials["password"] = req.password
    if req.email:
        credentials["email"] = req.email

    try:
        await broadcast_ws({
            "type": "batch_started",
            "platforms": req.platform_ids,
        })
        results = await engine.signup_batch(
            req.platform_ids,
            credentials or None,
            delay_seconds=req.delay_seconds,
        )
        await broadcast_ws({
            "type": "batch_completed",
            "total": len(results),
            "succeeded": sum(1 for r in results if r.success),
        })
        return [_to_dict(r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/signup/retry")
async def signup_with_retry(req: SignupRetryRequest):
    """Signup with automatic retry on transient failures."""
    if not get_platform(req.platform_id):
        raise HTTPException(status_code=404, detail=f"Unknown platform: {req.platform_id}")

    credentials = {}
    if req.password:
        credentials["password"] = req.password
    if req.email:
        credentials["email"] = req.email

    try:
        result = await engine.signup_with_retry(
            req.platform_id, credentials or None, max_retries=req.max_retries
        )
        return _to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ─── Platform Endpoints ──────────────────────────────────────────────────────


@app.get("/platforms")
async def list_platforms():
    """List all platforms with their status."""
    platforms = []
    for pid in get_all_platform_ids():
        p = get_platform(pid)
        account = engine.codex.get_account(pid)
        platforms.append({
            "platform_id": pid,
            "name": p.name,
            "category": p.category.value,
            "complexity": p.complexity.value,
            "status": account.get("status", "not_started") if account else "not_started",
            "monetization_potential": p.monetization_potential,
            "audience_size": p.audience_size,
        })
    return platforms


@app.get("/platform/{platform_id}")
async def get_platform_detail(platform_id: str):
    """Get detailed platform info + account status."""
    if not get_platform(platform_id):
        raise HTTPException(status_code=404, detail=f"Unknown platform: {platform_id}")
    return engine.get_platform_status(platform_id)


# ─── Profile Endpoints ───────────────────────────────────────────────────────


@app.post("/profile/generate")
async def generate_profile(req: ProfileGenerateRequest):
    """Generate profile content without executing signup (dry-run)."""
    if not get_platform(req.platform_id):
        raise HTTPException(status_code=404, detail=f"Unknown platform: {req.platform_id}")

    try:
        content = engine.generate_profile(req.platform_id)
        return _to_dict(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/profile/score/{platform_id}")
async def score_profile(platform_id: str):
    """Score an existing profile from the Codex."""
    score = engine.score_profile(platform_id)
    if not score:
        raise HTTPException(status_code=404, detail=f"No profile found for: {platform_id}")
    return _to_dict(score)


# ─── CAPTCHA Endpoints ───────────────────────────────────────────────────────


@app.post("/captcha/solve")
async def submit_captcha_solution(req: CaptchaSolutionRequest):
    """Submit a manual CAPTCHA solution."""
    success = engine.captcha.submit_solution(req.task_id, req.solution)
    if not success:
        raise HTTPException(status_code=404, detail=f"Unknown CAPTCHA task: {req.task_id}")
    return {"success": True, "task_id": req.task_id}


@app.get("/captcha/pending")
async def get_pending_captchas():
    """Get all CAPTCHAs waiting for human solving."""
    return engine.captcha.get_pending_tasks()


# ─── Analysis Endpoints ──────────────────────────────────────────────────────


@app.get("/analyze/{platform_id}")
async def analyze_platform(platform_id: str):
    """Analyze a platform for signup readiness (Scout)."""
    if not get_platform(platform_id):
        raise HTTPException(status_code=404, detail=f"Unknown platform: {platform_id}")
    try:
        result = engine.analyze_platform(platform_id)
        return _to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/prioritize")
async def prioritize_platforms():
    """Get prioritized list of platforms to sign up for (Oracle)."""
    try:
        recs = engine.prioritize()
        return [_to_dict(r) for r in recs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ─── Dashboard ────────────────────────────────────────────────────────────────


@app.get("/dashboard")
async def get_dashboard():
    """Get aggregate dashboard statistics."""
    try:
        stats = engine.get_dashboard()
        return _to_dict(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ─── WebSocket ────────────────────────────────────────────────────────────────


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Real-time browser session monitoring via WebSocket with heartbeat."""
    await websocket.accept()
    ws_connections.append(websocket)
    try:
        while True:
            try:
                # Wait for data with a timeout to send periodic pings
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=30.0
                )
                await websocket.send_text(json.dumps({"type": "ack", "data": data}))
            except asyncio.TimeoutError:
                # Send heartbeat ping
                try:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "ts": datetime.now().isoformat()})
                    )
                except Exception as ping_err:
                    logger.debug(f"WebSocket heartbeat send failed: {ping_err}")
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ws_connections:
            ws_connections.remove(websocket)


async def broadcast_ws(message: dict) -> None:
    """Broadcast a message to all connected WebSocket clients."""
    text = json.dumps(message, default=str)
    disconnected = []
    for ws in ws_connections:
        try:
            await ws.send_text(text)
        except Exception as e:
            logger.debug(f"WebSocket broadcast failed for a client: {e}")
            disconnected.append(ws)
    for ws in disconnected:
        if ws in ws_connections:
            ws_connections.remove(ws)


# ─── Health ───────────────────────────────────────────────────────────────────


# ─── Scheduler Endpoints ──────────────────────────────────────────────────────


@app.post("/schedule/batch")
async def schedule_batch(req: ScheduleBatchRequest):
    """Schedule a batch signup job for later execution."""
    invalid = [pid for pid in req.platform_ids if not get_platform(pid)]
    if invalid:
        raise HTTPException(status_code=404, detail=f"Unknown platforms: {invalid}")

    from datetime import timedelta as td
    start_after = None
    if req.start_after_minutes > 0:
        start_after = datetime.now() + td(minutes=req.start_after_minutes)

    credentials = {}
    if req.password:
        credentials["password"] = req.password
    if req.email:
        credentials["email"] = req.email

    job_id = scheduler.schedule_batch(
        platform_ids=req.platform_ids,
        credentials=credentials or None,
        delay_between_seconds=req.delay_seconds,
        start_after=start_after,
    )

    async def _signup_func(pid, creds):
        result = await engine.signup_async(pid, creds)
        return _to_dict(result)

    scheduler.start_job_background(job_id, _signup_func)
    return {"job_id": job_id, "platforms": len(req.platform_ids), "status": "scheduled"}


@app.get("/schedule/jobs")
async def list_jobs():
    """List all scheduled jobs."""
    jobs = scheduler.get_all_jobs()
    return [
        {
            "job_id": j.job_id,
            "status": j.status.value,
            "platforms": len(j.platform_ids),
            "completed": j.completed_count,
            "failed": j.failed_count,
            "current": j.current_platform,
            "scheduled_at": j.scheduled_at.isoformat() if j.scheduled_at else None,
        }
        for j in jobs
    ]


@app.get("/schedule/job/{job_id}")
async def get_job(job_id: str):
    """Get details of a scheduled job."""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "platform_ids": job.platform_ids,
        "completed_count": job.completed_count,
        "failed_count": job.failed_count,
        "current_platform": job.current_platform,
        "results": job.results,
        "scheduled_at": job.scheduled_at.isoformat() if job.scheduled_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@app.post("/schedule/job/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a running job."""
    if scheduler.pause_job(job_id):
        return {"success": True, "status": "paused"}
    raise HTTPException(status_code=400, detail="Job cannot be paused")


@app.post("/schedule/job/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused job."""
    if scheduler.resume_job(job_id):
        return {"success": True, "status": "running"}
    raise HTTPException(status_code=400, detail="Job cannot be resumed")


@app.post("/schedule/job/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a job."""
    if scheduler.cancel_job(job_id):
        return {"success": True, "status": "cancelled"}
    raise HTTPException(status_code=400, detail="Job cannot be cancelled")


# ─── Profile Sync Endpoints ─────────────────────────────────────────────────


@app.post("/sync")
async def sync_profiles(req: SyncRequest):
    """Sync profile content across platforms."""
    try:
        result = await engine.sync(
            changes=req.changes,
            platform_ids=req.platform_ids,
            browser=req.browser,
        )
        await broadcast_ws({"type": "sync_completed", "data": result})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/sync/preview")
async def sync_preview(req: SyncPreviewRequest):
    """Preview what a sync would change without executing."""
    try:
        return engine.sync_preview(
            changes=req.changes,
            platform_ids=req.platform_ids,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/sync/status")
async def sync_status():
    """Get profile consistency status across active platforms."""
    try:
        return engine.get_sync_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ─── Analytics Endpoints ─────────────────────────────────────────────────────


@app.get("/analytics/report")
async def get_analytics_report():
    """Get comprehensive analytics report."""
    try:
        report = analytics.generate_report()
        return _to_dict(report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/analytics/coverage")
async def get_coverage_map():
    """Get platform coverage map (all platforms with status)."""
    return analytics.get_coverage_map()


@app.get("/analytics/timeline")
async def get_timeline(days: int = 30):
    """Get signup activity timeline."""
    return analytics.get_timeline(days=days)


@app.post("/export")
async def export_data(req: ExportRequest):
    """Export all account data as JSON or CSV."""
    try:
        if req.format == "csv":
            from fastapi.responses import PlainTextResponse
            csv_data = analytics.export_csv()
            return PlainTextResponse(content=csv_data, media_type="text/csv")
        else:
            json_data = analytics.export_json()
            return json.loads(json_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ─── Platform Discovery Endpoints ────────────────────────────────────────────


@app.get("/platforms/category/{category}")
async def platforms_by_category(category: str):
    """Get platforms filtered by category."""
    try:
        cat = PlatformCategory(category)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Valid: {[c.value for c in PlatformCategory]}",
        ) from e
    platforms = get_platforms_by_category(cat)
    return [
        {
            "platform_id": p.platform_id,
            "name": p.name,
            "complexity": p.complexity.value,
            "monetization_potential": p.monetization_potential,
            "audience_size": p.audience_size,
            "seo_value": p.seo_value,
        }
        for p in platforms
    ]


@app.get("/platforms/easy-wins")
async def easy_wins():
    """Get platforms with best value-to-effort ratio."""
    from openclaw.knowledge.platforms import get_easy_wins
    wins = get_easy_wins()
    return [
        {
            "platform_id": p.platform_id,
            "name": p.name,
            "category": p.category.value,
            "complexity": p.complexity.value,
            "value_score": p.monetization_potential + p.audience_size + p.seo_value,
        }
        for p in wins[:15]
    ]


# ─── Email Verification Endpoints ────────────────────────────────────────────


@app.get("/email/stats")
async def email_verifier_stats():
    """Get email verification statistics."""
    return engine.email_verifier.get_stats()


@app.get("/email/verified")
async def email_verified_platforms():
    """Get list of platforms with verified emails."""
    return engine.email_verifier.get_verified_platforms()


# ─── Proxy Endpoints ────────────────────────────────────────────────────────


@app.get("/proxies/stats")
async def proxy_stats():
    """Get proxy pool statistics."""
    return engine.proxy_manager.get_stats()


# ─── Retry Engine Endpoints ─────────────────────────────────────────────────


@app.get("/retry/stats")
async def retry_stats():
    """Get retry engine statistics."""
    return engine.retry_engine.get_stats()


# ─── Rate Limiter Endpoints ─────────────────────────────────────────────────


@app.get("/ratelimit/stats")
async def rate_limit_stats():
    """Get rate limiter statistics."""
    return engine.rate_limiter.get_stats()


@app.get("/ratelimit/check/{platform_id}")
async def check_rate_limit(platform_id: str):
    """Check if a platform can be signed up now."""
    can_proceed, reason = engine.rate_limiter.can_proceed(platform_id)
    return {
        "platform_id": platform_id,
        "can_proceed": can_proceed,
        "reason": reason,
        "wait_seconds": engine.rate_limiter.wait_time(platform_id) if not can_proceed else 0,
    }


# ─── Health ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    active_jobs = len(scheduler.get_active_jobs())
    return {
        "status": "healthy",
        "service": "openclaw-agent",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "platforms_registered": len(get_all_platform_ids()),
        "active_jobs": active_jobs,
        "proxies_available": engine.proxy_manager.available_count,
        "email_verifier_configured": engine.email_verifier.is_configured,
        "scheduler_stats": scheduler.get_stats(),
        "rate_limiter": engine.rate_limiter.get_stats(),
        "daemon_running": _daemon is not None and _daemon._running,
    }


# ─── Daemon Endpoints ───────────────────────────────────────────────────────

_daemon = None  # HeartbeatDaemon singleton
_daemon_task = None


@app.post("/daemon/start")
async def daemon_start():
    """Start the heartbeat daemon."""
    global _daemon, _daemon_task
    from openclaw.daemon.heartbeat_daemon import HeartbeatDaemon

    if _daemon and _daemon._running:
        return {"status": "already_running", "uptime": _daemon.get_status().get("uptime_seconds", 0)}

    _daemon = HeartbeatDaemon(engine)
    _daemon_task = asyncio.create_task(_daemon.start())
    return {"status": "starting"}


@app.post("/daemon/stop")
async def daemon_stop():
    """Stop the heartbeat daemon."""
    global _daemon, _daemon_task
    if not _daemon or not _daemon._running:
        return {"status": "not_running"}

    await _daemon.stop()
    if _daemon_task:
        _daemon_task.cancel()
    _daemon = None
    _daemon_task = None
    return {"status": "stopped"}


@app.get("/daemon/status")
async def daemon_status():
    """Get daemon running state, uptime, tier stats."""
    from openclaw.daemon.heartbeat_daemon import HeartbeatDaemon
    if _daemon:
        return _daemon.get_status()
    return {
        "running": False,
        "started_at": None,
        "uptime_seconds": 0,
        "tier_runs": {},
        "cron_jobs": 0,
        "pending_actions": 0,
    }


# ─── Cron Endpoints ─────────────────────────────────────────────────────────


class CronJobRequest(BaseModel):
    name: str
    schedule: str
    action: str
    params: dict = {}


@app.get("/cron/jobs")
async def list_cron_jobs():
    """List all cron jobs."""
    from openclaw.daemon.cron_scheduler import CronScheduler
    cron = CronScheduler(engine.codex)
    jobs = cron.get_all()
    return [
        {
            "job_id": j.job_id,
            "name": j.name,
            "schedule": j.schedule,
            "action": j.action,
            "status": j.status.value,
            "run_count": j.run_count,
            "fail_count": j.fail_count,
            "last_run": j.last_run.isoformat() if j.last_run else None,
            "next_run": j.next_run.isoformat() if j.next_run else None,
        }
        for j in jobs
    ]


@app.post("/cron/jobs")
async def create_cron_job(req: CronJobRequest):
    """Create a new cron job."""
    from openclaw.daemon.cron_scheduler import CronScheduler
    cron = CronScheduler(engine.codex)
    job_id = cron.register(req.name, req.schedule, req.action, req.params)
    return {"job_id": job_id, "name": req.name, "status": "active"}


@app.get("/cron/job/{job_id}")
async def get_cron_job(job_id: str):
    """Get cron job details + history."""
    from openclaw.daemon.cron_scheduler import CronScheduler
    cron = CronScheduler(engine.codex)
    job = engine.codex.get_cron_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Cron job not found: {job_id}")
    history = cron.get_history(job_id, limit=10)
    return {
        "job_id": job.job_id,
        "name": job.name,
        "schedule": job.schedule,
        "action": job.action,
        "status": job.status.value,
        "run_count": job.run_count,
        "fail_count": job.fail_count,
        "last_run": job.last_run.isoformat() if job.last_run else None,
        "next_run": job.next_run.isoformat() if job.next_run else None,
        "history": history,
    }


@app.post("/cron/job/{job_id}/pause")
async def pause_cron_job(job_id: str):
    """Pause a cron job."""
    from openclaw.daemon.cron_scheduler import CronScheduler
    cron = CronScheduler(engine.codex)
    if cron.pause(job_id):
        return {"success": True, "status": "paused"}
    raise HTTPException(status_code=404, detail=f"Cron job not found: {job_id}")


@app.post("/cron/job/{job_id}/resume")
async def resume_cron_job(job_id: str):
    """Resume a paused cron job."""
    from openclaw.daemon.cron_scheduler import CronScheduler
    cron = CronScheduler(engine.codex)
    if cron.resume(job_id):
        return {"success": True, "status": "active"}
    raise HTTPException(status_code=404, detail=f"Cron job not found: {job_id}")


@app.delete("/cron/job/{job_id}")
async def disable_cron_job(job_id: str):
    """Disable a cron job."""
    from openclaw.daemon.cron_scheduler import CronScheduler
    cron = CronScheduler(engine.codex)
    if cron.disable(job_id):
        return {"success": True, "status": "disabled"}
    raise HTTPException(status_code=404, detail=f"Cron job not found: {job_id}")


# ─── Alert Endpoints ────────────────────────────────────────────────────────


@app.get("/alerts")
async def list_alerts(severity: str = None, source: str = None, limit: int = 50):
    """Get recent alerts with optional filtering."""
    from openclaw.models import AlertSeverity
    sev = AlertSeverity(severity) if severity else None
    return engine.codex.get_alerts(severity=sev, source=source, limit=limit)


@app.get("/alerts/stats")
async def alert_stats():
    """Get alert statistics."""
    return engine.codex.get_alert_stats()


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    if engine.codex.acknowledge_alert(alert_id):
        return {"success": True, "alert_id": alert_id}
    raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")


# ─── Empire Health Endpoints ────────────────────────────────────────────────


@app.get("/health/empire")
async def empire_health():
    """Full empire health (last check results per tier)."""
    return engine.codex.get_latest_checks()


@app.get("/health/history")
async def health_history(tier: str = None, limit: int = 50):
    """Health check history."""
    from openclaw.models import HeartbeatTier
    t = HeartbeatTier(tier) if tier else None
    return engine.codex.get_recent_checks(tier=t, limit=limit)


# ─── Daemon auto-start ──────────────────────────────────────────────────────


@app.router.on_startup.append
async def startup_event():
    """Auto-start daemon if OPENCLAW_DAEMON_MODE is set."""
    global _daemon, _daemon_task
    if os.environ.get("OPENCLAW_DAEMON_MODE", "").lower() in ("true", "1", "yes"):
        from openclaw.daemon.heartbeat_daemon import HeartbeatDaemon
        _daemon = HeartbeatDaemon(engine)
        _daemon_task = asyncio.create_task(_daemon.start())
        logger.info("Daemon auto-started via OPENCLAW_DAEMON_MODE")
