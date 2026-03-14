"""MissionDaemon — background queue processor for VibeCoder missions.

Polls the SQLite queue for pending missions and executes them through
the VibeCoderEngine pipeline. Runs as a coroutine alongside the
HeartbeatDaemon in asyncio.gather().

Pattern: openclaw/daemon/heartbeat_daemon.py (tier loop)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

from openclaw.vibecoder.models import MissionStatus

logger = logging.getLogger(__name__)

# Default poll interval in seconds
_DEFAULT_POLL_INTERVAL = 30
_MAX_CONCURRENT = 2


class MissionDaemon:
    """Background processor that picks up queued missions.

    Usage::

        daemon = MissionDaemon(vibecoder_engine)
        await daemon.start()  # runs until daemon.stop()

    Or integrated into HeartbeatDaemon::

        await asyncio.gather(
            heartbeat_daemon.start(),
            mission_daemon.start(),
        )
    """

    def __init__(
        self,
        engine: Any,  # VibeCoderEngine (lazy import avoids circular)
        poll_interval: int = _DEFAULT_POLL_INTERVAL,
        max_concurrent: int = _MAX_CONCURRENT,
    ):
        self.engine = engine
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self._running = False
        self._active_missions: set[str] = set()
        self._started_at: datetime | None = None
        self._missions_processed = 0
        self._missions_failed = 0

    async def start(self):
        """Start the daemon loop."""
        if self._running:
            logger.warning("MissionDaemon already running")
            return

        self._running = True
        self._started_at = datetime.now()
        logger.info(
            f"MissionDaemon starting — poll_interval={self.poll_interval}s, "
            f"max_concurrent={self.max_concurrent}"
        )

        while self._running:
            start = time.monotonic()
            try:
                await self._poll_and_execute()
            except Exception as e:
                logger.error(f"[MissionDaemon] Poll error: {e}", exc_info=True)

            elapsed = time.monotonic() - start
            sleep_time = max(0, self.poll_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        # Wait for active missions to complete (with timeout)
        if self._active_missions:
            logger.info(
                f"[MissionDaemon] Waiting for {len(self._active_missions)} "
                f"active mission(s) to complete..."
            )
            for _ in range(30):  # 30 second timeout
                if not self._active_missions:
                    break
                await asyncio.sleep(1)

        logger.info("MissionDaemon stopped")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        uptime = 0.0
        if self._started_at:
            uptime = (datetime.now() - self._started_at).total_seconds()
        return {
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "uptime_seconds": round(uptime, 0),
            "active_missions": list(self._active_missions),
            "missions_processed": self._missions_processed,
            "missions_failed": self._missions_failed,
            "poll_interval": self.poll_interval,
            "max_concurrent": self.max_concurrent,
        }

    async def _poll_and_execute(self):
        """Check for queued missions and execute them."""
        # Don't exceed concurrency limit
        available_slots = self.max_concurrent - len(self._active_missions)
        if available_slots <= 0:
            return

        queued = self.engine.codex.get_queued_missions(limit=available_slots)
        if not queued:
            return

        logger.info(
            f"[MissionDaemon] Found {len(queued)} queued mission(s), "
            f"executing up to {available_slots}"
        )

        for mission_row in queued:
            mission_id = mission_row["mission_id"]
            if mission_id in self._active_missions:
                continue

            self._active_missions.add(mission_id)
            # Fire and forget — execution happens in background
            asyncio.create_task(self._execute_mission(mission_id))

    async def _execute_mission(self, mission_id: str):
        """Execute a single mission with error handling."""
        try:
            logger.info(f"[MissionDaemon] Executing mission: {mission_id}")
            mission = await self.engine.execute_mission(mission_id)

            if mission.status == MissionStatus.COMPLETED:
                self._missions_processed += 1
                logger.info(
                    f"[MissionDaemon] Mission {mission_id} completed "
                    f"(cost=${mission.total_cost_usd:.4f}, "
                    f"duration={mission.duration_seconds:.1f}s)"
                )
            else:
                self._missions_failed += 1
                logger.warning(
                    f"[MissionDaemon] Mission {mission_id} ended with "
                    f"status={mission.status.value}: {mission.errors}"
                )

        except Exception as e:
            self._missions_failed += 1
            logger.error(
                f"[MissionDaemon] Mission {mission_id} crashed: {e}",
                exc_info=True,
            )
            # Update status in DB
            try:
                self.engine.codex.update_mission_status(
                    mission_id, MissionStatus.FAILED,
                    errors=[f"Daemon execution crash: {str(e)[:200]}"],
                )
            except Exception:
                pass

        finally:
            self._active_missions.discard(mission_id)
