"""MoneyClaw Daemon — runs the full autonomous system.

Starts:
1. Brain (decision engine)
2. Scheduler (heartbeat + cron jobs)
3. FastAPI server (HTTP endpoints + webhooks)
4. Telegram bot (customer channel)

This is the single entry point for production deployment.
"""

import asyncio
import logging
import signal
import sys
import threading

from moneyclaw.brain import Brain
from moneyclaw.memory import Memory
from moneyclaw.scheduler import Scheduler
from moneyclaw.agents import LunaGuardian
from moneyclaw.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("moneyclaw.daemon")


def start_api(config):
    """Start FastAPI in a thread."""
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=config.port,
        log_level="warning",
    )


def start_telegram(brain):
    """Telegram bot now starts via FastAPI lifespan — this is a no-op."""
    config = get_config()
    if not config.telegram.bot_token:
        logger.info("Telegram bot not configured (no TELEGRAM_BOT_TOKEN)")
    else:
        logger.info("Telegram bot will start via FastAPI lifespan")
    return None


def check_proactive_events(memory):
    """Check and log any due proactive events."""
    try:
        from moneyclaw.services.luna.proactive import ProactiveIntelligence
        proactive = ProactiveIntelligence(memory)
        due_events = proactive.get_due_events()
        for event in due_events:
            logger.info("Proactive event due: %s for user %s",
                        event.get("event_type"), event.get("user_id"))
            proactive.mark_sent(event["id"])
        if due_events:
            logger.info("Processed %d proactive events", len(due_events))
    except Exception as e:
        logger.debug("Proactive events check skipped: %s", e)


def main():
    config = get_config()
    logger.info("="*50)
    logger.info("  MONEYCLAW DAEMON STARTING")
    logger.info("="*50)
    logger.info("  Port: %d", config.port)
    logger.info("  Heartbeat: every %ds", config.heartbeat_interval_seconds)

    # Initialize core
    memory = Memory()
    brain = Brain(memory)
    scheduler = Scheduler(brain)

    logger.info("  State: %s", brain.state.value)
    logger.info("  Database: %s", memory.db_path)

    # Start scheduler
    scheduler.start()
    logger.info("  Scheduler: running (%d jobs)", len(scheduler._jobs))

    # Start Telegram bot
    tg_thread = start_telegram(brain)

    # Check proactive events on startup
    check_proactive_events(memory)

    # Start Luna Guardian agent in background thread
    guardian = LunaGuardian(
        base_url=f"http://localhost:{config.port}",
        vision_url="http://localhost:8002",
    )

    def run_guardian():
        import time
        time.sleep(15)  # Wait for API server to be ready
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(guardian.run())
        except Exception as e:
            logger.error("Guardian crashed: %s", e)
        finally:
            loop.close()

    guardian_thread = threading.Thread(target=run_guardian, daemon=True, name="luna-guardian")
    guardian_thread.start()
    logger.info("  Guardian: running (PULSE=5m, SCAN=30m, INTEL=6h, DAILY=24h)")

    # Start API (blocking)
    logger.info("  API: starting on port %d", config.port)
    logger.info("="*50)

    try:
        start_api(config)
    except KeyboardInterrupt:
        pass
    finally:
        guardian.stop()
        scheduler.stop()
        logger.info("MoneyClaw daemon stopped")


if __name__ == "__main__":
    main()
