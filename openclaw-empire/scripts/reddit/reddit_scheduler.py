"""Reddit automation orchestrator — daily plan generation + session executor.

Called by Task Scheduler every 45 minutes.
Each invocation: check if session is due, acquire phone lock, run session.

Usage:
  python scripts/reddit/reddit_scheduler.py              # Normal: run next due session
  python scripts/reddit/reddit_scheduler.py --generate    # Generate today's plan
  python scripts/reddit/reddit_scheduler.py --dry-run     # Simulate session (no phone)
  python scripts/reddit/reddit_scheduler.py --status      # Show plan + safety status
  python scripts/reddit/reddit_scheduler.py --maintenance # Cleanup old data
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import date, datetime
from pathlib import Path

# Ensure project root and scripts/ are on path so this works both as
# `python -m scripts.reddit.reddit_scheduler` AND as a direct script call
# from Task Scheduler: `python scripts/reddit/reddit_scheduler.py`
_project_root = str(Path(__file__).parent.parent.parent)
_scripts_dir = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

try:
    # When run as part of package (python -m scripts.reddit.reddit_scheduler)
    from .reddit_adb import PhoneLock
    from .reddit_engagement import run_session
    from .reddit_safety import SafetyEngine, PHASE_LIMITS
    from .reddit_state import RedditState
except ImportError:
    # When run directly (python scripts/reddit/reddit_scheduler.py)
    from reddit.reddit_adb import PhoneLock
    from reddit.reddit_engagement import run_session
    from reddit.reddit_safety import SafetyEngine, PHASE_LIMITS
    from reddit.reddit_state import RedditState

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "reddit"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PLAN_FILE = DATA_DIR / "daily_plan.json"
LOG_FILE = DATA_DIR / "scheduler.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger("reddit_scheduler")


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

def generate_daily_plan(state: RedditState, safety: SafetyEngine) -> dict:
    """Generate a daily session plan based on current phase and safety state."""
    phase = state.get_phase()
    limits = PHASE_LIMITS.get(phase, PHASE_LIMITS["lurk"])

    # Determine session types for this phase
    if phase == "lurk":
        session_pool = ["browse"] * 4
        max_sessions = 2
    elif phase == "comment":
        session_pool = ["browse"] * 3 + ["comment"] * 2
        max_sessions = 3
    elif phase == "active":
        session_pool = ["browse"] * 2 + ["comment"] * 3 + ["post"] * 1
        max_sessions = 3
    else:  # established
        session_pool = ["browse"] * 2 + ["comment"] * 3 + ["post"] * 2
        max_sessions = 4

    # Pick sessions for today
    num_sessions = random.randint(max(1, max_sessions - 1), max_sessions)
    sessions = []

    # Generate random times between now+1h and 10 PM (avoid scheduling in the past)
    now = datetime.now()
    min_hour = max(8, now.hour + 1)
    if min_hour > 21:
        # Too late to schedule anything meaningful
        logger.info("Too late in the day to generate new sessions")
        min_hour = 8  # Will generate for tomorrow's plan if re-run at 6:45 AM
    used_hours = set()
    for i in range(num_sessions):
        session_type = random.choice(session_pool)
        # Avoid consecutive sessions too close together
        hour = random.randint(min_hour, 22)
        while hour in used_hours or any(abs(hour - h) < 2 for h in used_hours):
            hour = random.randint(min_hour, 22)
            if len(used_hours) >= 7:
                break
        minute = random.randint(0, 45)
        used_hours.add(hour)

        sessions.append({
            "session_type": session_type,
            "hour": hour,
            "minute": minute,
            "scheduled_time": f"{hour:02d}:{minute:02d}",
            "completed": False,
            "completed_at": None,
            "result": None,
        })

    # Sort by time
    sessions.sort(key=lambda s: (s["hour"], s["minute"]))

    plan = {
        "date": date.today().isoformat(),
        "phase": phase,
        "account_age_days": state.get_account_age_days(),
        "generated_at": datetime.now().isoformat(),
        "sessions": sessions,
    }

    # Save plan
    PLAN_FILE.write_text(json.dumps(plan, indent=2))
    logger.info(f"Generated plan for {date.today()}: {len(sessions)} sessions ({phase} phase)")
    for s in sessions:
        logger.info(f"  {s['scheduled_time']} — {s['session_type']}")

    return plan


def load_plan() -> dict | None:
    """Load today's plan or return None if stale/missing."""
    if not PLAN_FILE.exists():
        return None
    try:
        plan = json.loads(PLAN_FILE.read_text())
        if plan.get("date") != date.today().isoformat():
            return None
        return plan
    except (json.JSONDecodeError, Exception):
        return None


# ---------------------------------------------------------------------------
# Session executor
# ---------------------------------------------------------------------------

def run_next_session(dry_run: bool = False):
    """Find and execute the next due session from today's plan."""
    state = RedditState()
    safety = SafetyEngine(state)

    # Pre-flight
    if safety.is_banned():
        logger.error("Account is banned. Exiting.")
        state.close()
        return
    if safety.is_rate_limited():
        logger.warning("Rate limited. Skipping this cycle.")
        state.close()
        return

    # Load or generate plan
    plan = load_plan()
    if not plan:
        logger.info("No plan for today, generating...")
        plan = generate_daily_plan(state, safety)

    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute

    # Find next due session (within +-30 min window)
    for session in plan["sessions"]:
        if session.get("completed"):
            continue

        scheduled = session["hour"] * 60 + session["minute"]
        diff = current_minutes - scheduled

        # Session is due if current time is 0-30 min past scheduled
        if 0 <= diff <= 30:
            session_type = session["session_type"]
            logger.info(
                f"Session due: {session_type} at {session['scheduled_time']} "
                f"(current: {now.strftime('%H:%M')})"
            )

            # Acquire phone lock
            if not dry_run:
                lock = PhoneLock()
                if not lock.acquire("reddit"):
                    logger.info("Phone locked by another automation, skipping")
                    state.close()
                    return
            else:
                lock = None

            try:
                # Small jitter before starting
                jitter = random.randint(0, 3)
                if jitter > 0 and not dry_run:
                    logger.info(f"Jitter: {jitter} min")
                    time.sleep(jitter * 60)

                result = run_session(session_type, dry_run=dry_run)

                session["completed"] = True
                session["completed_at"] = datetime.now().isoformat()
                session["result"] = result
                PLAN_FILE.write_text(json.dumps(plan, indent=2))

                logger.info(f"Session complete: {result}")
            finally:
                if lock:
                    lock.release()

            state.close()
            return  # Only ONE session per invocation

    # No sessions due
    next_session = None
    for session in plan["sessions"]:
        if not session.get("completed"):
            s_min = session["hour"] * 60 + session["minute"]
            if s_min > current_minutes:
                next_session = session
                break

    if next_session:
        logger.info(f"Next session: {next_session['session_type']} at {next_session['scheduled_time']}")
    else:
        completed = sum(1 for s in plan["sessions"] if s.get("completed"))
        logger.info(f"All sessions done for today ({completed}/{len(plan['sessions'])})")

    state.close()


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status():
    """Display full status: plan, safety, and state."""
    state = RedditState()
    safety = SafetyEngine(state)
    status = safety.get_status()

    print(f"\n{'='*60}")
    print(f"Reddit Automation Status — StillLabelingCables")
    print(f"{'='*60}\n")

    print(f"Phase: {status['phase']} (day {status['account_age_days']})")
    print(f"Karma estimate: {status['karma_estimate']}")
    print(f"Promo ratio: {status['promo_ratio']}")
    print(f"Allowed actions: {', '.join(status['allowed_actions'])}")
    print(f"Banned: {status['banned']} | Rate limited: {status['rate_limited']}")
    if status['blacklisted_subs']:
        print(f"Blacklisted subs: {', '.join(status['blacklisted_subs'])}")
    print()

    # Daily counts
    counts = status['daily_counts']
    if counts:
        print("Today's counts:")
        for k, v in sorted(counts.items()):
            print(f"  {k}: {v}")
        print()

    # Today's plan
    plan = load_plan()
    if plan:
        print(f"Today's Plan ({plan['date']}):")
        for s in plan["sessions"]:
            status_str = "DONE" if s.get("completed") else "PENDING"
            at = f" @ {s['completed_at'][11:16]}" if s.get("completed_at") else ""
            print(f"  {s['scheduled_time']} {s['session_type']:>8} [{status_str}]{at}")
        print()

    # Recent sessions
    recent = state.get_recent_sessions(5)
    if recent:
        print("Recent sessions:")
        for s in recent:
            ts = s["timestamp"][5:16]
            actions = json.loads(s.get("actions_json", "{}"))
            subs = json.loads(s.get("subreddits_visited", "[]"))
            print(f"  {ts} {s['session_type']:>8} — {s['duration_seconds']:.0f}s, "
                  f"votes={actions.get('upvotes', 0)}, "
                  f"comments={actions.get('comments', 0)}, "
                  f"subs={','.join(subs[:3])}")
        print()

    state.close()


def run_maintenance():
    """Cleanup old data and optimize database."""
    state = RedditState()
    state.cleanup_old_data(retention_days=90)
    state.close()
    logger.info("Maintenance complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reddit ForgeFiles Automation Scheduler")
    parser.add_argument("--generate", action="store_true",
                        help="Generate today's session plan")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run session without phone interaction")
    parser.add_argument("--status", action="store_true",
                        help="Show automation status")
    parser.add_argument("--maintenance", action="store_true",
                        help="Run database maintenance")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.maintenance:
        run_maintenance()
        return

    if args.generate:
        state = RedditState()
        safety = SafetyEngine(state)
        generate_daily_plan(state, safety)
        state.close()
        return

    run_next_session(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
