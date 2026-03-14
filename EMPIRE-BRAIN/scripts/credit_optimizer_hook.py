"""
Credit Optimizer Hook — Enhanced session tracking for Claude Max credit conservation.

Replaces the basic claude_cost_monitor.py with smarter tracking:
- Detects active model from environment/context
- Tracks session-to-session credit accumulation
- Generates warnings when approaching credit limits
- Produces daily optimization reports

Usage (in settings.json hooks):
    pythonw "D:\\Claude Code Projects\\EMPIRE-BRAIN\\scripts\\credit_optimizer_hook.py" --session-start
    pythonw "D:\\Claude Code Projects\\EMPIRE-BRAIN\\scripts\\credit_optimizer_hook.py" --session-end
    pythonw "D:\\Claude Code Projects\\EMPIRE-BRAIN\\scripts\\credit_optimizer_hook.py" --report
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOGS_DIR = Path(r"D:\Claude Code Projects\EMPIRE-BRAIN\logs")
COST_LOG = LOGS_DIR / "claude_code_costs.json"
CREDIT_REPORT = LOGS_DIR / "credit_optimization_report.json"
SESSION_STATE = LOGS_DIR / "current_session.json"

# Credit rates per message by model tier
CREDIT_RATES = {
    "opus": 1.0,
    "sonnet": 0.2,
    "haiku": 0.067,
}

# 5-hour rolling window limits (Claude Max plan)
WINDOW_LIMITS = {
    "opus": 45,
    "sonnet": 225,
    "haiku": 675,
}

# Daily approximate budget (based on 24hr / 5hr windows = ~4.8 windows)
DAILY_CREDIT_BUDGET = 216  # ~45 opus messages × 4.8 windows


def load_cost_log() -> dict:
    """Load persistent cost tracking."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if COST_LOG.exists():
        try:
            return json.loads(COST_LOG.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "sessions": [],
        "daily_totals": {},
        "lifetime": {
            "total_sessions": 0,
            "total_messages_est": 0,
            "total_credits_est": 0,
            "total_savings_est": 0,
        },
    }


def save_cost_log(data: dict):
    """Save cost tracking."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    COST_LOG.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def detect_model() -> str:
    """Detect which model the current session is likely using.

    Checks environment variables and process info for clues.
    Falls back to 'opus' (worst case for cost estimation).
    """
    # Check if CLAUDE_MODEL env var is set
    model_env = os.environ.get("CLAUDE_MODEL", "").lower()
    if "haiku" in model_env:
        return "haiku"
    if "sonnet" in model_env:
        return "sonnet"
    if "opus" in model_env:
        return "opus"

    # Check session state file (set by Claude Code if we can detect it)
    if SESSION_STATE.exists():
        try:
            state = json.loads(SESSION_STATE.read_text())
            if "model" in state:
                return state["model"]
        except Exception:
            pass

    # Default assumption: opus (conservative — overestimates cost)
    return "opus"


def estimate_messages(duration_min: float, model: str) -> int:
    """Estimate number of messages based on session duration and model.

    Different models have different response speeds:
    - Haiku: ~3 messages/min (fast responses)
    - Sonnet: ~1.5 messages/min (moderate)
    - Opus: ~0.8 messages/min (slow, long responses)
    """
    rates = {
        "haiku": 3.0,
        "sonnet": 1.5,
        "opus": 0.8,
    }
    rate = rates.get(model, 1.0)
    return max(1, int(duration_min * rate))


def session_start():
    """Record session start."""
    data = load_cost_log()
    model = detect_model()
    now = datetime.now(timezone.utc)

    session = {
        "id": f"session_{int(time.time())}",
        "started": now.isoformat(),
        "ended": None,
        "duration_min": 0,
        "model": model,
        "estimated_messages": 0,
        "estimated_credits": 0,
        "savings_vs_opus": 0,
    }
    data["sessions"].append(session)

    # Save session state for other tools to read
    SESSION_STATE.write_text(json.dumps({
        "session_id": session["id"],
        "started": session["started"],
        "model": model,
    }, indent=2))

    save_cost_log(data)

    # Print advisory to stderr (visible in hook output)
    today = now.strftime("%Y-%m-%d")
    today_data = data.get("daily_totals", {}).get(today, {})
    sessions_today = today_data.get("sessions", 0)
    credits_today = today_data.get("credits_est", 0)

    if credits_today > DAILY_CREDIT_BUDGET * 0.7:
        print(
            f"[CREDIT WARNING] {credits_today:.0f}/{DAILY_CREDIT_BUDGET} credits used today "
            f"({credits_today/DAILY_CREDIT_BUDGET*100:.0f}%). Consider using /model sonnet.",
            file=sys.stderr,
        )


def session_end():
    """Record session end and calculate costs."""
    data = load_cost_log()
    now = datetime.now(timezone.utc)

    if not data["sessions"]:
        return

    session = data["sessions"][-1]
    if session.get("ended") is not None:
        return  # Already ended

    session["ended"] = now.isoformat()

    # Calculate duration
    try:
        started = datetime.fromisoformat(session["started"].replace("Z", "+00:00"))
        duration = (now - started).total_seconds() / 60
        session["duration_min"] = round(duration, 1)
    except Exception:
        duration = 5.0
        session["duration_min"] = 5.0

    model = session.get("model", detect_model())
    est_messages = estimate_messages(duration, model)
    credit_rate = CREDIT_RATES.get(model, 1.0)
    est_credits = est_messages * credit_rate

    # Calculate savings vs all-opus
    opus_credits = est_messages * CREDIT_RATES["opus"]
    savings = opus_credits - est_credits

    session["estimated_messages"] = est_messages
    session["estimated_credits"] = round(est_credits, 2)
    session["savings_vs_opus"] = round(savings, 2)

    # Update daily totals
    today = now.strftime("%Y-%m-%d")
    if today not in data["daily_totals"]:
        data["daily_totals"][today] = {
            "sessions": 0,
            "messages_est": 0,
            "credits_est": 0,
            "savings_est": 0,
            "models_used": {},
        }

    day = data["daily_totals"][today]
    day["sessions"] += 1
    day["messages_est"] += est_messages
    day["credits_est"] = round(day.get("credits_est", 0) + est_credits, 2)
    day["savings_est"] = round(day.get("savings_est", 0) + savings, 2)

    # Track model distribution
    models_used = day.get("models_used", {})
    models_used[model] = models_used.get(model, 0) + 1
    day["models_used"] = models_used

    # Update lifetime
    data["lifetime"]["total_sessions"] += 1
    data["lifetime"]["total_messages_est"] += est_messages
    data["lifetime"]["total_credits_est"] = round(
        data["lifetime"].get("total_credits_est", 0) + est_credits, 2
    )
    data["lifetime"]["total_savings_est"] = round(
        data["lifetime"].get("total_savings_est", 0) + savings, 2
    )

    # Trim old sessions (keep last 100)
    if len(data["sessions"]) > 100:
        data["sessions"] = data["sessions"][-100:]

    # Trim old daily totals (keep last 30 days)
    cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    data["daily_totals"] = {
        k: v for k, v in data["daily_totals"].items() if k >= cutoff
    }

    save_cost_log(data)

    # Cleanup session state
    if SESSION_STATE.exists():
        SESSION_STATE.unlink(missing_ok=True)


def generate_report() -> dict:
    """Generate comprehensive credit optimization report."""
    data = load_cost_log()
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    today_data = data.get("daily_totals", {}).get(today, {
        "sessions": 0, "messages_est": 0, "credits_est": 0,
        "savings_est": 0, "models_used": {},
    })

    # Last 7 days
    week = {"sessions": 0, "messages": 0, "credits": 0, "savings": 0}
    for i in range(7):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        if day in data.get("daily_totals", {}):
            d = data["daily_totals"][day]
            week["sessions"] += d.get("sessions", 0)
            week["messages"] += d.get("messages_est", 0)
            week["credits"] += d.get("credits_est", 0)
            week["savings"] += d.get("savings_est", 0)

    # Model distribution (last 7 days)
    model_dist = {}
    for i in range(7):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        if day in data.get("daily_totals", {}):
            for model, count in data["daily_totals"][day].get("models_used", {}).items():
                model_dist[model] = model_dist.get(model, 0) + count

    # Recommendations based on actual patterns
    recommendations = []

    # Check if mostly using opus
    opus_pct = model_dist.get("opus", 0) / max(1, sum(model_dist.values())) * 100
    if opus_pct > 50:
        potential_savings = week["credits"] * 0.7  # ~70% could be cheaper
        recommendations.append({
            "priority": "CRITICAL",
            "action": f"Switch default to /model sonnet — {opus_pct:.0f}% of sessions use Opus",
            "impact": f"~{potential_savings:.0f} credits/week saved",
        })

    if today_data.get("credits_est", 0) > DAILY_CREDIT_BUDGET * 0.5:
        recommendations.append({
            "priority": "HIGH",
            "action": "Over 50% of daily credits consumed — use Haiku subagents for remaining work",
            "impact": "15x cheaper per subagent call",
        })

    if today_data.get("sessions", 0) > 8:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Many sessions today — consider batching related tasks",
            "impact": "Reduces context rebuilding overhead",
        })

    recommendations.append({
        "priority": "ALWAYS",
        "action": "Specify model:'haiku' on all search/explore subagent calls",
        "impact": "Each call costs 15x less than defaulting to parent model",
    })

    report = {
        "generated": now.isoformat(),
        "today": {
            "date": today,
            "sessions": today_data.get("sessions", 0),
            "messages_est": today_data.get("messages_est", 0),
            "credits_used": round(today_data.get("credits_est", 0), 1),
            "credits_remaining": round(DAILY_CREDIT_BUDGET - today_data.get("credits_est", 0), 1),
            "pct_used": round(today_data.get("credits_est", 0) / DAILY_CREDIT_BUDGET * 100, 1),
            "savings_vs_opus": round(today_data.get("savings_est", 0), 1),
            "models_used": today_data.get("models_used", {}),
        },
        "week": {
            "sessions": week["sessions"],
            "messages_est": week["messages"],
            "credits_used": round(week["credits"], 1),
            "savings_vs_opus": round(week["savings"], 1),
            "avg_daily_credits": round(week["credits"] / 7, 1),
            "model_distribution": model_dist,
        },
        "lifetime": data.get("lifetime", {}),
        "recommendations": recommendations,
        "credit_reference": {
            "daily_budget": DAILY_CREDIT_BUDGET,
            "opus_per_message": CREDIT_RATES["opus"],
            "sonnet_per_message": CREDIT_RATES["sonnet"],
            "haiku_per_message": CREDIT_RATES["haiku"],
            "5hr_window_opus": WINDOW_LIMITS["opus"],
            "5hr_window_sonnet": WINDOW_LIMITS["sonnet"],
            "5hr_window_haiku": WINDOW_LIMITS["haiku"],
        },
    }

    # Save report
    CREDIT_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def print_report(report: dict):
    """Pretty-print the credit report."""
    print("\n" + "=" * 60)
    print("  CLAUDE MAX CREDIT OPTIMIZER — Report")
    print("=" * 60)

    t = report["today"]
    print(f"\n  TODAY ({t['date']})")
    print(f"    Sessions:       {t['sessions']}")
    print(f"    Est. Messages:  {t['messages_est']}")
    print(f"    Credits Used:   {t['credits_used']:.1f} / {DAILY_CREDIT_BUDGET} ({t['pct_used']:.0f}%)")
    print(f"    Credits Left:   {t['credits_remaining']:.1f}")
    print(f"    Saved vs Opus:  {t['savings_vs_opus']:.1f} credits")
    if t.get("models_used"):
        print(f"    Models Used:    {t['models_used']}")

    w = report["week"]
    print(f"\n  LAST 7 DAYS")
    print(f"    Sessions:       {w['sessions']}")
    print(f"    Credits Used:   {w['credits_used']:.1f}")
    print(f"    Avg Daily:      {w['avg_daily_credits']:.1f} credits")
    print(f"    Saved vs Opus:  {w['savings_vs_opus']:.1f} credits")
    if w.get("model_distribution"):
        print(f"    Model Mix:      {w['model_distribution']}")

    lt = report.get("lifetime", {})
    if lt:
        print(f"\n  LIFETIME")
        print(f"    Sessions:       {lt.get('total_sessions', 0)}")
        print(f"    Total Credits:  {lt.get('total_credits_est', 0):.1f}")
        print(f"    Total Saved:    {lt.get('total_savings_est', 0):.1f}")

    print(f"\n  RECOMMENDATIONS")
    for r in report.get("recommendations", []):
        icon = {
            "CRITICAL": "!!!",
            "HIGH": " !!",
            "MEDIUM": "  !",
            "ALWAYS": "  *",
        }.get(r.get("priority", ""), "   ")
        print(f"    [{icon}] {r['action']}")
        if r.get("impact"):
            print(f"           Impact: {r['impact']}")

    print(f"\n  CREDIT MATH")
    print(f"    1 Opus msg    = 1.0 credits  (45/5hr)")
    print(f"    1 Sonnet msg  = 0.2 credits  (225/5hr)  — 5x cheaper")
    print(f"    1 Haiku msg   = 0.067 credits (675/5hr) — 15x cheaper")
    print()


if __name__ == "__main__":
    if "--session-start" in sys.argv:
        session_start()
    elif "--session-end" in sys.argv:
        session_end()
    elif "--report" in sys.argv:
        report = generate_report()
        print_report(report)
    else:
        # Auto-detect: if last session unclosed → end, else → start
        data = load_cost_log()
        if data["sessions"] and data["sessions"][-1].get("ended") is None:
            session_end()
        else:
            session_start()
