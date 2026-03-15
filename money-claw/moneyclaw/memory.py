"""Persistent memory and learning database for MoneyClaw.

SQLite-backed storage for:
- Customer interactions and satisfaction scores
- Revenue events and financial tracking
- Self-learning: what works, what doesn't, improvement experiments
- A/B test results and pricing optimization
- Skill performance metrics
- Agent decisions and outcomes
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DB_PATH

SCHEMA = """
-- Customer interactions (readings, support, feedback)
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    channel TEXT NOT NULL,           -- telegram, discord, whatsapp, web
    interaction_type TEXT NOT NULL,   -- reading, question, feedback, complaint
    service TEXT,                     -- tarot, astrology, ritual, skill
    spread_type TEXT,
    question TEXT,
    response TEXT,
    satisfaction_score REAL,          -- 1-5, from feedback or inferred
    revenue_cents INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    cost_cents INTEGER DEFAULT 0,    -- API cost
    duration_ms INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_interactions_customer ON interactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_interactions_type ON interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at);

-- Revenue ledger
CREATE TABLE IF NOT EXISTS revenue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,             -- readings, skills, subscriptions, tips
    amount_cents INTEGER NOT NULL,
    currency TEXT DEFAULT 'USD',
    customer_id TEXT,
    product TEXT,                     -- specific product/reading type
    stripe_payment_id TEXT,
    refunded INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_revenue_source ON revenue(source);
CREATE INDEX IF NOT EXISTS idx_revenue_created ON revenue(created_at);

-- Expenses
CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,           -- api_costs, tools, marketing, infrastructure
    amount_cents INTEGER NOT NULL,
    description TEXT,
    approved_by TEXT DEFAULT 'auto',  -- auto or nick
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);

-- Self-learning: what patterns work
CREATE TABLE IF NOT EXISTS learnings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,             -- pricing, readings, marketing, skills, customer_service
    pattern TEXT NOT NULL,            -- what was observed
    insight TEXT NOT NULL,            -- what was learned
    confidence REAL DEFAULT 0.5,     -- 0-1, increases with confirming evidence
    evidence_count INTEGER DEFAULT 1,
    impact TEXT,                      -- positive, negative, neutral
    applied INTEGER DEFAULT 0,       -- has this been acted on?
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_learnings_domain ON learnings(domain);
CREATE INDEX IF NOT EXISTS idx_learnings_confidence ON learnings(confidence DESC);

-- A/B test experiments
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    domain TEXT NOT NULL,             -- pricing, copy, spreads, delivery, marketing
    variant_a TEXT NOT NULL,
    variant_b TEXT NOT NULL,
    metric TEXT NOT NULL,             -- conversion_rate, satisfaction, revenue, retention
    samples_a INTEGER DEFAULT 0,
    samples_b INTEGER DEFAULT 0,
    value_a REAL DEFAULT 0,
    value_b REAL DEFAULT 0,
    status TEXT DEFAULT 'running',    -- running, concluded, abandoned
    winner TEXT,                      -- a, b, inconclusive
    min_samples INTEGER DEFAULT 20,
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    concluded_at TEXT
);

-- Agent decisions and outcomes (for self-improvement)
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,              -- brain, luna, skill_forge, revenue_ops, market_scout
    decision_type TEXT NOT NULL,      -- pricing, response_style, marketing, product, strategy
    context TEXT,                     -- what was the situation
    action TEXT NOT NULL,             -- what was decided
    expected_outcome TEXT,
    actual_outcome TEXT,
    success_score REAL,              -- 0-1, how well did it work
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    evaluated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON decisions(agent);
CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions(decision_type);

-- Skill performance tracking
CREATE TABLE IF NOT EXISTS skill_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,        -- downloads, stars, revenue, support_tickets
    value REAL NOT NULL,
    metadata TEXT DEFAULT '{}',
    recorded_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_skill_metrics_skill ON skill_metrics(skill_id);

-- Customer profiles (aggregated knowledge)
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY,              -- hash of identifier
    channel TEXT,
    first_seen TEXT DEFAULT (datetime('now')),
    last_seen TEXT DEFAULT (datetime('now')),
    total_spent_cents INTEGER DEFAULT 0,
    interaction_count INTEGER DEFAULT 0,
    avg_satisfaction REAL,
    preferred_services TEXT DEFAULT '[]',  -- JSON array
    notes TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}'
);

-- Self-upgrade log: tracks changes the system makes to itself
CREATE TABLE IF NOT EXISTS upgrades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component TEXT NOT NULL,          -- brain, luna, pricing, marketing, skills
    change_type TEXT NOT NULL,        -- prompt_update, strategy_shift, new_feature, optimization
    description TEXT NOT NULL,
    before_state TEXT,
    after_state TEXT,
    trigger TEXT,                     -- learning_id, experiment_id, manual, pattern
    impact_score REAL,               -- measured after applying
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    evaluated_at TEXT
);

-- Heartbeat log
CREATE TABLE IF NOT EXISTS heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    checks TEXT NOT NULL,             -- JSON: what was checked
    actions TEXT DEFAULT '[]',        -- JSON: what actions were taken
    health_score REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Reading credit bundles (one-time purchases)
CREATE TABLE IF NOT EXISTS reading_credits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    credits_remaining INTEGER NOT NULL DEFAULT 0,
    credits_total INTEGER NOT NULL DEFAULT 0,
    bundle_name TEXT,
    stripe_payment_id TEXT,
    purchased_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_reading_credits_user ON reading_credits(user_id);

-- Credit redemption history
CREATE TABLE IF NOT EXISTS credit_redemptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    credit_id INTEGER REFERENCES reading_credits(id),
    spread_type TEXT NOT NULL,
    reading_id INTEGER,
    redeemed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_credit_redemptions_user ON credit_redemptions(user_id);
"""


class Memory:
    """Persistent memory store with self-learning capabilities."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            # Run Luna Intelligence migration (adds 6 tables)
            from .migrations.luna_intelligence_002 import migrate
            migrate(conn)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- Interactions ---

    def log_interaction(self, customer_id: str, channel: str,
                        interaction_type: str, **kwargs) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO interactions
                   (customer_id, channel, interaction_type, service, spread_type,
                    question, response, satisfaction_score, revenue_cents,
                    tokens_used, cost_cents, duration_ms, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (customer_id, channel, interaction_type,
                 kwargs.get("service"), kwargs.get("spread_type"),
                 kwargs.get("question"), kwargs.get("response"),
                 kwargs.get("satisfaction_score"),
                 kwargs.get("revenue_cents", 0),
                 kwargs.get("tokens_used", 0),
                 kwargs.get("cost_cents", 0),
                 kwargs.get("duration_ms", 0),
                 json.dumps(kwargs.get("metadata", {})))
            )
            # Update customer profile
            self._upsert_customer(conn, customer_id, channel,
                                  kwargs.get("revenue_cents", 0),
                                  kwargs.get("satisfaction_score"))
            return cur.lastrowid

    def _upsert_customer(self, conn, customer_id: str, channel: str,
                         revenue_cents: int = 0, satisfaction: float | None = None):
        existing = conn.execute(
            "SELECT * FROM customers WHERE id = ?", (customer_id,)
        ).fetchone()

        if existing:
            updates = {
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "interaction_count": existing["interaction_count"] + 1,
                "total_spent_cents": existing["total_spent_cents"] + revenue_cents,
            }
            if satisfaction is not None:
                old_avg = existing["avg_satisfaction"] or satisfaction
                old_count = existing["interaction_count"]
                updates["avg_satisfaction"] = (
                    (old_avg * old_count + satisfaction) / (old_count + 1)
                )
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            conn.execute(
                f"UPDATE customers SET {set_clause} WHERE id = ?",
                (*updates.values(), customer_id)
            )
        else:
            conn.execute(
                """INSERT INTO customers (id, channel, total_spent_cents,
                   interaction_count, avg_satisfaction)
                   VALUES (?, ?, ?, 1, ?)""",
                (customer_id, channel, revenue_cents, satisfaction)
            )

    # --- Revenue ---

    def log_revenue(self, source: str, amount_cents: int, **kwargs) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO revenue (source, amount_cents, currency,
                   customer_id, product, stripe_payment_id, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (source, amount_cents, kwargs.get("currency", "USD"),
                 kwargs.get("customer_id"), kwargs.get("product"),
                 kwargs.get("stripe_payment_id"),
                 json.dumps(kwargs.get("metadata", {})))
            )
            return cur.lastrowid

    def log_expense(self, category: str, amount_cents: int,
                    description: str, **kwargs) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO expenses (category, amount_cents, description,
                   approved_by, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (category, amount_cents, description,
                 kwargs.get("approved_by", "auto"),
                 json.dumps(kwargs.get("metadata", {})))
            )
            return cur.lastrowid

    def get_revenue_summary(self, days: int = 30) -> dict:
        with self._conn() as conn:
            cutoff = f"-{days} days"
            total = conn.execute(
                """SELECT COALESCE(SUM(amount_cents), 0) as total
                   FROM revenue WHERE created_at > datetime('now', ?)
                   AND refunded = 0""", (cutoff,)
            ).fetchone()["total"]

            by_source = {
                row["source"]: row["total"]
                for row in conn.execute(
                    """SELECT source, SUM(amount_cents) as total
                       FROM revenue WHERE created_at > datetime('now', ?)
                       AND refunded = 0 GROUP BY source""", (cutoff,)
                )
            }

            expenses = conn.execute(
                """SELECT COALESCE(SUM(amount_cents), 0) as total
                   FROM expenses WHERE created_at > datetime('now', ?)""",
                (cutoff,)
            ).fetchone()["total"]

            return {
                "period_days": days,
                "total_revenue_cents": total,
                "total_expenses_cents": expenses,
                "net_profit_cents": total - expenses,
                "by_source": by_source,
            }

    # --- Self-Learning ---

    def record_learning(self, domain: str, pattern: str,
                        insight: str, **kwargs) -> int:
        with self._conn() as conn:
            # Check for existing similar learning
            existing = conn.execute(
                """SELECT id, confidence, evidence_count FROM learnings
                   WHERE domain = ? AND pattern = ?""",
                (domain, pattern)
            ).fetchone()

            if existing:
                new_confidence = min(
                    0.99, existing["confidence"] + 0.1
                )
                conn.execute(
                    """UPDATE learnings SET confidence = ?, evidence_count = ?,
                       insight = ?, updated_at = datetime('now')
                       WHERE id = ?""",
                    (new_confidence, existing["evidence_count"] + 1,
                     insight, existing["id"])
                )
                return existing["id"]
            else:
                cur = conn.execute(
                    """INSERT INTO learnings (domain, pattern, insight,
                       confidence, impact, metadata)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (domain, pattern, insight,
                     kwargs.get("confidence", 0.5),
                     kwargs.get("impact", "neutral"),
                     json.dumps(kwargs.get("metadata", {})))
                )
                return cur.lastrowid

    def get_top_learnings(self, domain: str | None = None,
                          limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            if domain:
                rows = conn.execute(
                    """SELECT * FROM learnings WHERE domain = ?
                       ORDER BY confidence DESC, evidence_count DESC
                       LIMIT ?""", (domain, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM learnings
                       ORDER BY confidence DESC, evidence_count DESC
                       LIMIT ?""", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_unapplied_learnings(self, min_confidence: float = 0.7) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM learnings WHERE applied = 0
                   AND confidence >= ?
                   ORDER BY confidence DESC""", (min_confidence,)
            ).fetchall()
            return [dict(r) for r in rows]

    # --- Experiments ---

    def create_experiment(self, name: str, domain: str,
                          variant_a: str, variant_b: str,
                          metric: str, min_samples: int = 20) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO experiments (name, domain, variant_a, variant_b,
                   metric, min_samples)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name, domain, variant_a, variant_b, metric, min_samples)
            )
            return cur.lastrowid

    def record_experiment_sample(self, experiment_id: int,
                                 variant: str, value: float):
        with self._conn() as conn:
            exp = conn.execute(
                "SELECT * FROM experiments WHERE id = ? AND status = 'running'",
                (experiment_id,)
            ).fetchone()
            if not exp:
                return

            if variant == "a":
                new_samples = exp["samples_a"] + 1
                new_value = (
                    (exp["value_a"] * exp["samples_a"] + value) / new_samples
                )
                conn.execute(
                    "UPDATE experiments SET samples_a = ?, value_a = ? WHERE id = ?",
                    (new_samples, new_value, experiment_id)
                )
            else:
                new_samples = exp["samples_b"] + 1
                new_value = (
                    (exp["value_b"] * exp["samples_b"] + value) / new_samples
                )
                conn.execute(
                    "UPDATE experiments SET samples_b = ?, value_b = ? WHERE id = ?",
                    (new_samples, new_value, experiment_id)
                )

            # Auto-conclude if enough samples
            exp = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if (exp["samples_a"] >= exp["min_samples"] and
                    exp["samples_b"] >= exp["min_samples"]):
                self._conclude_experiment(conn, exp)

    def _conclude_experiment(self, conn, exp):
        diff = abs(exp["value_a"] - exp["value_b"])
        avg = (exp["value_a"] + exp["value_b"]) / 2 if (exp["value_a"] + exp["value_b"]) else 1
        effect_size = diff / avg if avg else 0

        if effect_size < 0.05:
            winner = "inconclusive"
        elif exp["value_a"] > exp["value_b"]:
            winner = "a"
        else:
            winner = "b"

        conn.execute(
            """UPDATE experiments SET status = 'concluded', winner = ?,
               concluded_at = datetime('now') WHERE id = ?""",
            (winner, exp["id"])
        )

        # Auto-learn from concluded experiments (inline to avoid nested conn)
        winning_desc = exp["variant_a"] if winner == "a" else exp["variant_b"]
        if winner != "inconclusive":
            confidence = min(0.9, 0.5 + effect_size)
            existing = conn.execute(
                """SELECT id, confidence, evidence_count FROM learnings
                   WHERE domain = ? AND pattern = ?""",
                (exp["domain"], f"experiment:{exp['name']}")
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE learnings SET confidence = ?, evidence_count = ?,
                       insight = ?, updated_at = datetime('now')
                       WHERE id = ?""",
                    (min(0.99, existing["confidence"] + 0.1),
                     existing["evidence_count"] + 1,
                     f"'{winning_desc}' outperformed in {exp['metric']} "
                     f"(effect size: {effect_size:.1%})",
                     existing["id"])
                )
            else:
                conn.execute(
                    """INSERT INTO learnings (domain, pattern, insight,
                       confidence, impact, metadata)
                       VALUES (?, ?, ?, ?, ?, '{}')""",
                    (exp["domain"], f"experiment:{exp['name']}",
                     f"'{winning_desc}' outperformed in {exp['metric']} "
                     f"(effect size: {effect_size:.1%})",
                     confidence, "positive")
                )

    def get_active_experiments(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE status = 'running'"
            ).fetchall()
            return [dict(r) for r in rows]

    # --- Decisions ---

    def log_decision(self, agent: str, decision_type: str,
                     action: str, **kwargs) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO decisions (agent, decision_type, context,
                   action, expected_outcome, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (agent, decision_type, kwargs.get("context"),
                 action, kwargs.get("expected_outcome"),
                 json.dumps(kwargs.get("metadata", {})))
            )
            return cur.lastrowid

    def evaluate_decision(self, decision_id: int, actual_outcome: str,
                          success_score: float):
        with self._conn() as conn:
            conn.execute(
                """UPDATE decisions SET actual_outcome = ?, success_score = ?,
                   evaluated_at = datetime('now') WHERE id = ?""",
                (actual_outcome, success_score, decision_id)
            )

            # Learn from the decision (inline to avoid nested conn)
            decision = conn.execute(
                "SELECT * FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
            if decision and success_score is not None:
                impact = "positive" if success_score > 0.6 else (
                    "negative" if success_score < 0.4 else "neutral"
                )
                pattern = f"decision:{decision['action'][:80]}"
                domain = decision["decision_type"]
                insight = (f"{'Effective' if impact == 'positive' else 'Ineffective'}: "
                           f"{actual_outcome}")
                confidence = 0.4 + (success_score * 0.3)

                existing = conn.execute(
                    """SELECT id, confidence, evidence_count FROM learnings
                       WHERE domain = ? AND pattern = ?""",
                    (domain, pattern)
                ).fetchone()
                if existing:
                    conn.execute(
                        """UPDATE learnings SET confidence = ?, evidence_count = ?,
                           insight = ?, updated_at = datetime('now')
                           WHERE id = ?""",
                        (min(0.99, existing["confidence"] + 0.1),
                         existing["evidence_count"] + 1, insight, existing["id"])
                    )
                else:
                    conn.execute(
                        """INSERT INTO learnings (domain, pattern, insight,
                           confidence, impact, metadata)
                           VALUES (?, ?, ?, ?, ?, '{}')""",
                        (domain, pattern, insight, confidence, impact)
                    )

    def get_decision_success_rate(self, agent: str | None = None,
                                   decision_type: str | None = None) -> dict:
        with self._conn() as conn:
            where = ["evaluated_at IS NOT NULL"]
            params = []
            if agent:
                where.append("agent = ?")
                params.append(agent)
            if decision_type:
                where.append("decision_type = ?")
                params.append(decision_type)

            where_clause = " AND ".join(where)
            row = conn.execute(
                f"""SELECT COUNT(*) as total,
                    AVG(success_score) as avg_success,
                    SUM(CASE WHEN success_score > 0.6 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN success_score < 0.4 THEN 1 ELSE 0 END) as losses
                    FROM decisions WHERE {where_clause}""",
                params
            ).fetchone()
            return dict(row) if row else {}

    # --- Upgrades ---

    def log_upgrade(self, component: str, change_type: str,
                    description: str, **kwargs) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO upgrades (component, change_type, description,
                   before_state, after_state, trigger, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (component, change_type, description,
                 kwargs.get("before_state"), kwargs.get("after_state"),
                 kwargs.get("trigger"), json.dumps(kwargs.get("metadata", {})))
            )
            return cur.lastrowid

    # --- Heartbeat ---

    def log_heartbeat(self, agent: str, checks: dict,
                      actions: list, health_score: float) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO heartbeats (agent, checks, actions, health_score)
                   VALUES (?, ?, ?, ?)""",
                (agent, json.dumps(checks), json.dumps(actions), health_score)
            )
            return cur.lastrowid

    # --- Reading Credits ---

    def add_credits(self, user_id: str, credits: int, bundle_name: str,
                    stripe_payment_id: str = "") -> int:
        """Add reading credits for a user."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO reading_credits
                   (user_id, credits_remaining, credits_total, bundle_name, stripe_payment_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, credits, credits, bundle_name, stripe_payment_id)
            )
            return cur.lastrowid

    def get_credits(self, user_id: str) -> int:
        """Get total remaining credits for a user."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(credits_remaining), 0) as total FROM reading_credits WHERE user_id = ? AND credits_remaining > 0",
                (user_id,)
            ).fetchone()
            return row["total"]

    def redeem_credit(self, user_id: str, spread_type: str) -> bool:
        """Redeem one credit for a reading. Returns True if successful."""
        with self._conn() as conn:
            # Find oldest credit pack with remaining credits
            credit = conn.execute(
                "SELECT id, credits_remaining FROM reading_credits WHERE user_id = ? AND credits_remaining > 0 ORDER BY purchased_at ASC LIMIT 1",
                (user_id,)
            ).fetchone()
            if not credit:
                return False
            conn.execute(
                "UPDATE reading_credits SET credits_remaining = credits_remaining - 1 WHERE id = ?",
                (credit["id"],)
            )
            conn.execute(
                "INSERT INTO credit_redemptions (user_id, credit_id, spread_type) VALUES (?, ?, ?)",
                (user_id, credit["id"], spread_type)
            )
            return True

    # --- Analytics ---

    def get_customer_stats(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as total_customers,
                   AVG(avg_satisfaction) as avg_satisfaction,
                   SUM(total_spent_cents) as total_lifetime_value,
                   AVG(interaction_count) as avg_interactions
                   FROM customers"""
            ).fetchone()
            return dict(row) if row else {}

    def get_popular_services(self, days: int = 30) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT service, spread_type, COUNT(*) as count,
                   AVG(satisfaction_score) as avg_satisfaction,
                   SUM(revenue_cents) as total_revenue
                   FROM interactions
                   WHERE created_at > datetime('now', ?)
                   AND service IS NOT NULL
                   GROUP BY service, spread_type
                   ORDER BY count DESC""",
                (f"-{days} days",)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_system_health(self) -> dict:
        with self._conn() as conn:
            recent_heartbeats = conn.execute(
                """SELECT agent, health_score, created_at
                   FROM heartbeats
                   WHERE created_at > datetime('now', '-1 hour')
                   ORDER BY created_at DESC"""
            ).fetchall()

            revenue = self.get_revenue_summary(7)
            customers = self.get_customer_stats()
            learnings = conn.execute(
                "SELECT COUNT(*) as c FROM learnings WHERE confidence > 0.7"
            ).fetchone()["c"]
            experiments = conn.execute(
                "SELECT COUNT(*) as c FROM experiments WHERE status = 'running'"
            ).fetchone()["c"]

            return {
                "heartbeats": [dict(r) for r in recent_heartbeats],
                "revenue_7d": revenue,
                "customers": customers,
                "high_confidence_learnings": learnings,
                "active_experiments": experiments,
            }
