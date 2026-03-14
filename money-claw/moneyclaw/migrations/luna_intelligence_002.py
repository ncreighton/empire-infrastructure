"""Migration 002: Luna Intelligence System tables.

Adds 6 new tables for conversational memory, deep persona tracking,
proactive outreach, and reading pattern analysis.
"""

LUNA_INTELLIGENCE_SCHEMA = """
-- User entities extracted from conversations (people, pets, jobs, goals, locations)
CREATE TABLE IF NOT EXISTS user_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,      -- person, pet, job, goal, location, date_event
    entity_name TEXT NOT NULL,
    context TEXT DEFAULT '',
    first_mentioned TEXT DEFAULT (datetime('now')),
    last_mentioned TEXT DEFAULT (datetime('now')),
    mention_count INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_user_entities_user ON user_entities(user_id);
CREATE INDEX IF NOT EXISTS idx_user_entities_type ON user_entities(user_id, entity_type);

-- User conversation topics and sentiment tracking
CREATE TABLE IF NOT EXISTS user_topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    topic TEXT NOT NULL,             -- love, career, health, family, spiritual, grief, anxiety, money
    sentiment TEXT DEFAULT 'neutral', -- positive, negative, neutral, mixed
    first_raised TEXT DEFAULT (datetime('now')),
    last_raised TEXT DEFAULT (datetime('now')),
    times_discussed INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_user_topics_user ON user_topics(user_id);

-- Luna's follow-up promises — things she said she'd check back on
CREATE TABLE IF NOT EXISTS luna_followups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    promise_text TEXT NOT NULL,
    context TEXT DEFAULT '',
    promised_at TEXT DEFAULT (datetime('now')),
    due_at TEXT,
    fulfilled_at TEXT,
    status TEXT DEFAULT 'pending'    -- pending, fulfilled, expired
);
CREATE INDEX IF NOT EXISTS idx_luna_followups_user ON luna_followups(user_id);
CREATE INDEX IF NOT EXISTS idx_luna_followups_status ON luna_followups(status);

-- Every card drawn per user for pattern analysis
CREATE TABLE IF NOT EXISTS reading_card_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    card_name TEXT NOT NULL,
    reversed INTEGER DEFAULT 0,
    spread_type TEXT,
    position TEXT,
    reading_id INTEGER,
    drawn_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_card_log_user ON reading_card_log(user_id);
CREATE INDEX IF NOT EXISTS idx_card_log_card ON reading_card_log(card_name);

-- Significant life events and milestones per user
CREATE TABLE IF NOT EXISTS user_timeline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,        -- birthday, wedding, job_change, loss, achievement, move
    description TEXT NOT NULL,
    event_date TEXT,
    logged_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_user_timeline_user ON user_timeline(user_id);

-- Proactive outreach events scheduled by Luna
CREATE TABLE IF NOT EXISTS proactive_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,        -- moon_alert, card_pattern, streak_reminder, followup_check, etc.
    payload_json TEXT DEFAULT '{}',
    scheduled_for TEXT NOT NULL,
    sent_at TEXT,
    status TEXT DEFAULT 'pending'    -- pending, sent, skipped
);
CREATE INDEX IF NOT EXISTS idx_proactive_user ON proactive_events(user_id);
CREATE INDEX IF NOT EXISTS idx_proactive_status ON proactive_events(status, scheduled_for);
"""


def migrate(conn):
    """Run the Luna Intelligence migration. Safe to call multiple times."""
    conn.executescript(LUNA_INTELLIGENCE_SCHEMA)
