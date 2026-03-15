"""Authentication system — user registration, login, sessions.

Simple JWT-based auth with SQLite user storage.
"""

import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timezone
from contextlib import contextmanager

from .config import get_config, DB_PATH
from .memory import Memory

# JWT-like token (no external dependency needed)
TOKEN_SECRET = secrets.token_hex(32)
TOKEN_EXPIRY = 86400 * 30  # 30 days

AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'customer',  -- customer, admin
    stripe_customer_id TEXT,
    subscription_status TEXT DEFAULT 'none', -- none, active, canceled
    subscription_plan TEXT,
    subscription_tier TEXT NOT NULL DEFAULT 'free', -- free, seeker, mystic, inner_circle
    zodiac_sign TEXT,
    birth_date TEXT,
    preferences TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    last_login TEXT
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


class Auth:
    """User authentication and session management."""

    def __init__(self, memory: Memory | None = None):
        self.memory = memory or Memory()
        self._init_tables()

    def _init_tables(self):
        with self.memory._conn() as conn:
            conn.executescript(AUTH_SCHEMA)
            # Migrate: add subscription_tier column if it doesn't exist yet
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(users)").fetchall()
            }
            if "subscription_tier" not in cols:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN "
                    "subscription_tier TEXT NOT NULL DEFAULT 'free'"
                )

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return f"{salt}:{h.hex()}"

    def _verify_password(self, password: str, stored: str) -> bool:
        salt, hash_hex = stored.split(":")
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return hmac.compare_digest(h.hex(), hash_hex)

    def register(self, email: str, password: str, name: str = "",
                 zodiac_sign: str = "") -> dict:
        """Register a new user."""
        email = email.lower().strip()
        if not email or "@" not in email:
            return {"error": "Invalid email address"}
        if len(password) < 6:
            return {"error": "Password must be at least 6 characters"}

        with self.memory._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()
            if existing:
                return {"error": "Email already registered"}

            conn.execute(
                """INSERT INTO users (email, password_hash, name, zodiac_sign)
                   VALUES (?, ?, ?, ?)""",
                (email, self._hash_password(password), name, zodiac_sign)
            )
            user = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

        token = self._create_session(user["id"])
        return {
            "user": self._user_dict(user),
            "token": token,
        }

    def login(self, email: str, password: str) -> dict:
        """Login and return session token."""
        email = email.lower().strip()
        with self.memory._conn() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

        if not user or not self._verify_password(password, user["password_hash"]):
            return {"error": "Invalid email or password"}

        with self.memory._conn() as conn:
            conn.execute(
                "UPDATE users SET last_login = datetime('now') WHERE id = ?",
                (user["id"],)
            )

        token = self._create_session(user["id"])
        return {
            "user": self._user_dict(user),
            "token": token,
        }

    def _create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(48)
        expires = datetime.fromtimestamp(
            time.time() + TOKEN_EXPIRY, tz=timezone.utc
        ).isoformat()

        with self.memory._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires)
            )
        return token

    def get_user_by_token(self, token: str) -> dict | None:
        """Validate token and return user."""
        if not token:
            return None
        with self.memory._conn() as conn:
            session = conn.execute(
                """SELECT s.*, u.* FROM sessions s
                   JOIN users u ON s.user_id = u.id
                   WHERE s.token = ? AND s.expires_at > datetime('now')""",
                (token,)
            ).fetchone()
        if not session:
            return None
        return self._user_dict(session)

    def logout(self, token: str):
        with self.memory._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

    def update_user(self, user_id: int, **kwargs) -> dict:
        """Update user profile fields."""
        allowed = {"name", "zodiac_sign", "birth_date", "preferences",
                    "stripe_customer_id", "subscription_status", "subscription_plan",
                    "subscription_tier"}
        # Handle password separately (needs hashing)
        if "password" in kwargs:
            kwargs["password_hash"] = self._hash_password(kwargs.pop("password"))
            allowed = allowed | {"password_hash"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return {"error": "No valid fields to update"}

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        with self.memory._conn() as conn:
            conn.execute(
                f"UPDATE users SET {set_clause} WHERE id = ?",
                (*updates.values(), user_id)
            )
            user = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return {"user": self._user_dict(user)}

    # ── Subscription tier access control ─────────────────────────────────────

    # Features available at each tier and above
    _TIER_FEATURES: dict[str, set[str]] = {
        "free": {"yes_no", "daily_pull", "horoscopes"},
        "seeker": {
            "yes_no", "daily_pull", "horoscopes",
            # 1–3 card spreads
            "past_present_future", "new_moon", "full_moon",
            "luna_live", "reading_history",
        },
        "mystic": {
            "yes_no", "daily_pull", "horoscopes",
            "past_present_future", "new_moon", "full_moon",
            "luna_live", "reading_history",
            # expanded spreads + tools
            "celtic_cross", "year_ahead", "shadow_work",
            "love_relationships", "career_crossroads",
            "crystal_guides", "custom_rituals",
        },
        "inner_circle": {
            "yes_no", "daily_pull", "horoscopes",
            "past_present_future", "new_moon", "full_moon",
            "luna_live", "reading_history",
            "celtic_cross", "year_ahead", "shadow_work",
            "love_relationships", "career_crossroads",
            "crystal_guides", "custom_rituals",
            # premium
            "birth_chart", "compatibility", "monthly_deep_dive",
        },
    }

    # Map Stripe product names / price IDs to tier keys
    _PRODUCT_TO_TIER: dict[str, str] = {
        "mystic luna — seeker": "seeker",
        "mystic luna — mystic": "mystic",
        "mystic luna — inner circle": "inner_circle",
        "seeker": "seeker",
        "mystic": "mystic",
        "inner_circle": "inner_circle",
    }

    def update_subscription(self, user_id: int, tier: str) -> dict:
        """Set subscription_tier and subscription_status for a user.

        Args:
            user_id: DB user ID.
            tier:    One of "free", "seeker", "mystic", "inner_circle".

        Returns:
            Updated user dict, or {"error": ...} if user not found.
        """
        valid_tiers = {"free", "seeker", "mystic", "inner_circle"}
        if tier not in valid_tiers:
            return {"error": f"Invalid tier '{tier}'. Must be one of: {valid_tiers}"}

        status = "active" if tier != "free" else "none"
        with self.memory._conn() as conn:
            conn.execute(
                """UPDATE users
                   SET subscription_tier = ?,
                       subscription_status = ?,
                       subscription_plan = ?
                   WHERE id = ?""",
                (tier, status, tier if tier != "free" else None, user_id),
            )
            user = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        if not user:
            return {"error": "User not found"}
        return {"user": self._user_dict(user)}

    def check_access(self, user_id: int, feature: str) -> bool:
        """Return True if the user's subscription tier grants access to *feature*.

        Always returns True for "free" features regardless of tier.
        Falls back to "free" tier if the user record is missing.
        """
        with self.memory._conn() as conn:
            row = conn.execute(
                "SELECT subscription_tier FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        tier = row["subscription_tier"] if row else "free"
        allowed = self._TIER_FEATURES.get(tier, self._TIER_FEATURES["free"])
        return feature in allowed

    def get_user_by_stripe_customer(self, stripe_customer_id: str) -> dict | None:
        """Look up a user by their Stripe customer ID."""
        with self.memory._conn() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE stripe_customer_id = ?",
                (stripe_customer_id,),
            ).fetchone()
        return self._user_dict(user) if user else None

    def get_user_by_email(self, email: str) -> dict | None:
        """Look up a user by email address."""
        email = email.lower().strip()
        with self.memory._conn() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
        return self._user_dict(user) if user else None

    def get_user_readings(self, user_id: int, limit: int = 20) -> list[dict]:
        """Get a user's reading history."""
        with self.memory._conn() as conn:
            user = conn.execute(
                "SELECT email FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if not user:
                return []

            # Match by hashed customer_id pattern
            rows = conn.execute(
                """SELECT * FROM interactions
                   WHERE interaction_type = 'reading'
                   AND customer_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (self._customer_id(user["email"]), limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def _customer_id(self, email: str) -> str:
        return hashlib.sha256(f"web:{email}".encode()).hexdigest()[:16]

    def _user_dict(self, row) -> dict:
        return {
            "id": row["id"],
            "email": row["email"],
            "name": row["name"],
            "role": row["role"],
            "zodiac_sign": row["zodiac_sign"],
            "subscription_status": row["subscription_status"],
            "subscription_plan": row["subscription_plan"],
            "subscription_tier": row["subscription_tier"] if "subscription_tier" in row.keys() else "free",
            "created_at": row["created_at"],
        }

    def get_all_users(self, limit: int = 100) -> list[dict]:
        """Admin: get all users."""
        with self.memory._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [self._user_dict(r) for r in rows]

    def get_user_count(self) -> int:
        with self.memory._conn() as conn:
            return conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
