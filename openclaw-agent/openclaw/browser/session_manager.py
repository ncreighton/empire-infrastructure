"""SessionManager — persist and restore browser sessions (cookies/storage) per platform."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionManager:
    """Save and load browser session state (cookies, storage) per platform."""

    def __init__(self, sessions_dir: str | None = None):
        self.sessions_dir = Path(
            sessions_dir
            or os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "sessions",
            )
        )
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, platform_id: str) -> Path:
        """Get the session file path for a platform."""
        return self.sessions_dir / f"{platform_id}.json"

    def save_session(self, platform_id: str, state: dict[str, Any]) -> None:
        """Save browser session state for a platform."""
        path = self._session_path(platform_id)
        state["updated_at"] = datetime.now().isoformat()
        path.write_text(json.dumps(state, indent=2, default=str))
        logger.info(f"Session saved: {platform_id}")

    def load_session(self, platform_id: str) -> dict[str, Any] | None:
        """Load saved session state for a platform, or None if not found."""
        path = self._session_path(platform_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            logger.info(f"Session loaded: {platform_id}")
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load session for {platform_id}: {e}")
            return None

    def has_session(self, platform_id: str) -> bool:
        """Check if a saved session exists for a platform."""
        return self._session_path(platform_id).exists()

    def delete_session(self, platform_id: str) -> bool:
        """Delete saved session for a platform."""
        path = self._session_path(platform_id)
        if path.exists():
            path.unlink()
            logger.info(f"Session deleted: {platform_id}")
            return True
        return False

    def list_sessions(self) -> list[str]:
        """List all platform IDs with saved sessions."""
        return [
            p.stem for p in self.sessions_dir.glob("*.json")
        ]

    def get_session_age_hours(self, platform_id: str) -> float | None:
        """Get the age of a session in hours, or None if no session."""
        state = self.load_session(platform_id)
        if not state or "updated_at" not in state:
            return None
        try:
            updated = datetime.fromisoformat(state["updated_at"])
            return (datetime.now() - updated).total_seconds() / 3600
        except (ValueError, TypeError):
            return None

    def is_session_fresh(self, platform_id: str, max_age_hours: float = 24) -> bool:
        """Check if session is fresh enough to reuse."""
        age = self.get_session_age_hours(platform_id)
        if age is None:
            return False
        return age < max_age_hours

    def cleanup_stale(self, max_age_hours: float = 72) -> list[str]:
        """Delete sessions older than max_age_hours. Returns deleted platform IDs."""
        deleted = []
        for platform_id in self.list_sessions():
            age = self.get_session_age_hours(platform_id)
            if age is not None and age > max_age_hours:
                self.delete_session(platform_id)
                deleted.append(platform_id)
        if deleted:
            logger.info(f"Cleaned up {len(deleted)} stale sessions")
        return deleted
