"""Event Bus — Pub/Sub with JSONL Persistence

Merged from project-mesh-v2-omega's event_bus.py.
Provides event-driven communication between Brain components.
"""
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

EVENTS_DIR = Path(__file__).parent.parent / "events"
EVENT_LOG = EVENTS_DIR / "event_log.jsonl"
MAX_LOG_LINES = 50000


class EventBus:
    """Simple pub/sub event bus with JSONL persistence."""

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()
        EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type. Use '*' for all events."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type] if cb != callback
                ]

    def emit(self, event_type: str, data: dict, source: str = "brain"):
        """Emit an event to all subscribers and persist to log."""
        event = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }

        # Persist to JSONL
        self._persist(event)

        # Notify subscribers
        with self._lock:
            # Specific subscribers
            for callback in self._subscribers.get(event_type, []):
                try:
                    callback(event)
                except Exception as e:
                    print(f"Event handler error ({event_type}): {e}")

            # Wildcard subscribers
            for callback in self._subscribers.get("*", []):
                try:
                    callback(event)
                except Exception as e:
                    print(f"Wildcard handler error: {e}")

    def _persist(self, event: dict):
        """Append event to JSONL log, rotate if needed."""
        try:
            with open(EVENT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

            # Rotate if too large
            if EVENT_LOG.exists() and EVENT_LOG.stat().st_size > 10_000_000:  # 10MB
                self._rotate()
        except Exception as e:
            print(f"Event persistence error: {e}")

    def _rotate(self):
        """Rotate log file, keeping recent entries."""
        try:
            lines = EVENT_LOG.read_text(encoding="utf-8").strip().split("\n")
            # Keep last 25000 lines
            keep = lines[-25000:]
            EVENT_LOG.write_text("\n".join(keep) + "\n", encoding="utf-8")
            # Archive old
            archive = EVENTS_DIR / f"event_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            archive.write_text("\n".join(lines[:-25000]) + "\n", encoding="utf-8")
        except Exception:
            pass

    def recent(self, limit: int = 50, event_type: Optional[str] = None) -> list[dict]:
        """Get recent events from log."""
        try:
            if not EVENT_LOG.exists():
                return []
            lines = EVENT_LOG.read_text(encoding="utf-8").strip().split("\n")
            events = []
            for line in reversed(lines):
                if len(events) >= limit:
                    break
                try:
                    event = json.loads(line)
                    if event_type is None or event.get("type") == event_type:
                        events.append(event)
                except (json.JSONDecodeError, Exception):
                    continue
            return events
        except Exception:
            return []

    def stats(self) -> dict:
        """Get event statistics."""
        events = self.recent(limit=1000)
        type_counts = {}
        for e in events:
            t = e.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_recent": len(events),
            "types": type_counts,
            "log_size_mb": round(EVENT_LOG.stat().st_size / 1_000_000, 2) if EVENT_LOG.exists() else 0,
        }


# Global singleton
_bus = None

def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
