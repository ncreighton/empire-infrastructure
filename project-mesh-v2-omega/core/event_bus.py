"""
Event Bus   Pub/sub event system for cross-project communication.
Events are written to events/event_log.jsonl for history.

Event types:
- system.updated   a shared system was updated
- knowledge.new   new knowledge entry discovered
- pattern.detected   duplicated code found
- service.health   service went up/down
- drift.detected   implementation diverged from canonical
- project.bootstrapped   new project created
- scan.completed   code scan finished
- daemon.heartbeat   daemon is alive
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

log = logging.getLogger(__name__)

EVENT_LOG = Path(__file__).parent.parent / "events" / "event_log.jsonl"
MAX_LOG_LINES = 10000  # Rotate after this many entries


class EventBus:
    """Simple pub/sub event bus with file-based persistence."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or EVENT_LOG
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type. Callback receives (event_type, data)."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Remove a subscriber."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type] if cb != callback
                ]

    def publish(self, event_type: str, data: Optional[Dict] = None, source: str = ""):
        """Publish an event. Notifies subscribers and writes to log."""
        event = {
            "type": event_type,
            "data": data or {},
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }

        # Write to log
        self._write_log(event)

        # Notify subscribers
        with self._lock:
            callbacks = list(self._subscribers.get(event_type, []))
            # Also notify wildcard subscribers
            callbacks.extend(self._subscribers.get("*", []))

        for cb in callbacks:
            try:
                cb(event_type, event)
            except Exception as e:
                log.error(f"Event callback error for {event_type}: {e}")

    def _write_log(self, event: Dict):
        """Append event to JSONL log file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            log.error(f"Failed to write event log: {e}")

    def get_recent(self, count: int = 50, event_type: str = "") -> List[Dict]:
        """Get recent events from log."""
        if not self.log_path.exists():
            return []

        events = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        if event_type and ev.get("type") != event_type:
                            continue
                        events.append(ev)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []

        return events[-count:]

    def get_events_since(self, since: str) -> List[Dict]:
        """Get events after a given ISO timestamp."""
        events = self.get_recent(count=MAX_LOG_LINES)
        return [e for e in events if e.get("timestamp", "") > since]

    def rotate_log(self):
        """Rotate log if it exceeds MAX_LOG_LINES."""
        if not self.log_path.exists():
            return

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) > MAX_LOG_LINES:
                # Keep last half
                keep = lines[len(lines) // 2:]
                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.writelines(keep)
                log.info(f"Event log rotated: {len(lines)} -> {len(keep)} entries")
        except Exception as e:
            log.error(f"Failed to rotate event log: {e}")

    def stats(self) -> Dict:
        """Get event statistics."""
        events = self.get_recent(count=MAX_LOG_LINES)
        type_counts: Dict[str, int] = {}
        for e in events:
            t = e.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_events": len(events),
            "event_types": type_counts,
            "subscribers": {k: len(v) for k, v in self._subscribers.items()},
        }


# Global event bus instance
_bus: Optional[EventBus] = None


def get_bus() -> EventBus:
    """Get the global event bus instance."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def publish(event_type: str, data: Optional[Dict] = None, source: str = ""):
    """Convenience function to publish on the global bus."""
    get_bus().publish(event_type, data, source)
