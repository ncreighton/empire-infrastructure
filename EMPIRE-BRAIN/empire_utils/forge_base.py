"""Base FORGE module classes — shared pattern across all Empire intelligence systems.

The FORGE pattern (Scout/Sentinel/Oracle/Smith/Codex) appears in 4+ systems with
20+ modules total. These base classes eliminate ~80 lines of boilerplate per module.

Usage:
    from empire_utils.forge_base import BaseScout, BaseSentinel, BaseOracle, BaseSmith

    class SpellScout(BaseScout):
        DOMAIN = "grimoire"

        def analyze(self, intention: str, **kwargs) -> dict:
            correspondences = self._lookup_correspondences(intention)
            result = {"intention": intention, "correspondences": correspondences}
            self._emit("analysis_complete", result)
            return result

    class RitualSentinel(BaseSentinel):
        DOMAIN = "grimoire"

        def _run_checks(self, data: dict) -> list[dict]:
            checks = []
            checks.append(self._check("safety", data.get("safety_notes"), weight=25))
            checks.append(self._check("timing", data.get("moon_phase"), weight=15))
            return checks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional


class _ForgeBase(ABC):
    """Common base for all FORGE modules."""

    DOMAIN: str = "base"

    def __init__(self, db: Any = None, event_bus: Any = None):
        self.db = db
        self._event_bus = event_bus
        self.stats: dict[str, Any] = {}

    def _emit(self, action: str, data: dict | None = None):
        """Emit an event via the event bus or DB if available."""
        event_type = f"{self.DOMAIN}.{action}"
        if self.db and hasattr(self.db, "emit_event"):
            self.db.emit_event(event_type, data or {})
        if self._event_bus and hasattr(self._event_bus, "emit"):
            self._event_bus.emit(event_type, data or {})

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()


class BaseScout(_ForgeBase):
    """Discovery and analysis module. Finds data, patterns, opportunities.

    Subclasses must implement:
    - analyze(**kwargs) -> dict
    """

    @abstractmethod
    def analyze(self, **kwargs) -> dict:
        """Main analysis method. Returns discovery results."""
        ...


class BaseSentinel(_ForgeBase):
    """Health and quality monitoring module. Scores and validates.

    Subclasses must implement:
    - _run_checks(data) -> list[dict]

    Provides:
    - full_check(data) -> dict with score, checks, alerts, timestamp
    - _check(name, value, weight, threshold) -> dict helper
    """

    def full_check(self, data: dict | None = None) -> dict:
        """Run all checks and return scored result."""
        data = data or {}
        checks = self._run_checks(data)
        alerts = [c for c in checks if not c.get("passed", True)]
        total_weight = sum(c.get("weight", 10) for c in checks)
        earned = sum(c.get("weight", 10) for c in checks if c.get("passed", True))
        score = round((earned / total_weight * 100) if total_weight > 0 else 0, 1)

        result = {
            "timestamp": self._timestamp(),
            "domain": self.DOMAIN,
            "checks": checks,
            "alerts": alerts,
            "score": score,
            "is_healthy": score >= 70,
        }
        self._emit("health_check", {"score": score, "alerts": len(alerts)})
        return result

    @abstractmethod
    def _run_checks(self, data: dict) -> list[dict]:
        """Run domain-specific checks. Return list of check result dicts."""
        ...

    @staticmethod
    def _check(name: str, value: Any, *, weight: int = 10,
               threshold: Any = None, passed: bool | None = None) -> dict:
        """Helper to create a check result dict."""
        if passed is None:
            if threshold is not None:
                passed = value is not None and value >= threshold
            else:
                passed = bool(value)
        return {
            "name": name,
            "value": value,
            "weight": weight,
            "passed": passed,
        }


class BaseOracle(_ForgeBase):
    """Prediction and forecasting module. Identifies opportunities and risks.

    Subclasses must implement:
    - forecast(**kwargs) -> dict
    """

    @abstractmethod
    def forecast(self, **kwargs) -> dict:
        """Generate predictions. Returns opportunities, risks, recommendations."""
        ...

    def _score_opportunity(self, impact: float, effort: float,
                           strategic_value: float = 1.0) -> float:
        """Standard opportunity scoring formula used across all Empire systems."""
        return round(impact * (1 / max(effort, 0.1)) * strategic_value, 2)


class BaseSmith(_ForgeBase):
    """Generation and synthesis module. Creates solutions, content, artifacts.

    Subclasses must implement:
    - generate(**kwargs) -> dict
    """

    @abstractmethod
    def generate(self, **kwargs) -> dict:
        """Generate from spec. Returns generated artifact."""
        ...
