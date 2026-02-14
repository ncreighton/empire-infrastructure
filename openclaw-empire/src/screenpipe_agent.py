"""
Screenpipe Agent for OpenClaw Empire
Provides passive monitoring and historical context via Screenpipe (port 3030).

Connects to the Screenpipe service for OCR data, audio transcriptions, and
UI events. Designed for tracking automation progress, detecting errors in
screen recordings, and building activity timelines -- all without any UI
interaction.

Usage:
    from src.screenpipe_agent import ScreenpipeAgent

    agent = ScreenpipeAgent()
    results = await agent.search("login error", app_name="Chrome")
    state = await agent.get_current_state("GeeLark")
    errors = await agent.search_errors("Canvas", minutes_back=5)
    timeline = await agent.get_activity_timeline(minutes_back=30)

Sync usage:
    results = agent.search_sync("login error")
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ContentType(Enum):
    """Screenpipe content type filters."""
    ALL = "all"
    OCR = "ocr"
    AUDIO = "audio"
    UI = "ui"


class UIEventType(Enum):
    """UI event categories captured by Screenpipe."""
    CLICK = "click"
    TEXT = "text"
    SCROLL = "scroll"
    KEY = "key"
    APP_SWITCH = "app_switch"
    WINDOW_FOCUS = "window_focus"
    CLIPBOARD = "clipboard"


@dataclass
class SearchResult:
    """A single screenpipe search result."""
    content: str
    app_name: str = ""
    window_name: str = ""
    timestamp: str = ""
    content_type: str = ""
    frame_id: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def dt(self) -> Optional[datetime]:
        """Parse the timestamp into a datetime object."""
        if not self.timestamp:
            return None
        try:
            return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None


@dataclass
class UIEvent:
    """A captured UI event (click, keystroke, clipboard, etc.)."""
    event_type: str
    app_name: str = ""
    window_name: str = ""
    timestamp: str = ""
    text: str = ""
    element_info: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivityEntry:
    """An entry in the activity timeline."""
    app_name: str
    window_name: str
    start_time: str
    end_time: str = ""
    duration_seconds: float = 0.0
    text_snippets: List[str] = field(default_factory=list)


@dataclass
class PatternMatch:
    """A matched pattern found during monitoring."""
    pattern: str
    matched_text: str
    app_name: str = ""
    window_name: str = ""
    timestamp: str = ""
    context: str = ""


# ---------------------------------------------------------------------------
# FORGE Codex integration
# ---------------------------------------------------------------------------

class FORGECodex:
    """
    Integration with FORGE Codex -- the learning system.

    Feeds observed patterns (errors, successes, app behaviors) into the Codex
    so future automation runs can benefit from historical context.
    """

    def __init__(self):
        self._observations: List[Dict[str, Any]] = []

    def record_observation(
        self,
        category: str,
        data: Dict[str, Any],
        source: str = "screenpipe",
    ):
        """Record an observation for the Codex learning system."""
        entry = {
            "category": category,
            "data": data,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._observations.append(entry)
        logger.debug("FORGE Codex observation [%s]: %s", category, json.dumps(data)[:200])

    def get_observations(
        self,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve recorded observations, optionally filtered by category."""
        obs = self._observations
        if category:
            obs = [o for o in obs if o["category"] == category]
        return obs[-limit:]

    def get_error_patterns(self) -> List[Dict[str, Any]]:
        """Get all recorded error patterns for analysis."""
        return self.get_observations(category="error")

    def get_success_patterns(self) -> List[Dict[str, Any]]:
        """Get all recorded success patterns."""
        return self.get_observations(category="success")

    def summarize(self) -> Dict[str, int]:
        """Summarize observation counts by category."""
        counts: Dict[str, int] = {}
        for obs in self._observations:
            cat = obs["category"]
            counts[cat] = counts.get(cat, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Screenpipe Agent
# ---------------------------------------------------------------------------

class ScreenpipeAgent:
    """
    Passive monitoring agent using Screenpipe for OCR, audio, and UI events.

    Provides search, filtering, pattern monitoring, and activity timelines.
    Integrates with FORGE Codex to feed learned patterns into the automation
    system.

    Args:
        base_url: Screenpipe API base URL. Default ``http://localhost:3030``.
        timeout: Request timeout in seconds. Default 15.
        max_retries: Retry attempts on transient failures. Default 2.
        retry_delay: Base delay between retries in seconds. Default 0.5.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3030",
        timeout: int = 15,
        max_retries: int = 2,
        retry_delay: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.codex = FORGECodex()
        self._session: Optional[aiohttp.ClientSession] = None
        self._monitoring: bool = False

    # -- session management --------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create and return an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close the HTTP session and stop any active monitors."""
        self._monitoring = False
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # -- HTTP transport with retries -----------------------------------------

    async def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET from screenpipe with retry logic."""
        url = f"{self.base_url}{endpoint}"
        last_error: Optional[Exception] = None

        # Clean None values from params
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        for attempt in range(1, self.max_retries + 1):
            try:
                session = await self._get_session()
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    body = await resp.text()
                    if resp.status >= 500:
                        logger.warning(
                            "Screenpipe %s returned %d (attempt %d/%d): %s",
                            endpoint, resp.status, attempt, self.max_retries,
                            body[:200],
                        )
                        last_error = RuntimeError(
                            f"Screenpipe {resp.status}: {body[:200]}"
                        )
                    else:
                        raise RuntimeError(
                            f"Screenpipe {resp.status}: {body[:200]}"
                        )
            except aiohttp.ClientError as exc:
                logger.warning(
                    "Screenpipe connection error on %s (attempt %d/%d): %s",
                    endpoint, attempt, self.max_retries, exc,
                )
                last_error = exc
            except asyncio.TimeoutError:
                logger.warning(
                    "Screenpipe timeout on %s (attempt %d/%d)",
                    endpoint, attempt, self.max_retries,
                )
                last_error = TimeoutError(f"Timeout calling {endpoint}")

            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

        raise last_error or RuntimeError("Screenpipe request failed")

    # -- time helpers --------------------------------------------------------

    @staticmethod
    def _utc_now() -> datetime:
        """Current time in UTC."""
        return datetime.now(timezone.utc)

    @staticmethod
    def _to_iso(dt: datetime) -> str:
        """Format a datetime to ISO 8601 with Z suffix."""
        s = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return s

    def _minutes_ago(self, minutes: int) -> str:
        """ISO timestamp for N minutes ago."""
        return self._to_iso(self._utc_now() - timedelta(minutes=minutes))

    # -- core search ---------------------------------------------------------

    async def search(
        self,
        query: Optional[str] = None,
        app_name: Optional[str] = None,
        window_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        content_type: ContentType = ContentType.ALL,
        limit: int = 10,
        offset: int = 0,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search Screenpipe recorded content.

        Args:
            query: Text search query. Omit for all recent content.
            app_name: Filter by application name (e.g. "Google Chrome").
            window_name: Filter by window title.
            start_time: ISO 8601 UTC start time.
            end_time: ISO 8601 UTC end time.
            content_type: Filter by OCR, audio, UI, or all.
            limit: Maximum results to return.
            offset: Skip N results for pagination.
            min_length: Minimum content length in characters.
            max_length: Maximum content length in characters.

        Returns:
            List of SearchResult objects.
        """
        params: Dict[str, Any] = {
            "content_type": content_type.value,
            "limit": limit,
            "offset": offset,
        }
        if query:
            params["q"] = query
        if app_name:
            params["app_name"] = app_name
        if window_name:
            params["window_name"] = window_name
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if min_length is not None:
            params["min_length"] = min_length
        if max_length is not None:
            params["max_length"] = max_length

        raw = await self._get("/search", params)
        results = self._parse_search_results(raw)

        logger.info(
            "Screenpipe search: q=%s app=%s -> %d results",
            query, app_name, len(results),
        )
        return results

    def _parse_search_results(self, raw: Any) -> List[SearchResult]:
        """Parse raw Screenpipe API response into SearchResult objects."""
        results: List[SearchResult] = []
        if not isinstance(raw, dict):
            return results

        # Screenpipe returns data in a "data" array
        items = raw.get("data", [])
        if not isinstance(items, list):
            return results

        for item in items:
            content_obj = item.get("content", {})
            if isinstance(content_obj, str):
                text = content_obj
                sr = SearchResult(content=text, raw=item)
            elif isinstance(content_obj, dict):
                # OCR results have "text", audio has "transcription"
                text = content_obj.get("text", content_obj.get("transcription", ""))
                sr = SearchResult(
                    content=text,
                    app_name=content_obj.get("app_name", ""),
                    window_name=content_obj.get("window_name", ""),
                    timestamp=content_obj.get("timestamp", item.get("timestamp", "")),
                    content_type=item.get("type", ""),
                    frame_id=content_obj.get("frame_id"),
                    raw=item,
                )
            else:
                continue
            results.append(sr)

        return results

    # -- convenience methods -------------------------------------------------

    async def get_current_state(self, app_name: Optional[str] = None) -> List[SearchResult]:
        """
        Get what is currently on screen (last 30 seconds of OCR data).

        Args:
            app_name: Optionally filter to a specific app.

        Returns:
            Recent screen content results.
        """
        now = self._utc_now()
        start = self._to_iso(now - timedelta(seconds=30))
        end = self._to_iso(now)

        results = await self.search(
            app_name=app_name,
            start_time=start,
            end_time=end,
            content_type=ContentType.OCR,
            limit=5,
        )
        if results:
            self.codex.record_observation("current_state", {
                "app": app_name or "all",
                "text_preview": results[0].content[:200] if results else "",
                "result_count": len(results),
            })
        return results

    async def search_errors(
        self,
        app_name: Optional[str] = None,
        minutes_back: int = 10,
    ) -> List[SearchResult]:
        """
        Search for recent error-related text on screen.

        Looks for common error patterns: "error", "failed", "exception",
        "crash", "denied", "timeout", "not found", "401", "403", "500".

        Args:
            app_name: Filter to a specific app.
            minutes_back: How far back to search.

        Returns:
            Search results containing error text.
        """
        error_keywords = [
            "error", "failed", "exception", "crash", "denied",
            "timeout", "not found", "401", "403", "500",
            "permission", "unable to", "cannot",
        ]

        start_time = self._minutes_ago(minutes_back)
        all_results: List[SearchResult] = []

        # Search for each error keyword
        for keyword in error_keywords:
            try:
                results = await self.search(
                    query=keyword,
                    app_name=app_name,
                    start_time=start_time,
                    content_type=ContentType.OCR,
                    limit=5,
                )
                all_results.extend(results)
            except Exception as exc:
                logger.debug("Error searching for '%s': %s", keyword, exc)

        # Deduplicate by timestamp + content
        seen = set()
        unique: List[SearchResult] = []
        for r in all_results:
            key = (r.timestamp, r.content[:50])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        # Record errors in Codex
        for r in unique:
            self.codex.record_observation("error", {
                "app": r.app_name,
                "text": r.content[:300],
                "timestamp": r.timestamp,
            })

        logger.info(
            "Error search: app=%s, minutes=%d -> %d unique results",
            app_name, minutes_back, len(unique),
        )
        return unique

    async def get_activity_timeline(
        self,
        minutes_back: int = 30,
        app_name: Optional[str] = None,
    ) -> List[ActivityEntry]:
        """
        Build an activity timeline showing which apps were used and when.

        Groups consecutive OCR results by app_name to produce timeline entries
        with start/end times and sample text snippets.

        Args:
            minutes_back: How far back to look.
            app_name: Optionally filter to a single app.

        Returns:
            List of ActivityEntry objects sorted chronologically.
        """
        start_time = self._minutes_ago(minutes_back)
        results = await self.search(
            app_name=app_name,
            start_time=start_time,
            content_type=ContentType.OCR,
            limit=100,
        )

        if not results:
            return []

        # Group consecutive results by app_name
        timeline: List[ActivityEntry] = []
        current_app = ""
        current_window = ""
        current_start = ""
        current_snippets: List[str] = []

        for r in results:
            if r.app_name != current_app:
                # Close previous entry
                if current_app:
                    timeline.append(ActivityEntry(
                        app_name=current_app,
                        window_name=current_window,
                        start_time=current_start,
                        end_time=r.timestamp,
                        text_snippets=current_snippets[:5],
                    ))
                # Start new entry
                current_app = r.app_name
                current_window = r.window_name
                current_start = r.timestamp
                current_snippets = []

            current_window = r.window_name
            snippet = r.content.strip()[:100]
            if snippet and snippet not in current_snippets:
                current_snippets.append(snippet)

        # Close the last entry
        if current_app:
            timeline.append(ActivityEntry(
                app_name=current_app,
                window_name=current_window,
                start_time=current_start,
                end_time=results[-1].timestamp if results else current_start,
                text_snippets=current_snippets[:5],
            ))

        # Compute durations
        for entry in timeline:
            try:
                start = datetime.fromisoformat(entry.start_time.replace("Z", "+00:00"))
                end = datetime.fromisoformat(entry.end_time.replace("Z", "+00:00"))
                entry.duration_seconds = max(0, (end - start).total_seconds())
            except (ValueError, TypeError):
                entry.duration_seconds = 0.0

        # Record in Codex
        self.codex.record_observation("activity_timeline", {
            "minutes_back": minutes_back,
            "entries": len(timeline),
            "apps": list({e.app_name for e in timeline}),
        })

        logger.info(
            "Activity timeline: %d entries over %d minutes",
            len(timeline), minutes_back,
        )
        return timeline

    async def monitor_for_pattern(
        self,
        pattern: str,
        timeout: float = 60.0,
        poll_interval: float = 2.0,
        app_name: Optional[str] = None,
        callback: Optional[Callable[[PatternMatch], None]] = None,
    ) -> Optional[PatternMatch]:
        """
        Watch for specific text pattern on screen.

        Polls Screenpipe at regular intervals looking for the pattern in OCR
        results. Returns when the pattern is found or the timeout expires.

        Args:
            pattern: Regex pattern or plain text to match.
            timeout: Maximum seconds to watch.
            poll_interval: Seconds between polls.
            app_name: Optionally limit to a specific app.
            callback: Called when the pattern is found.

        Returns:
            PatternMatch if found, None on timeout.
        """
        self._monitoring = True
        compiled = re.compile(pattern, re.IGNORECASE)
        start_time = self._to_iso(self._utc_now())
        deadline = time.monotonic() + timeout

        logger.info(
            "Monitoring for pattern '%s' (timeout=%.0fs, app=%s)",
            pattern, timeout, app_name,
        )

        while self._monitoring and time.monotonic() < deadline:
            try:
                results = await self.search(
                    app_name=app_name,
                    start_time=start_time,
                    content_type=ContentType.OCR,
                    limit=10,
                )

                for r in results:
                    match = compiled.search(r.content)
                    if match:
                        pm = PatternMatch(
                            pattern=pattern,
                            matched_text=match.group(0),
                            app_name=r.app_name,
                            window_name=r.window_name,
                            timestamp=r.timestamp,
                            context=r.content[:300],
                        )
                        logger.info(
                            "Pattern '%s' matched in %s: '%s'",
                            pattern, r.app_name, match.group(0),
                        )

                        self.codex.record_observation("pattern_match", {
                            "pattern": pattern,
                            "matched": match.group(0),
                            "app": r.app_name,
                        })

                        if callback:
                            callback(pm)
                        self._monitoring = False
                        return pm

            except Exception as exc:
                logger.warning("Monitor poll error: %s", exc)

            await asyncio.sleep(poll_interval)

        logger.info("Pattern '%s' not found within %.0fs", pattern, timeout)
        self._monitoring = False
        return None

    def stop_monitoring(self):
        """Stop any active pattern monitoring loop."""
        self._monitoring = False

    async def get_typing_activity(
        self,
        minutes_back: int = 10,
        app_name: Optional[str] = None,
    ) -> List[UIEvent]:
        """
        Get recent keyboard input events.

        Returns text that was typed in the given time window, captured via
        Screenpipe's UI event monitoring (accessibility APIs).

        Args:
            minutes_back: How far back to search.
            app_name: Optionally filter to a specific app.

        Returns:
            List of UIEvent objects with typed text.
        """
        start_time = self._minutes_ago(minutes_back)
        params: Dict[str, Any] = {
            "event_type": "text",
            "limit": 50,
            "start_time": start_time,
        }
        if app_name:
            params["app_name"] = app_name

        try:
            raw = await self._get("/experimental/ui/events", params)
        except RuntimeError:
            # Fallback endpoint if the experimental one is not available
            try:
                raw = await self._get("/search", {
                    "content_type": "ui",
                    "start_time": start_time,
                    "app_name": app_name,
                    "limit": 50,
                })
            except Exception as exc:
                logger.warning("UI events endpoint not available: %s", exc)
                return []

        events = self._parse_ui_events(raw)
        logger.info(
            "Typing activity: app=%s, minutes=%d -> %d events",
            app_name, minutes_back, len(events),
        )
        return events

    async def search_ui_events(
        self,
        event_type: Optional[UIEventType] = None,
        query: Optional[str] = None,
        app_name: Optional[str] = None,
        window_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[UIEvent]:
        """
        Search for UI events (clicks, keyboard input, scrolls, etc.).

        Args:
            event_type: Filter by event type.
            query: Text search within event content.
            app_name: Filter by app name.
            window_name: Filter by window title.
            start_time: ISO 8601 start time.
            end_time: ISO 8601 end time.
            limit: Max results.
            offset: Pagination offset.

        Returns:
            List of UIEvent objects.
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if event_type:
            params["event_type"] = event_type.value
        if query:
            params["q"] = query
        if app_name:
            params["app_name"] = app_name
        if window_name:
            params["window_name"] = window_name
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        try:
            raw = await self._get("/experimental/ui/events", params)
        except RuntimeError:
            raw = await self._get("/search", {
                **params, "content_type": "ui",
            })

        events = self._parse_ui_events(raw)
        logger.info("UI events search: type=%s -> %d events",
                     event_type.value if event_type else "all", len(events))
        return events

    def _parse_ui_events(self, raw: Any) -> List[UIEvent]:
        """Parse raw Screenpipe UI events response."""
        events: List[UIEvent] = []
        if not isinstance(raw, dict):
            return events

        items = raw.get("data", raw.get("events", []))
        if not isinstance(items, list):
            return events

        for item in items:
            if isinstance(item, dict):
                content = item.get("content", {})
                if isinstance(content, dict):
                    events.append(UIEvent(
                        event_type=item.get("event_type", content.get("type", "")),
                        app_name=content.get("app_name", item.get("app_name", "")),
                        window_name=content.get("window_name", item.get("window_name", "")),
                        timestamp=content.get("timestamp", item.get("timestamp", "")),
                        text=content.get("text", content.get("value", "")),
                        element_info=content.get("element", ""),
                        raw=item,
                    ))
                else:
                    events.append(UIEvent(
                        event_type=item.get("event_type", ""),
                        app_name=item.get("app_name", ""),
                        window_name=item.get("window_name", ""),
                        timestamp=item.get("timestamp", ""),
                        text=str(content),
                        raw=item,
                    ))

        return events

    # -- phone screen monitoring (via scrcpy mirror) -------------------------

    async def get_phone_screen_text(
        self,
        minutes_back: int = 1,
        scrcpy_window: str = "scrcpy",
    ) -> List[SearchResult]:
        """
        Get OCR text from a mirrored phone screen (via scrcpy).

        When the Android phone screen is mirrored to the desktop using scrcpy,
        Screenpipe captures it as a regular window. This method filters to
        the scrcpy window for phone-specific screen content.

        Args:
            minutes_back: How far back to search.
            scrcpy_window: Window title substring for scrcpy. Default "scrcpy".

        Returns:
            Recent OCR results from the phone mirror window.
        """
        start_time = self._minutes_ago(minutes_back)
        results = await self.search(
            window_name=scrcpy_window,
            start_time=start_time,
            content_type=ContentType.OCR,
            limit=20,
        )
        logger.info("Phone screen text (scrcpy): %d results", len(results))
        return results

    async def monitor_phone_for_pattern(
        self,
        pattern: str,
        timeout: float = 60.0,
        poll_interval: float = 2.0,
        scrcpy_window: str = "scrcpy",
        callback: Optional[Callable[[PatternMatch], None]] = None,
    ) -> Optional[PatternMatch]:
        """
        Monitor the mirrored phone screen for a specific text pattern.

        Uses the scrcpy window filter to watch only the phone screen content.

        Args:
            pattern: Regex or plain text to match.
            timeout: Max wait time in seconds.
            poll_interval: Poll frequency in seconds.
            scrcpy_window: scrcpy window title substring.
            callback: Called when pattern is found.

        Returns:
            PatternMatch if found, None on timeout.
        """
        # We reuse monitor_for_pattern but it searches by query, not window.
        # For phone monitoring, we search OCR with window filter manually.
        self._monitoring = True
        compiled = re.compile(pattern, re.IGNORECASE)
        start_time = self._to_iso(self._utc_now())
        deadline = time.monotonic() + timeout

        logger.info("Monitoring phone (scrcpy) for pattern '%s'", pattern)

        while self._monitoring and time.monotonic() < deadline:
            try:
                results = await self.search(
                    window_name=scrcpy_window,
                    start_time=start_time,
                    content_type=ContentType.OCR,
                    limit=10,
                )
                for r in results:
                    match = compiled.search(r.content)
                    if match:
                        pm = PatternMatch(
                            pattern=pattern,
                            matched_text=match.group(0),
                            app_name=r.app_name,
                            window_name=r.window_name,
                            timestamp=r.timestamp,
                            context=r.content[:300],
                        )
                        logger.info("Phone pattern matched: '%s'", match.group(0))
                        self.codex.record_observation("phone_pattern_match", {
                            "pattern": pattern,
                            "matched": match.group(0),
                        })
                        if callback:
                            callback(pm)
                        self._monitoring = False
                        return pm
            except Exception as exc:
                logger.warning("Phone monitor poll error: %s", exc)

            await asyncio.sleep(poll_interval)

        logger.info("Phone pattern '%s' not found within %.0fs", pattern, timeout)
        self._monitoring = False
        return None

    # -- sync wrappers --------------------------------------------------------

    def search_sync(self, query: Optional[str] = None, **kwargs) -> List[SearchResult]:
        """Synchronous wrapper for search."""
        return self._run_sync(self.search(query=query, **kwargs))

    def get_current_state_sync(self, app_name: Optional[str] = None) -> List[SearchResult]:
        """Synchronous wrapper for get_current_state."""
        return self._run_sync(self.get_current_state(app_name))

    def search_errors_sync(self, app_name: Optional[str] = None, minutes_back: int = 10) -> List[SearchResult]:
        """Synchronous wrapper for search_errors."""
        return self._run_sync(self.search_errors(app_name, minutes_back))

    def get_activity_timeline_sync(self, minutes_back: int = 30, **kwargs) -> List[ActivityEntry]:
        """Synchronous wrapper for get_activity_timeline."""
        return self._run_sync(self.get_activity_timeline(minutes_back, **kwargs))

    def monitor_for_pattern_sync(self, pattern: str, **kwargs) -> Optional[PatternMatch]:
        """Synchronous wrapper for monitor_for_pattern."""
        return self._run_sync(self.monitor_for_pattern(pattern, **kwargs))

    def get_typing_activity_sync(self, minutes_back: int = 10, **kwargs) -> List[UIEvent]:
        """Synchronous wrapper for get_typing_activity."""
        return self._run_sync(self.get_typing_activity(minutes_back, **kwargs))

    def search_ui_events_sync(self, **kwargs) -> List[UIEvent]:
        """Synchronous wrapper for search_ui_events."""
        return self._run_sync(self.search_ui_events(**kwargs))

    @staticmethod
    def _run_sync(coro):
        """Run an async coroutine in a sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    # -- context manager ------------------------------------------------------

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Screenpipe Agent -- passive monitoring")
    sub = parser.add_subparsers(dest="command", required=True)

    # search
    p_search = sub.add_parser("search", help="Search screen content")
    p_search.add_argument("query", nargs="?", help="Search query")
    p_search.add_argument("--app", help="Filter by app name")
    p_search.add_argument("--minutes", type=int, default=10, help="Minutes back")
    p_search.add_argument("--limit", type=int, default=10, help="Max results")

    # state
    p_state = sub.add_parser("state", help="Get current screen state")
    p_state.add_argument("--app", help="Filter by app name")

    # errors
    p_errors = sub.add_parser("errors", help="Search for recent errors")
    p_errors.add_argument("--app", help="Filter by app name")
    p_errors.add_argument("--minutes", type=int, default=10, help="Minutes back")

    # timeline
    p_timeline = sub.add_parser("timeline", help="Get activity timeline")
    p_timeline.add_argument("--minutes", type=int, default=30, help="Minutes back")
    p_timeline.add_argument("--app", help="Filter by app name")

    # monitor
    p_monitor = sub.add_parser("monitor", help="Watch for a text pattern")
    p_monitor.add_argument("pattern", help="Text or regex pattern to watch for")
    p_monitor.add_argument("--app", help="Filter by app name")
    p_monitor.add_argument("--timeout", type=float, default=60, help="Timeout in seconds")

    # typing
    p_typing = sub.add_parser("typing", help="Get recent typing activity")
    p_typing.add_argument("--app", help="Filter by app name")
    p_typing.add_argument("--minutes", type=int, default=10, help="Minutes back")

    args = parser.parse_args()
    agent = ScreenpipeAgent()

    if args.command == "search":
        start_time = agent._minutes_ago(args.minutes) if args.minutes else None
        results = agent.search_sync(
            query=args.query, app_name=args.app,
            start_time=start_time, limit=args.limit,
        )
        for r in results:
            print(f"[{r.timestamp}] {r.app_name} | {r.content[:120]}")

    elif args.command == "state":
        results = agent.get_current_state_sync(app_name=args.app)
        for r in results:
            print(f"[{r.app_name}] {r.content[:200]}")

    elif args.command == "errors":
        results = agent.search_errors_sync(app_name=args.app, minutes_back=args.minutes)
        if results:
            for r in results:
                print(f"[{r.timestamp}] {r.app_name} | {r.content[:150]}")
        else:
            print("No errors found.")

    elif args.command == "timeline":
        entries = agent.get_activity_timeline_sync(
            minutes_back=args.minutes, app_name=args.app,
        )
        for e in entries:
            print(f"{e.start_time} -> {e.end_time} | {e.app_name} ({e.duration_seconds:.0f}s)")
            for s in e.text_snippets[:2]:
                print(f"    {s}")

    elif args.command == "monitor":
        print(f"Watching for pattern '{args.pattern}' (timeout={args.timeout}s)...")
        match = agent.monitor_for_pattern_sync(
            pattern=args.pattern, app_name=args.app, timeout=args.timeout,
        )
        if match:
            print(f"FOUND at {match.timestamp} in {match.app_name}:")
            print(f"  Matched: {match.matched_text}")
            print(f"  Context: {match.context[:200]}")
        else:
            print("Pattern not found within timeout.")

    elif args.command == "typing":
        events = agent.get_typing_activity_sync(
            minutes_back=args.minutes, app_name=args.app,
        )
        for ev in events:
            print(f"[{ev.timestamp}] {ev.app_name} | {ev.text[:120]}")

    asyncio.run(agent.close())
