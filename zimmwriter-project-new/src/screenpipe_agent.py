"""
Screenpipe Agent — ZimmWriter Desktop Automation

Integrates with Screenpipe (port 3030) to passively monitor ZimmWriter
through OCR screen capture and audio transcription. Provides a secondary
observation layer that runs alongside pywinauto, catching things that
UI automation might miss.

Key capabilities:
  - Passive OCR monitoring: reads ZimmWriter window text without touching the UI
  - Error detection: catches error dialogs and warning messages via OCR
  - Progress tracking: reads window titles and status text via screen capture
  - Activity correlation: matches OCR readings with automation job timelines
  - Historical search: queries past screen content for debugging

Enhanced by FORGE Intelligence:
  - Sentinel optimizes OCR query prompts
  - Codex stores and learns from OCR patterns
  - Oracle uses OCR history for failure prediction

Usage:
    from src.screenpipe_agent import ScreenpipeAgent
    agent = ScreenpipeAgent(forge_engine)

    # Monitor ZimmWriter passively
    state = agent.read_current_state()

    # Search for recent errors
    errors = agent.search_errors(minutes_back=10)

    # Track job progress via OCR
    progress = agent.track_progress(job_id)
"""

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .utils import setup_logger

logger = setup_logger("screenpipe_agent")

# Screenpipe configuration
SCREENPIPE_URL = "http://localhost:3030"
SCREENPIPE_TIMEOUT = 15  # seconds


class ScreenpipeAgent:
    """
    Passive monitoring layer that uses Screenpipe's OCR to observe
    ZimmWriter state without interfering with UI automation.
    """

    # ZimmWriter-specific OCR search terms
    ZIMMWRITER_KEYWORDS = {
        "progress": [
            "of articles", "Generating", "Processing", "Complete",
            "Finished", "1 of", "2 of", "3 of", "Writing article",
        ],
        "errors": [
            "Error", "Exception", "Failed", "Cannot", "Unable",
            "Timeout", "Connection refused", "Rate limit", "API error",
            "insufficient", "quota exceeded",
        ],
        "screens": [
            "Bulk Writer", "Options Menu", "WordPress", "SERP",
            "Deep Research", "Link Pack", "Style Mimic", "Custom Outline",
            "ZimmWriter v10",
        ],
        "completion": [
            "All articles completed", "Bulk Writer complete",
            "finished writing", "generation complete",
        ],
    }

    # Window names to filter for ZimmWriter
    ZIMMWRITER_WINDOWS = ["ZimmWriter", "Bulk Writer", "AutoIt", "Enable WordPress"]

    def __init__(self, forge_engine=None, screenpipe_url: str = None):
        self.screenpipe_url = (screenpipe_url or SCREENPIPE_URL).rstrip("/")
        self.forge = forge_engine
        self._available = None
        self._session = None
        self._monitoring_active = False
        self._job_start_times: Dict[str, str] = {}  # job_id -> ISO timestamp

    def _get_session(self):
        """Get or create a requests session."""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
            except ImportError:
                logger.warning("requests library not available — screenpipe agent disabled")
                return None
        return self._session

    def is_available(self) -> bool:
        """Check if Screenpipe is reachable."""
        if self._available is not None:
            return self._available

        session = self._get_session()
        if not session:
            self._available = False
            return False

        try:
            resp = session.get(f"{self.screenpipe_url}/health", timeout=5)
            self._available = resp.status_code == 200
            if self._available:
                logger.info(f"Screenpipe available at {self.screenpipe_url}")
            else:
                logger.warning(f"Screenpipe returned {resp.status_code}")
        except Exception as e:
            self._available = False
            logger.warning(f"Screenpipe not available: {e}")

        return self._available

    def _search(self, query: str = None, content_type: str = "ocr",
                start_time: str = None, end_time: str = None,
                app_name: str = None, window_name: str = None,
                limit: int = 10) -> List[Dict]:
        """
        Search Screenpipe content.

        Args:
            query: Search text (optional)
            content_type: "ocr", "audio", or "all"
            start_time: ISO 8601 UTC start time
            end_time: ISO 8601 UTC end time
            app_name: Filter by application name
            window_name: Filter by window title
            limit: Max results

        Returns:
            List of content entries from Screenpipe
        """
        session = self._get_session()
        if not session:
            return []

        params = {
            "content_type": content_type,
            "limit": limit,
        }
        if query:
            params["q"] = query
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if app_name:
            params["app_name"] = app_name
        if window_name:
            params["window_name"] = window_name

        try:
            resp = session.get(
                f"{self.screenpipe_url}/search",
                params=params,
                timeout=SCREENPIPE_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", [])
            logger.warning(f"Screenpipe search returned {resp.status_code}")
        except Exception as e:
            logger.error(f"Screenpipe search failed: {e}")
            self._available = None

        return []

    def _utc_now(self) -> str:
        """Current time as ISO 8601 UTC string."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _utc_minutes_ago(self, minutes: int) -> str:
        """Time N minutes ago as ISO 8601 UTC string."""
        t = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ─────────────────────────────────────────────
    # PASSIVE STATE READING
    # ─────────────────────────────────────────────

    def read_current_state(self) -> Dict:
        """
        Read the current ZimmWriter state via recent OCR captures.
        Returns the most recent screen text associated with ZimmWriter.

        Returns:
            {
                "available": bool,
                "window_title": str,
                "screen_text": str,
                "detected_screen": str,
                "timestamp": str,
                "has_errors": bool,
            }
        """
        if not self.is_available():
            return {"available": False, "reason": "Screenpipe unavailable"}

        # Search recent OCR for ZimmWriter windows
        results = self._search(
            query="ZimmWriter",
            content_type="ocr",
            start_time=self._utc_minutes_ago(2),
            limit=5,
        )

        if not results:
            # Try broader search
            results = self._search(
                content_type="ocr",
                app_name="AutoIt",
                start_time=self._utc_minutes_ago(2),
                limit=5,
            )

        if not results:
            return {
                "available": True,
                "window_title": "",
                "screen_text": "",
                "detected_screen": "unknown",
                "timestamp": self._utc_now(),
                "has_errors": False,
            }

        # Get the most recent result
        latest = results[0]
        content = latest.get("content", {})
        ocr_text = content.get("text", "")
        window_title = content.get("window_name", "")
        app_name = content.get("app_name", "")

        # Detect which screen we're on
        detected_screen = self._detect_screen(ocr_text, window_title)

        # Check for errors in the OCR text
        has_errors = self._check_for_errors(ocr_text)

        state = {
            "available": True,
            "window_title": window_title,
            "app_name": app_name,
            "screen_text": ocr_text[:1000],
            "detected_screen": detected_screen,
            "timestamp": content.get("timestamp", self._utc_now()),
            "has_errors": has_errors,
        }

        # Feed to FORGE Codex for learning
        if self.forge and has_errors:
            self.forge.codex.add_vision_tip(
                "detect_errors",
                f"Screenpipe OCR detected error in: {window_title}"
            )

        return state

    def _detect_screen(self, ocr_text: str, window_title: str) -> str:
        """Determine which ZimmWriter screen is showing based on OCR text."""
        combined = f"{window_title} {ocr_text}".lower()

        if "bulk writer" in combined or "start bulk" in combined:
            return "bulk_writer"
        if "options menu" in combined:
            return "options_menu"
        if "wordpress" in combined and "upload" in combined:
            return "wordpress_config"
        if "serp" in combined:
            return "serp_config"
        if "deep research" in combined:
            return "deep_research_config"
        if "style mimic" in combined:
            return "style_mimic_config"
        if "link pack" in combined:
            return "link_pack_config"
        if "zimmwriter" in combined:
            return "main_menu"

        return "unknown"

    def _check_for_errors(self, text: str) -> bool:
        """Check if OCR text contains error indicators."""
        text_lower = text.lower()
        for keyword in self.ZIMMWRITER_KEYWORDS["errors"]:
            if keyword.lower() in text_lower:
                return True
        return False

    # ─────────────────────────────────────────────
    # ERROR DETECTION
    # ─────────────────────────────────────────────

    def search_errors(self, minutes_back: int = 10) -> List[Dict]:
        """
        Search recent Screenpipe captures for ZimmWriter errors.

        Returns:
            List of error entries: [{timestamp, window_title, error_text, keyword_matched}]
        """
        if not self.is_available():
            return []

        errors_found = []
        start_time = self._utc_minutes_ago(minutes_back)

        for keyword in self.ZIMMWRITER_KEYWORDS["errors"]:
            results = self._search(
                query=keyword,
                content_type="ocr",
                start_time=start_time,
                limit=5,
            )

            for r in results:
                content = r.get("content", {})
                window = content.get("window_name", "")

                # Filter to ZimmWriter windows
                if not any(zw.lower() in window.lower() for zw in self.ZIMMWRITER_WINDOWS):
                    continue

                errors_found.append({
                    "timestamp": content.get("timestamp", ""),
                    "window_title": window,
                    "error_text": content.get("text", "")[:300],
                    "keyword_matched": keyword,
                })

        # Deduplicate by timestamp
        seen = set()
        unique_errors = []
        for e in errors_found:
            key = f"{e['timestamp']}:{e['keyword_matched']}"
            if key not in seen:
                seen.add(key)
                unique_errors.append(e)

        logger.info(f"Screenpipe error search: {len(unique_errors)} errors in last {minutes_back}min")
        return unique_errors

    # ─────────────────────────────────────────────
    # PROGRESS TRACKING
    # ─────────────────────────────────────────────

    def start_tracking(self, job_id: str):
        """Mark the start of a job for progress tracking."""
        self._job_start_times[job_id] = self._utc_now()
        self._monitoring_active = True
        logger.info(f"Screenpipe tracking started for job {job_id}")

    def stop_tracking(self, job_id: str):
        """Stop tracking a job."""
        self._job_start_times.pop(job_id, None)
        if not self._job_start_times:
            self._monitoring_active = False

    def track_progress(self, job_id: str) -> Dict:
        """
        Read job progress from Screenpipe OCR captures.

        Returns:
            {
                "job_id": str,
                "in_progress": bool,
                "last_seen_title": str,
                "progress_text": str,
                "articles_detected": int,
                "errors_detected": int,
                "last_update": str,
            }
        """
        if not self.is_available():
            return {"job_id": job_id, "in_progress": False, "reason": "Screenpipe unavailable"}

        start_time = self._job_start_times.get(job_id)
        if not start_time:
            start_time = self._utc_minutes_ago(30)

        # Search for progress indicators
        progress_results = []
        for keyword in self.ZIMMWRITER_KEYWORDS["progress"]:
            results = self._search(
                query=keyword,
                content_type="ocr",
                start_time=start_time,
                limit=3,
            )
            progress_results.extend(results)

        # Search for completion indicators
        completion_results = []
        for keyword in self.ZIMMWRITER_KEYWORDS["completion"]:
            results = self._search(
                query=keyword,
                content_type="ocr",
                start_time=start_time,
                limit=2,
            )
            completion_results.extend(results)

        # Search for errors since job start
        error_count = 0
        for keyword in self.ZIMMWRITER_KEYWORDS["errors"][:5]:
            results = self._search(
                query=keyword,
                content_type="ocr",
                start_time=start_time,
                limit=3,
            )
            error_count += len([
                r for r in results
                if any(
                    zw.lower() in r.get("content", {}).get("window_name", "").lower()
                    for zw in self.ZIMMWRITER_WINDOWS
                )
            ])

        # Determine current state
        is_complete = len(completion_results) > 0
        last_title = ""
        progress_text = ""
        articles_detected = 0

        if progress_results:
            latest = sorted(
                progress_results,
                key=lambda r: r.get("content", {}).get("timestamp", ""),
                reverse=True,
            )[0]
            content = latest.get("content", {})
            last_title = content.get("window_name", "")
            progress_text = content.get("text", "")[:200]

            # Try to extract article count from OCR
            import re
            match = re.search(r"(\d+)\s*of\s*(\d+)", progress_text + " " + last_title)
            if match:
                articles_detected = int(match.group(1))

        result = {
            "job_id": job_id,
            "in_progress": bool(progress_results) and not is_complete,
            "is_complete": is_complete,
            "last_seen_title": last_title,
            "progress_text": progress_text,
            "articles_detected": articles_detected,
            "errors_detected": error_count,
            "last_update": (
                progress_results[0].get("content", {}).get("timestamp", "")
                if progress_results else ""
            ),
        }

        logger.debug(
            f"Screenpipe progress: job={job_id}, "
            f"in_progress={result['in_progress']}, articles={articles_detected}"
        )
        return result

    # ─────────────────────────────────────────────
    # HISTORICAL SEARCH
    # ─────────────────────────────────────────────

    def search_history(self, query: str, hours_back: int = 24,
                       limit: int = 20) -> List[Dict]:
        """
        Search historical Screenpipe captures for ZimmWriter-related content.

        Returns:
            List of matching entries with timestamps and OCR text.
        """
        if not self.is_available():
            return []

        start_time = (
            datetime.now(timezone.utc) - timedelta(hours=hours_back)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        results = self._search(
            query=query,
            content_type="ocr",
            start_time=start_time,
            limit=limit,
        )

        # Filter to ZimmWriter-relevant results
        filtered = []
        for r in results:
            content = r.get("content", {})
            window = content.get("window_name", "")
            app = content.get("app_name", "")

            if any(
                zw.lower() in (window + " " + app).lower()
                for zw in self.ZIMMWRITER_WINDOWS
            ):
                filtered.append({
                    "timestamp": content.get("timestamp", ""),
                    "window_title": window,
                    "app_name": app,
                    "text": content.get("text", "")[:500],
                })

        return filtered

    def get_activity_timeline(self, hours_back: int = 4) -> List[Dict]:
        """
        Build a timeline of ZimmWriter activity from Screenpipe data.

        Returns:
            Chronological list of state changes and events.
        """
        if not self.is_available():
            return []

        start_time = (
            datetime.now(timezone.utc) - timedelta(hours=hours_back)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Get all ZimmWriter OCR captures
        results = self._search(
            query="ZimmWriter",
            content_type="ocr",
            start_time=start_time,
            limit=100,
        )

        timeline = []
        prev_screen = None

        for r in results:
            content = r.get("content", {})
            window = content.get("window_name", "")
            text = content.get("text", "")
            ts = content.get("timestamp", "")

            screen = self._detect_screen(text, window)
            has_error = self._check_for_errors(text)

            # Only add if state changed or error detected
            if screen != prev_screen or has_error:
                entry = {
                    "timestamp": ts,
                    "screen": screen,
                    "window_title": window,
                    "has_error": has_error,
                }
                if has_error:
                    entry["error_excerpt"] = text[:200]
                timeline.append(entry)
                prev_screen = screen

        return timeline

    # ─────────────────────────────────────────────
    # CORRELATION WITH AUTOMATION
    # ─────────────────────────────────────────────

    def correlate_with_action(self, action_name: str,
                              action_timestamp: str,
                              window_seconds: int = 5) -> Dict:
        """
        Correlate an automation action with Screenpipe OCR captures
        around the same time. Useful for debugging action failures.

        Args:
            action_name: Name of the automation action
            action_timestamp: ISO timestamp when the action was executed
            window_seconds: How many seconds before/after to search

        Returns:
            {
                "action": str,
                "ocr_before": list of OCR captures before the action,
                "ocr_after": list of OCR captures after the action,
                "state_changed": bool,
                "errors_after": bool,
            }
        """
        if not self.is_available():
            return {"action": action_name, "reason": "Screenpipe unavailable"}

        # Parse timestamp and create before/after windows
        try:
            action_time = datetime.fromisoformat(action_timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            action_time = datetime.now(timezone.utc)

        before_start = (action_time - timedelta(seconds=window_seconds)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        after_end = (action_time + timedelta(seconds=window_seconds)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        action_iso = action_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Get OCR before action
        before_results = self._search(
            content_type="ocr",
            start_time=before_start,
            end_time=action_iso,
            limit=5,
        )

        # Get OCR after action
        after_results = self._search(
            content_type="ocr",
            start_time=action_iso,
            end_time=after_end,
            limit=5,
        )

        # Determine if state changed
        before_screen = ""
        after_screen = ""
        if before_results:
            bc = before_results[-1].get("content", {})
            before_screen = self._detect_screen(
                bc.get("text", ""), bc.get("window_name", "")
            )
        if after_results:
            ac = after_results[0].get("content", {})
            after_screen = self._detect_screen(
                ac.get("text", ""), ac.get("window_name", "")
            )

        errors_after = any(
            self._check_for_errors(r.get("content", {}).get("text", ""))
            for r in after_results
        )

        return {
            "action": action_name,
            "ocr_before": [
                {
                    "timestamp": r.get("content", {}).get("timestamp", ""),
                    "text": r.get("content", {}).get("text", "")[:200],
                }
                for r in before_results
            ],
            "ocr_after": [
                {
                    "timestamp": r.get("content", {}).get("timestamp", ""),
                    "text": r.get("content", {}).get("text", "")[:200],
                }
                for r in after_results
            ],
            "screen_before": before_screen,
            "screen_after": after_screen,
            "state_changed": before_screen != after_screen,
            "errors_after": errors_after,
        }

    def get_stats(self) -> Dict:
        """Get Screenpipe agent statistics."""
        return {
            "available": self.is_available(),
            "monitoring_active": self._monitoring_active,
            "active_jobs": len(self._job_start_times),
            "screenpipe_url": self.screenpipe_url,
        }
