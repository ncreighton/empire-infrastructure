"""
Progress monitoring for ZimmWriter jobs.
Tracks window title changes, estimates time remaining, and can trigger notifications.
"""

import time
import re
import json
from datetime import datetime
from typing import Optional, Callable, Dict, List
from pathlib import Path
from .utils import setup_logger, ensure_output_dir

logger = setup_logger("monitor")


class JobMonitor:
    """Monitor ZimmWriter bulk generation progress."""

    def __init__(self, controller):
        self.controller = controller
        self.log: List[Dict] = []
        self.start_time: Optional[float] = None
        self.total_articles: Optional[int] = None

    def start(self, total_articles: int = None):
        """Begin monitoring a job."""
        self.start_time = time.time()
        self.total_articles = total_articles
        self.log = []
        logger.info(f"Monitoring started. Articles: {total_articles or 'unknown'}")

    def check(self) -> Dict:
        """Check current progress and return status dict."""
        title = self.controller.get_window_title()
        elapsed = time.time() - (self.start_time or time.time())

        # Try to parse progress from window title (ZimmWriter often shows "X of Y")
        progress = self._parse_progress(title)

        status = {
            "timestamp": datetime.now().isoformat(),
            "window_title": title,
            "elapsed_seconds": int(elapsed),
            "elapsed_human": self._format_time(elapsed),
            "completed": progress.get("completed", 0),
            "total": progress.get("total", self.total_articles),
            "is_finished": self._is_finished(title),
        }

        # Estimate remaining time
        if status["completed"] > 0 and status["total"]:
            rate = elapsed / status["completed"]
            remaining = rate * (status["total"] - status["completed"])
            status["eta_seconds"] = int(remaining)
            status["eta_human"] = self._format_time(remaining)

        self.log.append(status)
        return status

    def wait_until_done(self, check_interval: int = 30, timeout: int = 7200,
                         on_progress: Callable = None, on_complete: Callable = None) -> bool:
        """
        Block until job completes.
        on_progress(status_dict) called each interval.
        on_complete(status_dict) called when done.
        """
        while True:
            status = self.check()
            logger.info(
                f"[{status['elapsed_human']}] "
                f"{status['completed']}/{status.get('total', '?')} articles | "
                f"{status['window_title'][:60]}"
            )

            if on_progress:
                on_progress(status)

            if status["is_finished"]:
                logger.info(f"✅ Job complete in {status['elapsed_human']}")
                if on_complete:
                    on_complete(status)
                return True

            if status["elapsed_seconds"] > timeout:
                logger.warning(f"⏰ Timeout after {status['elapsed_human']}")
                return False

            time.sleep(check_interval)

    def save_log(self, filepath: str = None) -> str:
        """Save monitoring log to JSON."""
        if not filepath:
            filepath = str(ensure_output_dir() / f"job_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filepath, "w") as f:
            json.dump(self.log, f, indent=2)
        logger.info(f"Log saved: {filepath}")
        return filepath

    def _parse_progress(self, title: str) -> Dict:
        """Extract progress numbers from window title."""
        # Common patterns: "3 of 10", "3/10", "[3/10]"
        patterns = [
            r"(\d+)\s*of\s*(\d+)",
            r"(\d+)\s*/\s*(\d+)",
            r"\[(\d+)/(\d+)\]",
        ]
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                return {"completed": int(match.group(1)), "total": int(match.group(2))}
        return {}

    def _is_finished(self, title: str) -> bool:
        """Check if the window title indicates completion."""
        keywords = ["complete", "finished", "done", "all articles"]
        return any(kw in title.lower() for kw in keywords)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable."""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        return f"{s}s"
