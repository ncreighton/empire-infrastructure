"""Base step class for cascade pipeline."""

import logging
import time
from typing import Dict, Optional

log = logging.getLogger(__name__)


class BaseStep:
    """Base class for all cascade steps."""

    name: str = "base"
    description: str = "Base step"
    requires: list = []  # Step names that must complete first

    def execute(self, context: Dict) -> Dict:
        """
        Execute this step.

        Args:
            context: Dict with keys from previous steps + initial config.
                Always contains: site_slug, title, cascade_id

        Returns:
            Dict of output data to merge into context for subsequent steps.
        """
        raise NotImplementedError

    def dry_run(self, context: Dict) -> Dict:
        """Preview what this step would do without executing."""
        return {
            "step": self.name,
            "description": self.description,
            "would_use": self.requires,
            "status": "dry_run",
        }

    def _timed_execute(self, context: Dict) -> tuple:
        """Execute with timing. Returns (result, duration_ms)."""
        start = time.time()
        try:
            result = self.execute(context)
            duration = int((time.time() - start) * 1000)
            return result, duration
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            log.error(f"Step {self.name} failed: {e}")
            raise
