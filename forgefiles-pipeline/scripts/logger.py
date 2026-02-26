#!/usr/bin/env python3
"""
ForgeFiles Pipeline Logger
============================
Production logging with file + console output, per-model tracking,
batch progress reporting, and structured JSON log entries.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PIPELINE_ROOT / "logs"


class PipelineFormatter(logging.Formatter):
    """Clean console formatter with stage prefixes."""

    STAGE_COLORS = {
        "render": "\033[36m",     # cyan
        "composite": "\033[35m",  # magenta
        "caption": "\033[33m",    # yellow
        "thumbnail": "\033[32m",  # green
        "analyze": "\033[34m",    # blue
        "pipeline": "\033[97m",   # bright white
        "brand": "\033[93m",      # bright yellow
    }
    RESET = "\033[0m"
    LEVEL_COLORS = {
        "DEBUG": "\033[90m",
        "INFO": "",
        "WARNING": "\033[33m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[91;1m",
    }

    def __init__(self, use_color=True):
        super().__init__()
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record):
        stage = getattr(record, "stage", "pipeline")
        if self.use_color:
            stage_color = self.STAGE_COLORS.get(stage, "")
            level_color = self.LEVEL_COLORS.get(record.levelname, "")
            prefix = f"{stage_color}[{stage.upper():>10}]{self.RESET}"
            if record.levelno >= logging.WARNING:
                msg = f"{prefix} {level_color}{record.levelname}: {record.getMessage()}{self.RESET}"
            else:
                msg = f"{prefix} {record.getMessage()}"
        else:
            msg = f"[{stage.upper():>10}] {record.getMessage()}"
        return msg


class JsonFileHandler(logging.Handler):
    """Writes structured JSON log lines to a file."""

    def __init__(self, filepath):
        super().__init__()
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "stage": getattr(record, "stage", "pipeline"),
            "model": getattr(record, "model_name", None),
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_data", None)
        if extra:
            entry["data"] = extra
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


class BatchProgress:
    """Tracks batch processing progress with ETA estimation."""

    def __init__(self, total_items, logger=None):
        self.total = total_items
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.item_times = []
        self.logger = logger

    def item_started(self, name):
        self._item_start = time.time()
        self._item_name = name
        if self.logger:
            self.logger.info(
                f"[{self.completed + 1}/{self.total}] Starting: {name}",
                extra={"stage": "pipeline"}
            )

    def item_completed(self, name, success=True):
        elapsed = time.time() - self._item_start
        self.item_times.append(elapsed)
        if success:
            self.completed += 1
        else:
            self.failed += 1
        if self.logger:
            remaining = self.total - self.completed - self.failed
            eta = self._estimate_remaining(remaining)
            eta_str = str(timedelta(seconds=int(eta))) if eta else "unknown"
            self.logger.info(
                f"[{self.completed + self.failed}/{self.total}] "
                f"{'Done' if success else 'FAILED'}: {name} "
                f"({elapsed:.1f}s) | Remaining: {remaining} | ETA: {eta_str}",
                extra={"stage": "pipeline"}
            )

    def _estimate_remaining(self, remaining):
        if not self.item_times:
            return None
        avg = sum(self.item_times) / len(self.item_times)
        return avg * remaining

    def summary(self):
        total_time = time.time() - self.start_time
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "total_time_seconds": round(total_time, 1),
            "average_time_per_item": round(
                sum(self.item_times) / len(self.item_times), 1
            ) if self.item_times else 0,
        }


def get_logger(name="forgefiles", log_file=None, level=logging.INFO):
    """Get or create a pipeline logger with console + file output."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(PipelineFormatter(use_color=True))
    console.setLevel(level)
    logger.addHandler(console)

    if log_file is None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"pipeline_{timestamp}.jsonl"

    json_handler = JsonFileHandler(log_file)
    json_handler.setLevel(logging.DEBUG)
    logger.addHandler(json_handler)

    return logger


def log_stage(logger, stage, message, level=logging.INFO, model_name=None, **extra):
    """Log a message with stage context."""
    extra_dict = extra if extra else None
    logger.log(
        level, message,
        extra={"stage": stage, "model_name": model_name, "extra_data": extra_dict}
    )
