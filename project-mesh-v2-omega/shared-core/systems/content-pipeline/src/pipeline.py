"""
content-pipeline -- Reusable content generation pipeline.
Extracted from zimmwriter-project-new/src/orchestrator.py.

Provides:
- JobSpec: specification for a content generation job
- ContentPipeline: sequential job runner with status tracking
- CampaignPlanner: intelligent job planning with article type detection

This is the abstract pipeline -- project-specific controllers
(ZimmWriter, WordPress, etc.) plug in via the executor callback.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

log = logging.getLogger(__name__)


@dataclass
class JobSpec:
    """Specification for a single content generation job."""
    domain: str
    titles: Optional[List[str]] = None
    csv_path: Optional[str] = None
    profile_name: Optional[str] = None
    wait: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """Result of a single job execution."""
    domain: str
    index: str = ""
    status: str = "unknown"
    started: str = ""
    finished: str = ""
    elapsed_seconds: int = 0
    source: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Article type detection keywords (from CampaignEngine patterns)
CONTENT_TYPE_KEYWORDS = {
    "how_to": ["how to", "guide", "setup", "install", "step by step",
               "beginner", "tutorial", "diy"],
    "review": ["review", "worth it", "honest", "vs", "versus",
               "comparison", "tested", "best"],
    "listicle": ["top", "best", "worst", "most", "reasons", "ways",
                 "things", "tips", "hacks", "mistakes"],
    "informational": ["what is", "explained", "understanding",
                      "history of", "meaning of", "why"],
    "news": ["just announced", "breaking", "new release", "update",
             "launched", "announced"],
}


def detect_content_type(title: str) -> str:
    """Detect article content type from title keywords.

    Returns one of: how_to, review, listicle, informational, news, general.
    """
    title_lower = title.lower()
    scores = {}
    for content_type, keywords in CONTENT_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in title_lower)
        if score > 0:
            scores[content_type] = score
    if scores:
        return max(scores, key=scores.get)
    return "general"


class ContentPipeline:
    """Sequential content generation pipeline with status tracking.

    Usage:
        pipeline = ContentPipeline(executor=my_run_function)
        pipeline.add_job("site.com", titles=["Title 1", "Title 2"])
        results = pipeline.run_all()

    The executor callback receives (JobSpec) and returns a bool for success.
    """

    def __init__(self, executor: Optional[Callable[[JobSpec], bool]] = None,
                 output_dir: Optional[str] = None):
        self.executor = executor
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.jobs: List[JobSpec] = []
        self.results: List[JobResult] = []
        self._skip_domains: set = set()
        self._current_index: int = 0

    def add_job(self, domain: str, titles: Optional[List[str]] = None,
                csv_path: Optional[str] = None,
                profile_name: Optional[str] = None,
                wait: bool = True, **metadata):
        """Add a job to the queue."""
        self.jobs.append(JobSpec(
            domain=domain, titles=titles, csv_path=csv_path,
            profile_name=profile_name, wait=wait, metadata=metadata,
        ))
        count = len(titles or [])
        log.info("Queued: %s (%d titles / CSV: %s)", domain, count, csv_path)

    def skip(self, domain: str):
        """Mark a domain to be skipped in the next run."""
        self._skip_domains.add(domain)
        log.info("Marked for skip: %s", domain)

    def get_queue_status(self) -> List[Dict]:
        """Return status of all jobs in the queue."""
        status = []
        for i, job in enumerate(self.jobs):
            if i < self._current_index:
                matching = [r for r in self.results
                            if r.domain == job.domain]
                s = matching[-1].status if matching else "completed"
            elif i == self._current_index:
                s = "in_progress"
            elif job.domain in self._skip_domains:
                s = "skipped"
            else:
                s = "pending"
            status.append({
                "index": i + 1, "domain": job.domain, "status": s
            })
        return status

    def run_all(self, delay_between: int = 10) -> List[JobResult]:
        """Run all queued jobs sequentially. Returns list of results."""
        self.results = []
        total = len(self.jobs)
        log.info("Starting %d jobs...", total)

        for i, job in enumerate(self.jobs, 1):
            self._current_index = i - 1

            if job.domain in self._skip_domains:
                log.info("=== Job %d/%d: %s -- SKIPPED ===", i, total,
                         job.domain)
                self.results.append(JobResult(
                    domain=job.domain,
                    index=f"{i}/{total}",
                    status="skipped",
                    started=datetime.now().isoformat(),
                    finished=datetime.now().isoformat(),
                ))
                continue

            log.info("=== Job %d/%d: %s ===", i, total, job.domain)
            result = self._run_single(job, i, total)
            self.results.append(result)

            if i < total and delay_between > 0:
                log.info("Waiting %ds before next job...", delay_between)
                time.sleep(delay_between)

        self._current_index = total
        success = sum(1 for r in self.results if r.status == "completed")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        log.info("=== DONE: %d/%d completed, %d skipped ===",
                 success, total, skipped)

        self._save_results()
        return self.results

    def _run_single(self, job: JobSpec, index: int,
                    total: int) -> JobResult:
        """Run a single job through the executor."""
        start = time.time()
        result = JobResult(
            domain=job.domain,
            index=f"{index}/{total}",
            started=datetime.now().isoformat(),
        )

        if self.executor is None:
            result.status = "error"
            result.error = "No executor configured"
            return result

        try:
            success = self.executor(job)
            result.status = "completed" if success else "failed"
            result.source = (
                f"{len(job.titles or [])} titles"
                if job.titles else f"CSV: {job.csv_path}"
            )
        except Exception as e:
            result.status = "error"
            result.error = str(e)
            log.error("Job failed for %s: %s", job.domain, e)

        result.elapsed_seconds = int(time.time() - start)
        result.finished = datetime.now().isoformat()
        return result

    def _save_results(self):
        """Save orchestration results to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"pipeline_{ts}.json"
        data = [
            {k: v for k, v in r.__dict__.items() if v is not None}
            for r in self.results
        ]
        filepath.write_text(json.dumps(data, indent=2, default=str),
                            encoding="utf-8")
        log.info("Results saved: %s", filepath)
