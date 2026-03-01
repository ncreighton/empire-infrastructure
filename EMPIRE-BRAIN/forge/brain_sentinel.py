"""BrainSentinel — Empire Health Monitor

Monitors the entire empire for:
- Service health (ports, APIs, uptime)
- Project staleness (days since last change)
- Code quality drift (compliance scoring)
- Security concerns (exposed credentials, outdated deps)
- Performance anomalies
"""
import json
import httpx
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB
from config.settings import SERVICES, EMPIRE_ROOT


class BrainSentinel:
    """Monitors empire health and raises alerts."""

    SEVERITY_WEIGHTS = {"critical": 4, "high": 3, "warning": 2, "info": 1}

    def __init__(self, db: Optional[BrainDB] = None):
        self.db = db or BrainDB()
        self.alerts = []

    def full_health_check(self) -> dict:
        """Run comprehensive health check across the empire."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "services": self.check_services(),
            "projects": self.check_project_health(),
            "alerts": self.alerts,
            "overall_score": 0,
        }
        # Calculate overall score
        total = len(report["services"]) + len(report["projects"])
        healthy = sum(1 for s in report["services"].values() if s.get("status") == "up")
        healthy += sum(1 for p in report["projects"] if p.get("health_score", 0) >= 70)
        report["overall_score"] = round((healthy / max(total, 1)) * 100, 1)

        self.db.emit_event("sentinel.health_check", {
            "overall_score": report["overall_score"],
            "alerts_count": len(self.alerts),
            "services_up": healthy,
        })
        return report

    def check_services(self) -> dict:
        """Check health of all registered services."""
        results = {}
        for name, config in SERVICES.items():
            port = config["port"]
            health_path = config.get("health", "/health")
            url = f"http://localhost:{port}{health_path}"
            try:
                resp = httpx.get(url, timeout=5.0)
                results[name] = {
                    "status": "up" if resp.status_code < 400 else "degraded",
                    "port": port,
                    "response_time_ms": round(resp.elapsed.total_seconds() * 1000),
                    "status_code": resp.status_code,
                }
            except Exception as e:
                results[name] = {
                    "status": "down",
                    "port": port,
                    "error": str(e)[:200],
                }
                self.alerts.append({
                    "severity": "warning",
                    "source": "sentinel",
                    "message": f"Service '{name}' on port {port} is DOWN",
                    "timestamp": datetime.now().isoformat(),
                })
        return results

    def check_project_health(self) -> list[dict]:
        """Assess health of all projects."""
        projects = self.db.get_projects()
        results = []

        for proj in projects:
            path = Path(proj.get("path", ""))
            if not path.exists():
                continue

            health = {
                "slug": proj["slug"],
                "health_score": 100,
                "issues": [],
            }

            # Check staleness
            try:
                mtime = max(f.stat().st_mtime for f in path.rglob("*.py") if f.is_file() and not any(p in IGNORE_DIRS for p in f.parts))
                last_mod = datetime.fromtimestamp(mtime)
                days_stale = (datetime.now() - last_mod).days
                if days_stale > 30:
                    health["health_score"] -= 20
                    health["issues"].append(f"Stale: {days_stale} days since last Python change")
                elif days_stale > 14:
                    health["health_score"] -= 10
                    health["issues"].append(f"Aging: {days_stale} days since last Python change")
            except (ValueError, StopIteration):
                pass

            # Check for CLAUDE.md
            if not (path / "CLAUDE.md").exists():
                health["health_score"] -= 15
                health["issues"].append("Missing CLAUDE.md")

            # Check for tests
            has_tests = any(path.rglob("test_*.py")) or (path / "tests").exists()
            if not has_tests:
                health["health_score"] -= 10
                health["issues"].append("No tests found")

            # Check for requirements.txt or package.json
            has_deps = (path / "requirements.txt").exists() or (path / "package.json").exists()
            if not has_deps:
                health["health_score"] -= 5
                health["issues"].append("No dependency file")

            health["health_score"] = max(0, health["health_score"])

            # Update DB
            self.db.upsert_project({
                "slug": proj["slug"],
                "health_score": health["health_score"],
            })

            # Generate alerts for unhealthy projects
            if health["health_score"] < 50:
                self.alerts.append({
                    "severity": "warning",
                    "source": "sentinel",
                    "message": f"Project '{proj['slug']}' health score: {health['health_score']}/100 — {'; '.join(health['issues'])}",
                    "timestamp": datetime.now().isoformat(),
                })

            results.append(health)

        return results

    def detect_anomalies(self) -> list[dict]:
        """Detect unusual patterns in empire events."""
        anomalies = []
        conn = self.db._conn()

        # Check for sudden spikes in events
        recent = conn.execute("""
            SELECT event_type, COUNT(*) as cnt
            FROM events
            WHERE timestamp > datetime('now', '-1 hour')
            GROUP BY event_type
            HAVING cnt > 100
        """).fetchall()

        for row in recent:
            anomalies.append({
                "type": "event_spike",
                "detail": f"Event '{row['event_type']}' fired {row['cnt']} times in last hour",
                "severity": "warning",
            })

        # Check for projects that lost functions (possible breaking changes)
        # This would compare current vs historical scans
        conn.close()

        return anomalies

    def compliance_check(self) -> list[dict]:
        """Check compliance across all projects."""
        results = []
        projects = self.db.get_projects()

        for proj in projects:
            path = Path(proj.get("path", ""))
            if not path.exists():
                continue

            score = 100
            violations = []

            # Rule: No hardcoded API keys
            for py_file in path.rglob("*.py"):
                if any(part in IGNORE_DIRS for part in py_file.parts):
                    continue
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                    # Check for common API key patterns (but not env var lookups)
                    if ('api_key = "' in content.lower() or "api_key = '" in content.lower()) and \
                       "os.environ" not in content and "os.getenv" not in content:
                        score -= 20
                        violations.append(f"Possible hardcoded API key in {py_file.name}")
                        break
                except Exception:
                    pass

            # Rule: Has .gitignore
            if not (path / ".gitignore").exists():
                score -= 5
                violations.append("Missing .gitignore")

            results.append({
                "slug": proj["slug"],
                "compliance_score": max(0, score),
                "violations": violations,
            })

            self.db.upsert_project({
                "slug": proj["slug"],
                "compliance_score": max(0, score),
            })

        return results


# Import at module level for the check
from config.settings import IGNORE_DIRS
