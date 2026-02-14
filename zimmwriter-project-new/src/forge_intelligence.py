"""
FORGE Intelligence Engine — ZimmWriter Desktop Automation

Five specialized modules that continuously learn and improve the ZimmWriter
automation system. Each module handles a distinct responsibility:

  Scout    — Detects gaps in UI configuration before jobs start
  Sentinel — Optimizes prompts sent to Vision/AI analysis endpoints
  Oracle   — Predicts automation failures using historical patterns
  Smith    — Generates automatic fixes for known configuration issues
  Codex    — Persistent memory that learns from every job execution

The engine runs passively alongside the pywinauto controller. It reads
controller state, Vision Service results, and Screenpipe OCR data to build
an increasingly accurate model of ZimmWriter's behavior.

Usage:
    from src.forge_intelligence import ForgeEngine
    forge = ForgeEngine()
    # Before a job:
    issues = forge.scout.audit_config(site_config)
    fixes = forge.smith.auto_fix(issues)
    # During a job:
    prediction = forge.oracle.predict_failure(job_state)
    # After a job:
    forge.codex.record_outcome(job_id, outcome)
"""

import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .utils import setup_logger

logger = setup_logger("forge")

# Persistent storage for learning data
FORGE_DATA_DIR = Path(__file__).parent.parent / "data" / "forge"
FORGE_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# SCOUT — Configuration Gap Detection
# ═══════════════════════════════════════════════════════════

class Scout:
    """
    Inspects site presets, controller state, and job parameters to find
    configuration gaps BEFORE a job starts. Catches missing fields,
    incompatible settings, and known-bad combinations.
    """

    # Required fields for a valid Bulk Writer job
    REQUIRED_DROPDOWN_FIELDS = [
        "section_length", "voice", "ai_model", "h2_count",
    ]

    # Known bad setting combinations (discovered through past failures)
    INCOMPATIBLE_COMBOS = [
        {
            "condition": {"ai_model": "GPT-3.5 Turbo", "deep_research_settings": True},
            "reason": "GPT-3.5 Turbo cannot handle Deep Research — use GPT-4o or Claude",
            "severity": "error",
        },
        {
            "condition": {"section_length": "Very Short", "enable_h3": True},
            "reason": "Very Short sections rarely produce meaningful H3 subheadings",
            "severity": "warning",
        },
        {
            "condition": {"faq": "Disabled", "nuke_ai_words": False},
            "reason": "Without FAQ and Nuke AI Words, output may trigger AI detection",
            "severity": "warning",
        },
    ]

    # WordPress fields needed for upload
    WP_REQUIRED = ["site_url", "user_name"]

    def __init__(self, codex: "Codex" = None):
        self.codex = codex
        self._custom_rules: List[Dict] = []
        self._load_custom_rules()

    def _load_custom_rules(self):
        """Load user-defined audit rules from persistent storage."""
        rules_path = FORGE_DATA_DIR / "scout_rules.json"
        if rules_path.exists():
            try:
                self._custom_rules = json.loads(rules_path.read_text(encoding="utf-8"))
            except Exception:
                self._custom_rules = []

    def add_rule(self, condition: Dict, reason: str, severity: str = "warning"):
        """Add a custom validation rule that persists across sessions."""
        self._custom_rules.append({
            "condition": condition,
            "reason": reason,
            "severity": severity,
            "added": datetime.now().isoformat(),
        })
        rules_path = FORGE_DATA_DIR / "scout_rules.json"
        rules_path.write_text(json.dumps(self._custom_rules, indent=2), encoding="utf-8")

    def audit_config(self, config: Dict[str, Any]) -> List[Dict]:
        """
        Full audit of a site configuration dict. Returns list of issues:
        [{"field": ..., "issue": ..., "severity": "error"|"warning"|"info", "fix_hint": ...}]
        """
        issues = []

        # 1. Check required dropdowns
        for field in self.REQUIRED_DROPDOWN_FIELDS:
            val = config.get(field)
            if not val or val in ("", "Select", "None"):
                issues.append({
                    "field": field,
                    "issue": f"Required dropdown '{field}' is not set",
                    "severity": "error",
                    "fix_hint": f"Set {field} to a valid value",
                })

        # 2. Check WordPress settings if WP is enabled
        wp = config.get("wordpress_settings")
        if wp:
            for wf in self.WP_REQUIRED:
                if not wp.get(wf):
                    issues.append({
                        "field": f"wordpress_settings.{wf}",
                        "issue": f"WordPress field '{wf}' is missing",
                        "severity": "error",
                        "fix_hint": f"Add {wf} to wordpress_settings",
                    })
            # Safety: check article_status
            if wp.get("article_status", "").lower() == "publish":
                issues.append({
                    "field": "wordpress_settings.article_status",
                    "issue": "Article status set to 'publish' — articles go live immediately",
                    "severity": "warning",
                    "fix_hint": "Consider using 'draft' for safety",
                })

        # 3. Check known-bad combos
        for combo in self.INCOMPATIBLE_COMBOS + self._custom_rules:
            cond = combo["condition"]
            match = True
            for key, expected in cond.items():
                actual = config.get(key)
                if isinstance(expected, bool):
                    if expected and not actual:
                        match = False
                        break
                    if not expected and actual:
                        match = False
                        break
                elif actual != expected:
                    match = False
                    break
            if match:
                issues.append({
                    "field": ", ".join(cond.keys()),
                    "issue": combo["reason"],
                    "severity": combo.get("severity", "warning"),
                    "fix_hint": combo.get("fix_hint", ""),
                })

        # 4. Check titles aren't empty or duplicated
        titles = config.get("titles", [])
        if titles:
            seen = set()
            for t in titles:
                if t.strip() in seen:
                    issues.append({
                        "field": "titles",
                        "issue": f"Duplicate title: '{t[:50]}'",
                        "severity": "warning",
                        "fix_hint": "Remove duplicate titles",
                    })
                seen.add(t.strip())

        # 5. Codex-informed checks (patterns from past failures)
        if self.codex:
            domain = config.get("domain", "")
            failure_patterns = self.codex.get_failure_patterns(domain)
            for pattern in failure_patterns:
                field = pattern.get("field", "")
                if field and config.get(field) == pattern.get("bad_value"):
                    issues.append({
                        "field": field,
                        "issue": f"Codex: '{field}={pattern['bad_value']}' caused failures "
                                 f"{pattern['count']} times for {domain}",
                        "severity": "warning",
                        "fix_hint": pattern.get("suggested_fix", ""),
                    })

        logger.info(f"Scout audit: {len(issues)} issues found")
        return issues

    def audit_ui_state(self, controller_status: Dict) -> List[Dict]:
        """
        Audit the live UI state from controller.get_status().
        Checks for dropdown mismatches, unexpected checkbox states, etc.
        """
        issues = []

        # Check if any dropdowns show "Select" (unset)
        for dd in controller_status.get("dropdowns", []):
            selected = dd.get("selected", "")
            if selected in ("Select", "", "None") and dd.get("visible"):
                issues.append({
                    "field": dd.get("name", dd.get("auto_id", "unknown")),
                    "issue": f"Dropdown '{dd.get('name', '')}' is unset (shows '{selected}')",
                    "severity": "info",
                    "fix_hint": "Select a value before starting",
                })

        return issues


# ═══════════════════════════════════════════════════════════
# SENTINEL — Prompt & Analysis Optimization
# ═══════════════════════════════════════════════════════════

class Sentinel:
    """
    Optimizes prompts sent to the Vision Service and AI analysis endpoints.
    Learns which prompt patterns produce the best results for ZimmWriter
    UI analysis and adapts over time.
    """

    # Base prompt templates for Vision Service analysis
    VISION_PROMPTS = {
        "verify_screen": (
            "Analyze this ZimmWriter screenshot. Identify which screen is shown "
            "(Bulk Writer, Options Menu, Config Window, or other). "
            "List all visible UI elements: buttons, dropdowns, checkboxes, text fields. "
            "Report any error dialogs or popup windows."
        ),
        "verify_config": (
            "This is a ZimmWriter configuration window screenshot. "
            "Read ALL dropdown selections, checkbox states, and text field values. "
            "Report the exact current state of every visible control. "
            "Flag any fields that appear empty or set to default values."
        ),
        "detect_errors": (
            "Examine this ZimmWriter screenshot for any error conditions: "
            "error dialogs, warning popups, red text, disabled buttons that should be enabled, "
            "or any visual indication of a problem. "
            "Report: error_found (true/false), error_type, error_message, suggested_action."
        ),
        "verify_progress": (
            "This is a ZimmWriter Bulk Writer screenshot during content generation. "
            "Read the window title and any progress indicators. "
            "Report: current_article_number, total_articles, estimated_progress_percent, "
            "any error or warning messages visible."
        ),
        "compare_states": (
            "Compare these two ZimmWriter screenshots (before and after an action). "
            "Identify every difference: changed dropdown values, toggled checkboxes, "
            "new/closed windows, text changes, button state changes. "
            "Confirm whether the intended action was successful."
        ),
    }

    def __init__(self, codex: "Codex" = None):
        self.codex = codex
        self._prompt_scores: Dict[str, List[float]] = defaultdict(list)
        self._load_scores()

    def _load_scores(self):
        """Load prompt performance scores from persistent storage."""
        scores_path = FORGE_DATA_DIR / "sentinel_scores.json"
        if scores_path.exists():
            try:
                data = json.loads(scores_path.read_text(encoding="utf-8"))
                self._prompt_scores = defaultdict(list, {k: v for k, v in data.items()})
            except Exception:
                pass

    def _save_scores(self):
        """Persist prompt performance scores."""
        scores_path = FORGE_DATA_DIR / "sentinel_scores.json"
        scores_path.write_text(
            json.dumps(dict(self._prompt_scores), indent=2), encoding="utf-8"
        )

    def get_prompt(self, task: str, context: Dict = None) -> str:
        """
        Get the best prompt for a given Vision analysis task.
        Enriches the base template with contextual details.

        Args:
            task: One of VISION_PROMPTS keys or a custom task name
            context: Optional dict with {screen_name, expected_state, action_taken, site_domain}
        """
        base = self.VISION_PROMPTS.get(task, self.VISION_PROMPTS["verify_screen"])
        context = context or {}

        # Enrich with context
        enrichments = []
        if context.get("screen_name"):
            enrichments.append(f"Expected screen: {context['screen_name']}.")
        if context.get("expected_state"):
            enrichments.append(f"Expected state: {context['expected_state']}.")
        if context.get("action_taken"):
            enrichments.append(f"Action just taken: {context['action_taken']}.")
        if context.get("site_domain"):
            enrichments.append(f"Site being configured: {context['site_domain']}.")

        # Add Codex learnings if available
        if self.codex:
            tips = self.codex.get_vision_tips(task)
            if tips:
                enrichments.append(f"Known issues to watch for: {'; '.join(tips)}.")

        if enrichments:
            return base + "\n\nAdditional context:\n" + "\n".join(enrichments)
        return base

    def record_result(self, task: str, prompt: str, quality_score: float):
        """
        Record the quality of a Vision analysis result (0.0 - 1.0).
        Used to track which prompt variations perform best.
        """
        key = hashlib.md5(f"{task}:{prompt[:100]}".encode()).hexdigest()[:12]
        self._prompt_scores[key].append(quality_score)
        # Keep last 50 scores per prompt variant
        self._prompt_scores[key] = self._prompt_scores[key][-50:]
        self._save_scores()

    def enhance_prompt(self, raw_prompt: str, task_type: str = "general") -> str:
        """
        Apply the AMPLIFY-style enhancement to any prompt before sending
        it to an AI service. Adds structure, specificity, and output format.
        """
        enhanced = (
            f"{raw_prompt}\n\n"
            f"Respond in JSON format with these fields:\n"
            f"- success: boolean\n"
            f"- confidence: float (0.0-1.0)\n"
            f"- findings: list of strings\n"
            f"- errors: list of {{type, message, severity}}\n"
            f"- recommendations: list of strings"
        )
        return enhanced


# ═══════════════════════════════════════════════════════════
# ORACLE — Failure Prediction
# ═══════════════════════════════════════════════════════════

class Oracle:
    """
    Predicts automation failures before they happen using historical
    patterns from the Codex. Assigns risk scores to configurations
    and suggests preventive measures.
    """

    # Known risk factors with base weights
    RISK_FACTORS = {
        "config_window_interaction": 0.15,    # Config windows are fragile
        "file_dialog_interaction": 0.20,       # File dialogs can hang
        "wordpress_upload_enabled": 0.10,      # Network dependency
        "deep_research_enabled": 0.12,         # Slow, can timeout
        "many_titles": 0.08,                   # >10 titles increase failure chance
        "first_run_for_site": 0.15,            # No historical data
        "serp_scraping_enabled": 0.10,         # External API dependency
        "style_mimic_enabled": 0.05,           # Minor complexity
    }

    def __init__(self, codex: "Codex" = None):
        self.codex = codex

    def predict_failure(self, config: Dict, job_state: Dict = None) -> Dict:
        """
        Analyze a job configuration and return a risk assessment.

        Returns:
            {
                "risk_score": float (0.0-1.0),
                "risk_level": "low"|"medium"|"high"|"critical",
                "risk_factors": [{factor, weight, reason}],
                "preventive_actions": [str],
                "estimated_duration_minutes": int,
            }
        """
        factors = []
        total_risk = 0.0

        # Assess each risk factor
        if config.get("wordpress_settings"):
            w = self.RISK_FACTORS["wordpress_upload_enabled"]
            factors.append({
                "factor": "wordpress_upload",
                "weight": w,
                "reason": "WordPress uploads depend on network + REST/XML-RPC availability",
            })
            total_risk += w

        if config.get("deep_research_settings") or config.get("deep_research"):
            w = self.RISK_FACTORS["deep_research_enabled"]
            factors.append({
                "factor": "deep_research",
                "weight": w,
                "reason": "Deep Research adds significant processing time and web scraping",
            })
            total_risk += w

        if config.get("serp_settings") or config.get("serp_scraping"):
            w = self.RISK_FACTORS["serp_scraping_enabled"]
            factors.append({
                "factor": "serp_scraping",
                "weight": w,
                "reason": "SERP scraping depends on external search APIs",
            })
            total_risk += w

        # Check title count
        titles = config.get("titles", [])
        if len(titles) > 10:
            w = self.RISK_FACTORS["many_titles"]
            factors.append({
                "factor": "many_titles",
                "weight": w,
                "reason": f"{len(titles)} titles — longer jobs have higher failure rate",
            })
            total_risk += w

        # Check if this is first run for this domain
        domain = config.get("domain", "")
        if self.codex and domain:
            history = self.codex.get_domain_history(domain)
            if not history:
                w = self.RISK_FACTORS["first_run_for_site"]
                factors.append({
                    "factor": "first_run",
                    "weight": w,
                    "reason": f"No historical data for {domain} — higher uncertainty",
                })
                total_risk += w
            else:
                # Adjust risk based on historical success rate
                success_rate = history.get("success_rate", 0.5)
                if success_rate < 0.7:
                    adjustment = (1.0 - success_rate) * 0.3
                    factors.append({
                        "factor": "low_success_rate",
                        "weight": adjustment,
                        "reason": f"Historical success rate for {domain}: {success_rate:.0%}",
                    })
                    total_risk += adjustment

        # Clamp to [0, 1]
        total_risk = min(1.0, total_risk)

        # Determine risk level
        if total_risk < 0.15:
            level = "low"
        elif total_risk < 0.35:
            level = "medium"
        elif total_risk < 0.60:
            level = "high"
        else:
            level = "critical"

        # Generate preventive actions
        actions = []
        if any(f["factor"] == "wordpress_upload" for f in factors):
            actions.append("Verify WordPress REST API is accessible before starting")
            actions.append("Use article_status='draft' to prevent accidental publishing")
        if any(f["factor"] == "deep_research" for f in factors):
            actions.append("Increase job timeout to 30+ minutes per article")
            actions.append("Take pre-start screenshot for comparison")
        if any(f["factor"] == "first_run" for f in factors):
            actions.append("Run with a single test title first")
            actions.append("Enable progress monitoring with vision verification")
        if any(f["factor"] == "many_titles" for f in factors):
            actions.append("Consider splitting into smaller batches of 5-10 titles")

        # Estimate duration
        base_minutes = 5  # base per article
        title_count = max(len(titles), 1)
        if config.get("deep_research_settings"):
            base_minutes += 8
        if config.get("serp_settings"):
            base_minutes += 3
        est_duration = base_minutes * title_count

        result = {
            "risk_score": round(total_risk, 3),
            "risk_level": level,
            "risk_factors": factors,
            "preventive_actions": actions,
            "estimated_duration_minutes": est_duration,
        }

        logger.info(
            f"Oracle prediction: risk={total_risk:.2f} ({level}), "
            f"~{est_duration}min for {title_count} articles"
        )
        return result


# ═══════════════════════════════════════════════════════════
# SMITH — Automatic Fix Generation
# ═══════════════════════════════════════════════════════════

class Smith:
    """
    Generates automatic fixes for known configuration issues.
    Takes Scout audit results and produces actionable fix operations
    that the controller can execute.
    """

    # Fix strategies mapped to issue patterns
    FIX_STRATEGIES = {
        "required_dropdown_missing": {
            "action": "set_dropdown",
            "defaults": {
                "section_length": "Medium",
                "voice": "Third Person",
                "ai_model": "GPT-4o",
                "h2_count": "Auto",
            },
        },
        "wordpress_status_publish": {
            "action": "override_field",
            "field": "wordpress_settings.article_status",
            "value": "draft",
        },
        "duplicate_title": {
            "action": "deduplicate_titles",
        },
    }

    def __init__(self, codex: "Codex" = None):
        self.codex = codex

    def auto_fix(self, issues: List[Dict]) -> List[Dict]:
        """
        Generate fix operations for a list of Scout issues.

        Returns list of fix operations:
        [{"action": "set_dropdown", "field": ..., "value": ..., "reason": ...}]
        """
        fixes = []

        for issue in issues:
            field = issue.get("field", "")
            severity = issue.get("severity", "info")

            # Only auto-fix errors and warnings, not info
            if severity == "info":
                continue

            fix = self._generate_fix(field, issue)
            if fix:
                fixes.append(fix)

        logger.info(f"Smith generated {len(fixes)} fixes for {len(issues)} issues")
        return fixes

    def _generate_fix(self, field: str, issue: Dict) -> Optional[Dict]:
        """Generate a single fix operation."""
        issue_text = issue.get("issue", "")

        # Missing required dropdown
        if "Required dropdown" in issue_text and "not set" in issue_text:
            defaults = self.FIX_STRATEGIES["required_dropdown_missing"]["defaults"]
            if field in defaults:
                # Check Codex for domain-specific preference
                preferred = None
                if self.codex:
                    preferred = self.codex.get_preferred_value(field)

                value = preferred or defaults[field]
                return {
                    "action": "set_dropdown",
                    "field": field,
                    "value": value,
                    "reason": f"Auto-set '{field}' to '{value}' (was unset)",
                    "source": "codex" if preferred else "default",
                }

        # WordPress publish safety
        if "article_status" in field and "publish" in issue_text.lower():
            return {
                "action": "override_field",
                "field": "wordpress_settings.article_status",
                "value": "draft",
                "reason": "Safety override: changed publish to draft",
                "source": "safety",
            }

        # Duplicate titles
        if "Duplicate title" in issue_text:
            return {
                "action": "deduplicate_titles",
                "field": "titles",
                "value": None,
                "reason": "Remove duplicate titles from input list",
                "source": "cleanup",
            }

        return None

    def apply_fixes(self, config: Dict, fixes: List[Dict]) -> Dict:
        """
        Apply fix operations to a config dict and return the corrected config.
        Does NOT modify the original — returns a new dict.
        """
        fixed = json.loads(json.dumps(config))  # deep copy

        for fix in fixes:
            action = fix.get("action")
            field = fix.get("field", "")
            value = fix.get("value")

            if action == "set_dropdown":
                fixed[field] = value

            elif action == "override_field":
                # Handle nested fields like "wordpress_settings.article_status"
                parts = field.split(".")
                target = fixed
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value

            elif action == "deduplicate_titles":
                titles = fixed.get("titles", [])
                seen = set()
                unique = []
                for t in titles:
                    if t.strip() not in seen:
                        unique.append(t)
                        seen.add(t.strip())
                fixed["titles"] = unique

        return fixed


# ═══════════════════════════════════════════════════════════
# CODEX — Persistent Learning Memory
# ═══════════════════════════════════════════════════════════

class Codex:
    """
    Persistent memory that learns from every ZimmWriter job execution.
    Stores outcomes, failure patterns, successful configurations, and
    vision analysis results. All data persists to disk as JSON.
    """

    def __init__(self):
        self._jobs_path = FORGE_DATA_DIR / "codex_jobs.json"
        self._patterns_path = FORGE_DATA_DIR / "codex_patterns.json"
        self._preferences_path = FORGE_DATA_DIR / "codex_preferences.json"
        self._vision_tips_path = FORGE_DATA_DIR / "codex_vision_tips.json"

        self._jobs: List[Dict] = []
        self._patterns: Dict[str, List[Dict]] = {}
        self._preferences: Dict[str, Any] = {}
        self._vision_tips: Dict[str, List[str]] = {}

        self._load_all()

    def _load_all(self):
        """Load all persistent data."""
        for attr, path in [
            ("_jobs", self._jobs_path),
            ("_patterns", self._patterns_path),
            ("_preferences", self._preferences_path),
            ("_vision_tips", self._vision_tips_path),
        ]:
            if path.exists():
                try:
                    setattr(self, attr, json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    pass

    def _save(self, attr: str, path: Path):
        """Save a single data attribute to disk."""
        try:
            path.write_text(
                json.dumps(getattr(self, attr), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Codex save failed for {path.name}: {e}")

    def record_job(self, job_id: str, domain: str, config: Dict,
                   titles: List[str] = None):
        """Record the start of a new job."""
        entry = {
            "job_id": job_id,
            "domain": domain,
            "config_hash": hashlib.md5(
                json.dumps(config, sort_keys=True, default=str).encode()
            ).hexdigest()[:12],
            "title_count": len(titles or []),
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "outcome": None,
        }
        self._jobs.append(entry)
        # Keep last 500 jobs
        self._jobs = self._jobs[-500:]
        self._save("_jobs", self._jobs_path)
        logger.debug(f"Codex: recorded job {job_id} for {domain}")

    def record_outcome(self, job_id: str, success: bool, duration_seconds: int = 0,
                       error: str = None, articles_generated: int = 0,
                       vision_verified: bool = False):
        """Record the outcome of a completed job."""
        for job in reversed(self._jobs):
            if job["job_id"] == job_id:
                job["status"] = "completed" if success else "failed"
                job["outcome"] = {
                    "success": success,
                    "duration_seconds": duration_seconds,
                    "error": error,
                    "articles_generated": articles_generated,
                    "vision_verified": vision_verified,
                    "completed_at": datetime.now().isoformat(),
                }
                break
        self._save("_jobs", self._jobs_path)

        # Update failure patterns if job failed
        if not success and error:
            self._record_failure_pattern(job_id, error)

    def _record_failure_pattern(self, job_id: str, error: str):
        """Extract and store failure patterns for future predictions."""
        job = next((j for j in reversed(self._jobs) if j["job_id"] == job_id), None)
        if not job:
            return

        domain = job.get("domain", "unknown")
        if domain not in self._patterns:
            self._patterns[domain] = []

        self._patterns[domain].append({
            "error": error,
            "config_hash": job.get("config_hash"),
            "timestamp": datetime.now().isoformat(),
        })

        # Keep last 100 patterns per domain
        self._patterns[domain] = self._patterns[domain][-100:]
        self._save("_patterns", self._patterns_path)

    def get_failure_patterns(self, domain: str) -> List[Dict]:
        """Get known failure patterns for a domain, aggregated by frequency."""
        patterns = self._patterns.get(domain, [])
        if not patterns:
            return []

        # Aggregate by error message similarity
        error_counts = defaultdict(int)
        for p in patterns:
            # Normalize error message
            err = p.get("error", "")[:100]
            error_counts[err] += 1

        result = []
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            if count >= 2:  # Only report recurring patterns
                result.append({
                    "error_pattern": err,
                    "count": count,
                    "field": self._extract_field_from_error(err),
                    "bad_value": self._extract_value_from_error(err),
                    "suggested_fix": f"Review configuration related to: {err[:60]}",
                })

        return result

    def _extract_field_from_error(self, error: str) -> str:
        """Try to extract a config field name from an error message."""
        field_keywords = [
            "dropdown", "checkbox", "button", "wordpress", "serp",
            "deep_research", "link_pack", "config", "profile",
        ]
        error_lower = error.lower()
        for kw in field_keywords:
            if kw in error_lower:
                return kw
        return ""

    def _extract_value_from_error(self, error: str) -> str:
        """Try to extract a problematic value from an error message."""
        # Look for quoted values
        import re
        match = re.search(r"'([^']+)'", error)
        return match.group(1) if match else ""

    def get_domain_history(self, domain: str) -> Optional[Dict]:
        """Get historical job data for a domain."""
        domain_jobs = [j for j in self._jobs if j.get("domain") == domain]
        if not domain_jobs:
            return None

        total = len(domain_jobs)
        completed = [j for j in domain_jobs if j.get("outcome")]
        successes = [j for j in completed if j["outcome"].get("success")]

        return {
            "total_jobs": total,
            "completed": len(completed),
            "success_rate": len(successes) / max(len(completed), 1),
            "avg_duration": sum(
                j["outcome"].get("duration_seconds", 0) for j in completed
            ) / max(len(completed), 1),
            "last_run": domain_jobs[-1].get("started_at"),
        }

    def get_preferred_value(self, field: str) -> Optional[str]:
        """Get the most commonly successful value for a config field."""
        return self._preferences.get(field)

    def set_preference(self, field: str, value: str):
        """Record a preferred value for a field (from successful runs)."""
        self._preferences[field] = value
        self._save("_preferences", self._preferences_path)

    def get_vision_tips(self, task: str) -> List[str]:
        """Get learned tips for a specific vision analysis task."""
        return self._vision_tips.get(task, [])

    def add_vision_tip(self, task: str, tip: str):
        """Record a new tip learned from vision analysis."""
        if task not in self._vision_tips:
            self._vision_tips[task] = []
        if tip not in self._vision_tips[task]:
            self._vision_tips[task].append(tip)
            # Keep last 20 tips per task
            self._vision_tips[task] = self._vision_tips[task][-20:]
            self._save("_vision_tips", self._vision_tips_path)

    def get_stats(self) -> Dict:
        """Get overall Codex statistics."""
        total_jobs = len(self._jobs)
        completed = [j for j in self._jobs if j.get("outcome")]
        successes = [j for j in completed if j["outcome"].get("success")]

        domains = set(j.get("domain", "") for j in self._jobs)

        return {
            "total_jobs_recorded": total_jobs,
            "completed_jobs": len(completed),
            "success_rate": len(successes) / max(len(completed), 1),
            "domains_tracked": len(domains),
            "failure_patterns": sum(len(v) for v in self._patterns.values()),
            "preferences_stored": len(self._preferences),
            "vision_tips_stored": sum(len(v) for v in self._vision_tips.values()),
        }


# ═══════════════════════════════════════════════════════════
# FORGE ENGINE — Unified Interface
# ═══════════════════════════════════════════════════════════

class ForgeEngine:
    """
    Unified FORGE Intelligence Engine. Initializes all five modules
    with shared Codex memory and provides high-level orchestration methods.

    Usage:
        forge = ForgeEngine()

        # Pre-job analysis
        report = forge.pre_job_analysis(site_config, titles)

        # Post-job learning
        forge.post_job_learning(job_id, success=True, duration=300)
    """

    def __init__(self):
        self.codex = Codex()
        self.scout = Scout(codex=self.codex)
        self.sentinel = Sentinel(codex=self.codex)
        self.oracle = Oracle(codex=self.codex)
        self.smith = Smith(codex=self.codex)

        logger.info(
            f"FORGE Engine initialized | Codex stats: {self.codex.get_stats()}"
        )

    def pre_job_analysis(self, config: Dict, titles: List[str] = None) -> Dict:
        """
        Complete pre-job analysis: audit config, predict risks, generate fixes.

        Returns:
            {
                "config_issues": [...],
                "auto_fixes": [...],
                "fixed_config": {...},
                "risk_assessment": {...},
                "ready_to_start": bool,
            }
        """
        config_with_titles = {**config}
        if titles:
            config_with_titles["titles"] = titles

        # Scout: find issues
        issues = self.scout.audit_config(config_with_titles)

        # Smith: generate fixes
        fixes = self.smith.auto_fix(issues)
        fixed_config = self.smith.apply_fixes(config_with_titles, fixes)

        # Oracle: predict risk on the fixed config
        risk = self.oracle.predict_failure(fixed_config)

        # Determine readiness
        blocking_issues = [i for i in issues if i["severity"] == "error"]
        unfixed_blockers = []
        fixed_fields = {f["field"] for f in fixes}
        for bi in blocking_issues:
            if bi["field"] not in fixed_fields:
                unfixed_blockers.append(bi)

        ready = len(unfixed_blockers) == 0 and risk["risk_level"] != "critical"

        result = {
            "config_issues": issues,
            "auto_fixes": fixes,
            "fixed_config": fixed_config,
            "risk_assessment": risk,
            "ready_to_start": ready,
            "unfixed_blockers": unfixed_blockers,
        }

        logger.info(
            f"Pre-job analysis: {len(issues)} issues, {len(fixes)} auto-fixes, "
            f"risk={risk['risk_level']}, ready={ready}"
        )
        return result

    def start_job_tracking(self, job_id: str, domain: str, config: Dict,
                           titles: List[str] = None):
        """Record the start of a new job for Codex learning."""
        self.codex.record_job(job_id, domain, config, titles)

    def post_job_learning(self, job_id: str, success: bool,
                          duration_seconds: int = 0, error: str = None,
                          articles_generated: int = 0,
                          vision_verified: bool = False):
        """Record job outcome and update learning data."""
        self.codex.record_outcome(
            job_id, success, duration_seconds, error,
            articles_generated, vision_verified,
        )
        logger.info(
            f"Job {job_id} recorded: {'success' if success else 'failure'} "
            f"in {duration_seconds}s"
        )

    def get_vision_prompt(self, task: str, context: Dict = None) -> str:
        """Get an optimized Vision Service prompt via Sentinel."""
        return self.sentinel.get_prompt(task, context)

    def get_stats(self) -> Dict:
        """Get comprehensive FORGE Engine statistics."""
        return {
            "codex": self.codex.get_stats(),
            "scout_custom_rules": len(self.scout._custom_rules),
            "sentinel_tracked_prompts": len(self.sentinel._prompt_scores),
        }
