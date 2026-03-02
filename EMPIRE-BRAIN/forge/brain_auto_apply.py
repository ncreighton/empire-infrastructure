"""BrainAutoApply — Safe Auto-Application of High-Confidence Enhancements

Only auto-applies enhancements that meet ALL criteria:
- confidence >= 0.9
- severity = 'recommended' or 'important'
- enhancement_type = 'deprecated_pattern' (safest category — regex replacements)
- status = 'pending' for 24+ hours (cooling period)
- File still exists and contains the pattern

Creates a backup before modifying any file. Logs every action.
Never touches security-related, refactor, or uncertain findings.
"""
import logging
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB

log = logging.getLogger("evolution-engine")

# Only these types are safe to auto-apply
SAFE_TYPES = {"deprecated_pattern"}
MIN_CONFIDENCE = 0.9
MIN_AGE_HOURS = 24
SAFE_SEVERITIES = {"recommended", "important"}

# Exact replacement mappings (pattern → replacement regex)
REPLACEMENTS = {
    "datetime.utcnow()": {
        "find": r"datetime\.utcnow\(\)",
        "replace": "datetime.now(timezone.utc)",
        "add_import": "from datetime import timezone",
    },
    "datetime.utcfromtimestamp()": {
        "find": r"datetime\.utcfromtimestamp\(([^)]+)\)",
        "replace": r"datetime.fromtimestamp(\1, tz=timezone.utc)",
        "add_import": "from datetime import timezone",
    },
    "asyncio.get_event_loop()": {
        "find": r"asyncio\.get_event_loop\(\)",
        "replace": "asyncio.get_running_loop()",
    },
    "collections.MutableMapping": {
        "find": r"collections\.MutableMapping",
        "replace": "collections.abc.MutableMapping",
    },
}


class BrainAutoApply:
    """Auto-applies safe, high-confidence deprecated pattern fixes."""

    def __init__(self, db: Optional[BrainDB] = None, dry_run: bool = False):
        self.db = db or BrainDB()
        self.dry_run = dry_run
        self.applied = []
        self.skipped = []
        self.errors = []

    def find_eligible(self) -> list[dict]:
        """Find enhancements eligible for auto-apply."""
        cutoff = (datetime.now() - timedelta(hours=MIN_AGE_HOURS)).isoformat()
        all_pending = self.db.get_enhancements(
            status="pending",
            min_confidence=MIN_CONFIDENCE,
        )

        eligible = []
        for enh in all_pending:
            # Must be a safe type
            if enh.get("enhancement_type") not in SAFE_TYPES:
                continue
            # Must be safe severity
            if enh.get("severity") not in SAFE_SEVERITIES:
                continue
            # Must be old enough (cooling period)
            created = enh.get("created_at", "")
            if created > cutoff:
                continue
            # Must have a known replacement
            current = enh.get("current_code", "")
            if current not in REPLACEMENTS:
                continue
            # File must exist
            file_path = enh.get("file_path", "")
            if not file_path or not Path(file_path).exists():
                continue

            eligible.append(enh)

        return eligible

    def apply_one(self, enhancement: dict) -> bool:
        """Apply a single enhancement. Returns True on success."""
        file_path = Path(enhancement["file_path"])
        current = enhancement.get("current_code", "")
        replacement_info = REPLACEMENTS.get(current)

        if not replacement_info:
            self.skipped.append({"id": enhancement["id"], "reason": "no replacement mapping"})
            return False

        try:
            content = file_path.read_text(encoding="utf-8")

            # Verify the pattern still exists
            if not re.search(replacement_info["find"], content):
                self.skipped.append({"id": enhancement["id"], "reason": "pattern no longer in file"})
                self.db.update_enhancement_status(enhancement["id"], "applied")
                return False

            if self.dry_run:
                self.applied.append({
                    "id": enhancement["id"],
                    "file": str(file_path),
                    "pattern": current,
                    "dry_run": True,
                })
                return True

            # Backup the file
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            shutil.copy2(file_path, backup_path)

            # Apply replacement
            new_content = re.sub(replacement_info["find"], replacement_info["replace"], content)

            # Add import if needed and not already present
            add_import = replacement_info.get("add_import")
            if add_import and add_import not in new_content:
                # Insert after the last import line
                lines = new_content.split("\n")
                last_import = 0
                for i, line in enumerate(lines):
                    if line.startswith("import ") or line.startswith("from "):
                        last_import = i
                lines.insert(last_import + 1, add_import)
                new_content = "\n".join(lines)

            file_path.write_text(new_content, encoding="utf-8")

            # Clean up backup
            backup_path.unlink(missing_ok=True)

            # Mark as applied in DB
            self.db.update_enhancement_status(enhancement["id"], "applied")

            self.applied.append({
                "id": enhancement["id"],
                "file": str(file_path),
                "pattern": current,
                "project": enhancement.get("project_slug", ""),
            })
            log.info(f"[AutoApply] Applied: {current} in {file_path.name} ({enhancement.get('project_slug', '')})")
            return True

        except Exception as e:
            self.errors.append({
                "id": enhancement["id"],
                "file": str(file_path),
                "error": str(e),
            })
            log.error(f"[AutoApply] Failed on {file_path}: {e}")
            return False

    def run(self) -> dict:
        """Find and apply all eligible enhancements."""
        eligible = self.find_eligible()
        log.info(f"[AutoApply] Found {len(eligible)} eligible enhancements (dry_run={self.dry_run})")

        for enh in eligible:
            self.apply_one(enh)

        result = {
            "eligible": len(eligible),
            "applied": len(self.applied),
            "skipped": len(self.skipped),
            "errors": len(self.errors),
            "dry_run": self.dry_run,
            "details": {
                "applied": self.applied,
                "skipped": self.skipped,
                "errors": self.errors,
            },
        }

        if self.applied and not self.dry_run:
            self.db.emit_event("evolution.auto_apply", {
                "applied": len(self.applied),
                "files": [a["file"] for a in self.applied],
            }, source="auto_apply")

        log.info(f"[AutoApply] Complete: {len(self.applied)} applied, {len(self.skipped)} skipped, {len(self.errors)} errors")
        return result


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Auto-apply safe enhancements")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be applied without changing files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    auto = BrainAutoApply(dry_run=args.dry_run)
    result = auto.run()
    print(json.dumps(result, indent=2, default=str))
