"""OpenClaw CLI -- command-line interface for platform signup automation.

Usage:
    python cli.py signup gumroad --password "..."
    python cli.py signup-retry gumroad --password "..." --max-retries 3
    python cli.py batch gumroad,etsy --password "..."
    python cli.py status [platform_id]
    python cli.py dashboard
    python cli.py prioritize
    python cli.py easy-wins
    python cli.py generate gumroad
    python cli.py score gumroad
    python cli.py analyze gumroad
    python cli.py sync --bio "New bio text"
    python cli.py sync-status
    python cli.py export --format json -o accounts.json
    python cli.py analytics [--type report|coverage|timeline]
    python cli.py email-stats
    python cli.py email-verified
    python cli.py proxy-stats
    python cli.py retry-stats
    python cli.py ratelimit [platform_id]
    python cli.py schedule list
    python cli.py schedule batch --platforms gumroad,etsy --password "..."
    python cli.py schedule pause --job-id <id>
    python cli.py captcha pending
    python cli.py captcha solve --task-id <id> --solution <text>
    python cli.py platforms [--category ai_marketplace]
    python cli.py health
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openclaw.openclaw_engine import OpenClawEngine
from openclaw.knowledge.platforms import (
    get_platform,
    get_all_platform_ids,
    get_platforms_by_category,
    PLATFORMS,
)
from openclaw.forge.market_oracle import MarketOracle
from openclaw.models import (
    AccountStatus,
    PlatformCategory,
    OpenClawResult,
)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_DIVIDER = "-" * 72
_DIVIDER_THICK = "=" * 72


def _header(title: str) -> None:
    """Print a section header."""
    print()
    print(_DIVIDER_THICK)
    print(f"  {title}")
    print(_DIVIDER_THICK)


def _subheader(title: str) -> None:
    """Print a sub-section header."""
    print()
    print(f"  {title}")
    print(f"  {'-' * len(title)}")


def _kv(key: str, value: Any, indent: int = 2) -> None:
    """Print a key-value pair with consistent alignment."""
    prefix = " " * indent
    key_padded = f"{key}:".ljust(24)
    print(f"{prefix}{key_padded}{value}")


def _table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None) -> None:
    """Print a simple text table with column alignment."""
    if not rows:
        print("  (no data)")
        return

    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(min(max_w + 2, 40))

    # Header row
    header_line = "  "
    separator = "  "
    for i, h in enumerate(headers):
        header_line += h.ljust(col_widths[i])
        separator += "-" * col_widths[i]
    print(header_line)
    print(separator)

    # Data rows
    for row in rows:
        line = "  "
        for i, cell in enumerate(row):
            width = col_widths[i] if i < len(col_widths) else 20
            line += str(cell).ljust(width)
        print(line)


def _progress(step: int, total: int, message: str) -> None:
    """Print a progress step indicator."""
    print(f"  [{step}/{total}] {message}")


def _success(message: str) -> None:
    """Print a success message."""
    print(f"  [OK] {message}")


def _error(message: str) -> None:
    """Print an error message."""
    print(f"  [ERROR] {message}")


def _warn(message: str) -> None:
    """Print a warning message."""
    print(f"  [WARN] {message}")


def _info(message: str) -> None:
    """Print an info message."""
    print(f"  [INFO] {message}")


def _format_score(score: float) -> str:
    """Format a numeric score with visual grade bar."""
    filled = int(score / 5)  # 20 segments for 0-100
    empty = 20 - filled
    bar = "#" * filled + "." * empty
    return f"{score:5.1f}/100 [{bar}]"


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def _priority_label(priority_value: str) -> str:
    """Add visual weight to priority labels."""
    labels = {
        "critical": "[!!!] CRITICAL",
        "high": "[!! ] HIGH",
        "medium": "[!  ] MEDIUM",
        "low": "[   ] LOW",
        "skip": "[---] SKIP",
    }
    return labels.get(priority_value, priority_value.upper())


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_signup(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Execute a single-platform signup pipeline."""
    platform_id = args.platform_id
    platform = get_platform(platform_id)

    if not platform:
        _error(f"Unknown platform: {platform_id}")
        print(f"  Run 'python cli.py platforms' to see all supported platform IDs.")
        sys.exit(1)

    headless = args.headless and not args.visible
    engine.headless = headless

    _header(f"Signing up on {platform.name}")
    _kv("Platform ID", platform_id)
    _kv("Category", platform.category.value)
    _kv("Complexity", platform.complexity.value)
    _kv("Estimated time", f"{platform.estimated_signup_minutes} minutes")
    _kv("Browser mode", "headless" if headless else "visible")

    credentials: dict[str, str] = {"password": args.password}
    if args.email:
        credentials["email"] = args.email

    print()
    _info("Starting signup pipeline...")
    start = time.time()

    try:
        result = engine.signup(platform_id, credentials)
    except KeyboardInterrupt:
        print()
        _warn("Signup interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        _error(f"Signup pipeline crashed: {exc}")
        sys.exit(1)

    elapsed = time.time() - start

    _subheader("Result")
    _kv("Success", "YES" if result.success else "NO")
    _kv("Status", result.status.value)
    _kv("Steps completed", f"{result.steps_completed}/{result.steps_total}")
    _kv("Duration", _format_duration(elapsed))

    if result.profile_url:
        _kv("Profile URL", result.profile_url)
    if result.username:
        _kv("Username", result.username)

    if result.sentinel_score:
        _kv("Profile score", _format_score(result.sentinel_score.total_score))
        _kv("Grade", result.sentinel_score.grade.value)

    if result.errors:
        _subheader("Errors")
        for err in result.errors:
            _error(err)

    if result.warnings:
        _subheader("Warnings")
        for w in result.warnings:
            _warn(w)

    print()
    if result.success:
        _success(f"Signup complete for {platform.name}!")
    else:
        _error(f"Signup failed for {platform.name}.")
        sys.exit(1)


def cmd_batch(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Execute batch signup across multiple platforms."""
    raw_ids = [pid.strip() for pid in args.platform_ids.split(",") if pid.strip()]

    if not raw_ids:
        _error("No platform IDs provided.")
        sys.exit(1)

    # Validate all platform IDs upfront
    valid_ids = []
    for pid in raw_ids:
        if get_platform(pid):
            valid_ids.append(pid)
        else:
            _warn(f"Unknown platform '{pid}' -- skipping.")

    if not valid_ids:
        _error("No valid platform IDs found.")
        sys.exit(1)

    _header(f"Batch Signup: {len(valid_ids)} platforms")
    for i, pid in enumerate(valid_ids, 1):
        p = get_platform(pid)
        print(f"  {i}. {p.name} ({pid}) -- {p.complexity.value}")

    credentials: dict[str, str] = {"password": args.password}
    if args.email:
        credentials["email"] = args.email

    delay = args.delay
    _info(f"Delay between signups: {delay}s")
    print()

    results: list[OpenClawResult] = []
    start = time.time()

    for i, pid in enumerate(valid_ids, 1):
        platform = get_platform(pid)
        _subheader(f"[{i}/{len(valid_ids)}] {platform.name}")

        try:
            result = engine.signup(pid, credentials)
            results.append(result)

            status_str = "SUCCESS" if result.success else "FAILED"
            _kv("Result", status_str)
            if result.profile_url:
                _kv("Profile URL", result.profile_url)
            if result.sentinel_score:
                _kv("Score", f"{result.sentinel_score.total_score:.1f}")
            if result.errors:
                for err in result.errors:
                    _error(err)

        except Exception as exc:
            _error(f"Pipeline error: {exc}")
            results.append(OpenClawResult(
                platform_id=pid,
                platform_name=platform.name,
                success=False,
                errors=[str(exc)],
            ))

        # Delay between signups (skip after last one)
        if i < len(valid_ids):
            _info(f"Waiting {delay}s before next signup...")
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                print()
                _warn("Batch interrupted by user.")
                break

    # Summary
    total_time = time.time() - start
    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded

    _header("Batch Summary")
    _kv("Total platforms", len(valid_ids))
    _kv("Succeeded", succeeded)
    _kv("Failed", failed)
    _kv("Total duration", _format_duration(total_time))

    if failed > 0:
        _subheader("Failed Platforms")
        for r in results:
            if not r.success:
                errors = "; ".join(r.errors) if r.errors else "unknown error"
                print(f"    {r.platform_name} ({r.platform_id}): {errors}")

    print()


def cmd_status(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Display account status for one or all platforms."""
    if args.platform_id:
        # Single platform detail view
        pid = args.platform_id
        status_data = engine.get_platform_status(pid)

        if args.format == "json":
            print(json.dumps(status_data, indent=2, default=str))
            return

        _header(f"Status: {pid}")

        p_info = status_data.get("platform", {})
        _kv("Name", p_info.get("name", "Unknown"))
        _kv("Category", p_info.get("category", ""))
        _kv("Complexity", p_info.get("complexity", ""))

        account = status_data.get("account")
        if account:
            _subheader("Account")
            _kv("Status", account.get("status", "not_started"))
            _kv("Username", account.get("username", "(none)"))
            _kv("Profile URL", account.get("profile_url", "(none)"))
            _kv("Created", account.get("created_at", ""))
            _kv("Updated", account.get("updated_at", ""))
        else:
            _info("No account record found.")

        profile = status_data.get("profile")
        if profile:
            _subheader("Profile")
            _kv("Sentinel score", f"{profile.get('sentinel_score', 0):.1f}")
            _kv("Grade", profile.get("grade", "F"))
            content = profile.get("content", {})
            _kv("Bio length", f"{len(content.get('bio', ''))} chars")
            _kv("Tagline", content.get("tagline", "(none)"))
            _kv("Website URL", content.get("website_url", "(none)"))

        log = status_data.get("signup_log", [])
        if log:
            _subheader(f"Signup Log ({len(log)} steps)")
            log_rows = []
            for entry in log[-10:]:  # Last 10 steps
                log_rows.append([
                    str(entry.get("step_number", "")),
                    entry.get("step_type", ""),
                    entry.get("status", ""),
                    entry.get("description", "")[:40],
                ])
            _table(["Step", "Type", "Status", "Description"], log_rows)

        print()
        return

    # All platforms overview
    accounts = engine.codex.get_all_accounts()

    if args.format == "json":
        print(json.dumps(accounts, indent=2, default=str))
        return

    _header("All Accounts")

    if not accounts:
        _info("No accounts tracked yet. Run 'python cli.py signup <platform>' to start.")
        print()
        return

    # Retrieve profile scores to join in
    all_profiles = engine.codex.get_all_profiles()
    score_map: dict[str, tuple[float, str]] = {}
    for p in all_profiles:
        score_map[p["platform_id"]] = (p.get("sentinel_score", 0.0), p.get("grade", "F"))

    rows = []
    for acct in accounts:
        pid = acct["platform_id"]
        score_val, grade = score_map.get(pid, (0.0, "-"))
        score_str = f"{score_val:.0f} ({grade})" if score_val > 0 else "-"
        updated = acct.get("updated_at", "")
        if updated and len(updated) >= 10:
            updated = updated[:10]  # Date only
        rows.append([
            pid,
            acct.get("status", "unknown"),
            acct.get("username", "") or "-",
            score_str,
            updated,
        ])

    _table(
        ["Platform", "Status", "Username", "Score", "Updated"],
        rows,
        col_widths=[24, 22, 18, 12, 12],
    )

    # Summary counts
    by_status: dict[str, int] = {}
    for acct in accounts:
        s = acct.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    _subheader("Summary")
    for s, count in sorted(by_status.items()):
        _kv(s, count)
    _kv("Total", len(accounts))
    print()


def cmd_prioritize(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Display prioritized platform recommendations from MarketOracle."""
    _header("Platform Recommendations (MarketOracle)")

    recommendations = engine.prioritize()

    if not recommendations:
        _info("All platforms completed or no platforms available.")
        print()
        return

    # Show top 10
    top = recommendations[:10]
    _subheader(f"Top {len(top)} Recommendations")

    for i, rec in enumerate(top, 1):
        priority_str = _priority_label(rec.priority.value)
        print()
        print(f"  {i}. {rec.platform_name}")
        _kv("Platform ID", rec.platform_id, indent=5)
        _kv("Priority", priority_str, indent=5)
        _kv("Composite score", _format_score(rec.score), indent=5)
        _kv("Monetization", f"{rec.monetization_score:.0f}/40", indent=5)
        _kv("Audience", f"{rec.audience_score:.0f}/25", indent=5)
        _kv("SEO", f"{rec.seo_score:.0f}/20", indent=5)
        _kv("Effort penalty", f"{rec.effort_score:.0f}/10", indent=5)
        _kv("Reasoning", rec.reasoning[:80], indent=5)
        if len(rec.reasoning) > 80:
            print(f"{'':>29}{rec.reasoning[80:]}")

    # Category summary
    _subheader("Category Coverage")
    oracle = MarketOracle()
    completed = set()
    for acct in engine.codex.get_all_accounts():
        if acct.get("status") in (AccountStatus.ACTIVE.value, AccountStatus.PROFILE_COMPLETE.value):
            completed.add(acct["platform_id"])

    cat_summary = oracle.get_category_summary(completed=completed)
    cat_rows = []
    for cat, data in sorted(cat_summary.items()):
        cat_rows.append([
            cat,
            str(data["total"]),
            str(data["completed"]),
            str(data["remaining"]),
            f"{data['avg_monetization']:.1f}",
        ])
    _table(
        ["Category", "Total", "Done", "Remaining", "Avg Monetization"],
        cat_rows,
        col_widths=[24, 8, 8, 12, 18],
    )
    print()


def cmd_generate(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Generate and display profile content for a platform (dry-run)."""
    platform_id = args.platform_id
    platform = get_platform(platform_id)

    if not platform:
        _error(f"Unknown platform: {platform_id}")
        sys.exit(1)

    _header(f"Generated Profile: {platform.name}")

    try:
        content = engine.generate_profile(platform_id)
    except ValueError as exc:
        _error(str(exc))
        sys.exit(1)

    _kv("Username", content.username)
    _kv("Display name", content.display_name)
    _kv("Email", content.email)

    _subheader("Tagline")
    print(f"    {content.tagline}")

    _subheader("Bio")
    for line in (content.bio or "").split("\n"):
        print(f"    {line}")

    _subheader("Description")
    for line in (content.description or "").split("\n"):
        print(f"    {line}")

    _subheader("Links")
    _kv("Website", content.website_url or "(none)")
    if content.social_links:
        for name, url in content.social_links.items():
            if url:
                _kv(name, url, indent=4)
    else:
        _info("No social links configured.")

    _subheader("SEO Keywords")
    if content.seo_keywords:
        print(f"    {', '.join(content.seo_keywords)}")
    else:
        _info("No SEO keywords configured.")

    _subheader("Assets")
    _kv("Avatar", content.avatar_path or "(none)")
    _kv("Banner", content.banner_path or "(none)")

    _subheader("Platform Limits")
    _kv("Username max", f"{platform.username_max_length} chars (using {len(content.username)})")
    _kv("Bio max", f"{platform.bio_max_length} chars (using {len(content.bio or '')})")
    _kv("Tagline max", f"{platform.tagline_max_length} chars (using {len(content.tagline or '')})")
    _kv("Description max", f"{platform.description_max_length} chars (using {len(content.description or '')})")
    _kv("Max links", f"{platform.max_links} (using {len(content.social_links)})")

    print()
    _info("This is a dry-run. No browser actions taken. No data saved.")
    print()


def cmd_score(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Score an existing profile from the Codex database."""
    platform_id = args.platform_id
    platform = get_platform(platform_id)

    if not platform:
        _error(f"Unknown platform: {platform_id}")
        sys.exit(1)

    score = engine.score_profile(platform_id)
    if not score:
        _warn(f"No stored profile found for {platform_id}.")
        _info("Run 'python cli.py signup <platform>' first, or check 'python cli.py status'.")
        sys.exit(1)

    _header(f"Profile Score: {platform.name}")
    _kv("Total score", _format_score(score.total_score))
    _kv("Grade", score.grade.value)

    _subheader("Score Breakdown (6 Criteria)")
    criteria = [
        ("Completeness", score.completeness, 20),
        ("SEO Quality", score.seo_quality, 20),
        ("Brand Consistency", score.brand_consistency, 15),
        ("Link Presence", score.link_presence, 15),
        ("Bio Quality", score.bio_quality, 15),
        ("Avatar Quality", score.avatar_quality, 15),
    ]
    for name, val, max_val in criteria:
        pct = (val / max_val * 100) if max_val > 0 else 0
        bar_len = int(pct / 5)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"    {name:<22} {val:5.1f}/{max_val:2d}  [{bar}]")

    if score.feedback:
        _subheader("Feedback")
        for fb in score.feedback:
            print(f"    - {fb}")

    if score.enhancements:
        _subheader("Enhancement Suggestions")
        for i, enh in enumerate(score.enhancements, 1):
            print(f"    {i}. {enh}")

    print()


def cmd_analyze(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Analyze a platform for signup readiness via PlatformScout."""
    platform_id = args.platform_id
    platform = get_platform(platform_id)

    if not platform:
        _error(f"Unknown platform: {platform_id}")
        sys.exit(1)

    _header(f"Scout Analysis: {platform.name}")

    try:
        result = engine.analyze_platform(platform_id)
    except ValueError as exc:
        _error(str(exc))
        sys.exit(1)

    _kv("Complexity", result.complexity.value)
    _kv("Estimated time", f"{result.estimated_minutes} minutes")
    _kv("CAPTCHA type", result.captcha_type.value)
    _kv("Readiness score", _format_score(result.completeness_score))

    _subheader("Required Fields")
    if result.required_fields:
        for f in result.required_fields:
            print(f"    - {f}")
    else:
        _info("No required fields beyond email/password.")

    _subheader("Optional Fields")
    if result.optional_fields:
        for f in result.optional_fields:
            print(f"    - {f}")
    else:
        _info("No optional fields configured.")

    _subheader("Readiness Checklist")
    for item in result.readiness_checklist:
        status_icon = "[READY]" if item["ready"] else "[NEED ]"
        print(f"    {status_icon} {item['item']}: {item['note']}")

    _subheader("Risks")
    if result.risks:
        for risk in result.risks:
            print(f"    - {risk}")
    else:
        _success("No significant risks identified.")

    _subheader("Tips")
    if result.tips:
        for tip in result.tips:
            print(f"    - {tip}")

    print()


def cmd_sync(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Synchronize profile content across active platforms using ProfileSync."""
    from openclaw.automation.profile_sync import ProfileSync

    changes: dict[str, str] = {}
    if args.bio:
        changes["bio"] = args.bio
    if args.tagline:
        changes["tagline"] = args.tagline
    if args.website:
        changes["website_url"] = args.website

    if not changes:
        _error("No changes specified. Use --bio, --tagline, or --website.")
        sys.exit(1)

    target_ids = None
    if args.platforms:
        target_ids = [pid.strip() for pid in args.platforms.split(",") if pid.strip()]

    sync = ProfileSync(codex=engine.codex, sentinel=engine.sentinel)

    _header("Profile Sync")

    _subheader("Changes to Apply")
    for field_name, new_val in changes.items():
        display_val = new_val[:60] + "..." if len(new_val) > 60 else new_val
        _kv(field_name, display_val)

    plan = sync.plan_sync(changes, platform_ids=target_ids)

    _subheader("Target Platforms")
    if not plan.target_platforms:
        _info("No active platforms found to sync.")
        print()
        return

    preview = sync.preview_sync(plan)
    for entry in preview:
        print(f"    {entry['platform_name']} ({entry['platform_id']})")
        for change in entry.get("changes", []):
            old_display = change["old"][:30] if change["old"] else "(empty)"
            new_display = change["new"][:30] if change["new"] else "(empty)"
            print(f"      {change['field']}: {old_display} -> {new_display}")

    # Execute local sync (update Codex records)
    _subheader("Updating Local Records")
    sync.update_local(plan)

    for result in plan.results:
        if result.success:
            _success(f"{result.platform_name}: {', '.join(result.changes_applied)}")
        else:
            errors = "; ".join(result.errors) if result.errors else "unknown error"
            _error(f"{result.platform_name}: {errors}")

    _subheader("Consistency Check")
    consistency = sync.get_sync_status()
    _kv("Total active platforms", consistency.get("total_active", 0))
    _kv("Consistent fields", consistency.get("consistent_fields", 0))
    _kv("Mismatched fields", consistency.get("mismatched_fields", 0))

    mismatches = consistency.get("mismatches", [])
    if mismatches:
        print()
        for mm in mismatches[:5]:
            _warn(f"{mm['field']}: {mm['count']} platforms differ")

    print()
    _info("Local Codex records updated. Browser sync not executed.")
    _info("To push changes to live platforms, use the API: POST /sync")
    print()


def cmd_export(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Export all account and profile data to JSON or CSV."""
    _header("Exporting Account Data")

    accounts = engine.codex.get_all_accounts()
    all_profiles = engine.codex.get_all_profiles()

    # Build a lookup of profiles
    profile_map: dict[str, dict] = {}
    for p in all_profiles:
        profile_map[p["platform_id"]] = p

    # Combine data
    export_data = []
    for acct in accounts:
        pid = acct["platform_id"]
        profile = profile_map.get(pid, {})
        content = profile.get("content", {})

        entry = {
            "platform_id": pid,
            "platform_name": acct.get("platform_name", ""),
            "status": acct.get("status", ""),
            "username": acct.get("username", ""),
            "profile_url": acct.get("profile_url", ""),
            "sentinel_score": profile.get("sentinel_score", 0.0),
            "grade": profile.get("grade", ""),
            "bio": content.get("bio", ""),
            "tagline": content.get("tagline", ""),
            "website_url": content.get("website_url", ""),
            "created_at": acct.get("created_at", ""),
            "updated_at": acct.get("updated_at", ""),
        }
        export_data.append(entry)

    output_path = args.output

    if args.format == "json":
        if not output_path.endswith(".json"):
            output_path += ".json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

    elif args.format == "csv":
        if not output_path.endswith(".csv"):
            output_path = output_path.rsplit(".", 1)[0] + ".csv"
        if export_data:
            fieldnames = list(export_data[0].keys())
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_data)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("")

    _kv("Format", args.format)
    _kv("Records exported", len(export_data))
    _kv("Output file", os.path.abspath(output_path))
    print()
    _success(f"Export complete: {output_path}")
    print()


def cmd_platforms(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """List all supported platforms with metadata."""
    _header("Supported Platforms")

    if args.category:
        # Validate category
        try:
            cat_enum = PlatformCategory(args.category)
        except ValueError:
            _error(f"Unknown category: {args.category}")
            valid = ", ".join(c.value for c in PlatformCategory)
            _info(f"Valid categories: {valid}")
            sys.exit(1)

        platforms = get_platforms_by_category(cat_enum)
        _info(f"Filtering by category: {args.category}")
    else:
        platforms = list(PLATFORMS.values())

    if not platforms:
        _info("No platforms found matching the filter.")
        print()
        return

    # Group by category
    by_category: dict[str, list] = {}
    for p in platforms:
        cat = p.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(p)

    total = 0
    for cat, cat_platforms in sorted(by_category.items()):
        _subheader(f"{cat} ({len(cat_platforms)} platforms)")
        rows = []
        for p in sorted(cat_platforms, key=lambda x: x.name):
            rows.append([
                p.platform_id,
                p.name,
                p.complexity.value,
                str(p.monetization_potential),
                str(p.audience_size),
                str(p.seo_value),
                p.captcha_type.value,
            ])
            total += 1
        _table(
            ["ID", "Name", "Complexity", "Mon", "Aud", "SEO", "CAPTCHA"],
            rows,
            col_widths=[22, 22, 14, 6, 6, 6, 14],
        )
        print()

    _info(f"Total: {total} platforms across {len(by_category)} categories")
    print()


def cmd_signup_retry(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Execute signup with automatic retry on transient failures."""
    platform_id = args.platform_id
    platform = get_platform(platform_id)
    if not platform:
        _error(f"Unknown platform: {platform_id}")
        sys.exit(1)

    _header(f"Signup with Retry: {platform.name}")
    _kv("Max retries", args.max_retries)

    credentials: dict[str, str] = {"password": args.password}
    if args.email:
        credentials["email"] = args.email

    start = time.time()
    try:
        result = asyncio.run(
            engine.signup_with_retry(platform_id, credentials, max_retries=args.max_retries)
        )
    except Exception as exc:
        _error(f"Signup failed after retries: {exc}")
        sys.exit(1)

    elapsed = time.time() - start
    _kv("Success", "YES" if result.success else "NO")
    _kv("Status", result.status.value)
    _kv("Duration", _format_duration(elapsed))
    if result.errors:
        for err in result.errors:
            _error(err)
    print()


def cmd_dashboard(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Display aggregate dashboard statistics."""
    stats = engine.get_dashboard()

    if args.format == "json":
        from dataclasses import asdict
        print(json.dumps(asdict(stats), indent=2, default=str))
        return

    _header("Dashboard")
    _kv("Total platforms", stats.total_platforms)
    _kv("Active accounts", stats.active_accounts)
    _kv("Pending signups", stats.pending_signups)
    _kv("Failed signups", stats.failed_signups)
    _kv("Avg profile score", f"{stats.avg_profile_score:.1f}")

    if stats.platforms_by_category:
        _subheader("By Category")
        for cat, count in sorted(stats.platforms_by_category.items()):
            _kv(cat, count)

    if stats.platforms_by_status:
        _subheader("By Status")
        for status, count in sorted(stats.platforms_by_status.items()):
            _kv(status, count)

    if stats.recent_activity:
        _subheader(f"Recent Activity ({len(stats.recent_activity)} items)")
        for item in stats.recent_activity[:5]:
            print(f"    {item.get('platform_id', '?')} — {item.get('status', '?')} — {item.get('updated_at', '')}")
    print()


def cmd_easy_wins(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Show platforms with the best value-to-effort ratio."""
    from openclaw.knowledge.platforms import get_easy_wins
    wins = get_easy_wins()

    _header("Easy Wins — Best Value-to-Effort")
    rows = []
    for p in wins[:15]:
        value = p.monetization_potential + p.audience_size + p.seo_value
        rows.append([p.platform_id, p.name, p.category.value, p.complexity.value, str(value)])
    _table(["ID", "Name", "Category", "Complexity", "Value"], rows,
           col_widths=[22, 22, 18, 14, 8])
    print()


def cmd_analytics(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Display analytics report, coverage map, or timeline."""
    from openclaw.automation.analytics import Analytics
    analytics = Analytics(codex=engine.codex)

    report_type = args.type

    if report_type == "coverage":
        _header("Platform Coverage Map")
        coverage = analytics.get_coverage_map()
        if args.format == "json":
            print(json.dumps(coverage, indent=2, default=str))
        else:
            for entry in coverage:
                status = entry.get("status", "not_started")
                print(f"  {entry['platform_id']:<24} {status}")
        print()
        return

    if report_type == "timeline":
        _header(f"Activity Timeline (last {args.days} days)")
        timeline = analytics.get_timeline(days=args.days)
        if args.format == "json":
            print(json.dumps(timeline, indent=2, default=str))
        else:
            for entry in timeline:
                print(f"  {entry.get('date', '?')} — {entry.get('platform_id', '?')} — {entry.get('event', '?')}")
        print()
        return

    # Default: full report
    _header("Analytics Report")
    report = analytics.generate_report()
    if args.format == "json":
        print(json.dumps(report, indent=2, default=str))
    else:
        if isinstance(report, dict):
            for key, val in report.items():
                _kv(key, val)
        else:
            print(f"  {report}")
    print()


def cmd_email_stats(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Show email verification statistics."""
    _header("Email Verification Stats")
    stats = engine.email_verifier.get_stats()
    if args.format == "json":
        print(json.dumps(stats, indent=2, default=str))
    else:
        for key, val in stats.items():
            _kv(key, val)
    print()


def cmd_email_verified(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Show platforms with verified emails."""
    _header("Verified Email Platforms")
    verified = engine.email_verifier.get_verified_platforms()
    if not verified:
        _info("No platforms have verified emails yet.")
    else:
        for pid in verified:
            print(f"  {pid}")
    print()


def cmd_proxy_stats(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Show proxy pool statistics."""
    _header("Proxy Pool Stats")
    stats = engine.proxy_manager.get_stats()
    if args.format == "json":
        print(json.dumps(stats, indent=2, default=str))
    else:
        for key, val in stats.items():
            _kv(key, val)
    print()


def cmd_retry_stats(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Show retry engine statistics."""
    _header("Retry Engine Stats")
    stats = engine.retry_engine.get_stats()
    if args.format == "json":
        print(json.dumps(stats, indent=2, default=str))
    else:
        for key, val in stats.items():
            _kv(key, val)
    print()


def cmd_ratelimit(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Show rate limiter stats or check a specific platform."""
    if args.platform_id:
        can_proceed, reason = engine.rate_limiter.can_proceed(args.platform_id)
        _header(f"Rate Limit Check: {args.platform_id}")
        _kv("Can proceed", "YES" if can_proceed else "NO")
        _kv("Reason", reason)
        if not can_proceed:
            wait = engine.rate_limiter.wait_time(args.platform_id)
            _kv("Wait seconds", f"{wait:.0f}")
    else:
        _header("Rate Limiter Stats")
        stats = engine.rate_limiter.get_stats()
        if args.format == "json":
            print(json.dumps(stats, indent=2, default=str))
        else:
            for key, val in stats.items():
                _kv(key, val)
    print()


def cmd_schedule(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Manage scheduled batch jobs."""
    from openclaw.automation.scheduler import Scheduler
    scheduler = Scheduler()

    action = args.action

    if action == "list":
        _header("Scheduled Jobs")
        jobs = scheduler.get_all_jobs()
        if not jobs:
            _info("No scheduled jobs.")
        else:
            rows = []
            for j in jobs:
                rows.append([
                    j.job_id[:12],
                    j.status.value,
                    str(len(j.platform_ids)),
                    str(j.completed_count),
                    str(j.failed_count),
                    j.current_platform or "-",
                ])
            _table(["Job ID", "Status", "Platforms", "Done", "Failed", "Current"], rows)
        print()
        return

    if action in ("pause", "resume", "cancel"):
        if not args.job_id:
            _error(f"--job-id required for '{action}'")
            sys.exit(1)
        method = getattr(scheduler, f"{action}_job")
        success = method(args.job_id)
        if success:
            _success(f"Job {args.job_id} {action}d.")
        else:
            _error(f"Cannot {action} job {args.job_id}.")
        print()
        return

    if action == "batch":
        if not args.platforms:
            _error("--platforms required for 'batch'")
            sys.exit(1)
        platform_ids = [pid.strip() for pid in args.platforms.split(",")]
        credentials = {}
        if args.password:
            credentials["password"] = args.password

        job_id = scheduler.schedule_batch(
            platform_ids=platform_ids,
            credentials=credentials or None,
            delay_between_seconds=args.delay,
        )
        _header("Batch Scheduled")
        _kv("Job ID", job_id)
        _kv("Platforms", len(platform_ids))
        _kv("Delay", f"{args.delay}s")
        _info("Job created. Start the API server to execute scheduled jobs.")
        print()
        return

    _error(f"Unknown schedule action: {action}")
    sys.exit(1)


def cmd_sync_status(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Show profile consistency status across active platforms."""
    status = engine.get_sync_status()

    if args.format == "json":
        print(json.dumps(status, indent=2, default=str))
        return

    _header("Profile Sync Status")
    _kv("Active platforms", status.get("total_active", 0))
    _kv("Consistent fields", status.get("consistent_fields", 0))
    _kv("Mismatched fields", status.get("mismatched_fields", 0))

    mismatches = status.get("mismatches", [])
    if mismatches:
        _subheader("Mismatches")
        for mm in mismatches:
            _warn(f"{mm['field']}: {mm['count']} different values across platforms")
    print()


def cmd_captcha(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """View pending CAPTCHAs or submit a solution."""
    if args.action == "pending":
        _header("Pending CAPTCHAs")
        pending = engine.captcha.get_pending_tasks()
        if not pending:
            _info("No CAPTCHAs pending.")
        else:
            for task in pending:
                print(f"  Task: {task.get('task_id', '?')} — Platform: {task.get('platform_id', '?')} — Type: {task.get('type', '?')}")
        print()
    elif args.action == "solve":
        if not args.task_id or not args.solution:
            _error("--task-id and --solution required for 'solve'")
            sys.exit(1)
        success = engine.captcha.submit_solution(args.task_id, args.solution)
        if success:
            _success(f"Solution submitted for task {args.task_id}")
        else:
            _error(f"Unknown task: {args.task_id}")
        print()


def cmd_health(engine: OpenClawEngine, args: argparse.Namespace) -> None:
    """Check system health: API dependencies, DB, email, proxy."""
    _header("System Health Check")

    checks_passed = 0
    checks_total = 0

    # 1. Python version
    checks_total += 1
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    _kv("Python version", py_version)
    checks_passed += 1

    # 2. SQLite database
    checks_total += 1
    db_path = engine.codex.db_path
    db_exists = os.path.exists(db_path)
    if db_exists:
        db_size = os.path.getsize(db_path)
        db_size_str = f"{db_size / 1024:.1f} KB" if db_size < 1024 * 1024 else f"{db_size / 1024 / 1024:.1f} MB"
        _kv("Database", f"OK ({db_size_str})")
        checks_passed += 1
    else:
        _kv("Database", f"MISSING ({db_path})")

    # 3. Database stats
    checks_total += 1
    try:
        stats = engine.codex.get_stats()
        _kv("Total accounts", stats.get("total_accounts", 0))
        _kv("Active accounts", stats.get("active_accounts", 0))
        _kv("Avg sentinel score", f"{stats.get('avg_sentinel_score', 0):.1f}")
        checks_passed += 1
    except Exception as exc:
        _kv("Database query", f"ERROR: {exc}")

    # 4. Email configuration
    checks_total += 1
    email = os.environ.get("OPENCLAW_EMAIL", "")
    if email:
        masked = email[:3] + "***" + email[email.index("@"):] if "@" in email else email[:3] + "***"
        _kv("Email configured", f"YES ({masked})")
        checks_passed += 1
    else:
        _kv("Email configured", "NO (set OPENCLAW_EMAIL)")

    # 5. Brand configuration
    checks_total += 1
    from openclaw.knowledge.brand_config import get_brand
    brand = get_brand()
    brand_fields = 0
    if brand.name:
        brand_fields += 1
    if brand.email:
        brand_fields += 1
    if brand.website:
        brand_fields += 1
    if brand.username_base:
        brand_fields += 1
    if brand.has_avatar:
        brand_fields += 1
    _kv("Brand config", f"{brand_fields}/5 fields set")
    if brand_fields >= 3:
        checks_passed += 1

    # 6. Anthropic API key
    checks_total += 1
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:]
        _kv("Anthropic API key", f"SET ({masked_key})")
        checks_passed += 1
    else:
        _kv("Anthropic API key", "NOT SET (needed for browser-use Agent)")

    # 7. 2Captcha API key
    checks_total += 1
    captcha_key = os.environ.get("TWOCAPTCHA_API_KEY", "")
    if captcha_key:
        _kv("2Captcha API key", "SET")
        checks_passed += 1
    else:
        _kv("2Captcha API key", "NOT SET (optional, for auto CAPTCHA solving)")
        checks_passed += 1  # Optional, so still passes

    # 8. Encryption key
    checks_total += 1
    enc_key = os.environ.get("OPENCLAW_ENCRYPTION_KEY", "")
    if enc_key:
        _kv("Encryption key", "SET")
        checks_passed += 1
    else:
        _kv("Encryption key", "NOT SET (auto-generated key will be used)")
        checks_passed += 1  # Fallback exists

    # 9. Required packages
    _subheader("Package Dependencies")
    packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("browser-use", "browser_use"),
        ("playwright", "playwright"),
        ("httpx", "httpx"),
        ("cryptography", "cryptography"),
        ("pydantic", "pydantic"),
    ]
    for display_name, import_name in packages:
        checks_total += 1
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "installed")
            _kv(display_name, f"OK ({version})")
            checks_passed += 1
        except ImportError:
            _kv(display_name, "NOT INSTALLED")

    # 10. Platform knowledge base
    checks_total += 1
    all_ids = get_all_platform_ids()
    _kv("Platforms loaded", f"{len(all_ids)}")
    if len(all_ids) > 0:
        checks_passed += 1

    # 11. Data directories
    _subheader("Data Directories")
    data_dirs = [
        ("data/", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")),
        ("data/sessions/", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sessions")),
        ("data/screenshots/", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "screenshots")),
    ]
    for label, path in data_dirs:
        checks_total += 1
        exists = os.path.isdir(path)
        if exists:
            _kv(label, f"OK ({path})")
            checks_passed += 1
        else:
            _kv(label, f"MISSING (will be auto-created)")
            checks_passed += 1  # Auto-created on first use

    # Summary
    _subheader("Result")
    _kv("Checks passed", f"{checks_passed}/{checks_total}")

    if checks_passed == checks_total:
        _success("All health checks passed.")
    elif checks_passed >= checks_total * 0.7:
        _warn("Most checks passed. Review warnings above.")
    else:
        _error("Multiple issues detected. Fix the items above before running signups.")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the appropriate command handler."""
    parser = argparse.ArgumentParser(
        prog="openclaw",
        description="OpenClaw Agent -- autonomous platform profile manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python cli.py signup gumroad --password 'secret123'\n"
            "  python cli.py batch gumroad,etsy --password 'secret123'\n"
            "  python cli.py status\n"
            "  python cli.py prioritize\n"
            "  python cli.py generate gumroad\n"
            "  python cli.py platforms --category ai_marketplace\n"
            "  python cli.py health\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- signup --
    signup_parser = subparsers.add_parser("signup", help="Sign up on a platform")
    signup_parser.add_argument("platform_id", help="Platform to sign up on")
    signup_parser.add_argument("--password", "-p", required=True, help="Account password")
    signup_parser.add_argument("--email", "-e", help="Override email from brand config")
    signup_parser.add_argument(
        "--headless", action="store_true", default=True,
        help="Run browser in headless mode (default)",
    )
    signup_parser.add_argument(
        "--visible", action="store_true",
        help="Show browser window (overrides --headless)",
    )

    # -- batch --
    batch_parser = subparsers.add_parser("batch", help="Batch signup on multiple platforms")
    batch_parser.add_argument("platform_ids", help="Comma-separated platform IDs")
    batch_parser.add_argument("--password", "-p", required=True, help="Account password")
    batch_parser.add_argument("--email", "-e", help="Override email from brand config")
    batch_parser.add_argument(
        "--delay", type=int, default=60,
        help="Seconds between signups (default: 60)",
    )

    # -- status --
    status_parser = subparsers.add_parser("status", help="View account status")
    status_parser.add_argument("platform_id", nargs="?", help="Specific platform (optional)")
    status_parser.add_argument(
        "--format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # -- prioritize --
    subparsers.add_parser("prioritize", help="Get ranked platform recommendations")

    # -- generate --
    gen_parser = subparsers.add_parser("generate", help="Generate profile content (dry-run)")
    gen_parser.add_argument("platform_id", help="Platform to generate content for")

    # -- score --
    score_parser = subparsers.add_parser("score", help="Score an existing profile")
    score_parser.add_argument("platform_id", help="Platform to score")

    # -- analyze --
    analyze_parser = subparsers.add_parser("analyze", help="Analyze platform signup readiness")
    analyze_parser.add_argument("platform_id", help="Platform to analyze")

    # -- sync --
    sync_parser = subparsers.add_parser("sync", help="Sync profile content across platforms")
    sync_parser.add_argument("--bio", help="New bio text to propagate")
    sync_parser.add_argument("--tagline", help="New tagline to propagate")
    sync_parser.add_argument("--website", help="New website URL to propagate")
    sync_parser.add_argument(
        "--platforms",
        help="Comma-separated platform IDs (default: all active)",
    )

    # -- export --
    export_parser = subparsers.add_parser("export", help="Export account data")
    export_parser.add_argument(
        "--format", choices=["json", "csv"], default="json",
        help="Export format (default: json)",
    )
    export_parser.add_argument(
        "--output", "-o", default="openclaw_export.json",
        help="Output file path (default: openclaw_export.json)",
    )

    # -- platforms --
    platforms_parser = subparsers.add_parser("platforms", help="List all supported platforms")
    platforms_parser.add_argument(
        "--category", "-c",
        help="Filter by category (e.g., ai_marketplace, digital_product)",
    )

    # -- signup-retry --
    retry_parser = subparsers.add_parser("signup-retry", help="Signup with auto-retry")
    retry_parser.add_argument("platform_id", help="Platform to sign up on")
    retry_parser.add_argument("--password", "-p", required=True, help="Account password")
    retry_parser.add_argument("--email", "-e", help="Override email")
    retry_parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts (default: 3)")

    # -- dashboard --
    dash_parser = subparsers.add_parser("dashboard", help="View aggregate dashboard stats")
    dash_parser.add_argument("--format", choices=["table", "json"], default="table")

    # -- easy-wins --
    subparsers.add_parser("easy-wins", help="Show best value-to-effort platforms")

    # -- analytics --
    analytics_parser = subparsers.add_parser("analytics", help="View analytics report/coverage/timeline")
    analytics_parser.add_argument("--type", choices=["report", "coverage", "timeline"], default="report")
    analytics_parser.add_argument("--days", type=int, default=30, help="Timeline days (default: 30)")
    analytics_parser.add_argument("--format", choices=["table", "json"], default="table")

    # -- email-stats --
    email_stats_parser = subparsers.add_parser("email-stats", help="Email verification statistics")
    email_stats_parser.add_argument("--format", choices=["table", "json"], default="table")

    # -- email-verified --
    subparsers.add_parser("email-verified", help="List platforms with verified emails")

    # -- proxy-stats --
    proxy_parser = subparsers.add_parser("proxy-stats", help="Proxy pool statistics")
    proxy_parser.add_argument("--format", choices=["table", "json"], default="table")

    # -- retry-stats --
    retry_stats_parser = subparsers.add_parser("retry-stats", help="Retry engine statistics")
    retry_stats_parser.add_argument("--format", choices=["table", "json"], default="table")

    # -- ratelimit --
    rl_parser = subparsers.add_parser("ratelimit", help="Rate limiter stats or check a platform")
    rl_parser.add_argument("platform_id", nargs="?", help="Platform to check (optional)")
    rl_parser.add_argument("--format", choices=["table", "json"], default="table")

    # -- schedule --
    sched_parser = subparsers.add_parser("schedule", help="Manage scheduled batch jobs")
    sched_parser.add_argument("action", choices=["list", "batch", "pause", "resume", "cancel"])
    sched_parser.add_argument("--job-id", help="Job ID (for pause/resume/cancel)")
    sched_parser.add_argument("--platforms", help="Comma-separated platform IDs (for batch)")
    sched_parser.add_argument("--password", "-p", help="Account password (for batch)")
    sched_parser.add_argument("--delay", type=int, default=60, help="Delay between signups (default: 60)")

    # -- sync-status --
    sync_status_parser = subparsers.add_parser("sync-status", help="Profile consistency status")
    sync_status_parser.add_argument("--format", choices=["table", "json"], default="table")

    # -- captcha --
    captcha_parser = subparsers.add_parser("captcha", help="View/solve pending CAPTCHAs")
    captcha_parser.add_argument("action", choices=["pending", "solve"])
    captcha_parser.add_argument("--task-id", help="CAPTCHA task ID (for solve)")
    captcha_parser.add_argument("--solution", help="CAPTCHA solution text (for solve)")

    # -- health --
    subparsers.add_parser("health", help="Check system health and dependencies")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Initialize engine
    engine = OpenClawEngine()

    # Dispatch
    commands = {
        "signup": cmd_signup,
        "signup-retry": cmd_signup_retry,
        "batch": cmd_batch,
        "status": cmd_status,
        "dashboard": cmd_dashboard,
        "prioritize": cmd_prioritize,
        "generate": cmd_generate,
        "score": cmd_score,
        "analyze": cmd_analyze,
        "easy-wins": cmd_easy_wins,
        "sync": cmd_sync,
        "sync-status": cmd_sync_status,
        "export": cmd_export,
        "analytics": cmd_analytics,
        "email-stats": cmd_email_stats,
        "email-verified": cmd_email_verified,
        "proxy-stats": cmd_proxy_stats,
        "retry-stats": cmd_retry_stats,
        "ratelimit": cmd_ratelimit,
        "schedule": cmd_schedule,
        "captcha": cmd_captcha,
        "platforms": cmd_platforms,
        "health": cmd_health,
    }

    handler = commands.get(args.command)
    if handler:
        handler(engine, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
