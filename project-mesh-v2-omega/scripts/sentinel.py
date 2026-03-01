#!/usr/bin/env python3
"""
PROJECT MESH v2.0: THE SENTINEL   Monitoring & Alerting
=========================================================
Real-time health monitoring, alert rules evaluation,
anomaly detection, and notification dispatch.

Usage:
  python sentinel.py --monitor               # Run all checks
  python sentinel.py --alerts                 # Show active alerts
  python sentinel.py --anomalies              # Run anomaly detection
  python sentinel.py --compliance             # Compliance scan
  python sentinel.py --compliance --project X # Scan specific project
"""

import json, os, sys, re, argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

DEFAULT_HUB_PATH = r"D:\Claude Code Projects\project-mesh-v2-omega"
SCAN_EXTENSIONS = {".js", ".ts", ".py", ".php", ".jsx", ".tsx"}
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__", ".project-mesh", "vendor"}

def load_json(p):
    if not Path(p).exists(): return {}
    try: return json.loads(Path(p).read_text("utf-8"))
    except: return {}

def save_json(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=2, default=str), "utf-8")

# ============================================================================
# ALERT RULES ENGINE
# ============================================================================

DEFAULT_ALERT_RULES = [
    {
        "name": "stale-project",
        "severity": "warning",
        "description": "Project hasn't been synced recently",
        "check": "staleness"
    },
    {
        "name": "critical-system-outdated",
        "severity": "critical",
        "description": "Critical-priority system is outdated",
        "check": "critical_outdated"
    },
    {
        "name": "claude-md-stale",
        "severity": "warning",
        "description": "CLAUDE.md hasn't been recompiled recently",
        "check": "claude_md_stale"
    },
    {
        "name": "deprecation-exception-expiring",
        "severity": "warning",
        "description": "Deprecation exception expires soon",
        "check": "exception_expiring"
    },
    {
        "name": "low-compliance",
        "severity": "warning",
        "description": "Project compliance score below threshold",
        "check": "low_compliance"
    }
]

def evaluate_alerts(hub_path: Path) -> List[Dict]:
    """Evaluate all alert rules and return active alerts."""
    alerts = []
    manifests_dir = hub_path / "registry" / "manifests"
    if not manifests_dir.exists():
        return alerts
    
    for mf_path in manifests_dir.glob("*.manifest.json"):
        proj_name = mf_path.stem.replace(".manifest", "")
        manifest = load_json(mf_path)
        proj_path = hub_path.parent / proj_name
        
        # Check sync staleness
        sync_status = load_json(proj_path / ".project-mesh" / "sync-status.json")
        last_sync = sync_status.get("last_sync", "")
        if last_sync:
            try:
                ls = datetime.fromisoformat(last_sync.replace("Z", "+00:00"))
                days = (datetime.now(timezone.utc) - ls).days
                if days > 14:
                    alerts.append({
                        "rule": "stale-project",
                        "severity": "warning",
                        "project": proj_name,
                        "message": f"No sync in {days} days",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            except: pass
        
        # Check critical systems outdated
        for sys in manifest.get("consumes", {}).get("shared-core", []):
            if sys.get("criticality") == "critical":
                consumed_v = sys.get("version", "")
                sys_dir = hub_path / "shared-core" / "systems" / sys.get("system", "") / "VERSION"
                current_v = sys_dir.read_text("utf-8").strip() if sys_dir.exists() else ""
                if consumed_v and current_v and consumed_v != current_v:
                    alerts.append({
                        "rule": "critical-system-outdated",
                        "severity": "critical",
                        "project": proj_name,
                        "message": f"CRITICAL system {sys['system']}: v{consumed_v}  v{current_v}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
        
        # Check CLAUDE.md staleness
        compile_meta = load_json(proj_path / ".project-mesh" / "compile-meta.json")
        if compile_meta.get("compiled_at"):
            try:
                ca = datetime.fromisoformat(compile_meta["compiled_at"])
                hours = (datetime.now() - ca).total_seconds() / 3600
                if hours > 72:
                    alerts.append({
                        "rule": "claude-md-stale",
                        "severity": "warning",
                        "project": proj_name,
                        "message": f"CLAUDE.md is {hours:.0f}h old (>72h threshold)",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            except: pass
    
    # Check deprecation exceptions
    exc_data = load_json(hub_path / "deprecated" / "exceptions" / "exceptions-registry.json")
    for exc in exc_data.get("exceptions", []):
        if exc.get("status") == "active" and exc.get("expires_date"):
            try:
                exp = datetime.strptime(exc["expires_date"], "%Y-%m-%d")
                days_left = (exp - datetime.now()).days
                if days_left <= 7:
                    alerts.append({
                        "rule": "deprecation-exception-expiring",
                        "severity": "warning" if days_left > 0 else "critical",
                        "project": exc.get("project", "?"),
                        "message": f"Exception for {exc.get('deprecated_item','?')} "
                                   f"{'expires in ' + str(days_left) + ' days' if days_left > 0 else 'HAS EXPIRED'}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            except: pass
    
    return alerts


# ============================================================================
# COMPLIANCE SCANNER
# ============================================================================

def scan_compliance(hub_path: Path, project_name: Optional[str] = None) -> Dict:
    """Scan projects for deprecated pattern violations and compliance scoring."""
    
    # Load deprecated patterns
    patterns_file = hub_path / "deprecated" / "patterns" / "code-patterns.json"
    code_patterns = load_json(patterns_file).get("patterns", [])
    
    # Also build patterns from BLACKLIST.md
    blacklist = hub_path / "deprecated" / "BLACKLIST.md"
    
    results = {}
    manifests_dir = hub_path / "registry" / "manifests"
    
    if not manifests_dir.exists():
        return results
    
    for mf_path in manifests_dir.glob("*.manifest.json"):
        pn = mf_path.stem.replace(".manifest", "")
        if project_name and pn != project_name:
            continue
        
        proj_path = hub_path.parent / pn
        if not proj_path.exists():
            continue
        
        manifest = load_json(mf_path)
        violations = []
        files_scanned = 0
        
        # Scan code files
        for root, dirs, files in os.walk(proj_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                if Path(fname).suffix not in SCAN_EXTENSIONS:
                    continue
                fpath = Path(root) / fname
                try:
                    content = fpath.read_text("utf-8", errors="ignore")
                except:
                    continue
                
                files_scanned += 1
                rel_path = str(fpath.relative_to(proj_path))
                
                for pattern in code_patterns:
                    try:
                        matches = re.findall(pattern["regex"], content)
                        if matches:
                            for match in matches:
                                # Find line number
                                for i, line in enumerate(content.split("\n"), 1):
                                    if re.search(pattern["regex"], line):
                                        violations.append({
                                            "file": rel_path,
                                            "line": i,
                                            "pattern": pattern["name"],
                                            "severity": pattern.get("severity", "medium"),
                                            "description": pattern.get("description", ""),
                                            "replacement": pattern.get("replacement", ""),
                                            "auto_fixable": pattern.get("auto_fixable", False)
                                        })
                                        break
                    except re.error:
                        pass
        
        # Compute compliance score
        consumed = manifest.get("consumes", {}).get("shared-core", [])
        total_systems = len(consumed)
        
        # Count systems using latest version
        current_systems = 0
        for s in consumed:
            sv = hub_path / "shared-core" / "systems" / s.get("system","") / "VERSION"
            if sv.exists() and sv.read_text("utf-8").strip() == s.get("version",""):
                current_systems += 1
        
        # Score components
        violation_penalty = min(50, len(violations) * 2)  # Max 50 point penalty
        version_score = (current_systems / total_systems * 40) if total_systems > 0 else 40
        doc_score = 10  # Base documentation score
        
        # Check documentation
        if (proj_path / "CLAUDE.md").exists():
            doc_score += 10
        if (proj_path / "CLAUDE.local.md").exists():
            doc_score += 5
        if (proj_path / ".project-mesh" / "manifest.json").exists() or mf_path.exists():
            doc_score += 5
        
        total_score = max(0, min(100, 100 - violation_penalty + version_score + doc_score - 50))
        
        high_violations = len([v for v in violations if v["severity"] == "high"])
        medium_violations = len([v for v in violations if v["severity"] == "medium"])
        low_violations = len([v for v in violations if v["severity"] == "low"])
        auto_fixable = len([v for v in violations if v["auto_fixable"]])
        
        results[pn] = {
            "compliance_score": round(total_score),
            "files_scanned": files_scanned,
            "total_violations": len(violations),
            "high": high_violations,
            "medium": medium_violations,
            "low": low_violations,
            "auto_fixable": auto_fixable,
            "version_currency": f"{current_systems}/{total_systems}",
            "violations": violations[:50],  # Limit detail
            "recommendations": []
        }
        
        # Generate recommendations
        if high_violations:
            results[pn]["recommendations"].append(
                f"Fix {high_violations} high-severity violations (critical)"
            )
        if auto_fixable:
            results[pn]["recommendations"].append(
                f"Run auto-fix to resolve {auto_fixable} violations automatically"
            )
        if current_systems < total_systems:
            outdated = total_systems - current_systems
            results[pn]["recommendations"].append(
                f"Update {outdated} outdated system(s) to latest versions"
            )
    
    return results


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

def detect_anomalies(hub_path: Path) -> List[Dict]:
    """Detect unusual patterns in mesh operations."""
    anomalies = []
    
    # Analyze sync frequency
    sync_log = load_json(hub_path / "sync" / "sync-log.json")
    entries = sync_log.get("entries", [])
    
    if entries:
        # Count syncs per system in last 24h
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent = defaultdict(int)
        for e in entries:
            try:
                ts = datetime.fromisoformat(e.get("timestamp", "").replace("Z", "+00:00"))
                if ts >= cutoff:
                    for s in e.get("systems_synced", []):
                        recent[s.get("system", "")] += 1
            except: pass
        
        for sys_name, count in recent.items():
            if count > 5:  # More than 5 syncs in 24h for same system
                anomalies.append({
                    "type": "sync-frequency-spike",
                    "severity": "warning",
                    "message": f"{sys_name} synced {count} times in 24h (unusual)",
                    "recommendation": "Consider batching changes or stabilizing the system"
                })
        
        # Check for sync failures
        recent_failures = [e for e in entries[:20] if e.get("status") == "failed"]
        if len(recent_failures) >= 3:
            anomalies.append({
                "type": "sync-failure-cluster",
                "severity": "critical",
                "message": f"{len(recent_failures)} sync failures in recent history",
                "recommendation": "Investigate sync errors and run mesh-doctor"
            })
    
    # Check for rapid system evolution
    systems_dir = hub_path / "shared-core" / "systems"
    if systems_dir.exists():
        for sys_dir in systems_dir.iterdir():
            if not sys_dir.is_dir():
                continue
            changelog = sys_dir / "CHANGELOG.md"
            if changelog.exists():
                content = changelog.read_text("utf-8", errors="ignore")
                # Count version entries in last 30 days
                recent_versions = content.count(datetime.now().strftime("%Y-%m"))
                if recent_versions > 5:
                    anomalies.append({
                        "type": "rapid-evolution",
                        "severity": "info",
                        "message": f"{sys_dir.name} has {recent_versions} releases this month",
                        "recommendation": "High velocity may indicate instability   consider stabilization"
                    })
    
    return anomalies


# ============================================================================
# REPORTING
# ============================================================================

def print_alerts(alerts: List[Dict]):
    if not alerts:
        print("[OK] No active alerts   empire is healthy")
        return
    
    critical = [a for a in alerts if a["severity"] == "critical"]
    warnings = [a for a in alerts if a["severity"] == "warning"]
    info = [a for a in alerts if a["severity"] == "info"]
    
    print(f"\n[ALERT] ACTIVE ALERTS: {len(alerts)} total\n")
    
    if critical:
        print("  [FAIL] CRITICAL:")
        for a in critical:
            print(f"     [{a.get('project','empire')}] {a['message']}")
    
    if warnings:
        print("  [WARN]  WARNINGS:")
        for a in warnings:
            print(f"     [{a.get('project','empire')}] {a['message']}")
    
    if info:
        print("    INFO:")
        for a in info:
            print(f"     [{a.get('project','empire')}] {a['message']}")


def print_compliance(results: Dict):
    print(f"\n[LIST] COMPLIANCE REPORT\n")
    print(f"  {'Project':<32} {'Score':>5} {'Violations':>10} {'Auto-Fix':>8} {'Versions':>10}")
    print(f"  {'-'*32} {'-'*5} {'-'*10} {'-'*8} {'-'*10}")
    
    for pn, data in sorted(results.items(), key=lambda x: x[1]["compliance_score"]):
        score = data["compliance_score"]
        icon = "[GREEN]" if score >= 85 else "[YELLOW]" if score >= 60 else "[RED]"
        print(f"  {icon} {pn[:30]:<30} {score:>5} {data['total_violations']:>10} "
              f"{data['auto_fixable']:>8} {data['version_currency']:>10}")
    
    # Summary
    avg = sum(d["compliance_score"] for d in results.values()) / len(results) if results else 0
    total_violations = sum(d["total_violations"] for d in results.values())
    total_fixable = sum(d["auto_fixable"] for d in results.values())
    
    print(f"\n  Average score: {avg:.1f}/100")
    print(f"  Total violations: {total_violations} ({total_fixable} auto-fixable)")
    
    if total_fixable:
        print(f"\n  [IDEA] Run auto-fix to resolve {total_fixable} violations automatically")


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Project Mesh Sentinel v2.0")
    p.add_argument("--monitor", action="store_true", help="Full monitoring check")
    p.add_argument("--alerts", action="store_true", help="Show active alerts")
    p.add_argument("--anomalies", action="store_true", help="Run anomaly detection")
    p.add_argument("--compliance", action="store_true", help="Compliance scan")
    p.add_argument("--project", "-p", help="Target specific project")
    p.add_argument("--hub", default=DEFAULT_HUB_PATH)
    
    args = p.parse_args()
    hub = Path(args.hub)
    
    if not hub.exists():
        print(f"[FAIL] Hub not found: {hub}")
        sys.exit(1)
    
    print("[GUARD]  Project Mesh Sentinel v2.0\n")
    
    if args.monitor:
        # Run everything
        alerts = evaluate_alerts(hub)
        print_alerts(alerts)
        
        anomalies = detect_anomalies(hub)
        if anomalies:
            print(f"\n[SEARCH] ANOMALIES DETECTED:\n")
            for a in anomalies:
                icon = {"critical":"[FAIL]","warning":"[WARN]","info":""}.get(a["severity"],"")
                print(f"  {icon} [{a['type']}] {a['message']}")
                print(f"      {a['recommendation']}")
        
        compliance = scan_compliance(hub)
        if compliance:
            print_compliance(compliance)
        
        # Save full report
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alerts": alerts,
            "anomalies": anomalies,
            "compliance": {k: {**v, "violations": v["violations"][:10]} for k, v in compliance.items()}
        }
        save_json(hub / "sentinel" / "latest-report.json", report)
        print(f"\n[DOC] Full report saved to sentinel/latest-report.json")
    
    elif args.alerts:
        alerts = evaluate_alerts(hub)
        print_alerts(alerts)
    
    elif args.anomalies:
        anomalies = detect_anomalies(hub)
        if anomalies:
            for a in anomalies:
                icon = {"critical":"[FAIL]","warning":"[WARN]","info":""}.get(a["severity"],"")
                print(f"  {icon} [{a['type']}] {a['message']}")
        else:
            print("  [OK] No anomalies detected")
    
    elif args.compliance:
        compliance = scan_compliance(hub, args.project)
        print_compliance(compliance)
        
        # Show detail for single project
        if args.project and args.project in compliance:
            data = compliance[args.project]
            if data["violations"]:
                print(f"\n  VIOLATIONS for {args.project}:\n")
                for v in data["violations"][:20]:
                    icon = {"high":"[FAIL]","medium":"[WARN]","low":""}.get(v["severity"],"")
                    fix = " [auto-fixable]" if v["auto_fixable"] else ""
                    print(f"    {icon} {v['file']}:{v['line']}   {v['pattern']}{fix}")
                    print(f"        {v['replacement']}")
    
    else:
        p.print_help()

if __name__ == "__main__":
    main()
