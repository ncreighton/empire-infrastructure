#!/usr/bin/env python3
"""
PROJECT MESH v2.0: THE ORACLE   Predictive Intelligence
=========================================================
Forecasts drift, predicts update needs, identifies optimization
opportunities, and generates actionable recommendations.

Usage:
  python oracle.py --forecast              # Full weekly forecast
  python oracle.py --drift-risk            # Drift risk analysis
  python oracle.py --optimize              # Optimization opportunities
  python oracle.py --recommend             # Top recommendations
  python oracle.py --trends                # Trend analysis
"""

import json, os, sys, argparse, math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from collections import defaultdict

DEFAULT_HUB_PATH = r"D:\Claude Code Projects\project-mesh-v2-omega"

def load_json(p):
    if not Path(p).exists(): return {}
    try: return json.loads(Path(p).read_text("utf-8"))
    except Exception: return {}

def save_json(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=2, default=str), "utf-8")

def load_manifests(hub):
    m = {}
    d = Path(hub)/"registry"/"manifests"
    if d.exists():
        for f in d.glob("*.manifest.json"):
            m[f.stem.replace(".manifest","")] = load_json(f)
    return m

def get_sys_version(hub, name):
    v = Path(hub)/"shared-core"/"systems"/name/"VERSION"
    return v.read_text("utf-8").strip() if v.exists() else "0.0.0"


# ============================================================================
# DRIFT RISK PREDICTION
# ============================================================================

def predict_drift_risk(hub: Path, manifests: Dict) -> List[Dict]:
    """Predict which projects are most at risk of drifting out of sync."""
    risks = []
    
    sync_log = load_json(hub / "sync" / "sync-log.json")
    entries = sync_log.get("entries", [])
    
    # Build sync history per project
    sync_history = defaultdict(list)
    for e in entries:
        proj = e.get("project", "")
        if proj:
            sync_history[proj].append({
                "timestamp": e.get("timestamp", ""),
                "status": e.get("status", ""),
                "systems": len(e.get("systems_synced", []))
            })
    
    for proj_name, manifest in manifests.items():
        consumed = manifest.get("consumes", {}).get("shared-core", [])
        priority = manifest.get("project", {}).get("priority", "normal")
        
        # Factor 1: Current outdated count
        outdated = 0
        critical_outdated = 0
        for s in consumed:
            cv = get_sys_version(hub, s.get("system", ""))
            if s.get("version", "") != cv:
                outdated += 1
                if s.get("criticality") == "critical":
                    critical_outdated += 1
        
        # Factor 2: Sync frequency (lower = higher risk)
        history = sync_history.get(proj_name, [])
        recent_syncs = len([h for h in history if _within_days(h.get("timestamp",""), 30)])
        sync_velocity = recent_syncs / 30.0  # syncs per day
        
        # Factor 3: Days since last sync
        days_since_sync = 999
        if history:
            try:
                last = max(h["timestamp"] for h in history if h.get("timestamp"))
                ls = datetime.fromisoformat(last.replace("Z", "+00:00"))
                days_since_sync = (datetime.now(timezone.utc) - ls).days
            except Exception: pass
        
        # Factor 4: System evolution velocity (how fast consumed systems change)
        system_velocity = 0
        for s in consumed:
            sys_name = s.get("system", "")
            cl = hub / "shared-core" / "systems" / sys_name / "CHANGELOG.md"
            if cl.exists():
                content = cl.read_text("utf-8", errors="ignore")
                # Count recent version entries
                month = datetime.now().strftime("%Y-%m")
                system_velocity += content.count(month)
        
        # Factor 5: Number of consumed systems (more = more drift surface)
        surface_area = len(consumed)
        
        # Compute risk score (0-100)
        risk = 0
        risk += min(30, outdated * 10)                        # Current outdated
        risk += min(20, critical_outdated * 20)               # Critical outdated
        risk += min(15, max(0, 15 - recent_syncs * 3))        # Low sync frequency
        risk += min(15, max(0, (days_since_sync - 7) * 1.5))  # Days since sync
        risk += min(10, system_velocity * 2)                   # Fast-moving systems
        risk += min(10, surface_area * 0.5)                    # Large surface area
        
        # Priority multiplier
        priority_mult = {"critical": 1.5, "high": 1.2, "normal": 1.0, "low": 0.8}.get(priority, 1.0)
        risk = min(100, risk * priority_mult)
        
        risks.append({
            "project": proj_name,
            "drift_risk_score": round(risk, 1),
            "risk_level": "critical" if risk >= 70 else "high" if risk >= 50 else "medium" if risk >= 30 else "low",
            "factors": {
                "outdated_systems": outdated,
                "critical_outdated": critical_outdated,
                "days_since_sync": days_since_sync,
                "sync_velocity": round(sync_velocity, 3),
                "system_evolution_velocity": system_velocity,
                "surface_area": surface_area
            },
            "priority": priority,
            "prediction": _predict_drift_timeline(risk, sync_velocity, system_velocity)
        })
    
    return sorted(risks, key=lambda x: -x["drift_risk_score"])


def _predict_drift_timeline(risk: float, sync_vel: float, sys_vel: float) -> str:
    if risk >= 70:
        return "Already drifting   immediate sync needed"
    elif risk >= 50:
        days = max(1, int(7 / max(0.1, sys_vel)))
        return f"Likely to drift within ~{days} days"
    elif risk >= 30:
        return "May drift within 2-3 weeks without sync"
    else:
        return "Low risk   stable for foreseeable future"


# ============================================================================
# OPTIMIZATION OPPORTUNITIES
# ============================================================================

def find_optimizations(hub: Path, manifests: Dict) -> List[Dict]:
    """Identify optimization opportunities across the empire."""
    optimizations = []
    
    # 1. Find systems consumed by only 1 project (candidates for project-local)
    system_consumers = defaultdict(list)
    for proj_name, manifest in manifests.items():
        for s in manifest.get("consumes", {}).get("shared-core", []):
            system_consumers[s.get("system", "")].append(proj_name)
    
    for sys_name, consumers in system_consumers.items():
        if len(consumers) == 1:
            optimizations.append({
                "type": "single-consumer-system",
                "severity": "low",
                "title": f"{sys_name} only used by {consumers[0]}",
                "description": "Consider inlining this system into the project, "
                              "or promote adoption across other projects",
                "effort": "low",
                "impact": "low",
                "affected": consumers
            })
    
    # 2. Find projects consuming many systems (complexity risk)
    for proj_name, manifest in manifests.items():
        consumed = manifest.get("consumes", {}).get("shared-core", [])
        if len(consumed) > 10:
            optimizations.append({
                "type": "high-dependency-count",
                "severity": "medium",
                "title": f"{proj_name} consumes {len(consumed)} systems",
                "description": "High dependency count increases sync complexity. "
                              "Consider bundling related systems.",
                "effort": "medium",
                "impact": "medium",
                "affected": [proj_name]
            })
    
    # 3. Find systems with no consumers (dead systems)
    systems_dir = hub / "shared-core" / "systems"
    if systems_dir.exists():
        all_systems = {s.name for s in systems_dir.iterdir() if s.is_dir()}
        consumed_systems = set(system_consumers.keys())
        dead_systems = all_systems - consumed_systems
        
        for ds in dead_systems:
            optimizations.append({
                "type": "dead-system",
                "severity": "medium",
                "title": f"{ds} has no consumers",
                "description": "System exists but no project consumes it. "
                              "Archive or promote for adoption.",
                "effort": "low",
                "impact": "low",
                "affected": []
            })
    
    # 4. Find projects with no shared-core consumption (not integrated)
    for proj_name, manifest in manifests.items():
        consumed = manifest.get("consumes", {}).get("shared-core", [])
        if not consumed:
            optimizations.append({
                "type": "unintegrated-project",
                "severity": "high",
                "title": f"{proj_name} uses no shared systems",
                "description": "Project is isolated from the mesh. Likely has local "
                              "implementations that duplicate shared-core systems.",
                "effort": "high",
                "impact": "high",
                "affected": [proj_name]
            })
    
    # 5. Find systems that could be merged (similar names/purposes)
    system_names = list(system_consumers.keys())
    for i, s1 in enumerate(system_names):
        for s2 in system_names[i+1:]:
            # Simple similarity: common word stems
            words1 = set(s1.lower().replace("-", " ").split())
            words2 = set(s2.lower().replace("-", " ").split())
            common = words1 & words2
            if len(common) >= 2:
                optimizations.append({
                    "type": "merge-candidate",
                    "severity": "low",
                    "title": f"Consider merging {s1} + {s2}",
                    "description": f"Systems share keywords: {common}. "
                                  "May have overlapping functionality.",
                    "effort": "high",
                    "impact": "medium",
                    "affected": list(set(system_consumers[s1] + system_consumers[s2]))
                })
    
    # 6. Category-level patterns
    category_systems = defaultdict(lambda: defaultdict(int))
    for proj_name, manifest in manifests.items():
        cat = manifest.get("project", {}).get("category", "uncategorized")
        for s in manifest.get("consumes", {}).get("shared-core", []):
            category_systems[cat][s.get("system", "")] += 1
    
    for cat, systems in category_systems.items():
        projects_in_cat = [pn for pn, m in manifests.items() 
                          if m.get("project", {}).get("category") == cat]
        for sys_name, count in systems.items():
            if count < len(projects_in_cat) and len(projects_in_cat) > 2:
                missing = len(projects_in_cat) - count
                if count >= len(projects_in_cat) * 0.5:  # At least half use it
                    optimizations.append({
                        "type": "category-adoption-gap",
                        "severity": "low",
                        "title": f"{sys_name} not adopted by {missing} project(s) in {cat}",
                        "description": f"Most {cat} projects use {sys_name}. "
                                      f"Consider standardizing across the category.",
                        "effort": "low",
                        "impact": "medium",
                        "affected": [pn for pn in projects_in_cat 
                                    if not any(s.get("system") == sys_name 
                                              for s in manifests[pn].get("consumes", {}).get("shared-core", []))]
                    })
    
    return sorted(optimizations, key=lambda x: 
        {"high": 0, "medium": 1, "low": 2}.get(x["severity"], 3))


# ============================================================================
# RECOMMENDATIONS ENGINE
# ============================================================================

def generate_recommendations(hub: Path, manifests: Dict) -> List[Dict]:
    """Generate prioritized recommendations."""
    recs = []
    
    drift_risks = predict_drift_risk(hub, manifests)
    optimizations = find_optimizations(hub, manifests)
    
    # Critical drift risks  immediate action
    for risk in drift_risks:
        if risk["risk_level"] in ("critical", "high"):
            recs.append({
                "priority": 1 if risk["risk_level"] == "critical" else 2,
                "category": "sync",
                "title": f"Sync {risk['project']} (drift risk: {risk['drift_risk_score']})",
                "action": f"python sync_engine_v2.py --sync --project {risk['project']}",
                "effort": "low",
                "impact": "high",
                "reason": risk["prediction"]
            })
    
    # High-severity optimizations  planned action
    for opt in optimizations:
        if opt["severity"] == "high":
            recs.append({
                "priority": 3,
                "category": "optimization",
                "title": opt["title"],
                "action": opt["description"],
                "effort": opt["effort"],
                "impact": opt["impact"],
                "reason": f"Affects: {', '.join(opt['affected'][:5])}"
            })
    
    # Stale CLAUDE.md  recompile
    stale_projects = []
    for proj_name in manifests:
        proj_path = hub.parent / proj_name
        cm = load_json(proj_path / ".project-mesh" / "compile-meta.json")
        if cm.get("compiled_at"):
            try:
                age = (datetime.now() - datetime.fromisoformat(cm["compiled_at"])).total_seconds() / 3600
                if age > 72:
                    stale_projects.append(proj_name)
            except Exception: pass
    
    if stale_projects:
        recs.append({
            "priority": 2,
            "category": "compile",
            "title": f"Recompile {len(stale_projects)} stale CLAUDE.md files",
            "action": "python claude_md_compiler_v2.py --all",
            "effort": "low",
            "impact": "high",
            "reason": f"Projects: {', '.join(stale_projects[:5])}"
        })
    
    return sorted(recs, key=lambda x: x["priority"])


# ============================================================================
# TREND ANALYSIS
# ============================================================================

def analyze_trends(hub: Path, manifests: Dict) -> Dict:
    """Analyze historical trends in mesh health."""
    sync_log = load_json(hub / "sync" / "sync-log.json")
    entries = sync_log.get("entries", [])
    
    # Sync volume over time
    weekly_syncs = defaultdict(int)
    weekly_failures = defaultdict(int)
    for e in entries:
        ts = e.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                week = dt.strftime("%Y-W%W")
                weekly_syncs[week] += 1
                if e.get("status") == "failed":
                    weekly_failures[week] += 1
            except Exception: pass
    
    # Most synced systems
    system_sync_count = defaultdict(int)
    for e in entries:
        for s in e.get("systems_synced", []):
            system_sync_count[s.get("system", "")] += 1
    
    top_synced = sorted(system_sync_count.items(), key=lambda x: -x[1])[:10]
    
    # System growth
    systems_dir = hub / "shared-core" / "systems"
    total_systems = len(list(systems_dir.iterdir())) if systems_dir.exists() else 0
    
    # Projects growth
    total_projects = len(manifests)
    
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "empire_size": {
            "total_projects": total_projects,
            "total_systems": total_systems,
            "total_sync_entries": len(entries)
        },
        "sync_volume": {
            "weekly": dict(sorted(weekly_syncs.items())[-8:]),  # Last 8 weeks
            "failures": dict(sorted(weekly_failures.items())[-8:])
        },
        "most_active_systems": [{"system": s, "sync_count": c} for s, c in top_synced],
        "health_indicators": {
            "avg_syncs_per_week": sum(weekly_syncs.values()) / max(1, len(weekly_syncs)),
            "failure_rate": sum(weekly_failures.values()) / max(1, sum(weekly_syncs.values())) * 100,
            "systems_per_project": total_systems / max(1, total_projects)
        }
    }


def _within_days(ts, days):
    try:
        dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        return dt >= datetime.now(timezone.utc) - timedelta(days=days)
    except Exception: return False


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def print_forecast(hub, manifests):
    print("----------------------------------------------------------------")
    print("-           [CRYSTAL] ORACLE   WEEKLY FORECAST                      -")
    print(f"-           {datetime.now().strftime('%Y-%m-%d %H:%M')}                                -")
    print("----------------------------------------------------------------")
    
    # Drift risks
    risks = predict_drift_risk(hub, manifests)
    critical = [r for r in risks if r["risk_level"] in ("critical", "high")]
    
    print("-                                                              -")
    print(f"-  [WARN]  DRIFT RISK: {len(critical)} project(s) at risk                      -")
    print("-                                                              -")
    
    for r in risks[:8]:
        icon = {"critical":"[RED]","high":"","medium":"[YELLOW]","low":"[GREEN]"}.get(r["risk_level"],"")
        name = r["project"][:28].ljust(28)
        score = f"{r['drift_risk_score']:5.1f}"
        print(f"-  {icon} {name} Risk: {score}  -")
    
    print("-                                                              -")
    print("----------------------------------------------------------------")
    
    # Recommendations
    recs = generate_recommendations(hub, manifests)
    print("-                                                              -")
    print(f"-  [IDEA] TOP RECOMMENDATIONS ({len(recs)})                              -")
    print("-                                                              -")
    
    for r in recs[:5]:
        icon = {1:"[RED]",2:"",3:"[YELLOW]"}.get(r["priority"],"")
        title = r["title"][:50]
        print(f"-  {icon} {title:<50}        -")
        if r.get("action"):
            act = r["action"][:52]
            print(f"-      {act:<52}      -")
    
    print("-                                                              -")
    print("----------------------------------------------------------------")
    
    # Optimizations
    opts = find_optimizations(hub, manifests)
    print("-                                                              -")
    print(f"-  [FIX] OPTIMIZATIONS ({len(opts)} found)                              -")
    print("-                                                              -")
    
    for o in opts[:5]:
        icon = {"high":"[RED]","medium":"[YELLOW]","low":"[GREEN]"}.get(o["severity"],"")
        title = o["title"][:52]
        print(f"-  {icon} {title:<52}        -")
    
    print("-                                                              -")
    print("----------------------------------------------------------------")
    
    # Save forecast
    forecast = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "drift_risks": risks,
        "recommendations": recs,
        "optimizations": opts
    }
    save_json(hub / "oracle" / "latest-forecast.json", forecast)
    print(f"\n[DOC] Full forecast saved to oracle/latest-forecast.json")


def main():
    p = argparse.ArgumentParser(description="Project Mesh Oracle v2.0")
    p.add_argument("--forecast", action="store_true", help="Full weekly forecast")
    p.add_argument("--drift-risk", action="store_true", help="Drift risk analysis")
    p.add_argument("--optimize", action="store_true", help="Optimization opportunities")
    p.add_argument("--recommend", action="store_true", help="Top recommendations")
    p.add_argument("--trends", action="store_true", help="Trend analysis")
    p.add_argument("--hub", default=DEFAULT_HUB_PATH)
    args = p.parse_args()
    hub = Path(args.hub)
    if not hub.exists(): print(f"[FAIL] Hub not found: {hub}"); sys.exit(1)
    manifests = load_manifests(hub)
    print(f"[CRYSTAL] Oracle v2.0   {len(manifests)} projects\n")
    
    if args.forecast:
        print_forecast(hub, manifests)
    elif args.drift_risk:
        risks = predict_drift_risk(hub, manifests)
        for r in risks:
            icon = {"critical":"[RED]","high":"","medium":"[YELLOW]","low":"[GREEN]"}.get(r["risk_level"],"")
            print(f"  {icon} {r['project']:<30} Risk: {r['drift_risk_score']:5.1f}   {r['prediction']}")
    elif args.optimize:
        opts = find_optimizations(hub, manifests)
        for o in opts:
            icon = {"high":"[RED]","medium":"[YELLOW]","low":"[GREEN]"}.get(o["severity"],"")
            print(f"  {icon} [{o['type']}] {o['title']}")
            print(f"     {o['description']}")
            print()
    elif args.recommend:
        recs = generate_recommendations(hub, manifests)
        for r in recs:
            icon = {1:"[RED]",2:"",3:"[YELLOW]"}.get(r["priority"],"")
            print(f"  {icon} P{r['priority']} [{r['category']}] {r['title']}")
            if r.get("action"):
                print(f"      {r['action']}")
            print()
    elif args.trends:
        trends = analyze_trends(hub, manifests)
        print(json.dumps(trends, indent=2))
    else:
        p.print_help()

if __name__ == "__main__": main()
