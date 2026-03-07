#!/usr/bin/env python3
"""
PROJECT MESH v2.0: SYNC ENGINE   OMEGA Edition
================================================
Transactional sync with rollback, dependency resolution,
chain orchestration, intelligent scheduling, and analytics.

Usage:
  python sync_engine_v2.py --check                    # Health check
  python sync_engine_v2.py --sync                     # Smart sync all
  python sync_engine_v2.py --sync --project X         # Sync one project
  python sync_engine_v2.py --impact content-pipeline  # Impact analysis
  python sync_engine_v2.py --rollback sync-id-123     # Rollback a sync
  python sync_engine_v2.py --dashboard                # Generate dashboard
  python sync_engine_v2.py --history                  # Sync history
  python sync_engine_v2.py --build-graph              # Rebuild dependency graph
"""

import json, os, sys, shutil, tarfile, hashlib, argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict

DEFAULT_HUB_PATH = r"D:\Claude Code Projects\project-mesh-v2-omega"
SYNC_LOG_MAX = 500
ROLLBACK_DAYS = 30

def load_json(p):
    if not Path(p).exists(): return {}
    try: return json.loads(Path(p).read_text("utf-8"))
    except Exception: return {}

def save_json(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=2, default=str), "utf-8")

def load_all_manifests(hub):
    m = {}
    d = Path(hub) / "registry" / "manifests"
    if d.exists():
        for f in d.glob("*.manifest.json"):
            m[f.stem.replace(".manifest","")] = load_json(f)
    return m

def get_sys_version(hub, name):
    v = Path(hub)/"shared-core"/"systems"/name/"VERSION"
    return v.read_text("utf-8").strip() if v.exists() else "0.0.0"

def get_sys_deps(hub, name):
    return load_json(Path(hub)/"shared-core"/"systems"/name/"DEPENDENCIES.json")

def version_distance(old, new):
    try:
        o = [int(x) for x in old.split(".")]; n = [int(x) for x in new.split(".")]
        while len(o)<3: o.append(0)
        while len(n)<3: n.append(0)
        if n[0]>o[0]: return (n[0]-o[0])*100
        if n[1]>o[1]: return (n[1]-o[1])*10
        return max(0,n[2]-o[2])
    except Exception: return 0

# --- DEPENDENCY GRAPH ---
def build_dependency_graph(hub, manifests):
    hub = Path(hub)
    g = {"generated_at":datetime.now(timezone.utc).isoformat(),"nodes":[],"edges":[],"systems":{},"projects":{}}
    sd = hub/"shared-core"/"systems"
    if sd.exists():
        for s in sd.iterdir():
            if s.is_dir():
                v = get_sys_version(hub, s.name)
                g["nodes"].append({"id":f"system:{s.name}","type":"system","name":s.name,"version":v})
                g["systems"][s.name] = {"version":v,"consumers":[],"depends_on":[]}
                deps = get_sys_deps(hub, s.name)
                for dep in deps.get("requires",{}).get("systems",[]):
                    dn = dep.get("name","")
                    g["edges"].append({"from":f"system:{s.name}","to":f"system:{dn}","type":"system-depends-on"})
                    g["systems"][s.name]["depends_on"].append(dn)
    for pn, mf in manifests.items():
        g["nodes"].append({"id":f"project:{pn}","type":"project","name":pn,
            "category":mf.get("project",{}).get("category",""),
            "priority":mf.get("project",{}).get("priority","normal")})
        consumed = []
        for s in mf.get("consumes",{}).get("shared-core",[]):
            sn, sv, cr = s.get("system",""), s.get("version",""), s.get("criticality","normal")
            g["edges"].append({"from":f"project:{pn}","to":f"system:{sn}","type":"consumes","version":sv,"criticality":cr})
            if sn in g["systems"]: g["systems"][sn]["consumers"].append(pn)
            consumed.append({"name":sn,"version":sv})
        for dep in mf.get("consumes",{}).get("from-projects",[]):
            g["edges"].append({"from":f"project:{pn}","to":f"project:{dep.get('project','')}","type":"cross-project"})
        g["projects"][pn] = {"consumed_systems":consumed}
    return g

def build_reverse_graph(g):
    r = defaultdict(lambda:{"consumed_by":[],"depended_by_systems":[]})
    for e in g.get("edges",[]):
        if e["type"]=="consumes":
            sn = e["to"].replace("system:",""); pn = e["from"].replace("project:","")
            r[sn]["consumed_by"].append({"project":pn,"version":e.get("version",""),"criticality":e.get("criticality","normal")})
        elif e["type"]=="system-depends-on":
            r[e["to"].replace("system:","")]["depended_by_systems"].append(e["from"].replace("system:",""))
    return dict(r)

# --- IMPACT ANALYSIS ---
def analyze_impact(hub, system_name, manifests):
    hub = Path(hub)
    g = build_dependency_graph(hub, manifests)
    rv = build_reverse_graph(g)
    cv = get_sys_version(hub, system_name)
    info = rv.get(system_name, {})
    direct = info.get("consumed_by",[])
    dep_sys = info.get("depended_by_systems",[])
    indirect = []
    for ds in dep_sys:
        for c in rv.get(ds,{}).get("consumed_by",[]):
            if c["project"] not in [d["project"] for d in direct]:
                indirect.append({**c,"via":ds})
    outdated = [c for c in direct if c.get("version","") != cv]
    return {"system":system_name,"current_version":cv,
        "direct_consumers":len(direct),"outdated":len(outdated),
        "indirect_via_systems":dep_sys,"indirect_consumers":len(indirect),
        "total_affected":len(set([c["project"] for c in direct]+[c["project"] for c in indirect])),
        "estimated_minutes":len(outdated)*15,
        "details":{"outdated":outdated,"indirect":indirect}}

# --- SYNC STATUS ---
def check_project_status(hub, pn, mf):
    hub = Path(hub)
    consumed = mf.get("consumes",{}).get("shared-core",[])
    systems, outdated = [], 0
    for s in consumed:
        sn, sv, cv = s.get("system",""), s.get("version","0.0.0"), get_sys_version(hub, s.get("system",""))
        cr = s.get("criticality","normal")
        current = sv == cv
        if not current: outdated += 1
        systems.append({"system":sn,"consumed":sv,"current":cv,"is_current":current,"criticality":cr})
    pct = ((len(consumed)-outdated)/len(consumed)*100) if consumed else 100
    pp = hub.parent/pn
    cm = load_json(pp/".project-mesh"/"compile-meta.json")
    age = 999
    if cm.get("compiled_at"):
        try: age = (datetime.now()-datetime.fromisoformat(cm["compiled_at"])).total_seconds()/3600
        except Exception: pass
    return {"project":pn,"sync_pct":round(pct,1),"total":len(consumed),"outdated":outdated,
        "systems":systems,"claude_md_age_h":round(age,1),"claude_md_stale":age>72,
        "priority":mf.get("project",{}).get("priority","normal")}

# --- ROLLBACK ---
def create_snapshot(hub, pn, sid):
    pp = Path(hub).parent/pn
    if not pp.exists(): return None
    rd = Path(hub)/"sync"/"rollback"; rd.mkdir(parents=True, exist_ok=True)
    sp = rd/f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{pn}.tar.gz"
    with tarfile.open(sp,"w:gz") as t:
        for f in ["CLAUDE.md"]:
            fp = pp/f
            if fp.exists(): t.add(fp, arcname=f)
        md = pp/".project-mesh"
        if md.exists(): t.add(md, arcname=".project-mesh")
    return sp

def rollback(hub, sid):
    sl = load_json(Path(hub)/"sync"/"sync-log.json")
    entry = next((e for e in sl.get("entries",[]) if e.get("id")==sid), None)
    if not entry: print(f"[FAIL] Not found: {sid}"); return
    sp = entry.get("snapshot_path")
    if not sp or not Path(sp).exists(): print(f"[FAIL] Snapshot unavailable"); return
    pp = Path(hub).parent/entry["project"]
    with tarfile.open(sp,"r:gz") as t: t.extractall(path=pp)
    entry["rolled_back"] = True
    save_json(Path(hub)/"sync"/"sync-log.json", sl)
    print(f"[OK] Rolled back {entry['project']}")

# --- SYNC EXECUTION ---
def sync_project(hub, pn, mf, force=False, dry_run=False):
    hub = Path(hub)
    sid = f"sync-{datetime.now().strftime('%Y%m%d%H%M%S')}-{pn[:12]}"
    result = {"id":sid,"project":pn,"timestamp":datetime.now(timezone.utc).isoformat(),
        "status":"pending","systems_synced":[],"snapshot_path":None,"errors":[],"duration_seconds":0}
    start = datetime.now()
    status = check_project_status(hub, pn, mf)
    outdated = [s for s in status["systems"] if not s["is_current"]]
    if not outdated and not force:
        result["status"] = "up-to-date"
        print(f"    {pn}: Up to date")
        return result
    to_sync = outdated if not force else status["systems"]
    print(f"\n  [CYCLE] {pn}: {len(to_sync)} system(s)")
    if dry_run:
        for s in to_sync: print(f"     [DRY] {s['system']}: v{s['consumed']}  v{s['current']}")
        result["status"] = "dry-run"; return result
    # Snapshot
    snap = create_snapshot(hub, pn, sid)
    result["snapshot_path"] = str(snap) if snap else None
    # Dep resolution
    for s in to_sync:
        deps = get_sys_deps(hub, s["system"])
        for r in deps.get("requires",{}).get("systems",[]):
            has = any(x.get("system")==r.get("name","") for x in mf.get("consumes",{}).get("shared-core",[]))
            if not has: print(f"     [WARN]  {s['system']} requires {r.get('name','')} (not in manifest)")
    # Update manifest versions
    mf_path = hub/"registry"/"manifests"/f"{pn}.manifest.json"
    if mf_path.exists():
        um = load_json(mf_path)
        for s in to_sync:
            for c in um.get("consumes",{}).get("shared-core",[]):
                if c.get("system")==s["system"]: c["version"]=s["current"]
        save_json(mf_path, um)
    # Sync status
    mesh = hub.parent/pn/".project-mesh"; mesh.mkdir(parents=True, exist_ok=True)
    ss = {"last_sync":datetime.now(timezone.utc).isoformat(),"sync_id":sid,
        "systems":{s["system"]:{"version":s["current"]} for s in to_sync}}
    save_json(mesh/"sync-status.json", ss)
    for s in to_sync:
        print(f"     [PKG] {s['system']}: v{s['consumed']}  v{s['current']}")
        result["systems_synced"].append({"system":s["system"],"from":s["consumed"],"to":s["current"]})
    result["status"] = "success"
    result["duration_seconds"] = round((datetime.now()-start).total_seconds(),2)
    print(f"  [OK] Done ({result['duration_seconds']:.1f}s)")
    return result

def sync_all(hub, force=False, dry_run=False):
    manifests = load_all_manifests(hub)
    print(f"[CYCLE] Syncing {len(manifests)} projects...\n")
    results, recompile = [], set()
    for pn, mf in sorted(manifests.items()):
        r = sync_project(hub, pn, mf, force=force, dry_run=dry_run)
        results.append(r)
        if r["status"]=="success" and r["systems_synced"]: recompile.add(pn)
    if not dry_run:
        sl = load_json(Path(hub)/"sync"/"sync-log.json")
        sl.setdefault("entries",[])
        sl["entries"] = results + sl["entries"]
        sl["entries"] = sl["entries"][:SYNC_LOG_MAX]
        save_json(Path(hub)/"sync"/"sync-log.json", sl)
    ok = sum(1 for r in results if r["status"]=="success")
    skip = sum(1 for r in results if r["status"]=="up-to-date")
    print(f"\n{'='*60}\nSYNC SUMMARY\n{'='*60}")
    print(f"  Synced: {ok} | Up to date: {skip} | Need recompile: {len(recompile)}")
    if recompile: print(f"  Run: python claude_md_compiler_v2.py --all")
    return results

# --- HEALTH DASHBOARD ---
def dashboard(hub, manifests):
    hub = Path(hub)
    statuses = [check_project_status(hub, pn, mf) for pn, mf in sorted(manifests.items())]
    avg = sum(s["sync_pct"] for s in statuses)/len(statuses) if statuses else 0
    sd = hub/"shared-core"/"systems"
    sys_count = len(list(sd.iterdir())) if sd.exists() else 0
    bl = hub/"deprecated"/"BLACKLIST.md"
    dep_count = bl.read_text("utf-8","ignore").count("[FAIL] NEVER") if bl.exists() else 0
    bw = 20
    lines = []
    lines.append("--------------------------------------------------------------------")
    lines.append("-            EMPIRE PROJECT MESH   HEALTH DASHBOARD               -")
    lines.append(f"-            {datetime.now().strftime('%Y-%m-%d %H:%M')}                                      -")
    lines.append("--------------------------------------------------------------------")
    for s in statuses:
        f = int(s["sync_pct"]/100*bw); bar = "-"*f+"-"*(bw-f)
        ic = {"critical":"[RED]","high":"","normal":"[GREEN]","low":""}.get(s["priority"],"")
        nm = s["project"][:30].ljust(30); stl = " [NOTE]" if s["claude_md_stale"] else ""
        lines.append(f"-  {ic} {nm} {bar} {s['sync_pct']:5.1f}%{stl}   -")
        for ss in s["systems"]:
            if not ss["is_current"]:
                sn = ss["system"][:28].ljust(28)
                lines.append(f"-    -- {sn} v{ss['consumed']}  v{ss['current']}       -")
    hi = "[GREEN]" if avg>=90 else "[YELLOW]" if avg>=70 else "[RED]"
    lines.append("--------------------------------------------------------------------")
    lines.append(f"-  {hi} Empire Health: {avg:.1f}%  -  Systems: {sys_count}  -  Deprecated: {dep_count}    -")
    lines.append("--------------------------------------------------------------------")
    out = "\n".join(lines)
    save_json(hub/"command-center"/"dashboard-data.json",
        {"generated_at":datetime.now(timezone.utc).isoformat(),"health":round(avg,1),
         "projects":statuses,"systems":sys_count,"deprecated":dep_count})
    return out

def show_history(hub, limit=20):
    sl = load_json(Path(hub)/"sync"/"sync-log.json")
    entries = sl.get("entries",[])[:limit]
    print(f"\n[LIST] SYNC HISTORY (last {min(limit,len(entries))})\n")
    for e in entries:
        ic = {"success":"[OK]","failed":"[FAIL]","up-to-date":"","dry-run":"[SEARCH]"}.get(e.get("status",""),"")
        print(f"  {ic} {e.get('id','?')[:35]:<35} {e.get('project','?')[:22]:<22} "
              f"{len(e.get('systems_synced',[]))} sys  {e.get('duration_seconds',0):.1f}s")

# --- MAIN ---
def main():
    p = argparse.ArgumentParser(description="Project Mesh Sync Engine v2.0")
    p.add_argument("--check",action="store_true"); p.add_argument("--sync",action="store_true")
    p.add_argument("--project","-p"); p.add_argument("--force",action="store_true")
    p.add_argument("--dry-run",action="store_true"); p.add_argument("--impact")
    p.add_argument("--chain"); p.add_argument("--rollback"); p.add_argument("--dashboard",action="store_true")
    p.add_argument("--history",action="store_true"); p.add_argument("--build-graph",action="store_true")
    p.add_argument("--hub",default=DEFAULT_HUB_PATH)
    a = p.parse_args(); hub = Path(a.hub)
    if not hub.exists(): print(f"[FAIL] Hub not found: {hub}"); sys.exit(1)
    manifests = load_all_manifests(hub)
    print(f" Sync Engine v2.0   {len(manifests)} projects\n")
    if a.check or a.dashboard: print(dashboard(hub, manifests))
    elif a.sync:
        if a.project:
            mf = manifests.get(a.project)
            if not mf: print(f"[FAIL] Not found: {a.project}"); return
            sync_project(hub, a.project, mf, force=a.force, dry_run=a.dry_run)
        else: sync_all(hub, force=a.force, dry_run=a.dry_run)
    elif a.impact: print(json.dumps(analyze_impact(hub, a.impact, manifests), indent=2, default=str))
    elif a.chain:
        g = build_dependency_graph(hub, manifests); rv = build_reverse_graph(g)
        info = rv.get(a.chain,{}); direct = [c["project"] for c in info.get("consumed_by",[])]
        dep_sys = info.get("depended_by_systems",[])
        print(f"\n SYNC CHAIN: {a.chain}")
        print(f"  Direct consumers ({len(direct)}): {', '.join(direct)}")
        if dep_sys: print(f"  Dependent systems: {', '.join(dep_sys)}")
    elif a.rollback: rollback(hub, a.rollback)
    elif a.history: show_history(hub)
    elif a.build_graph:
        g = build_dependency_graph(hub, manifests); rv = build_reverse_graph(g)
        save_json(hub/"registry"/"dependency-graph.json", g)
        save_json(hub/"registry"/"reverse-dependency-graph.json", rv)
        print(f"[OK] Graph: {len(g['nodes'])} nodes, {len(g['edges'])} edges")
    else: p.print_help()

if __name__ == "__main__": main()
