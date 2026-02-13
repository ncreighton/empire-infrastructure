"""
Empire Architect - Skill Deduplication Tool
Finds duplicate skills and helps consolidate them into the skill library.
"""
import json
import hashlib
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Tuple

PROJECTS_PATH = Path(r"C:\Claude Code Projects")
LIBRARY_PATH = Path(r"C:\Claude Code Projects\empire-skill-library\skills")

def hash_content(content: str) -> str:
    """Create a hash of content, normalized for comparison"""
    # Normalize whitespace and line endings
    normalized = ' '.join(content.split())
    return hashlib.md5(normalized.encode()).hexdigest()

def find_all_skills() -> List[Dict[str, Any]]:
    """Find all skills across all projects"""
    skills = []
    skip_dirs = {'__pycache__', 'node_modules', 'venv', '.venv', 'site-packages', '.git'}

    def scan_dir(path: Path):
        for item in path.iterdir():
            if item.name in skip_dirs or item.name.startswith('.'):
                continue
            if item.is_dir():
                if item.name == 'skills' or item.name == 'commands':
                    # Found a skills directory
                    for skill_file in item.rglob("*.md"):
                        try:
                            content = skill_file.read_text(encoding='utf-8', errors='replace')
                            skills.append({
                                "name": skill_file.stem,
                                "path": str(skill_file),
                                "project": get_project_name(skill_file),
                                "content": content,
                                "size": len(content),
                                "hash": hash_content(content),
                                "modified": datetime.fromtimestamp(skill_file.stat().st_mtime).isoformat()
                            })
                        except Exception as e:
                            pass
                else:
                    scan_dir(item)

    scan_dir(PROJECTS_PATH)
    return skills

def get_project_name(skill_path: Path) -> str:
    """Get project name from skill path"""
    parts = skill_path.parts
    try:
        # Find 'skills' in path and get parent
        for i, part in enumerate(parts):
            if part == 'skills' or part == 'commands':
                if i > 0:
                    return parts[i-1]
        return parts[-3] if len(parts) >= 3 else "unknown"
    except:
        return "unknown"

def find_duplicates(skills: List[Dict]) -> Dict[str, List[Dict]]:
    """Group skills by name to find duplicates"""
    by_name = defaultdict(list)
    for skill in skills:
        by_name[skill["name"]].append(skill)

    # Only return skills that appear more than once
    return {name: instances for name, instances in by_name.items() if len(instances) > 1}

def find_exact_duplicates(skills: List[Dict]) -> Dict[str, List[Dict]]:
    """Group skills by content hash to find exact duplicates"""
    by_hash = defaultdict(list)
    for skill in skills:
        by_hash[skill["hash"]].append(skill)

    return {h: instances for h, instances in by_hash.items() if len(instances) > 1}

def choose_canonical(instances: List[Dict]) -> Dict:
    """Choose the best version of a skill to be canonical"""
    # Prefer: larger content, more recently modified, from library
    def score(skill):
        s = 0
        if 'empire-skill-library' in skill['path']:
            s += 1000
        if 'empire-master' in skill['path']:
            s += 500
        s += skill['size']  # Larger is usually more complete
        return s

    return max(instances, key=score)

def analyze_duplicates():
    """Main analysis function"""
    print("\n" + "="*60)
    print("EMPIRE ARCHITECT - SKILL DEDUPLICATION ANALYSIS")
    print("="*60 + "\n")

    # Find all skills
    print("Scanning for skills...")
    skills = find_all_skills()
    print(f"Found {len(skills)} skill files\n")

    # Find duplicates by name
    name_dupes = find_duplicates(skills)
    print(f"Skills with same name: {len(name_dupes)}")

    # Find exact duplicates
    exact_dupes = find_exact_duplicates(skills)
    exact_count = sum(len(v) - 1 for v in exact_dupes.values())
    print(f"Exact duplicates (identical content): {exact_count}\n")

    # Detailed report
    print("-"*60)
    print("DUPLICATE SKILLS BY NAME")
    print("-"*60)

    recommendations = []

    for name, instances in sorted(name_dupes.items()):
        print(f"\n{name} ({len(instances)} copies):")

        # Check if they're identical
        hashes = set(i["hash"] for i in instances)
        if len(hashes) == 1:
            print("  [IDENTICAL CONTENT]")
        else:
            print(f"  [{len(hashes)} different versions]")

        canonical = choose_canonical(instances)

        for inst in instances:
            marker = " <- CANONICAL" if inst == canonical else ""
            print(f"  - {inst['project']}: {inst['size']} bytes{marker}")
            print(f"    {inst['path']}")

        recommendations.append({
            "skill_name": name,
            "instances": len(instances),
            "identical": len(hashes) == 1,
            "canonical_path": canonical["path"],
            "canonical_project": canonical["project"],
            "locations": [i["project"] for i in instances]
        })

    # Summary
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    # Skills to add to library
    library_skills = set()
    if LIBRARY_PATH.exists():
        library_skills = {f.stem for f in LIBRARY_PATH.glob("*.md")}

    print(f"\nSkill Library currently has: {len(library_skills)} skills")

    missing_from_library = []
    for rec in recommendations:
        if rec["skill_name"] not in library_skills:
            missing_from_library.append(rec)

    print(f"Duplicated skills NOT in library: {len(missing_from_library)}")
    for rec in missing_from_library[:10]:
        print(f"  - {rec['skill_name']} (from {rec['canonical_project']})")

    # Save report
    report = {
        "generated": datetime.now().isoformat(),
        "total_skills": len(skills),
        "duplicate_names": len(name_dupes),
        "exact_duplicates": exact_count,
        "recommendations": recommendations
    }

    report_path = PROJECTS_PATH / "empire-mcp-server" / "duplicate_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")

    print("\n" + "="*60)
    print("ACTIONS YOU CAN TAKE")
    print("="*60)
    print("""
1. Add canonical skills to library:
   python library.py add <skill_name> --from <canonical_path>

2. Propagate from library to all projects:
   python library.py propagate <skill_name>

3. Remove redundant copies after propagation:
   (Manual review recommended before deletion)
""")

    return report

if __name__ == "__main__":
    analyze_duplicates()
