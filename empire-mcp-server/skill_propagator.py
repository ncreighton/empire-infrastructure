"""
Skill Propagator - Automatically copy recommended skills to projects
Identifies canonical skills and propagates them to projects that need them.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import hashlib

PROJECTS_DIR = Path(r"C:\Claude Code Projects")
CANONICAL_PROJECT = "empire-master"  # The "gold standard" project

# Skills that every project should have
RECOMMENDED_SKILLS = [
    "commit",
    "review-pr",
    "test",
    "lint",
]

# Skills grouped by project type
SKILL_PROFILES = {
    "python": ["pytest", "pip-install", "uvicorn"],
    "node": ["npm-install", "npm-test", "npm-build"],
    "wordpress": ["wp-publish", "wp-update", "wp-api"],
    "mcp": ["mcp-test", "mcp-debug"],
}


def get_skill_hash(content: str) -> str:
    """Generate hash of skill content"""
    normalized = content.strip().replace('\r\n', '\n')
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def find_skills_dir(project_path: Path) -> Path | None:
    """Find skills directory in a project"""
    candidates = [
        project_path / ".claude" / "skills",
        project_path / "skills",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def ensure_skills_dir(project_path: Path) -> Path:
    """Create skills directory if it doesn't exist"""
    skills_dir = project_path / ".claude" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    return skills_dir


def load_canonical_skills() -> dict:
    """Load skills from the canonical project"""
    canonical_path = PROJECTS_DIR / CANONICAL_PROJECT
    skills_dir = find_skills_dir(canonical_path)

    if not skills_dir:
        print(f"Warning: No skills found in canonical project {CANONICAL_PROJECT}")
        return {}

    skills = {}
    for skill_file in skills_dir.glob("*.md"):
        skill_name = skill_file.stem
        content = skill_file.read_text(encoding='utf-8', errors='ignore')
        skills[skill_name] = {
            'path': str(skill_file),
            'content': content,
            'hash': get_skill_hash(content),
            'size': len(content),
        }

    return skills


def detect_project_type(project_path: Path) -> list:
    """Detect what type of project this is"""
    types = []

    if (project_path / "requirements.txt").exists() or \
       (project_path / "pyproject.toml").exists() or \
       (project_path / "setup.py").exists():
        types.append("python")

    if (project_path / "package.json").exists():
        types.append("node")

    if "wordpress" in project_path.name.lower() or \
       (project_path / "wp-config.php").exists():
        types.append("wordpress")

    if "mcp" in project_path.name.lower() or \
       (project_path / "server.py").exists() and \
       (project_path / ".claude" / "mcp.json").exists():
        types.append("mcp")

    return types


def get_recommended_skills_for_project(project_path: Path) -> set:
    """Get list of recommended skills for a project"""
    recommended = set(RECOMMENDED_SKILLS)

    project_types = detect_project_type(project_path)
    for ptype in project_types:
        if ptype in SKILL_PROFILES:
            recommended.update(SKILL_PROFILES[ptype])

    return recommended


def scan_project_skills(project_path: Path) -> dict:
    """Scan existing skills in a project"""
    skills_dir = find_skills_dir(project_path)
    if not skills_dir:
        return {}

    skills = {}
    for skill_file in skills_dir.glob("*.md"):
        content = skill_file.read_text(encoding='utf-8', errors='ignore')
        skills[skill_file.stem] = {
            'path': str(skill_file),
            'hash': get_skill_hash(content),
        }

    return skills


def propagate_skill(skill_name: str, canonical_skills: dict, project_path: Path,
                    dry_run: bool = True) -> dict:
    """
    Propagate a skill to a project.
    Returns status dict with result information.
    """
    if skill_name not in canonical_skills:
        return {'status': 'error', 'message': f'Skill {skill_name} not found in canonical project'}

    source = canonical_skills[skill_name]
    skills_dir = ensure_skills_dir(project_path)
    target_path = skills_dir / f"{skill_name}.md"

    # Check if skill already exists
    if target_path.exists():
        existing_content = target_path.read_text(encoding='utf-8', errors='ignore')
        existing_hash = get_skill_hash(existing_content)

        if existing_hash == source['hash']:
            return {'status': 'skipped', 'message': 'Skill already exists with same content'}
        else:
            # Skill exists but different content
            if dry_run:
                return {'status': 'would_update', 'message': 'Would update existing skill'}

            # Backup existing
            backup_path = target_path.with_suffix('.md.bak')
            shutil.copy(target_path, backup_path)

    if dry_run:
        return {'status': 'would_create', 'message': f'Would create {target_path}'}

    # Write skill with propagation header
    header = f"""<!--
Propagated from: {CANONICAL_PROJECT}
Propagated on: {datetime.now().isoformat()[:19]}
Source hash: {source['hash']}
-->

"""
    target_path.write_text(header + source['content'], encoding='utf-8')

    return {'status': 'created', 'message': f'Created {target_path}'}


def analyze_project(project_path: Path, canonical_skills: dict) -> dict:
    """Analyze a project for missing/outdated skills"""
    project_name = project_path.name
    existing_skills = scan_project_skills(project_path)
    recommended = get_recommended_skills_for_project(project_path)
    project_types = detect_project_type(project_path)

    # Filter recommended to only those we have in canonical
    available_recommended = recommended.intersection(set(canonical_skills.keys()))

    missing = []
    outdated = []
    current = []

    for skill_name in available_recommended:
        if skill_name not in existing_skills:
            missing.append(skill_name)
        else:
            if existing_skills[skill_name]['hash'] != canonical_skills[skill_name]['hash']:
                outdated.append(skill_name)
            else:
                current.append(skill_name)

    return {
        'project': project_name,
        'path': str(project_path),
        'types': project_types,
        'existing_count': len(existing_skills),
        'recommended_count': len(available_recommended),
        'missing': missing,
        'outdated': outdated,
        'current': current,
        'score': len(current) / max(1, len(available_recommended)) * 100,
    }


def scan_all_projects(exclude_archived: bool = True) -> list:
    """Scan all projects and return analysis results"""
    canonical_skills = load_canonical_skills()
    print(f"Loaded {len(canonical_skills)} canonical skills from {CANONICAL_PROJECT}")

    results = []

    for item in PROJECTS_DIR.iterdir():
        if not item.is_dir():
            continue

        # Skip certain directories
        if item.name.startswith('.') or item.name.startswith('_'):
            if exclude_archived:
                continue

        if item.name == CANONICAL_PROJECT:
            continue

        # Check if it's a project
        has_claudemd = (item / "CLAUDE.md").exists()
        has_skills = find_skills_dir(item) is not None
        has_git = (item / ".git").exists()

        if not (has_claudemd or has_skills or has_git):
            continue

        analysis = analyze_project(item, canonical_skills)
        results.append(analysis)

    return results


def propagate_to_project(project_path: Path, skills: list = None, dry_run: bool = True) -> list:
    """Propagate skills to a specific project"""
    canonical_skills = load_canonical_skills()
    project_name = project_path.name

    if skills is None:
        # Get recommended skills for this project
        analysis = analyze_project(project_path, canonical_skills)
        skills = analysis['missing'] + analysis['outdated']

    results = []
    for skill_name in skills:
        result = propagate_skill(skill_name, canonical_skills, project_path, dry_run)
        results.append({
            'skill': skill_name,
            'project': project_name,
            **result
        })

    return results


def propagate_everywhere(skills: list = None, dry_run: bool = True) -> dict:
    """Propagate skills to all projects"""
    canonical_skills = load_canonical_skills()

    if skills is None:
        skills = list(RECOMMENDED_SKILLS)

    summary = {
        'total_projects': 0,
        'skills_created': 0,
        'skills_updated': 0,
        'skills_skipped': 0,
        'details': []
    }

    for item in PROJECTS_DIR.iterdir():
        if not item.is_dir():
            continue

        if item.name.startswith('.') or item.name.startswith('_'):
            continue

        if item.name == CANONICAL_PROJECT:
            continue

        # Check if it's a project
        if not ((item / "CLAUDE.md").exists() or (item / ".git").exists()):
            continue

        summary['total_projects'] += 1

        for skill_name in skills:
            if skill_name not in canonical_skills:
                continue

            result = propagate_skill(skill_name, canonical_skills, item, dry_run)

            if result['status'] in ['created', 'would_create']:
                summary['skills_created'] += 1
            elif result['status'] in ['updated', 'would_update']:
                summary['skills_updated'] += 1
            else:
                summary['skills_skipped'] += 1

            summary['details'].append({
                'project': item.name,
                'skill': skill_name,
                **result
            })

    return summary


def generate_report() -> dict:
    """Generate a comprehensive propagation report"""
    canonical_skills = load_canonical_skills()
    projects = scan_all_projects()

    report = {
        'generated': datetime.now().isoformat()[:19],
        'canonical_project': CANONICAL_PROJECT,
        'canonical_skills': list(canonical_skills.keys()),
        'total_projects': len(projects),
        'projects': sorted(projects, key=lambda x: x['score']),
        'summary': {
            'total_missing': sum(len(p['missing']) for p in projects),
            'total_outdated': sum(len(p['outdated']) for p in projects),
            'avg_score': sum(p['score'] for p in projects) / max(1, len(projects)),
            'projects_needing_skills': len([p for p in projects if p['missing'] or p['outdated']]),
        }
    }

    return report


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Skill Propagator')
    parser.add_argument('--report', action='store_true', help='Generate analysis report')
    parser.add_argument('--propagate', action='store_true', help='Propagate skills')
    parser.add_argument('--project', help='Target specific project')
    parser.add_argument('--skill', help='Specific skill to propagate')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Dry run (default)')
    parser.add_argument('--execute', action='store_true', help='Actually execute changes')

    args = parser.parse_args()
    dry_run = not args.execute

    if args.report:
        report = generate_report()
        print(json.dumps(report, indent=2))
        return

    if args.propagate:
        if args.project:
            project_path = PROJECTS_DIR / args.project
            if not project_path.exists():
                print(f"Project not found: {args.project}")
                return

            skills = [args.skill] if args.skill else None
            results = propagate_to_project(project_path, skills, dry_run)
        else:
            skills = [args.skill] if args.skill else None
            results = propagate_everywhere(skills, dry_run)

        print(json.dumps(results, indent=2))
        return

    # Default: show analysis
    results = scan_all_projects()
    for r in sorted(results, key=lambda x: x['score']):
        missing = ', '.join(r['missing'][:5]) or 'None'
        if len(r['missing']) > 5:
            missing += f' (+{len(r["missing"])-5} more)'
        print(f"{r['project']}: {r['score']:.0f}% | Missing: {missing}")


if __name__ == "__main__":
    main()
