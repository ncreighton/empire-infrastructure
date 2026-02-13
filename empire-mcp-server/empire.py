#!/usr/bin/env python
"""
Empire CLI - Unified command interface for Empire Architect
Quick commands for managing projects, skills, sync, and more.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

PROJECTS_DIR = Path(r"C:\Claude Code Projects")


def cmd_sync(args):
    """Trigger a full sync"""
    from sync_agent import main as sync_main
    print("Starting full sync...")
    sync_main()


def cmd_status(args):
    """Show quick status"""
    import requests

    webhook = "https://vmi2976539.contaboserver.net/webhook/empire/db-query"

    try:
        resp = requests.post(webhook, json={
            "query": """
                SELECT 'projects' as type, COUNT(*) as count FROM claude_projects
                UNION ALL SELECT 'skills', COUNT(*) FROM claude_skills
                UNION ALL SELECT 'agents', COUNT(*) FROM claude_agents
                UNION ALL SELECT 'workflows', COUNT(*) FROM claude_workflows
            """
        }, timeout=10)

        if resp.status_code == 200:
            data = {r['type']: r['count'] for r in resp.json()}
            print(f"""
Empire Architect Status
-----------------------
Projects:  {data.get('projects', 0)}
Skills:    {data.get('skills', 0)}
Agents:    {data.get('agents', 0)}
Workflows: {data.get('workflows', 0)}
            """)
        else:
            print(f"Error: {resp.status_code}")
    except Exception as e:
        print(f"Connection error: {e}")


def cmd_health(args):
    """Show project health scores"""
    import requests

    webhook = "https://vmi2976539.contaboserver.net/webhook/empire/db-query"

    try:
        resp = requests.post(webhook, json={
            "query": """
                SELECT project_name, is_git_repo, has_mcp_config
                FROM claude_projects
                LIMIT 20
            """
        }, timeout=10)

        if resp.status_code == 200:
            try:
                data = resp.json()
                if not data:
                    print("No projects found")
                    return
                print(f"{'Project':<35} {'Git':<6} {'MCP':<6}")
                print("-" * 50)
                for p in data:
                    git = "Yes" if p.get('is_git_repo') else "No"
                    mcp = "Yes" if p.get('has_mcp_config') else "No"
                    print(f"{p['project_name']:<35} {git:<6} {mcp:<6}")
            except json.JSONDecodeError:
                print(f"Invalid response: {resp.text[:200]}")
        else:
            print(f"Error: HTTP {resp.status_code}")
    except Exception as e:
        print(f"Connection error: {e}")


def cmd_search(args):
    """Search across projects and skills"""
    import requests

    query = args.query
    webhook = "https://vmi2976539.contaboserver.net/webhook/empire/db-query"

    resp = requests.post(webhook, json={
        "query": f"""
            SELECT 'project' as type, project_name as name
            FROM claude_projects
            WHERE project_name ILIKE '%{query}%' OR claude_md_content ILIKE '%{query}%'
            UNION ALL
            SELECT 'skill' as type, skill_name || ' (' || project_name || ')' as name
            FROM claude_skills
            WHERE skill_name ILIKE '%{query}%' OR content ILIKE '%{query}%'
            LIMIT 20
        """
    }, timeout=10)

    if resp.status_code == 200:
        results = resp.json()
        if results:
            print(f"Found {len(results)} results for '{query}':")
            for r in results:
                print(f"  [{r['type']}] {r['name']}")
        else:
            print(f"No results for '{query}'")


def cmd_duplicates(args):
    """Show duplicate skills"""
    from deduplicate_skills import analyze_duplicates
    result = analyze_duplicates()

    print(f"Total skills: {result['total_skills']}")
    print(f"Duplicate names: {result['duplicate_names']}")
    print(f"Exact duplicates: {result['exact_duplicates']}")

    if args.verbose:
        print("\nTop duplicates:")
        for name, info in list(result['by_name'].items())[:10]:
            if len(info['projects']) > 1:
                print(f"  {name}: {', '.join(info['projects'])}")


def cmd_propagate(args):
    """Propagate skills"""
    from skill_propagator import propagate_everywhere, propagate_to_project

    if args.project:
        project_path = PROJECTS_DIR / args.project
        result = propagate_to_project(project_path, dry_run=not args.execute)
    else:
        result = propagate_everywhere(dry_run=not args.execute)

    if isinstance(result, dict):
        print(f"Projects: {result.get('total_projects', 0)}")
        print(f"Would create: {result.get('skills_created', 0)}")
        print(f"Would update: {result.get('skills_updated', 0)}")
        print(f"Skipped: {result.get('skills_skipped', 0)}")
    else:
        for r in result:
            print(f"  {r['skill']}: {r['status']}")

    if not args.execute:
        print("\n(Dry run - use --execute to apply)")


def cmd_template(args):
    """Create project from template"""
    from project_templates import list_templates, create_project_from_template

    if args.list:
        for t in list_templates():
            print(f"{t['name']}: {t['description']}")
        return

    if not args.name:
        print("Error: --name required")
        return

    result = create_project_from_template(
        args.template,
        args.name,
        dry_run=not args.execute
    )

    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Template: {result['template']}")
        print(f"Target: {result['target']}")
        print(f"Files: {len(result['files_created'])}")
        for f in result['files_created']:
            print(f"  - {f}")

        if result['dry_run']:
            print("\n(Dry run - use --execute to create)")


def cmd_projects(args):
    """List projects"""
    import requests

    webhook = "https://vmi2976539.contaboserver.net/webhook/empire/db-query"

    try:
        resp = requests.post(webhook, json={
            "query": f"""
                SELECT project_name, is_git_repo, has_mcp_config
                FROM claude_projects
                LIMIT {args.limit or 20}
            """
        }, timeout=10)

        if resp.status_code == 200:
            projects = resp.json()
            if not projects:
                print("No projects found")
                return
            print(f"{'Project':<40} {'Git':<6} {'MCP':<6}")
            print("-" * 55)
            for p in projects:
                git = "Yes" if p.get('is_git_repo') else "No"
                mcp = "Yes" if p.get('has_mcp_config') else "No"
                print(f"{p['project_name']:<40} {git:<6} {mcp:<6}")
        else:
            print(f"Error: HTTP {resp.status_code}")
    except Exception as e:
        print(f"Connection error: {e}")


def cmd_skills(args):
    """List skills"""
    import requests

    webhook = "https://vmi2976539.contaboserver.net/webhook/empire/db-query"

    where = ""
    if args.project:
        where = f"WHERE project_name = '{args.project}'"

    resp = requests.post(webhook, json={
        "query": f"""
            SELECT skill_name, project_name, content_length
            FROM claude_skills
            {where}
            ORDER BY project_name, skill_name
            LIMIT {args.limit or 50}
        """
    }, timeout=10)

    if resp.status_code == 200:
        skills = resp.json()
        print(f"{'Skill':<30} {'Project':<30} {'Size':<10}")
        print("-" * 70)
        for s in skills:
            size = f"{(s.get('content_length', 0) or 0) / 1024:.1f}KB"
            print(f"{s['skill_name']:<30} {s['project_name']:<30} {size:<10}")


def cmd_open(args):
    """Open project in explorer/editor"""
    import subprocess

    project_path = PROJECTS_DIR / args.project

    if not project_path.exists():
        print(f"Project not found: {args.project}")
        return

    if args.code:
        subprocess.run(['code', str(project_path)], shell=True)
    else:
        subprocess.run(['explorer', str(project_path)], shell=True)

    print(f"Opened: {project_path}")


def cmd_dashboard(args):
    """Start the dashboard"""
    import subprocess
    import webbrowser

    dashboard_dir = PROJECTS_DIR / "empire-dashboard"
    port = args.port or 5000

    print(f"Starting dashboard on port {port}...")
    subprocess.Popen(
        ['python', 'app.py'],
        cwd=str(dashboard_dir),
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    import time
    time.sleep(2)
    webbrowser.open(f"http://localhost:{port}")


def main():
    parser = argparse.ArgumentParser(
        description='Empire CLI - Manage Claude Code projects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  empire status              Show overall status
  empire sync                Trigger full sync
  empire health              Show project health
  empire search "wordpress"  Search across everything
  empire projects            List projects
  empire skills --project empire-master
  empire duplicates          Show duplicate skills
  empire propagate           Preview skill propagation
  empire template --list     Show available templates
  empire open myproject      Open project in explorer
  empire dashboard           Start the web dashboard
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # status
    subparsers.add_parser('status', help='Show quick status')

    # sync
    subparsers.add_parser('sync', help='Trigger full sync')

    # health
    subparsers.add_parser('health', help='Show project health scores')

    # search
    p = subparsers.add_parser('search', help='Search projects and skills')
    p.add_argument('query', help='Search query')

    # duplicates
    p = subparsers.add_parser('duplicates', help='Show duplicate skills')
    p.add_argument('-v', '--verbose', action='store_true')

    # propagate
    p = subparsers.add_parser('propagate', help='Propagate skills')
    p.add_argument('--project', help='Target specific project')
    p.add_argument('--execute', action='store_true', help='Actually apply changes')

    # template
    p = subparsers.add_parser('template', help='Create from template')
    p.add_argument('template', nargs='?', help='Template name')
    p.add_argument('--list', action='store_true', help='List templates')
    p.add_argument('--name', help='New project name')
    p.add_argument('--execute', action='store_true', help='Actually create')

    # projects
    p = subparsers.add_parser('projects', help='List projects')
    p.add_argument('--limit', type=int, default=20)

    # skills
    p = subparsers.add_parser('skills', help='List skills')
    p.add_argument('--project', help='Filter by project')
    p.add_argument('--limit', type=int, default=50)

    # open
    p = subparsers.add_parser('open', help='Open project')
    p.add_argument('project', help='Project name')
    p.add_argument('--code', action='store_true', help='Open in VS Code')

    # dashboard
    p = subparsers.add_parser('dashboard', help='Start web dashboard')
    p.add_argument('--port', type=int, default=5000)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Dispatch
    commands = {
        'status': cmd_status,
        'sync': cmd_sync,
        'health': cmd_health,
        'search': cmd_search,
        'duplicates': cmd_duplicates,
        'propagate': cmd_propagate,
        'template': cmd_template,
        'projects': cmd_projects,
        'skills': cmd_skills,
        'open': cmd_open,
        'dashboard': cmd_dashboard,
    }

    if args.command in commands:
        try:
            commands[args.command](args)
        except Exception as e:
            print(f"Error: {e}")
            if os.environ.get('DEBUG'):
                raise
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
