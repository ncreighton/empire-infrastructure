"""
Project Templates - Generate templates from best projects
Creates reusable project templates based on high-scoring projects.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import re

PROJECTS_DIR = Path(r"C:\Claude Code Projects")
TEMPLATES_DIR = PROJECTS_DIR / "project-templates"

# Template definitions
TEMPLATES = {
    "python-api": {
        "description": "Python FastAPI service with MCP support",
        "source_projects": ["empire-dashboard", "empire-mcp-server"],
        "include_patterns": [
            "requirements.txt",
            "pyproject.toml",
            ".gitignore",
            "CLAUDE.md",
            ".claude/skills/*.md",
            ".claude/mcp.json",
        ],
        "structure": {
            "main.py": "# FastAPI application\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/')\ndef root():\n    return {'status': 'ok'}\n",
            "requirements.txt": "fastapi>=0.109.0\nuvicorn[standard]>=0.27.0\nhttpx>=0.26.0\npython-dotenv>=1.0.0\n",
            ".gitignore": "__pycache__/\n*.pyc\n.env\nvenv/\n.venv/\n*.egg-info/\ndist/\nbuild/\n",
        },
    },
    "mcp-server": {
        "description": "Model Context Protocol server",
        "source_projects": ["empire-mcp-server"],
        "include_patterns": [
            "server.py",
            "requirements.txt",
            "CLAUDE.md",
            ".claude/mcp.json",
        ],
        "structure": {
            "server.py": '''"""MCP Server Template"""
import asyncio
from mcp.server import Server
from mcp.types import Tool

app = Server("my-mcp-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(name="example", description="Example tool", inputSchema={"type": "object"})
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "example":
        return {"result": "Hello from MCP!"}
    raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    asyncio.run(app.run())
''',
            "requirements.txt": "mcp>=1.0.0\n",
            ".claude/mcp.json": '{\n  "mcpServers": {}\n}\n',
        },
    },
    "wordpress-site": {
        "description": "WordPress site management project",
        "source_projects": ["mythicalarchives", "witchcraftforbeginners"],
        "include_patterns": [
            "CLAUDE.md",
            ".claude/skills/*.md",
        ],
        "structure": {
            "CLAUDE.md": "# {{PROJECT_NAME}}\n\nWordPress site project.\n\n## Site Details\n- Domain: example.com\n- WP Admin: /wp-admin/\n\n## Content Guidelines\n- Target audience: \n- Tone: \n- SEO focus keywords:\n",
        },
    },
    "n8n-integration": {
        "description": "n8n workflow integration project",
        "source_projects": ["zimm-command-center-archived"],
        "include_patterns": [
            "CLAUDE.md",
            "workflows/*.json",
            ".claude/skills/*.md",
        ],
        "structure": {
            "CLAUDE.md": "# {{PROJECT_NAME}}\n\nn8n workflow integration.\n\n## Webhook Endpoints\n- https://your-n8n-instance.com/webhook/example\n\n## Workflows\n| ID | Name | Description |\n|----|------|-------------|\n",
            "workflows/.gitkeep": "",
        },
    },
    "minimal": {
        "description": "Minimal Claude Code project",
        "source_projects": [],
        "include_patterns": [],
        "structure": {
            "CLAUDE.md": "# {{PROJECT_NAME}}\n\n## Overview\nProject description here.\n\n## Quick Start\n```bash\n# Commands to get started\n```\n",
            ".gitignore": "*.pyc\n__pycache__/\nnode_modules/\n.env\n",
        },
    },
}


def extract_template_from_project(project_name: str) -> dict:
    """Extract template data from an existing project"""
    project_path = PROJECTS_DIR / project_name

    if not project_path.exists():
        return {"error": f"Project {project_name} not found"}

    template = {
        "source": project_name,
        "extracted": datetime.now().isoformat()[:19],
        "files": {},
        "structure": [],
    }

    # Get CLAUDE.md as template
    claude_md = project_path / "CLAUDE.md"
    if claude_md.exists():
        content = claude_md.read_text(encoding='utf-8', errors='ignore')
        # Anonymize
        content = re.sub(r'https?://[^\s]+', 'https://example.com', content)
        content = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'email@example.com', content)
        template['files']['CLAUDE.md'] = content

    # Get skills
    skills_dir = project_path / ".claude" / "skills"
    if not skills_dir.exists():
        skills_dir = project_path / "skills"

    if skills_dir.exists():
        for skill in skills_dir.glob("*.md"):
            relative = f".claude/skills/{skill.name}"
            content = skill.read_text(encoding='utf-8', errors='ignore')
            template['files'][relative] = content

    # Get MCP config (anonymized)
    mcp_config = project_path / ".claude" / "mcp.json"
    if mcp_config.exists():
        try:
            config = json.loads(mcp_config.read_text())
            # Keep structure but clear sensitive data
            if 'mcpServers' in config:
                for server in config['mcpServers'].values():
                    if 'args' in server:
                        server['args'] = ['path/to/server.py']
            template['files']['.claude/mcp.json'] = json.dumps(config, indent=2)
        except:
            pass

    # Get directory structure
    for item in project_path.rglob('*'):
        if item.is_file():
            relative = item.relative_to(project_path)
            # Skip certain paths
            if any(p in str(relative) for p in ['.git', '__pycache__', 'node_modules', '.env']):
                continue
            template['structure'].append(str(relative))

    return template


def create_project_from_template(template_name: str, project_name: str,
                                  target_dir: Path = None, dry_run: bool = True) -> dict:
    """Create a new project from a template"""
    if template_name not in TEMPLATES:
        return {"error": f"Template {template_name} not found"}

    template = TEMPLATES[template_name]
    target = target_dir or (PROJECTS_DIR / project_name)

    if target.exists() and not dry_run:
        return {"error": f"Directory {target} already exists"}

    results = {
        "template": template_name,
        "project": project_name,
        "target": str(target),
        "files_created": [],
        "dry_run": dry_run,
    }

    # Create files from structure
    for file_path, content in template['structure'].items():
        # Replace placeholders
        content = content.replace("{{PROJECT_NAME}}", project_name)

        full_path = target / file_path

        if dry_run:
            results['files_created'].append(str(file_path))
        else:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            results['files_created'].append(str(full_path))

    return results


def list_templates() -> list:
    """List available templates"""
    return [
        {
            "name": name,
            "description": t["description"],
            "source_projects": t["source_projects"],
            "files": list(t["structure"].keys()),
        }
        for name, t in TEMPLATES.items()
    ]


def save_template(name: str, template: dict):
    """Save a custom template"""
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    template_file = TEMPLATES_DIR / f"{name}.json"
    template_file.write_text(json.dumps(template, indent=2), encoding='utf-8')
    return str(template_file)


def load_custom_templates() -> dict:
    """Load custom templates from disk"""
    custom = {}
    if TEMPLATES_DIR.exists():
        for template_file in TEMPLATES_DIR.glob("*.json"):
            try:
                custom[template_file.stem] = json.loads(template_file.read_text())
            except:
                pass
    return custom


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Project Templates')
    parser.add_argument('--list', action='store_true', help='List available templates')
    parser.add_argument('--create', metavar='TEMPLATE', help='Create project from template')
    parser.add_argument('--name', help='New project name')
    parser.add_argument('--extract', metavar='PROJECT', help='Extract template from project')
    parser.add_argument('--save', metavar='NAME', help='Save extracted template as')
    parser.add_argument('--execute', action='store_true', help='Actually create files')

    args = parser.parse_args()

    if args.list:
        templates = list_templates()
        print("Available Templates:")
        print("-" * 50)
        for t in templates:
            print(f"\n{t['name']}: {t['description']}")
            print(f"  Files: {', '.join(t['files'])}")
            if t['source_projects']:
                print(f"  Based on: {', '.join(t['source_projects'])}")
        return

    if args.extract:
        template = extract_template_from_project(args.extract)
        if args.save:
            path = save_template(args.save, template)
            print(f"Template saved to: {path}")
        else:
            print(json.dumps(template, indent=2))
        return

    if args.create:
        if not args.name:
            print("Error: --name required when creating project")
            return

        result = create_project_from_template(
            args.create,
            args.name,
            dry_run=not args.execute
        )
        print(json.dumps(result, indent=2))
        return

    # Default: list templates
    templates = list_templates()
    print(f"Found {len(templates)} templates. Use --list for details.")


if __name__ == "__main__":
    main()
