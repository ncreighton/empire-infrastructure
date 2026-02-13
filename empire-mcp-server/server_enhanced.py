"""
Empire Architect MCP Server - Enhanced Edition
Comprehensive tools for managing the entire Claude Code empire.
"""

import json
import sys
import os
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
import requests

PROTOCOL_VERSION = "2024-11-05"
N8N_WEBHOOK = "https://vmi2976539.contaboserver.net/webhook/empire/db-query"
N8N_BASE = "https://vmi2976539.contaboserver.net"
PROJECTS_PATH = Path(r"C:\Claude Code Projects")
LIBRARY_PATH = Path(r"C:\Claude Code Projects\empire-skill-library\skills")


def query_db(sql: str) -> list:
    """Execute SQL via n8n webhook"""
    try:
        resp = requests.post(N8N_WEBHOOK, json={"query": sql}, timeout=30)
        if resp.status_code == 200:
            # Handle empty response body (no results)
            if not resp.text or resp.text.strip() == "":
                return []
            return resp.json()
        return []
    except Exception as e:
        return [{"error": str(e)}]


def send_response(response: dict):
    print(json.dumps(response), flush=True)


def send_error(id: Any, code: int, message: str):
    send_response({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })


def handle_initialize(id: Any, params: dict):
    send_response({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {
                "name": "empire-architect",
                "version": "2.0.0"
            },
            "capabilities": {
                "tools": {}
            }
        }
    })


def get_all_tools():
    """Define all available tools"""
    return [
        # ═══════════════════════════════════════════════════════════════
        # SEARCH & DISCOVERY
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_search_skills",
            "description": "Search for skills across all Claude Code projects by name or content",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"},
                    "project": {"type": "string", "description": "Filter by project name"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"}
                }
            }
        },
        {
            "name": "empire_search_content",
            "description": "Full-text search across all CLAUDE.md files and skill content",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for"},
                    "type": {"type": "string", "enum": ["all", "projects", "skills"], "description": "What to search"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "empire_search_code",
            "description": "Search for code patterns across all projects (grep-like)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Code pattern or regex to find"},
                    "file_type": {"type": "string", "description": "File extension filter (e.g., 'py', 'js')"},
                    "project": {"type": "string", "description": "Limit to specific project"}
                },
                "required": ["pattern"]
            }
        },

        # ═══════════════════════════════════════════════════════════════
        # SKILL MANAGEMENT
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_get_skill",
            "description": "Get the full content of a specific skill",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Name of the skill"},
                    "project": {"type": "string", "description": "Specific project (optional)"}
                },
                "required": ["skill_name"]
            }
        },
        {
            "name": "empire_create_skill",
            "description": "Create a new skill in a project",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project name"},
                    "skill_name": {"type": "string", "description": "Name for the skill file (without .md)"},
                    "content": {"type": "string", "description": "Skill content/instructions"}
                },
                "required": ["project", "skill_name", "content"]
            }
        },
        {
            "name": "empire_copy_skill",
            "description": "Copy a skill from one project to another",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Name of skill to copy"},
                    "from_project": {"type": "string", "description": "Source project"},
                    "to_project": {"type": "string", "description": "Destination project"}
                },
                "required": ["skill_name", "from_project", "to_project"]
            }
        },
        {
            "name": "empire_find_duplicates",
            "description": "Find duplicate skills (same content) across projects",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "empire_skill_usage",
            "description": "Find all projects that have a specific skill",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Skill name to search for"}
                },
                "required": ["skill_name"]
            }
        },
        {
            "name": "empire_recommend_skills",
            "description": "Get skill recommendations for a project based on similar projects",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Project to get recommendations for"}
                },
                "required": ["project_name"]
            }
        },

        # ═══════════════════════════════════════════════════════════════
        # PROJECT MANAGEMENT
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_list_projects",
            "description": "List all Claude Code projects with metadata",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filter": {"type": "string", "description": "Filter by name"},
                    "has_skills": {"type": "boolean", "description": "Only projects with skills"},
                    "has_mcp": {"type": "boolean", "description": "Only projects with MCP config"},
                    "is_git": {"type": "boolean", "description": "Only git repositories"},
                    "sort_by": {"type": "string", "enum": ["name", "skills", "updated"], "description": "Sort order"}
                }
            }
        },
        {
            "name": "empire_get_project",
            "description": "Get detailed info about a project including CLAUDE.md content",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Project name"}
                },
                "required": ["project_name"]
            }
        },
        {
            "name": "empire_create_project",
            "description": "Create a new project from a template",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "New project name"},
                    "template": {"type": "string", "enum": ["basic", "python-project", "web-project", "wordpress-site", "automation-project", "empire-module"], "description": "Template to use"},
                    "description": {"type": "string", "description": "Project description for CLAUDE.md"}
                },
                "required": ["name", "template"]
            }
        },
        {
            "name": "empire_compare_projects",
            "description": "Compare two projects - skills, structure, configuration",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project1": {"type": "string", "description": "First project"},
                    "project2": {"type": "string", "description": "Second project"}
                },
                "required": ["project1", "project2"]
            }
        },
        {
            "name": "empire_analyze_project",
            "description": "Analyze a project for completeness, best practices, and suggestions",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Project to analyze"}
                },
                "required": ["project_name"]
            }
        },

        # ═══════════════════════════════════════════════════════════════
        # STATISTICS & ANALYTICS
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_get_stats",
            "description": "Get overall statistics about the Empire",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "empire_recent_changes",
            "description": "Get recently modified projects, skills, and files",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "hours": {"type": "integer", "description": "Look back hours (default 24)"},
                    "type": {"type": "string", "enum": ["all", "projects", "skills", "files"], "description": "What to show"}
                }
            }
        },
        {
            "name": "empire_skill_distribution",
            "description": "Show skill distribution across projects",
            "inputSchema": {"type": "object", "properties": {}}
        },

        # ═══════════════════════════════════════════════════════════════
        # WORKFLOW & AUTOMATION
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_list_workflows",
            "description": "List all n8n workflows across projects",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Filter by project"}
                }
            }
        },
        {
            "name": "empire_trigger_sync",
            "description": "Trigger a manual sync of all projects to the database",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "empire_get_sync_status",
            "description": "Get the last sync time and status",
            "inputSchema": {"type": "object", "properties": {}}
        },

        # ═══════════════════════════════════════════════════════════════
        # SKILL LIBRARY
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_library_list",
            "description": "List all skills in the central skill library",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "description": "Filter by tag"}
                }
            }
        },
        {
            "name": "empire_library_add",
            "description": "Add a skill to the central library",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Skill to add (from a project)"},
                    "from_project": {"type": "string", "description": "Source project"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for the skill"}
                },
                "required": ["skill_name", "from_project"]
            }
        },
        {
            "name": "empire_library_install",
            "description": "Install a skill from the library to a project",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Skill to install"},
                    "to_project": {"type": "string", "description": "Destination project"}
                },
                "required": ["skill_name", "to_project"]
            }
        },

        # ═══════════════════════════════════════════════════════════════
        # TEMPLATES
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_template_list",
            "description": "List available project templates",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "empire_template_create",
            "description": "Create a template from an existing project",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Project to create template from"},
                    "template_name": {"type": "string", "description": "Name for the new template"}
                },
                "required": ["project_name", "template_name"]
            }
        },

        # ═══════════════════════════════════════════════════════════════
        # WORDPRESS SITES (Empire-specific)
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_list_sites",
            "description": "List all WordPress sites in the empire with their status",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "empire_site_config",
            "description": "Get configuration for a specific WordPress site",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "site_id": {"type": "string", "description": "Site identifier (e.g., 'smarthomewizards')"}
                },
                "required": ["site_id"]
            }
        },

        # ═══════════════════════════════════════════════════════════════
        # MCP & HOOKS
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_list_mcp_configs",
            "description": "List all projects with MCP server configurations",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "empire_get_mcp_config",
            "description": "Get MCP configuration for a specific project",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Project name"}
                },
                "required": ["project_name"]
            }
        },

        # ═══════════════════════════════════════════════════════════════
        # FILE OPERATIONS
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_read_file",
            "description": "Read a file from any project in the empire",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project name"},
                    "file_path": {"type": "string", "description": "Relative path within project"}
                },
                "required": ["project", "file_path"]
            }
        },
        {
            "name": "empire_list_files",
            "description": "List files in a project directory",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project name"},
                    "path": {"type": "string", "description": "Subdirectory (optional)"},
                    "pattern": {"type": "string", "description": "Glob pattern filter"}
                },
                "required": ["project"]
            }
        },

        # ═══════════════════════════════════════════════════════════════
        # SCREENPIPE (Screen Intelligence)
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_screen_search",
            "description": "Search Screenpipe screen/audio history for content. Queries OCR text and audio transcriptions from your screen recordings.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (text seen on screen or spoken)"},
                    "content_type": {"type": "string", "enum": ["ocr", "audio", "all"], "description": "Type of content to search (default: all)"},
                    "limit": {"type": "integer", "description": "Max results (default 10)"},
                    "start_time": {"type": "string", "description": "ISO8601 start time filter"},
                    "end_time": {"type": "string", "description": "ISO8601 end time filter"},
                    "app_name": {"type": "string", "description": "Filter by application name"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "empire_screen_timeline",
            "description": "Get recent screen activity timeline from Screenpipe. Shows what apps were used and what was on screen.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of recent entries (default 20)"},
                    "content_type": {"type": "string", "enum": ["ocr", "audio", "all"], "description": "Content type filter"},
                    "app_name": {"type": "string", "description": "Filter by app name"}
                }
            }
        },
        {
            "name": "empire_monitor_check",
            "description": "Health check on Screenpipe recording status. Returns whether Screenpipe is running, recording, and basic stats.",
            "inputSchema": {"type": "object", "properties": {}}
        },

        # ═══════════════════════════════════════════════════════════════
        # GIT OPERATIONS
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "empire_git_status",
            "description": "Get git status for all repositories or a specific project",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Specific project (optional)"}
                }
            }
        },
        {
            "name": "empire_git_log",
            "description": "Get recent git commits for a project",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project name"},
                    "count": {"type": "integer", "description": "Number of commits (default 10)"}
                },
                "required": ["project"]
            }
        },
    ]


def handle_tools_list(id: Any):
    send_response({
        "jsonrpc": "2.0",
        "id": id,
        "result": {"tools": get_all_tools()}
    })


def execute_tool(name: str, args: dict) -> Any:
    """Execute a tool and return the result"""

    # ═══════════════════════════════════════════════════════════════
    # SEARCH & DISCOVERY
    # ═══════════════════════════════════════════════════════════════

    if name == "empire_search_skills":
        query = args.get("query", "")
        project = args.get("project", "")
        limit = args.get("limit", 20)

        where = []
        if query:
            where.append(f"(skill_name ILIKE '%{query}%' OR content ILIKE '%{query}%')")
        if project:
            where.append(f"project_name ILIKE '%{project}%'")

        where_clause = f"WHERE {' AND '.join(where)}" if where else ""

        return query_db(f"""
            SELECT skill_name, project_name, content_length,
                   LEFT(content, 500) as preview
            FROM claude_skills
            {where_clause}
            ORDER BY skill_name
            LIMIT {limit}
        """)

    elif name == "empire_search_content":
        query = args.get("query", "")
        search_type = args.get("type", "all")

        results = []
        if search_type in ["all", "projects"]:
            projects = query_db(f"""
                SELECT 'project' as type, project_name as name,
                       LEFT(claude_md_content, 300) as preview
                FROM claude_projects
                WHERE claude_md_content ILIKE '%{query}%'
                LIMIT 10
            """)
            results.extend(projects)

        if search_type in ["all", "skills"]:
            skills = query_db(f"""
                SELECT 'skill' as type,
                       skill_name || ' (' || project_name || ')' as name,
                       LEFT(content, 300) as preview
                FROM claude_skills
                WHERE content ILIKE '%{query}%' OR skill_name ILIKE '%{query}%'
                LIMIT 10
            """)
            results.extend(skills)

        return results

    elif name == "empire_search_code":
        pattern = args.get("pattern", "")
        file_type = args.get("file_type", "")
        project = args.get("project", "")

        search_path = PROJECTS_PATH / project if project else PROJECTS_PATH
        glob_pattern = f"*.{file_type}" if file_type else "*"

        try:
            result = subprocess.run(
                ["grep", "-r", "-l", "-I", "--include=" + glob_pattern, pattern, str(search_path)],
                capture_output=True, text=True, timeout=30
            )
            files = result.stdout.strip().split("\n")[:20]
            return [{"file": f, "pattern": pattern} for f in files if f]
        except:
            return [{"error": "Search failed or grep not available"}]

    # ═══════════════════════════════════════════════════════════════
    # SKILL MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    elif name == "empire_get_skill":
        skill_name = args.get("skill_name", "")
        project = args.get("project", "")

        where = f"AND project_name = '{project}'" if project else ""
        return query_db(f"""
            SELECT skill_name, project_name, skill_path, content,
                   content_length, content_hash, updated_at
            FROM claude_skills
            WHERE skill_name ILIKE '%{skill_name}%' {where}
            LIMIT 5
        """)

    elif name == "empire_create_skill":
        project = args.get("project", "")
        skill_name = args.get("skill_name", "")
        content = args.get("content", "")

        # Find project path
        project_path = PROJECTS_PATH / project
        if not project_path.exists():
            # Try to find it
            matches = list(PROJECTS_PATH.glob(f"*{project}*"))
            if matches:
                project_path = matches[0]
            else:
                return {"error": f"Project '{project}' not found"}

        skills_dir = project_path / ".claude" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        skill_file = skills_dir / f"{skill_name}.md"
        skill_file.write_text(content)

        return {"success": True, "path": str(skill_file), "message": f"Created skill '{skill_name}' in {project}"}

    elif name == "empire_copy_skill":
        skill_name = args.get("skill_name", "")
        from_project = args.get("from_project", "")
        to_project = args.get("to_project", "")

        # Get skill content from database
        skills = query_db(f"""
            SELECT content FROM claude_skills
            WHERE skill_name = '{skill_name}' AND project_name ILIKE '%{from_project}%'
            LIMIT 1
        """)

        if not skills or not skills[0].get("content"):
            return {"error": f"Skill '{skill_name}' not found in '{from_project}'"}

        # Create in target project
        return execute_tool("empire_create_skill", {
            "project": to_project,
            "skill_name": skill_name,
            "content": skills[0]["content"]
        })

    elif name == "empire_find_duplicates":
        return query_db("""
            SELECT content_hash, COUNT(*) as copies,
                   array_agg(skill_name) as skill_names,
                   array_agg(project_name) as projects
            FROM claude_skills
            WHERE content_hash IS NOT NULL
            GROUP BY content_hash
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 20
        """)

    elif name == "empire_skill_usage":
        skill_name = args.get("skill_name", "")
        return query_db(f"""
            SELECT skill_name, project_name, skill_path, content_length
            FROM claude_skills
            WHERE skill_name ILIKE '%{skill_name}%'
            ORDER BY project_name
        """)

    elif name == "empire_recommend_skills":
        project_name = args.get("project_name", "")
        return query_db(f"""
            WITH project_skills AS (
                SELECT skill_name FROM claude_skills
                WHERE project_name ILIKE '%{project_name}%'
            ),
            popular_skills AS (
                SELECT skill_name, COUNT(DISTINCT project_name) as usage_count
                FROM claude_skills
                WHERE skill_name NOT IN (SELECT skill_name FROM project_skills)
                GROUP BY skill_name
                HAVING COUNT(DISTINCT project_name) >= 2
            )
            SELECT ps.skill_name, ps.usage_count,
                   array_agg(DISTINCT cs.project_name) as used_in
            FROM popular_skills ps
            JOIN claude_skills cs ON cs.skill_name = ps.skill_name
            GROUP BY ps.skill_name, ps.usage_count
            ORDER BY ps.usage_count DESC
            LIMIT 10
        """)

    # ═══════════════════════════════════════════════════════════════
    # PROJECT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    elif name == "empire_list_projects":
        filter_name = args.get("filter", "")
        has_skills = args.get("has_skills", False)
        has_mcp = args.get("has_mcp", False)
        is_git = args.get("is_git", False)
        sort_by = args.get("sort_by", "name")

        where = []
        if filter_name:
            where.append(f"p.project_name ILIKE '%{filter_name}%'")
        if has_skills:
            where.append("(SELECT COUNT(*) FROM claude_skills s WHERE s.project_name = p.project_name) > 0")
        if has_mcp:
            where.append("p.has_mcp_config = true")
        if is_git:
            where.append("p.is_git_repo = true")

        where_clause = f"WHERE {' AND '.join(where)}" if where else ""

        order = {
            "name": "p.project_name",
            "skills": "skill_count DESC",
            "updated": "p.updated_at DESC"
        }.get(sort_by, "p.project_name")

        return query_db(f"""
            SELECT p.project_name, p.project_path,
                   (SELECT COUNT(*) FROM claude_skills s WHERE s.project_name = p.project_name) as skill_count,
                   (SELECT COUNT(*) FROM claude_agents a WHERE a.project_name = p.project_name) as agent_count,
                   (SELECT COUNT(*) FROM claude_workflows w WHERE w.project_name = p.project_name) as workflow_count,
                   p.has_mcp_config, p.is_git_repo, p.git_branch, p.updated_at
            FROM claude_projects p
            {where_clause}
            ORDER BY {order}
            LIMIT 50
        """)

    elif name == "empire_get_project":
        project_name = args.get("project_name", "")
        return query_db(f"""
            SELECT p.project_name, p.project_path, p.claude_md_content,
                   (SELECT COUNT(*) FROM claude_skills s WHERE s.project_name = p.project_name) as skill_count,
                   (SELECT array_agg(s.skill_name) FROM claude_skills s WHERE s.project_name = p.project_name) as skill_names,
                   (SELECT COUNT(*) FROM claude_agents a WHERE a.project_name = p.project_name) as agent_count,
                   (SELECT array_agg(a.agent_name) FROM claude_agents a WHERE a.project_name = p.project_name) as agent_names,
                   (SELECT COUNT(*) FROM claude_workflows w WHERE w.project_name = p.project_name) as workflow_count,
                   (SELECT array_agg(w.workflow_name) FROM claude_workflows w WHERE w.project_name = p.project_name) as workflow_names,
                   p.has_mcp_config, p.mcp_servers,
                   p.is_git_repo, p.git_branch, p.git_last_commit_message, p.updated_at
            FROM claude_projects p
            WHERE p.project_name ILIKE '%{project_name}%'
            LIMIT 1
        """)

    elif name == "empire_create_project":
        name = args.get("name", "")
        template = args.get("template", "basic")
        description = args.get("description", "")

        try:
            result = subprocess.run(
                ["python", str(PROJECTS_PATH / "empire-templates" / "templates.py"),
                 "create", template, name],
                capture_output=True, text=True, timeout=30
            )
            return {"success": True, "output": result.stdout, "path": str(PROJECTS_PATH / name)}
        except Exception as e:
            return {"error": str(e)}

    elif name == "empire_compare_projects":
        p1 = args.get("project1", "")
        p2 = args.get("project2", "")

        proj1 = query_db(f"SELECT * FROM claude_projects WHERE project_name ILIKE '%{p1}%' LIMIT 1")
        proj2 = query_db(f"SELECT * FROM claude_projects WHERE project_name ILIKE '%{p2}%' LIMIT 1")

        skills1 = query_db(f"SELECT skill_name FROM claude_skills WHERE project_name ILIKE '%{p1}%'")
        skills2 = query_db(f"SELECT skill_name FROM claude_skills WHERE project_name ILIKE '%{p2}%'")

        s1_names = {s.get("skill_name") for s in skills1}
        s2_names = {s.get("skill_name") for s in skills2}

        return {
            "project1": proj1[0] if proj1 else None,
            "project2": proj2[0] if proj2 else None,
            "common_skills": list(s1_names & s2_names),
            "only_in_project1": list(s1_names - s2_names),
            "only_in_project2": list(s2_names - s1_names)
        }

    elif name == "empire_analyze_project":
        project_name = args.get("project_name", "")

        project = query_db(f"""
            SELECT * FROM claude_projects WHERE project_name ILIKE '%{project_name}%' LIMIT 1
        """)

        if not project:
            return {"error": "Project not found"}

        p = project[0]
        score = 0
        suggestions = []

        # Scoring
        if p.get("claude_md_content"):
            score += 20
        else:
            suggestions.append("Add a CLAUDE.md file with project documentation")

        if p.get("skill_count", 0) > 0:
            score += 20
        else:
            suggestions.append("Add skills to help Claude understand project patterns")

        if p.get("has_mcp_config"):
            score += 15
        else:
            suggestions.append("Consider adding MCP server configuration")

        if p.get("is_git_repo"):
            score += 15
            if not p.get("git_is_dirty"):
                score += 10
            else:
                suggestions.append("Commit uncommitted changes")
        else:
            suggestions.append("Initialize git repository for version control")

        if p.get("workflow_count", 0) > 0:
            score += 10

        if p.get("agent_count", 0) > 0:
            score += 10

        return {
            "project": p.get("project_name"),
            "health_score": score,
            "max_score": 100,
            "grade": "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D",
            "suggestions": suggestions,
            "stats": {
                "skills": p.get("skill_count", 0),
                "agents": p.get("agent_count", 0),
                "workflows": p.get("workflow_count", 0),
                "has_mcp": p.get("has_mcp_config", False),
                "is_git": p.get("is_git_repo", False)
            }
        }

    # ═══════════════════════════════════════════════════════════════
    # STATISTICS & ANALYTICS
    # ═══════════════════════════════════════════════════════════════

    elif name == "empire_get_stats":
        return query_db("""
            SELECT
                (SELECT COUNT(*) FROM claude_projects) as total_projects,
                (SELECT COUNT(*) FROM claude_skills) as total_skills,
                (SELECT COUNT(DISTINCT skill_name) FROM claude_skills) as unique_skill_names,
                (SELECT COUNT(*) FROM claude_agents) as total_agents,
                (SELECT COUNT(*) FROM claude_workflows) as total_workflows,
                (SELECT COUNT(*) FROM claude_mcp_configs) as projects_with_mcp,
                (SELECT COUNT(*) FROM claude_projects WHERE is_git_repo = true) as git_repos,
                (SELECT COUNT(*) FROM claude_hooks) as total_hooks
        """)

    elif name == "empire_recent_changes":
        hours = args.get("hours", 24)
        change_type = args.get("type", "all")

        results = {}

        if change_type in ["all", "projects"]:
            results["projects"] = query_db(f"""
                SELECT project_name, updated_at FROM claude_projects
                WHERE updated_at > NOW() - INTERVAL '{hours} hours'
                ORDER BY updated_at DESC LIMIT 10
            """)

        if change_type in ["all", "skills"]:
            results["skills"] = query_db(f"""
                SELECT skill_name, project_name, updated_at FROM claude_skills
                WHERE updated_at > NOW() - INTERVAL '{hours} hours'
                ORDER BY updated_at DESC LIMIT 10
            """)

        return results

    elif name == "empire_skill_distribution":
        return query_db("""
            SELECT project_name, COUNT(*) as skill_count,
                   SUM(content_length) as total_content_size
            FROM claude_skills
            GROUP BY project_name
            ORDER BY skill_count DESC
        """)

    # ═══════════════════════════════════════════════════════════════
    # WORKFLOW & AUTOMATION
    # ═══════════════════════════════════════════════════════════════

    elif name == "empire_list_workflows":
        project = args.get("project", "")
        where = f"WHERE project_name ILIKE '%{project}%'" if project else ""
        return query_db(f"""
            SELECT workflow_name, project_name, workflow_path, last_modified
            FROM claude_workflows
            {where}
            ORDER BY project_name, workflow_name
            LIMIT 50
        """)

    elif name == "empire_trigger_sync":
        try:
            result = subprocess.Popen(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File",
                 str(PROJECTS_PATH / "empire-claude-agent.ps1"), "-once"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return {"status": "triggered", "message": "Sync started in background"}
        except Exception as e:
            return {"error": str(e)}

    elif name == "empire_get_sync_status":
        state_file = Path(os.environ.get("LOCALAPPDATA", "")) / "EmpireArchitect" / "state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())
        return {"status": "unknown", "message": "No sync state found"}

    # ═══════════════════════════════════════════════════════════════
    # SKILL LIBRARY
    # ═══════════════════════════════════════════════════════════════

    elif name == "empire_library_list":
        tag = args.get("tag", "")
        index_file = LIBRARY_PATH / "index.json"

        if not index_file.exists():
            return {"skills": [], "message": "Library not initialized"}

        index = json.loads(index_file.read_text())
        skills = list(index.get("skills", {}).values())

        if tag:
            skills = [s for s in skills if tag in s.get("tags", [])]

        return {"skills": skills, "total": len(skills)}

    elif name == "empire_library_add":
        skill_name = args.get("skill_name", "")
        from_project = args.get("from_project", "")
        tags = args.get("tags", [])

        # Get skill from database
        skills = query_db(f"""
            SELECT content FROM claude_skills
            WHERE skill_name = '{skill_name}' AND project_name ILIKE '%{from_project}%'
            LIMIT 1
        """)

        if not skills:
            return {"error": "Skill not found"}

        try:
            result = subprocess.run(
                ["python", str(PROJECTS_PATH / "empire-skill-library" / "library.py"),
                 "add", skill_name, "-"],
                input=skills[0].get("content", ""),
                capture_output=True, text=True, timeout=30
            )
            return {"success": True, "output": result.stdout}
        except Exception as e:
            return {"error": str(e)}

    elif name == "empire_library_install":
        skill_name = args.get("skill_name", "")
        to_project = args.get("to_project", "")

        try:
            result = subprocess.run(
                ["python", str(PROJECTS_PATH / "empire-skill-library" / "library.py"),
                 "install", skill_name, str(PROJECTS_PATH / to_project)],
                capture_output=True, text=True, timeout=30
            )
            return {"success": True, "output": result.stdout}
        except Exception as e:
            return {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════
    # TEMPLATES
    # ═══════════════════════════════════════════════════════════════

    elif name == "empire_template_list":
        try:
            result = subprocess.run(
                ["python", str(PROJECTS_PATH / "empire-templates" / "templates.py"), "list"],
                capture_output=True, text=True, timeout=30
            )
            return {"templates": result.stdout}
        except Exception as e:
            return {"error": str(e)}

    elif name == "empire_template_create":
        project_name = args.get("project_name", "")
        template_name = args.get("template_name", "")

        try:
            result = subprocess.run(
                ["python", str(PROJECTS_PATH / "empire-templates" / "templates.py"),
                 "save-from", str(PROJECTS_PATH / project_name), template_name],
                capture_output=True, text=True, timeout=30
            )
            return {"success": True, "output": result.stdout}
        except Exception as e:
            return {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════
    # WORDPRESS SITES
    # ═══════════════════════════════════════════════════════════════

    elif name == "empire_list_sites":
        sites_config = PROJECTS_PATH / "config" / "sites.json"
        if sites_config.exists():
            config = json.loads(sites_config.read_text())
            sites = config.get("sites", config)
            return [{"id": k, "domain": v.get("domain"), "brand": v.get("brand_name")}
                    for k, v in sites.items()]
        return {"error": "Sites config not found"}

    elif name == "empire_site_config":
        site_id = args.get("site_id", "")
        sites_config = PROJECTS_PATH / "config" / "sites.json"

        if sites_config.exists():
            config = json.loads(sites_config.read_text())
            sites = config.get("sites", config)
            if site_id in sites:
                site = sites[site_id].copy()
                # Don't expose passwords
                if "wp_app_password" in site:
                    site["wp_app_password"] = "***"
                return site
        return {"error": f"Site '{site_id}' not found"}

    # ═══════════════════════════════════════════════════════════════
    # MCP & HOOKS
    # ═══════════════════════════════════════════════════════════════

    elif name == "empire_list_mcp_configs":
        return query_db("""
            SELECT project_name, servers, server_count, updated_at
            FROM claude_mcp_configs
            ORDER BY project_name
        """)

    elif name == "empire_get_mcp_config":
        project_name = args.get("project_name", "")
        return query_db(f"""
            SELECT project_name, config_content, servers, updated_at
            FROM claude_mcp_configs
            WHERE project_name ILIKE '%{project_name}%'
            LIMIT 1
        """)

    # ═══════════════════════════════════════════════════════════════
    # FILE OPERATIONS
    # ═══════════════════════════════════════════════════════════════

    elif name == "empire_read_file":
        project = args.get("project", "")
        file_path = args.get("file_path", "")

        full_path = PROJECTS_PATH / project / file_path
        if not full_path.exists():
            # Try to find project
            matches = list(PROJECTS_PATH.glob(f"*{project}*"))
            if matches:
                full_path = matches[0] / file_path

        if full_path.exists():
            try:
                content = full_path.read_text()
                if len(content) > 10000:
                    content = content[:10000] + "\n... [truncated]"
                return {"path": str(full_path), "content": content}
            except:
                return {"error": "Could not read file (possibly binary)"}
        return {"error": f"File not found: {file_path}"}

    elif name == "empire_list_files":
        project = args.get("project", "")
        subpath = args.get("path", "")
        pattern = args.get("pattern", "*")

        project_path = PROJECTS_PATH / project
        if not project_path.exists():
            matches = list(PROJECTS_PATH.glob(f"*{project}*"))
            if matches:
                project_path = matches[0]
            else:
                return {"error": f"Project '{project}' not found"}

        search_path = project_path / subpath if subpath else project_path
        files = list(search_path.glob(pattern))[:50]

        return [{"name": f.name, "is_dir": f.is_dir(), "size": f.stat().st_size if f.is_file() else 0}
                for f in files]

    # ═══════════════════════════════════════════════════════════════
    # SCREENPIPE (Screen Intelligence)
    # ═══════════════════════════════════════════════════════════════

    elif name == "empire_screen_search":
        query = args.get("query", "")
        content_type = args.get("content_type", "all")
        limit = args.get("limit", 10)
        start_time = args.get("start_time", "")
        end_time = args.get("end_time", "")
        app_name = args.get("app_name", "")

        params = {"q": query, "limit": limit}
        if content_type != "all":
            params["content_type"] = content_type
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if app_name:
            params["app_name"] = app_name

        try:
            resp = requests.get("http://localhost:3030/search", params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for item in data.get("data", []):
                    content = item.get("content", {})
                    results.append({
                        "timestamp": content.get("timestamp", ""),
                        "text": content.get("text", "")[:500],
                        "app_name": content.get("app_name", ""),
                        "window_name": content.get("window_name", ""),
                        "type": item.get("type", ""),
                    })
                return {"results": results, "total": len(results)}
            return {"error": f"Screenpipe returned {resp.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"error": "Screenpipe not running. Start with: screenpipe"}
        except Exception as e:
            return {"error": str(e)}

    elif name == "empire_screen_timeline":
        limit = args.get("limit", 20)
        content_type = args.get("content_type", "all")
        app_name = args.get("app_name", "")

        params = {"limit": limit, "q": ""}
        if content_type != "all":
            params["content_type"] = content_type
        if app_name:
            params["app_name"] = app_name

        try:
            resp = requests.get("http://localhost:3030/search", params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                timeline = []
                for item in data.get("data", []):
                    content = item.get("content", {})
                    timeline.append({
                        "timestamp": content.get("timestamp", ""),
                        "app_name": content.get("app_name", ""),
                        "window_name": content.get("window_name", ""),
                        "text_preview": content.get("text", "")[:200],
                    })
                return {"timeline": timeline}
            return {"error": f"Screenpipe returned {resp.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"error": "Screenpipe not running. Start with: screenpipe"}
        except Exception as e:
            return {"error": str(e)}

    elif name == "empire_monitor_check":
        try:
            resp = requests.get("http://localhost:3030/health", timeout=5)
            if resp.status_code == 200:
                health = resp.json()
                return {
                    "status": "running",
                    "health": health,
                    "api_url": "http://localhost:3030",
                }
            return {"status": "unhealthy", "http_code": resp.status_code}
        except requests.exceptions.ConnectionError:
            return {"status": "not_running", "message": "Screenpipe is not running. Start with: screenpipe"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ═══════════════════════════════════════════════════════════════
    # GIT OPERATIONS
    # ═══════════════════════════════════════════════════════════════

    elif name == "empire_git_status":
        project = args.get("project", "")

        if project:
            return query_db(f"""
                SELECT project_name, git_branch, git_last_commit,
                       git_last_commit_message, git_is_dirty, git_remote_url
                FROM claude_projects
                WHERE project_name ILIKE '%{project}%' AND is_git_repo = true
                LIMIT 1
            """)
        else:
            return query_db("""
                SELECT project_name, git_branch, git_is_dirty
                FROM claude_projects
                WHERE is_git_repo = true
                ORDER BY project_name
            """)

    elif name == "empire_git_log":
        project = args.get("project", "")
        count = args.get("count", 10)

        project_path = PROJECTS_PATH / project
        if not project_path.exists():
            matches = list(PROJECTS_PATH.glob(f"*{project}*"))
            if matches:
                project_path = matches[0]

        try:
            result = subprocess.run(
                ["git", "-C", str(project_path), "log", f"-{count}",
                 "--pretty=format:%h|%s|%an|%ar"],
                capture_output=True, text=True, timeout=30
            )
            commits = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("|")
                    if len(parts) >= 4:
                        commits.append({
                            "hash": parts[0],
                            "message": parts[1],
                            "author": parts[2],
                            "when": parts[3]
                        })
            return commits
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unknown tool: {name}"}


def handle_tool_call(id: Any, params: dict):
    tool_name = params.get("name")
    args = params.get("arguments", {})

    try:
        result = execute_tool(tool_name, args)
        send_response({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2, default=str)
                    }
                ]
            }
        })
    except Exception as e:
        send_response({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True
            }
        })


def main():
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            method = request.get("method")
            id = request.get("id")
            params = request.get("params", {})

            if method == "initialize":
                handle_initialize(id, params)
            elif method == "initialized":
                pass
            elif method == "tools/list":
                handle_tools_list(id)
            elif method == "tools/call":
                handle_tool_call(id, params)
            elif method == "notifications/cancelled":
                pass
            else:
                if id is not None:
                    send_error(id, -32601, f"Method not found: {method}")

        except json.JSONDecodeError:
            continue
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
