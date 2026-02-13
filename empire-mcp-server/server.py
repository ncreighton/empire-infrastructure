"""
Empire Architect MCP Server
Allows Claude Code to query projects, skills, agents, and workflows directly.

Install: Add to your Claude Code MCP config (.claude/mcp.json)
"""

import json
import sys
import requests
from typing import Any

# MCP Protocol version
PROTOCOL_VERSION = "2024-11-05"

N8N_WEBHOOK = "https://vmi2976539.contaboserver.net/webhook/empire/db-query"


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
    """Send JSON-RPC response"""
    print(json.dumps(response), flush=True)


def send_error(id: Any, code: int, message: str):
    """Send JSON-RPC error"""
    send_response({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })


def handle_initialize(id: Any, params: dict):
    """Handle initialize request"""
    send_response({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {
                "name": "empire-architect",
                "version": "1.0.0"
            },
            "capabilities": {
                "tools": {}
            }
        }
    })


def handle_tools_list(id: Any):
    """Return available tools"""
    tools = [
        {
            "name": "empire_search_skills",
            "description": "Search for skills across all Claude Code projects. Returns skill names, content, and which projects they belong to.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term to find in skill names or content"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: filter by project name"
                    }
                },
                "required": []
            }
        },
        {
            "name": "empire_get_skill",
            "description": "Get the full content of a specific skill by name",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The name of the skill to retrieve"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: specific project to get skill from"
                    }
                },
                "required": ["skill_name"]
            }
        },
        {
            "name": "empire_list_projects",
            "description": "List all Claude Code projects with their metadata (skills count, git status, MCP config, etc.)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional: filter projects by name"
                    },
                    "has_skills": {
                        "type": "boolean",
                        "description": "Only show projects with skills"
                    },
                    "has_mcp": {
                        "type": "boolean",
                        "description": "Only show projects with MCP config"
                    }
                },
                "required": []
            }
        },
        {
            "name": "empire_get_project",
            "description": "Get detailed info about a specific project including its CLAUDE.md content",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "The project name to look up"
                    }
                },
                "required": ["project_name"]
            }
        },
        {
            "name": "empire_find_duplicates",
            "description": "Find duplicate skills across projects (same content in multiple places)",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "empire_get_stats",
            "description": "Get overall statistics about the Empire (project count, skill count, etc.)",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "empire_search_content",
            "description": "Search across all CLAUDE.md files and skill content for specific text",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "empire_list_workflows",
            "description": "List all n8n workflows across projects",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Optional: filter by project"
                    }
                },
                "required": []
            }
        },
        {
            "name": "empire_recommend_skills",
            "description": "Get skill recommendations for a project based on what similar projects have",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "The project to get recommendations for"
                    }
                },
                "required": ["project_name"]
            }
        }
    ]

    send_response({
        "jsonrpc": "2.0",
        "id": id,
        "result": {"tools": tools}
    })


def handle_tool_call(id: Any, params: dict):
    """Handle tool execution"""
    tool_name = params.get("name")
    args = params.get("arguments", {})

    result = None

    try:
        if tool_name == "empire_search_skills":
            query = args.get("query", "")
            project = args.get("project", "")

            where = []
            if query:
                where.append(f"(skill_name ILIKE '%{query}%' OR content ILIKE '%{query}%')")
            if project:
                where.append(f"project_name ILIKE '%{project}%'")

            where_clause = f"WHERE {' AND '.join(where)}" if where else ""

            result = query_db(f"""
                SELECT skill_name, project_name, content_length,
                       LEFT(content, 500) as preview
                FROM claude_skills
                {where_clause}
                ORDER BY skill_name
                LIMIT 20
            """)

        elif tool_name == "empire_get_skill":
            skill_name = args.get("skill_name", "")
            project = args.get("project", "")

            where = [f"skill_name ILIKE '%{skill_name}%'"]
            if project:
                where.append(f"project_name ILIKE '%{project}%'")

            result = query_db(f"""
                SELECT skill_name, project_name, skill_path, content, content_length, updated_at
                FROM claude_skills
                WHERE {' AND '.join(where)}
                LIMIT 5
            """)

        elif tool_name == "empire_list_projects":
            filter_name = args.get("filter", "")
            has_skills = args.get("has_skills", False)
            has_mcp = args.get("has_mcp", False)

            where = []
            if filter_name:
                where.append(f"p.project_name ILIKE '%{filter_name}%'")
            if has_skills:
                where.append("(SELECT COUNT(*) FROM claude_skills s WHERE s.project_name = p.project_name) > 0")
            if has_mcp:
                where.append("p.has_mcp_config = true")

            where_clause = f"WHERE {' AND '.join(where)}" if where else ""

            result = query_db(f"""
                SELECT p.project_name, p.project_path,
                       (SELECT COUNT(*) FROM claude_skills s WHERE s.project_name = p.project_name) as skill_count,
                       (SELECT COUNT(*) FROM claude_agents a WHERE a.project_name = p.project_name) as agent_count,
                       (SELECT COUNT(*) FROM claude_workflows w WHERE w.project_name = p.project_name) as workflow_count,
                       p.has_mcp_config, p.mcp_servers, p.is_git_repo, p.git_branch, p.updated_at
                FROM claude_projects p
                {where_clause}
                ORDER BY skill_count DESC, p.project_name
                LIMIT 50
            """)

        elif tool_name == "empire_get_project":
            project_name = args.get("project_name", "")
            result = query_db(f"""
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

        elif tool_name == "empire_find_duplicates":
            result = query_db("""
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

        elif tool_name == "empire_get_stats":
            result = query_db("""
                SELECT
                    (SELECT COUNT(*) FROM claude_projects) as total_projects,
                    (SELECT COUNT(*) FROM claude_skills) as total_skills,
                    (SELECT COUNT(DISTINCT skill_name) FROM claude_skills) as unique_skill_names,
                    (SELECT COUNT(*) FROM claude_agents) as total_agents,
                    (SELECT COUNT(*) FROM claude_workflows) as total_workflows,
                    (SELECT COUNT(*) FROM claude_mcp_configs) as projects_with_mcp,
                    (SELECT COUNT(*) FROM claude_projects WHERE is_git_repo = true) as git_repos
            """)

        elif tool_name == "empire_search_content":
            query = args.get("query", "")
            result = query_db(f"""
                SELECT 'project' as type, project_name as name, LEFT(claude_md_content, 300) as preview
                FROM claude_projects
                WHERE claude_md_content ILIKE '%{query}%'
                UNION ALL
                SELECT 'skill' as type, skill_name || ' (' || project_name || ')' as name, LEFT(content, 300) as preview
                FROM claude_skills
                WHERE content ILIKE '%{query}%'
                LIMIT 20
            """)

        elif tool_name == "empire_list_workflows":
            project = args.get("project", "")
            where = f"WHERE project_name ILIKE '%{project}%'" if project else ""
            result = query_db(f"""
                SELECT workflow_name, project_name, workflow_path, last_modified
                FROM claude_workflows
                {where}
                ORDER BY project_name, workflow_name
                LIMIT 50
            """)

        elif tool_name == "empire_recommend_skills":
            project_name = args.get("project_name", "")
            # Find skills in similar projects that this project doesn't have
            result = query_db(f"""
                WITH project_skills AS (
                    SELECT skill_name FROM claude_skills WHERE project_name ILIKE '%{project_name}%'
                ),
                popular_skills AS (
                    SELECT skill_name, COUNT(DISTINCT project_name) as usage_count
                    FROM claude_skills
                    WHERE skill_name NOT IN (SELECT skill_name FROM project_skills)
                    GROUP BY skill_name
                    HAVING COUNT(DISTINCT project_name) >= 2
                )
                SELECT ps.skill_name, ps.usage_count,
                       array_agg(DISTINCT cs.project_name) as used_in_projects
                FROM popular_skills ps
                JOIN claude_skills cs ON cs.skill_name = ps.skill_name
                GROUP BY ps.skill_name, ps.usage_count
                ORDER BY ps.usage_count DESC
                LIMIT 10
            """)

        else:
            send_error(id, -32601, f"Unknown tool: {tool_name}")
            return

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
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {str(e)}"
                    }
                ],
                "isError": True
            }
        })


def main():
    """Main MCP server loop"""
    # Send initialized notification is not needed - wait for initialize request

    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            method = request.get("method")
            id = request.get("id")
            params = request.get("params", {})

            if method == "initialize":
                handle_initialize(id, params)
            elif method == "initialized":
                pass  # Notification, no response needed
            elif method == "tools/list":
                handle_tools_list(id)
            elif method == "tools/call":
                handle_tool_call(id, params)
            elif method == "notifications/cancelled":
                pass  # Handle cancellation
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
