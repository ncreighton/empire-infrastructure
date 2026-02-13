# Empire Architect MCP Server

Query your Claude Code projects, skills, agents, and workflows directly from conversations.

## Installation

Add to your `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "empire-architect": {
      "command": "python",
      "args": ["C:\\Claude Code Projects\\empire-mcp-server\\server.py"]
    }
  }
}
```

## Available Tools

### empire_search_skills
Search for skills across all projects.
```
"Find skills related to image generation"
```

### empire_get_skill
Get the full content of a specific skill.
```
"Show me the content of the 'commit' skill"
```

### empire_list_projects
List all projects with metadata.
```
"List projects that have MCP configs"
```

### empire_get_project
Get detailed info about a project including CLAUDE.md.
```
"What's in the empire-master project?"
```

### empire_find_duplicates
Find duplicate skills across projects.
```
"Show me duplicate skills"
```

### empire_get_stats
Get overall statistics.
```
"How many skills do I have total?"
```

### empire_search_content
Search across all content.
```
"Search for 'WordPress' in all projects"
```

### empire_list_workflows
List n8n workflows.
```
"List all workflows in the empire-master project"
```

### empire_recommend_skills
Get skill recommendations for a project.
```
"What skills should I add to empire-dashboard?"
```

## Example Usage

Once installed, you can ask Claude Code:
- "Search my empire for skills about SEO"
- "What projects have MCP configs?"
- "Show me duplicate skills I should clean up"
- "What skills does empire-master have?"
- "Recommend skills for my new project"

---

## API Cost Optimization Rules

### Model Selection (MANDATORY)
When generating code that calls Anthropic's API:

1. **Default to Sonnet** (`claude-sonnet-4-20250514`) for most tasks
2. **Use Haiku** (`claude-haiku-4-5-20251001`) for:
   - Classification tasks
   - Intent detection
   - Simple data extraction
   - Yes/no decisions
   - Formatting/conversion
   - Tag generation
3. **Reserve Opus** (`claude-opus-4-20250514`) ONLY for:
   - Complex multi-step reasoning
   - Critical business decisions
   - Nuanced editorial judgment

### Prompt Caching (ALWAYS ENABLE)
When system prompts exceed 2,048 tokens, ALWAYS use cache_control:

```python
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    system=[
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{"role": "user", "content": user_input}]
)
```

### Token Limits
| Output Type | max_tokens |
|-------------|------------|
| Yes/no, classification | 50-100 |
| Short response | 200-500 |
| Article section | 1000-2000 |
| Full article | 3000-4096 |

### Quick Reference
```
Model Strings (Dec 2025):
- claude-haiku-4-5-20251001    → Simple tasks
- claude-sonnet-4-20250514     → Default
- claude-opus-4-20250514       → Complex only

Pricing per 1M tokens:
- Haiku:  $0.80 in / $4.00 out
- Sonnet: $3.00 in / $15.00 out
- Opus:   $15.00 in / $75.00 out
- Cache reads: 90% discount
- Batch API: 50% discount
```
