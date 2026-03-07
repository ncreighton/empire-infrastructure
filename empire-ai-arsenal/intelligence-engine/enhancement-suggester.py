#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
EMPIRE AI ARSENAL — Enhancement Suggester
═══════════════════════════════════════════════════════════════
Analyzes your current stack and suggests:
- New integrations between existing tools
- Workflow automations you're missing
- Cost optimization opportunities
- Performance improvements

Usage:
    python enhancement-suggester.py --analyze
    python enhancement-suggester.py --suggest-workflows
═══════════════════════════════════════════════════════════════
"""

import json
import os
from datetime import datetime

# ── Current Stack Definition ──
CURRENT_STACK = {
    "llm_gateway": {
        "tool": "LiteLLM",
        "port": 4000,
        "integrates_with": ["langfuse", "ollama", "redis"],
    },
    "vector_db": {
        "tool": "Qdrant",
        "port": 6333,
        "integrates_with": ["litellm", "dify", "mem0"],
    },
    "database": {
        "tool": "PostgreSQL + pgvector",
        "port": 5432,
        "integrates_with": ["litellm", "langfuse", "n8n", "dify", "authentik"],
    },
    "cache": {
        "tool": "Redis",
        "port": 6379,
        "integrates_with": ["litellm", "dify", "authentik"],
    },
    "local_llm": {
        "tool": "Ollama",
        "port": 11434,
        "integrates_with": ["open-webui", "litellm", "dify", "n8n"],
    },
    "chat_ui": {
        "tool": "Open WebUI",
        "port": 3000,
        "integrates_with": ["ollama", "litellm", "searxng"],
    },
    "workflow": {
        "tool": "n8n",
        "port": 5678,
        "integrates_with": ["litellm", "postgres", "crawl4ai", "searxng"],
    },
    "app_platform": {
        "tool": "Dify",
        "port": 5001,
        "integrates_with": ["postgres", "redis", "qdrant", "ollama"],
    },
    "crawler": {
        "tool": "Crawl4AI",
        "port": 11235,
        "integrates_with": ["browserless", "n8n"],
    },
    "search": {
        "tool": "SearXNG",
        "port": 8080,
        "integrates_with": ["open-webui", "n8n"],
    },
    "browser": {
        "tool": "Browserless",
        "port": 3002,
        "integrates_with": ["crawl4ai", "firecrawl", "n8n"],
    },
    "scraper": {
        "tool": "Firecrawl",
        "port": 3003,
        "integrates_with": ["browserless", "redis"],
    },
    "tracing": {
        "tool": "Langfuse",
        "port": 3004,
        "integrates_with": ["litellm", "postgres"],
    },
    "monitoring": {
        "tool": "Uptime Kuma",
        "port": 3005,
        "integrates_with": [],
    },
    "proxy": {
        "tool": "Traefik",
        "port": 443,
        "integrates_with": ["authentik"],
    },
    "auth": {
        "tool": "Authentik",
        "port": 9000,
        "integrates_with": ["traefik", "postgres", "redis"],
    },
    "voice": {
        "tool": "Speaches",
        "port": 8100,
        "integrates_with": ["litellm"],
    },
    "docs": {
        "tool": "Docling",
        "port": 5002,
        "integrates_with": ["qdrant"],
    },
}

# ── Workflow Templates ──
WORKFLOW_SUGGESTIONS = [
    {
        "name": "Content Research Pipeline",
        "description": "Auto-research any topic using search + crawling + LLM analysis",
        "tools": ["searxng", "crawl4ai", "litellm", "qdrant"],
        "n8n_flow": [
            "Trigger: webhook or schedule",
            "SearXNG: search for topic",
            "Crawl4AI: deep-crawl top results",
            "LiteLLM: extract key insights",
            "Qdrant: store embeddings for RAG",
            "Output: structured research brief",
        ],
    },
    {
        "name": "Document Knowledge Base Builder",
        "description": "Upload any document, auto-convert and build searchable knowledge base",
        "tools": ["docling", "qdrant", "litellm", "ollama"],
        "n8n_flow": [
            "Trigger: file upload webhook",
            "Docling: convert PDF/DOCX/PPTX to markdown",
            "LiteLLM: chunk and generate embeddings",
            "Qdrant: store vectors with metadata",
            "Result: instant RAG over your documents",
        ],
    },
    {
        "name": "Competitor Monitor",
        "description": "Track competitor websites and get alerts on changes",
        "tools": ["crawl4ai", "litellm", "n8n", "postgres"],
        "n8n_flow": [
            "Trigger: daily schedule",
            "Crawl4AI: crawl competitor sites",
            "LiteLLM: compare with previous crawl",
            "PostgreSQL: store diffs",
            "n8n: send alert if significant changes",
        ],
    },
    {
        "name": "Voice Note Processor",
        "description": "Record voice notes, transcribe, summarize, and create action items",
        "tools": ["speaches", "litellm", "n8n", "postgres"],
        "n8n_flow": [
            "Trigger: audio file upload",
            "Speaches: transcribe audio to text",
            "LiteLLM: summarize and extract action items",
            "PostgreSQL: store notes and actions",
            "n8n: create tasks in your project management tool",
        ],
    },
    {
        "name": "Site Content Auditor",
        "description": "Audit all content across your 14 WordPress sites",
        "tools": ["crawl4ai", "litellm", "langfuse", "n8n"],
        "n8n_flow": [
            "Trigger: weekly schedule",
            "Crawl4AI: crawl each site's sitemap",
            "LiteLLM: analyze content quality, SEO, freshness",
            "Langfuse: track analysis costs",
            "n8n: generate report with recommendations",
        ],
    },
    {
        "name": "Cross-Empire Knowledge Linker",
        "description": "Find content opportunities by linking knowledge across all 14 niches",
        "tools": ["qdrant", "litellm", "postgres", "n8n"],
        "n8n_flow": [
            "Trigger: after new content published",
            "Qdrant: find semantically similar content across sites",
            "LiteLLM: generate cross-linking suggestions",
            "PostgreSQL: store link graph",
            "Output: internal linking recommendations",
        ],
    },
    {
        "name": "AI Model Cost Optimizer",
        "description": "Automatically route to cheapest model that meets quality threshold",
        "tools": ["litellm", "langfuse", "ollama"],
        "n8n_flow": [
            "LiteLLM: route simple tasks to Haiku/local models",
            "LiteLLM: route complex tasks to Sonnet",
            "Langfuse: track cost per task type",
            "Weekly report: cost savings analysis",
        ],
    },
    {
        "name": "GitHub Trending Scanner",
        "description": "Daily scan of GitHub trending for tools relevant to your stack",
        "tools": ["searxng", "litellm", "n8n", "postgres"],
        "n8n_flow": [
            "Trigger: daily schedule",
            "GitHub API: fetch trending repos",
            "LiteLLM: evaluate relevance to empire stack",
            "PostgreSQL: track candidates over time",
            "n8n: weekly digest email with top picks",
        ],
    },
]


def analyze_stack():
    """Analyze current stack for gaps and integration opportunities."""
    print("═══════════════════════════════════════════════════════════")
    print("  ARSENAL ENHANCEMENT SUGGESTER — Stack Analysis")
    print(f"  {datetime.utcnow().isoformat()} UTC")
    print("═══════════════════════════════════════════════════════════")

    print(f"\n  Current stack: {len(CURRENT_STACK)} services")

    # Find underconnected services
    print("\n  INTEGRATION GAPS:")
    for key, svc in CURRENT_STACK.items():
        connections = len(svc["integrates_with"])
        if connections <= 1:
            print(f"    {svc['tool']:20s} — only {connections} integration(s)")
            print(f"      Suggestion: Connect to n8n for workflow automation")

    # Resource allocation suggestions
    print("\n  RESOURCE OPTIMIZATION (128GB RAM):")
    print("    Ollama:     64GB max (can run 70B models)")
    print("    Qdrant:      8GB (handles millions of vectors)")
    print("    PostgreSQL:  4GB (more than enough)")
    print("    Redis:       2GB (caching layer)")
    print("    Crawl4AI:    4GB (parallel crawling)")
    print("    Browserless: 4GB (10 concurrent browsers)")
    print("    Speaches:    4GB (Whisper large-v3-turbo)")
    print("    Docling:     4GB (document processing)")
    print("    Everything else: ~10GB total")
    print("    Remaining:   ~24GB free for new services")


def suggest_workflows():
    """Print workflow suggestions."""
    print("═══════════════════════════════════════════════════════════")
    print("  ARSENAL ENHANCEMENT SUGGESTER — Workflow Ideas")
    print("═══════════════════════════════════════════════════════════")

    for i, wf in enumerate(WORKFLOW_SUGGESTIONS, 1):
        print(f"\n  {i}. {wf['name']}")
        print(f"     {wf['description']}")
        print(f"     Tools: {', '.join(wf['tools'])}")
        print(f"     n8n Flow:")
        for step in wf["n8n_flow"]:
            print(f"       → {step}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--analyze"

    if cmd == "--analyze":
        analyze_stack()
    elif cmd == "--suggest-workflows":
        suggest_workflows()
    else:
        print("Usage:")
        print("  python enhancement-suggester.py --analyze")
        print("  python enhancement-suggester.py --suggest-workflows")
