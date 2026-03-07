#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
EMPIRE AI ARSENAL — Intelligence Engine: Tool Discovery
═══════════════════════════════════════════════════════════════
Automated discovery of new AI tools, frameworks, and platforms.
Runs daily via cron or n8n workflow. Scores tools against your
existing stack and sends weekly digests.

Usage:
    python tool-discovery.py --discover    # Run discovery scan
    python tool-discovery.py --digest      # Generate weekly digest
    python tool-discovery.py --evaluate URL # Evaluate a specific tool
═══════════════════════════════════════════════════════════════
"""

import json
import os
import sys
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
import urllib.request
import urllib.parse


@dataclass
class ToolCandidate:
    name: str
    url: str
    description: str
    category: str
    github_stars: int = 0
    docker_available: bool = False
    has_api: bool = False
    has_mcp: bool = False
    self_hostable: bool = False
    score: float = 0.0
    discovered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = ""
    evaluated: bool = False
    verdict: str = ""


# ── Categories we care about ──
CATEGORIES = [
    "ai-agents", "rag-knowledge", "llm-tools", "browser-automation",
    "data-pipelines", "document-processing", "voice-audio",
    "monitoring-observability", "search-engines", "mcp-servers",
    "code-generation", "workflow-automation", "vector-databases",
    "identity-auth", "content-generation"
]

# ── Sources to scan ──
GITHUB_TRENDING_URL = "https://api.github.com/search/repositories"
HN_API = "https://hacker-news.firebaseio.com/v0"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates.json")
DIGEST_FILE = os.path.join(DATA_DIR, "weekly-digest.json")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_candidates() -> list[dict]:
    if os.path.exists(CANDIDATES_FILE):
        with open(CANDIDATES_FILE) as f:
            return json.load(f)
    return []


def save_candidates(candidates: list[dict]):
    ensure_data_dir()
    with open(CANDIDATES_FILE, "w") as f:
        json.dump(candidates, f, indent=2)


def fetch_json(url: str, headers: dict = None) -> dict:
    """Fetch JSON from URL with basic error handling."""
    req = urllib.request.Request(url)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return {}


def search_github_trending(query: str, min_stars: int = 500) -> list[ToolCandidate]:
    """Search GitHub for trending repos matching query."""
    candidates = []
    # Recent repos with good traction
    since = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    params = urllib.parse.urlencode({
        "q": f"{query} created:>{since} stars:>{min_stars}",
        "sort": "stars",
        "order": "desc",
        "per_page": 10
    })
    url = f"{GITHUB_TRENDING_URL}?{params}"
    data = fetch_json(url, {"Accept": "application/vnd.github.v3+json"})

    for repo in data.get("items", []):
        candidate = ToolCandidate(
            name=repo["name"],
            url=repo["html_url"],
            description=repo.get("description", "")[:200],
            category=categorize_repo(repo),
            github_stars=repo.get("stargazers_count", 0),
            self_hostable=True,
            source="github_trending",
        )
        candidates.append(candidate)

    return candidates


def categorize_repo(repo: dict) -> str:
    """Auto-categorize a GitHub repo based on topics and description."""
    text = f"{repo.get('description', '')} {' '.join(repo.get('topics', []))}".lower()

    category_keywords = {
        "ai-agents": ["agent", "autonomous", "multi-agent", "crew", "swarm"],
        "rag-knowledge": ["rag", "retrieval", "knowledge", "graph", "embedding"],
        "llm-tools": ["llm", "language model", "gpt", "claude", "inference"],
        "browser-automation": ["browser", "playwright", "puppeteer", "selenium", "scraping"],
        "data-pipelines": ["pipeline", "etl", "data flow", "orchestrat", "airflow"],
        "document-processing": ["document", "pdf", "ocr", "parsing", "markdown"],
        "voice-audio": ["voice", "tts", "stt", "speech", "whisper", "audio"],
        "monitoring-observability": ["monitor", "observ", "trac", "metric", "log"],
        "search-engines": ["search", "index", "query"],
        "mcp-servers": ["mcp", "model context", "tool server"],
        "code-generation": ["code gen", "copilot", "coding assistant", "ide"],
        "workflow-automation": ["workflow", "automat", "n8n", "zapier"],
        "vector-databases": ["vector", "qdrant", "pinecone", "weaviate", "chroma"],
        "identity-auth": ["auth", "identity", "sso", "oauth", "oidc"],
        "content-generation": ["content", "article", "blog", "writing"],
    }

    for category, keywords in category_keywords.items():
        if any(kw in text for kw in keywords):
            return category

    return "other"


def score_candidate(candidate: ToolCandidate) -> float:
    """Score a tool candidate based on relevance to our stack."""
    score = 0.0

    # GitHub stars (log scale)
    if candidate.github_stars > 0:
        import math
        score += min(math.log10(candidate.github_stars) * 10, 50)

    # Key features
    if candidate.docker_available:
        score += 15
    if candidate.has_api:
        score += 15
    if candidate.has_mcp:
        score += 20
    if candidate.self_hostable:
        score += 10

    # Category relevance
    high_value_categories = ["ai-agents", "rag-knowledge", "mcp-servers", "browser-automation"]
    if candidate.category in high_value_categories:
        score += 10

    return round(score, 1)


def discover():
    """Run full discovery scan."""
    print("═══════════════════════════════════════════════════════════")
    print("  ARSENAL INTELLIGENCE ENGINE — Discovery Scan")
    print(f"  {datetime.utcnow().isoformat()} UTC")
    print("═══════════════════════════════════════════════════════════")

    existing = load_candidates()
    existing_urls = {c["url"] for c in existing}
    new_candidates = []

    # Search queries targeting our categories
    queries = [
        "self-hosted ai agent framework",
        "mcp server model context protocol",
        "rag knowledge graph self-hosted",
        "ai browser automation",
        "llm observability open source",
        "document processing ai",
        "voice ai self-hosted tts stt",
        "workflow automation ai",
        "vector database self-hosted",
        "web scraping ai crawler",
    ]

    for query in queries:
        print(f"\n  Searching: {query}")
        candidates = search_github_trending(query, min_stars=200)
        for c in candidates:
            if c.url not in existing_urls:
                c.score = score_candidate(c)
                new_candidates.append(asdict(c))
                existing_urls.add(c.url)
                print(f"    NEW: {c.name} ({c.github_stars} stars, score: {c.score})")

    # Merge and save
    all_candidates = existing + new_candidates
    # Sort by score descending
    all_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    save_candidates(all_candidates)

    print(f"\n  Total candidates: {len(all_candidates)}")
    print(f"  New this scan: {len(new_candidates)}")
    print(f"  Saved to: {CANDIDATES_FILE}")


def generate_digest():
    """Generate a weekly digest of top tool candidates."""
    candidates = load_candidates()
    if not candidates:
        print("No candidates found. Run --discover first.")
        return

    # Top 20 by score
    top = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:20]

    digest = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_candidates": len(candidates),
        "top_picks": [],
        "by_category": {},
    }

    for c in top:
        digest["top_picks"].append({
            "name": c["name"],
            "url": c["url"],
            "description": c["description"],
            "category": c["category"],
            "stars": c.get("github_stars", 0),
            "score": c.get("score", 0),
        })

    # Group by category
    for c in candidates:
        cat = c.get("category", "other")
        if cat not in digest["by_category"]:
            digest["by_category"][cat] = []
        digest["by_category"][cat].append(c["name"])

    ensure_data_dir()
    with open(DIGEST_FILE, "w") as f:
        json.dump(digest, f, indent=2)

    # Print digest
    print("═══════════════════════════════════════════════════════════")
    print("  ARSENAL INTELLIGENCE ENGINE — Weekly Digest")
    print("═══════════════════════════════════════════════════════════")
    print(f"\n  Total tools tracked: {len(candidates)}")
    print(f"\n  TOP 10 PICKS:")
    for i, pick in enumerate(digest["top_picks"][:10], 1):
        print(f"    {i}. {pick['name']} ({pick['stars']} stars, score: {pick['score']})")
        print(f"       {pick['description'][:80]}")
        print(f"       {pick['url']}")
    print(f"\n  Full digest: {DIGEST_FILE}")


def evaluate_tool(url: str):
    """Evaluate a specific tool URL."""
    print(f"Evaluating: {url}")

    # Extract GitHub info if it's a GitHub URL
    if "github.com" in url:
        parts = url.rstrip("/").split("/")
        if len(parts) >= 5:
            owner, repo = parts[-2], parts[-1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            data = fetch_json(api_url, {"Accept": "application/vnd.github.v3+json"})

            if data:
                candidate = ToolCandidate(
                    name=data.get("name", repo),
                    url=url,
                    description=data.get("description", ""),
                    category=categorize_repo(data),
                    github_stars=data.get("stargazers_count", 0),
                    self_hostable=True,
                    source="manual_evaluation",
                )
                candidate.score = score_candidate(candidate)

                print(f"\n  Name:        {candidate.name}")
                print(f"  Stars:       {candidate.github_stars}")
                print(f"  Category:    {candidate.category}")
                print(f"  Description: {candidate.description}")
                print(f"  Score:       {candidate.score}")
                print(f"  Language:    {data.get('language', 'Unknown')}")
                print(f"  License:     {data.get('license', {}).get('spdx_id', 'Unknown')}")
                print(f"  Last Push:   {data.get('pushed_at', 'Unknown')}")
                print(f"  Topics:      {', '.join(data.get('topics', []))}")

                # Add to candidates
                candidates = load_candidates()
                existing_urls = {c["url"] for c in candidates}
                if url not in existing_urls:
                    candidates.append(asdict(candidate))
                    save_candidates(candidates)
                    print(f"\n  Added to candidate database")
    else:
        print("  Only GitHub URLs supported for auto-evaluation currently")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tool-discovery.py --discover     # Scan for new tools")
        print("  python tool-discovery.py --digest        # Generate weekly digest")
        print("  python tool-discovery.py --evaluate URL  # Evaluate a specific tool")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "--discover":
        discover()
    elif cmd == "--digest":
        generate_digest()
    elif cmd == "--evaluate" and len(sys.argv) > 2:
        evaluate_tool(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
