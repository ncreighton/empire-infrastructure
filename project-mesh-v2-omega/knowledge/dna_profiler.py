"""
DNA Profiler   Generates DNA profiles for every project.
A DNA profile captures what a project CAN do, what patterns it uses,
what APIs it integrates with, and how reusable its code is.

Used by the bootstrapper to find similar projects and clone their capabilities.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from knowledge.graph_engine import KnowledgeGraph

log = logging.getLogger(__name__)

# Capability categories detected from function/class names and imports
CAPABILITY_PATTERNS = {
    "seo-optimization": ["seo", "keyword", "schema", "rankmath", "meta_tag", "serp", "sitemap"],
    "wordpress-api": ["wp_rest", "wordpress", "wp_admin", "post_create", "media_upload", "featured_image"],
    "image-generation": ["image_gen", "generate_image", "create_image", "pillow", "PIL", "fal_ai"],
    "video-creation": ["video", "creatomate", "render_script", "storyboard", "scene"],
    "tts-generation": ["elevenlabs", "text_to_speech", "tts", "voice", "narration"],
    "content-pipeline": ["article", "blog_post", "content_gen", "outline", "draft"],
    "email-marketing": ["email", "newsletter", "campaign", "smtp", "substack"],
    "social-media": ["pinterest", "instagram", "linkedin", "twitter", "social"],
    "ecommerce": ["etsy", "product", "listing", "shop", "order", "printable"],
    "analytics": ["analytics", "ga4", "gsc", "search_console", "pageview"],
    "automation": ["n8n", "webhook", "cron", "schedule", "workflow", "adb"],
    "ai-llm": ["openrouter", "anthropic", "claude", "llm", "prompt", "completion"],
    "database": ["sqlite", "codex", "database", "sql", "query"],
    "api-service": ["fastapi", "uvicorn", "endpoint", "router", "api"],
    "web-scraping": ["scrape", "crawl", "browser", "selenium", "playwright"],
    "affiliate": ["affiliate", "amazon", "paapi", "commission", "link_manager"],
}

PATTERN_NAMES = {
    "forge-amplify": ["forge", "amplify", "scout", "sentinel", "smith", "codex", "enrich", "fortify"],
    "fastapi-service": ["fastapi", "uvicorn", "app.get", "app.post", "router"],
    "sqlite-codex": ["sqlite", "codex", "knowledge_db", "BaseCodex"],
    "retry-backoff": ["retry", "backoff", "exponential", "max_retries"],
    "brand-config": ["site_config", "brand_color", "sites.json"],
}


class DNAProfiler:
    """Generates capability DNA profiles for projects."""

    def __init__(self, graph: Optional[KnowledgeGraph] = None):
        self.graph = graph or KnowledgeGraph()

    def profile_project(self, slug: str) -> Dict:
        """Generate a DNA profile for a project."""
        project = self.graph.get_project(slug)
        if not project:
            return {"error": f"Project {slug} not found in graph"}

        project_id = project["id"]

        # Get all functions for this project
        with self.graph._conn() as conn:
            functions = [dict(r) for r in conn.execute(
                "SELECT * FROM functions WHERE project_id = ?", (project_id,)
            ).fetchall()]

            classes = [dict(r) for r in conn.execute(
                "SELECT * FROM classes WHERE project_id = ?", (project_id,)
            ).fetchall()]

            endpoints = [dict(r) for r in conn.execute(
                "SELECT * FROM api_endpoints WHERE project_id = ?", (project_id,)
            ).fetchall()]

            api_keys = [dict(r) for r in conn.execute(
                "SELECT * FROM api_keys_used WHERE project_id = ?", (project_id,)
            ).fetchall()]

            configs = [dict(r) for r in conn.execute(
                "SELECT * FROM configs WHERE project_id = ?", (project_id,)
            ).fetchall()]

        # Detect capabilities
        capabilities = self._detect_capabilities(functions, classes, endpoints)

        # Detect patterns
        patterns = self._detect_patterns(functions, classes)

        # APIs integrated
        apis = list(set(k["service_name"] for k in api_keys))

        # Tech stack
        tech_stack = ["python"]
        if any(e for e in endpoints):
            tech_stack.append("fastapi")
        if any("sqlite" in f["name"].lower() for f in functions):
            tech_stack.append("sqlite")

        # Code reuse score (0-100)
        total_funcs = len(functions)
        documented = sum(1 for f in functions if f.get("docstring"))
        reuse_score = min(100, int(
            (documented / max(total_funcs, 1)) * 40 +  # Documentation
            len(capabilities) * 5 +  # Breadth
            len(patterns) * 10 +  # Pattern usage
            min(len(endpoints), 10) * 2  # API surface
        ))

        # Knowledge entries count
        knowledge_count = self.graph.count_knowledge_for_project(slug) if hasattr(self.graph, 'count_knowledge_for_project') else 0

        return {
            "project": slug,
            "name": project.get("name", slug),
            "category": project.get("category", ""),
            "capabilities": sorted(capabilities),
            "patterns": sorted(patterns),
            "apis_integrated": sorted(apis),
            "tech_stack": sorted(tech_stack),
            "code_reuse_score": reuse_score,
            "knowledge_entries": knowledge_count,
            "stats": {
                "functions": total_funcs,
                "classes": len(classes),
                "endpoints": len(endpoints),
                "configs": len(configs),
            },
            "profiled_at": datetime.now().isoformat(),
        }

    def profile_all(self) -> Dict[str, Dict]:
        """Profile all projects."""
        projects = self.graph.list_projects()
        profiles = {}
        for p in projects:
            slug = p["slug"]
            try:
                profiles[slug] = self.profile_project(slug)
            except Exception as e:
                log.error(f"Error profiling {slug}: {e}")
        return profiles

    def find_similar(self, slug: str, top_n: int = 5) -> List[Dict]:
        """Find projects most similar to the given one."""
        target = self.profile_project(slug)
        if "error" in target:
            return []

        target_caps = set(target["capabilities"])
        target_patterns = set(target["patterns"])
        target_apis = set(target["apis_integrated"])

        all_profiles = self.profile_all()
        scores = []

        for other_slug, profile in all_profiles.items():
            if other_slug == slug:
                continue
            other_caps = set(profile["capabilities"])
            other_patterns = set(profile["patterns"])
            other_apis = set(profile["apis_integrated"])

            # Jaccard similarity
            cap_sim = len(target_caps & other_caps) / max(len(target_caps | other_caps), 1)
            pat_sim = len(target_patterns & other_patterns) / max(len(target_patterns | other_patterns), 1)
            api_sim = len(target_apis & other_apis) / max(len(target_apis | other_apis), 1)

            score = cap_sim * 0.5 + pat_sim * 0.3 + api_sim * 0.2
            scores.append({
                "project": other_slug,
                "similarity": round(score * 100, 1),
                "shared_capabilities": sorted(target_caps & other_caps),
                "shared_patterns": sorted(target_patterns & other_patterns),
            })

        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_n]

    def _detect_capabilities(self, functions, classes, endpoints) -> List[str]:
        """Detect capabilities from function/class names."""
        capabilities = set()
        all_names = [f["name"].lower() for f in functions] + [c["name"].lower() for c in classes]
        all_text = " ".join(all_names)

        for cap, keywords in CAPABILITY_PATTERNS.items():
            if any(kw in all_text for kw in keywords):
                capabilities.add(cap)

        # API endpoints indicate api-service
        if endpoints:
            capabilities.add("api-service")

        return list(capabilities)

    def _detect_patterns(self, functions, classes) -> List[str]:
        """Detect architectural patterns used."""
        patterns = set()
        all_names = [f["name"].lower() for f in functions] + [c["name"].lower() for c in classes]
        all_text = " ".join(all_names)

        for pattern, keywords in PATTERN_NAMES.items():
            matches = sum(1 for kw in keywords if kw.lower() in all_text)
            if matches >= 2:
                patterns.add(pattern)

        return list(patterns)

    def print_profile(self, profile: Dict):
        """Pretty-print a DNA profile."""
        print(f"\n{'='*60}")
        print(f"  DNA Profile: {profile['name']}")
        print(f"  Category: {profile['category']}")
        print(f"  Reuse Score: {profile['code_reuse_score']}/100")
        print(f"{'='*60}")

        print(f"\n  Capabilities ({len(profile['capabilities'])}):")
        for cap in profile["capabilities"]:
            print(f"    - {cap}")

        print(f"\n  Patterns ({len(profile['patterns'])}):")
        for pat in profile["patterns"]:
            print(f"    - {pat}")

        print(f"\n  APIs ({len(profile['apis_integrated'])}):")
        for api in profile["apis_integrated"]:
            print(f"    - {api}")

        print(f"\n  Stats:")
        for k, v in profile["stats"].items():
            print(f"    {k}: {v}")
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DNA Profiler")
    parser.add_argument("--project", help="Profile a specific project")
    parser.add_argument("--all", action="store_true", help="Profile all projects")
    parser.add_argument("--similar", help="Find similar projects")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    profiler = DNAProfiler()

    if args.project:
        profile = profiler.profile_project(args.project)
        if args.json:
            print(json.dumps(profile, indent=2))
        else:
            profiler.print_profile(profile)
    elif args.similar:
        results = profiler.find_similar(args.similar)
        for r in results:
            print(f"  {r['project']:30s} {r['similarity']:5.1f}%  shared: {', '.join(r['shared_capabilities'][:3])}")
    elif args.all:
        profiles = profiler.profile_all()
        if args.json:
            print(json.dumps(profiles, indent=2))
        else:
            for slug, profile in profiles.items():
                print(f"  {slug:30s} score={profile['code_reuse_score']:3d}  caps={len(profile['capabilities'])}  patterns={len(profile['patterns'])}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
