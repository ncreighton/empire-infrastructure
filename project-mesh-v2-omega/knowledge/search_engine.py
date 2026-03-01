"""
Semantic Search Engine   Full-text search over the knowledge graph.
Supports searching functions, classes, endpoints, knowledge entries, and patterns.
Category-aware scoring with cross-project relevance ranking.
"""

import re
import logging
from typing import Dict, List, Optional
from knowledge.graph_engine import KnowledgeGraph

log = logging.getLogger(__name__)


class SearchEngine:
    """Semantic search over the knowledge graph."""

    def __init__(self, graph: Optional[KnowledgeGraph] = None):
        self.graph = graph or KnowledgeGraph()

    def search(self, query: str, category: str = "", limit: int = 30,
               search_type: str = "all") -> List[Dict]:
        """
        Search the knowledge graph.

        Args:
            query: Search query string
            category: Filter by category (optional)
            limit: Max results
            search_type: 'all', 'functions', 'classes', 'endpoints', 'knowledge', 'patterns'

        Returns:
            List of scored results with type, name, file_path, detail, project, score
        """
        terms = self._tokenize(query)
        if not terms:
            return []

        results = []

        if search_type in ("all", "functions"):
            results.extend(self._search_functions(terms, limit))

        if search_type in ("all", "classes"):
            results.extend(self._search_classes(terms, limit))

        if search_type in ("all", "endpoints"):
            results.extend(self._search_endpoints(terms, limit))

        if search_type in ("all", "knowledge"):
            results.extend(self._search_knowledge(terms, category, limit))

        if search_type in ("all", "patterns"):
            results.extend(self._search_patterns(terms, limit))

        # Score and sort
        scored = []
        for r in results:
            r["score"] = self._score_result(r, terms, category)
            scored.append(r)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def find_implementations(self, capability: str) -> List[Dict]:
        """Find all implementations of a capability across projects."""
        results = self.search(capability, search_type="functions", limit=50)
        results.extend(self.search(capability, search_type="classes", limit=50))
        # Deduplicate by file_path + name
        seen = set()
        unique = []
        for r in results:
            key = f"{r.get('file_path', '')}:{r.get('name', '')}"
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def find_project_for(self, need: str) -> List[Dict]:
        """Find which project has a system for a given need."""
        results = self.search(need, limit=20)
        # Group by project
        projects = {}
        for r in results:
            proj = r.get("project", "unknown")
            if proj not in projects:
                projects[proj] = {"project": proj, "matches": 0, "top_match": r}
            projects[proj]["matches"] += 1
        return sorted(projects.values(), key=lambda x: x["matches"], reverse=True)

    def _tokenize(self, query: str) -> List[str]:
        """Split query into searchable tokens."""
        # Split on spaces, underscores, hyphens, camelCase
        words = re.split(r'[\s_\-]+', query.lower())
        # Also split camelCase
        expanded = []
        for w in words:
            parts = re.findall(r'[a-z]+|[A-Z][a-z]*', w)
            if parts:
                expanded.extend(p.lower() for p in parts)
            else:
                expanded.append(w)
        return [w for w in expanded if len(w) >= 2]

    def _search_functions(self, terms: List[str], limit: int) -> List[Dict]:
        results = []
        for term in terms:
            for r in self.graph.find_functions(term, limit=limit // len(terms)):
                results.append({
                    "type": "function",
                    "name": r["name"],
                    "file_path": r["file_path"],
                    "detail": r.get("signature", ""),
                    "project": r.get("project_slug", ""),
                    "line": r.get("line_number", 0),
                    "docstring": r.get("docstring", ""),
                })
        return results

    def _search_classes(self, terms: List[str], limit: int) -> List[Dict]:
        results = []
        for term in terms:
            for r in self.graph.find_classes(term, limit=limit // len(terms)):
                results.append({
                    "type": "class",
                    "name": r["name"],
                    "file_path": r["file_path"],
                    "detail": r.get("bases", ""),
                    "project": r.get("project_slug", ""),
                    "line": r.get("line_number", 0),
                    "docstring": r.get("docstring", ""),
                })
        return results

    def _search_endpoints(self, terms: List[str], limit: int) -> List[Dict]:
        results = []
        for term in terms:
            for r in self.graph.find_endpoints(term, limit=limit // len(terms)):
                results.append({
                    "type": "endpoint",
                    "name": f"{r['method']} {r['path']}",
                    "file_path": r.get("file_path", ""),
                    "detail": r.get("handler", ""),
                    "project": r.get("project_slug", ""),
                    "line": r.get("line_number", 0),
                })
        return results

    def _search_knowledge(self, terms: List[str], category: str, limit: int) -> List[Dict]:
        query = " ".join(terms)
        results = []
        for r in self.graph.search_knowledge(query, category=category, limit=limit):
            results.append({
                "type": "knowledge",
                "name": r["text"][:80],
                "file_path": r.get("source_file", ""),
                "detail": r.get("category", ""),
                "project": r.get("source_project", ""),
                "confidence": r.get("confidence", 0.5),
            })
        return results

    def _search_patterns(self, terms: List[str], limit: int) -> List[Dict]:
        query = " ".join(terms)
        with self.graph._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM patterns WHERE name LIKE ? OR description LIKE ? LIMIT ?",
                (f"%{query}%", f"%{query}%", limit)
            ).fetchall()
        results = []
        for r in rows:
            results.append({
                "type": "pattern",
                "name": r["name"],
                "file_path": r.get("canonical_source", ""),
                "detail": r.get("description", ""),
                "project": "",
            })
        return results

    def _score_result(self, result: Dict, terms: List[str], category: str) -> float:
        """Score a result based on term relevance."""
        score = 0.0
        name = result.get("name", "").lower()
        detail = result.get("detail", "").lower()
        docstring = result.get("docstring", "").lower()

        for term in terms:
            # Exact name match (highest)
            if term == name:
                score += 50
            elif term in name:
                score += 20
            # Detail/signature match
            if term in detail:
                score += 10
            # Docstring match
            if term in docstring:
                score += 5

        # Type boost
        type_boosts = {
            "function": 1.2,
            "class": 1.1,
            "endpoint": 1.3,
            "knowledge": 1.0,
            "pattern": 1.4,
        }
        score *= type_boosts.get(result.get("type", ""), 1.0)

        # Category match boost
        if category and result.get("detail", "") == category:
            score *= 1.5

        # Confidence boost for knowledge entries
        if result.get("type") == "knowledge":
            score *= result.get("confidence", 0.5) + 0.5

        return round(score, 2)

    def print_results(self, results: List[Dict]):
        """Pretty-print search results."""
        if not results:
            print("  No results found.")
            return

        type_icons = {
            "function": "fn",
            "class": "cls",
            "endpoint": "api",
            "knowledge": "kb",
            "pattern": "pat",
        }

        print(f"\n  Found {len(results)} results:\n")
        for i, r in enumerate(results[:20], 1):
            icon = type_icons.get(r["type"], "?")
            proj = r.get("project", "")
            score = r.get("score", 0)
            print(f"  {i:2d}. [{icon:3s}] {r['name']}")
            if proj:
                print(f"       Project: {proj}  |  Score: {score}")
            if r.get("file_path"):
                line = r.get("line", "")
                loc = f"{r['file_path']}:{line}" if line else r["file_path"]
                print(f"       {loc}")
            if r.get("detail"):
                print(f"       {r['detail'][:100]}")
            print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Empire Search Engine")
    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("--type", choices=["all", "functions", "classes", "endpoints", "knowledge", "patterns"],
                        default="all")
    parser.add_argument("--category", default="")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--find-impl", help="Find all implementations of a capability")
    parser.add_argument("--find-project", help="Find which project has a system for X")
    args = parser.parse_args()

    engine = SearchEngine()

    if args.find_impl:
        results = engine.find_implementations(args.find_impl)
        engine.print_results(results)
    elif args.find_project:
        projects = engine.find_project_for(args.find_project)
        for p in projects:
            print(f"  {p['project']:30s} ({p['matches']} matches)")
    elif args.query:
        query_str = " ".join(args.query)
        results = engine.search(query_str, category=args.category,
                                limit=args.limit, search_type=args.type)
        engine.print_results(results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
