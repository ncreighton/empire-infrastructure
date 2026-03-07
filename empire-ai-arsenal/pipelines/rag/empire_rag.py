#!/usr/bin/env python3
"""
Empire Arsenal — RAG Knowledge Base System v2.0
================================================
Crawls all 16 WordPress sites into Qdrant vector embeddings,
builds site-specific knowledge bases, powers Dify AI assistants,
and enables cross-empire content gap analysis.

Architecture:
  Crawl4AI (site crawler) → LiteLLM (embeddings via nomic-embed-text)
  → Qdrant (vector DB) → Dify (chatbot builder) → WordPress (widget)

Usage:
  # Crawl and index a single site
  python empire_rag.py crawl --site witchcraftforbeginners

  # Crawl all sites
  python empire_rag.py crawl --all

  # Search across all sites
  python empire_rag.py search "best crystals for protection"

  # Find content gaps between sites
  python empire_rag.py gaps --site witchcraftforbeginners

  # Export Dify knowledge base
  python empire_rag.py export-dify --site smarthomewizards

  # Stats on all collections
  python empire_rag.py stats
"""

import argparse
import json
import hashlib
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

ARSENAL_IP = os.getenv("ARSENAL_IP", "89.116.29.33")
QDRANT_URL = os.getenv("QDRANT_URL", f"http://{ARSENAL_IP}:6333")
LITELLM_URL = os.getenv("LITELLM_URL", f"http://{ARSENAL_IP}:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-arsenal-fec2dfe2b1256586b84b962c9d25e4e9")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", f"http://{ARSENAL_IP}:11235")
OLLAMA_URL = os.getenv("OLLAMA_URL", f"http://{ARSENAL_IP}:11434")
DIFY_URL = os.getenv("DIFY_URL", f"http://{ARSENAL_IP}:3001")

# Embedding config
EMBEDDING_MODEL = "nomic-embed-text"   # Via Ollama (384 dimensions)
EMBEDDING_DIM = 768                     # nomic-embed-text output dim
CHUNK_SIZE = 500                        # Words per chunk
CHUNK_OVERLAP = 50                      # Word overlap between chunks
MAX_CRAWL_PAGES = 200                   # Max pages to crawl per site

SITES_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "sites.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("empire-rag")

# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SiteInfo:
    site_id: str
    name: str
    domain: str
    niche: str
    wp_user: str
    wp_password: str

    @property
    def wp_api_url(self):
        return f"https://{self.domain}/wp-json/wp/v2"

    @property
    def collection_name(self):
        return f"site_{self.site_id.replace('-', '_')}"


@dataclass
class ContentChunk:
    chunk_id: str
    site_id: str
    url: str
    title: str
    content: str
    word_count: int
    chunk_index: int
    total_chunks: int
    post_id: Optional[int] = None
    categories: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    published_date: Optional[str] = None
    embedding: Optional[list] = None


@dataclass
class SearchResult:
    chunk: ContentChunk
    score: float
    site_id: str


# ──────────────────────────────────────────────────────────────────────────────
# Site Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_sites() -> dict[str, SiteInfo]:
    """Load site configurations."""
    config_path = SITES_CONFIG_PATH
    if not config_path.exists():
        config_path = Path("D:/Claude Code Projects/config/sites.json")

    with open(config_path) as f:
        data = json.load(f)

    sites = {}
    raw = data.get("sites", data)
    for sid, cfg in raw.items():
        wp = cfg.get("wordpress", {})
        brand = cfg.get("brand", {})
        sites[sid] = SiteInfo(
            site_id=sid,
            name=cfg.get("name", sid),
            domain=cfg.get("domain", ""),
            niche=brand.get("voice", "General"),
            wp_user=wp.get("user", ""),
            wp_password=wp.get("app_password", cfg.get("wp_app_password", "")),
        )
    return sites


# ──────────────────────────────────────────────────────────────────────────────
# Embedding Engine
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """Generate embeddings via Ollama's nomic-embed-text model."""

    def __init__(self):
        self.ollama_url = OLLAMA_URL
        self.model = EMBEDDING_MODEL

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def embed_batch(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                emb = self.embed(text)
                all_embeddings.append(emb)
            if i + batch_size < len(texts):
                time.sleep(0.1)  # Rate limiting
        return all_embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Qdrant Vector Store
# ──────────────────────────────────────────────────────────────────────────────

class QdrantStore:
    """Manage Qdrant collections for the knowledge base."""

    def __init__(self):
        self.base_url = QDRANT_URL

    def create_collection(self, name: str, dim: int = EMBEDDING_DIM):
        """Create a Qdrant collection if it doesn't exist."""
        # Check if exists
        resp = requests.get(f"{self.base_url}/collections/{name}", timeout=10)
        if resp.status_code == 200:
            logger.info(f"Collection '{name}' already exists")
            return

        # Create
        resp = requests.put(
            f"{self.base_url}/collections/{name}",
            json={
                "vectors": {
                    "size": dim,
                    "distance": "Cosine",
                },
            },
            timeout=15,
        )
        resp.raise_for_status()
        logger.info(f"Created collection: {name}")

    def upsert_points(self, collection: str, points: list[dict]):
        """Upsert vectors into a collection."""
        if not points:
            return

        # Batch in groups of 100
        for i in range(0, len(points), 100):
            batch = points[i:i + 100]
            resp = requests.put(
                f"{self.base_url}/collections/{collection}/points",
                json={"points": batch},
                timeout=60,
            )
            resp.raise_for_status()
        logger.info(f"Upserted {len(points)} points to '{collection}'")

    def search(self, collection: str, vector: list[float], limit: int = 10,
               filters: Optional[dict] = None) -> list[dict]:
        """Search for similar vectors."""
        payload = {
            "vector": vector,
            "limit": limit,
            "with_payload": True,
        }
        if filters:
            payload["filter"] = filters

        resp = requests.post(
            f"{self.base_url}/collections/{collection}/points/search",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("result", [])

    def count(self, collection: str) -> int:
        """Get point count in a collection."""
        try:
            resp = requests.post(
                f"{self.base_url}/collections/{collection}/points/count",
                json={"exact": True},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("result", {}).get("count", 0)
        except Exception:
            return 0

    def list_collections(self) -> list[str]:
        """List all collections."""
        resp = requests.get(f"{self.base_url}/collections", timeout=10)
        resp.raise_for_status()
        return [c["name"] for c in resp.json().get("result", {}).get("collections", [])]

    def delete_collection(self, name: str):
        """Delete a collection."""
        resp = requests.delete(f"{self.base_url}/collections/{name}", timeout=10)
        resp.raise_for_status()
        logger.info(f"Deleted collection: {name}")


# ──────────────────────────────────────────────────────────────────────────────
# Content Crawler
# ──────────────────────────────────────────────────────────────────────────────

class SiteCrawler:
    """Crawl WordPress sites and extract content."""

    def __init__(self):
        self.crawl4ai_url = CRAWL4AI_URL

    def get_all_posts(self, site: SiteInfo, max_posts: int = MAX_CRAWL_PAGES) -> list[dict]:
        """Fetch all published posts from a WordPress site."""
        posts = []
        page = 1
        per_page = 100

        import base64
        creds = base64.b64encode(f"{site.wp_user}:{site.wp_password}".encode()).decode()
        headers = {"Authorization": f"Basic {creds}"}

        while len(posts) < max_posts:
            try:
                resp = requests.get(
                    f"{site.wp_api_url}/posts",
                    params={
                        "per_page": per_page,
                        "page": page,
                        "status": "publish",
                        "_fields": "id,title,content,excerpt,slug,date,categories,tags,link",
                    },
                    headers=headers,
                    timeout=30,
                )

                if resp.status_code == 400:
                    break  # No more pages

                resp.raise_for_status()
                batch = resp.json()

                if not batch:
                    break

                posts.extend(batch)
                page += 1

                if len(batch) < per_page:
                    break

            except Exception as e:
                logger.warning(f"Failed to fetch page {page} from {site.domain}: {e}")
                break

        logger.info(f"Fetched {len(posts)} posts from {site.domain}")
        return posts[:max_posts]

    def crawl_url_content(self, url: str) -> str:
        """Crawl a single URL for content using Crawl4AI."""
        try:
            resp = requests.post(
                f"{self.crawl4ai_url}/crawl",
                json={
                    "urls": [url],
                    "word_count_threshold": 50,
                    "extraction_strategy": "NoExtractionStrategy",
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("markdown", data[0].get("cleaned_html", ""))
            return data.get("markdown", "")
        except Exception as e:
            logger.warning(f"Crawl4AI failed for {url}: {e}")
            return ""

    def extract_text(self, html: str) -> str:
        """Strip HTML tags and clean text."""
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&[^;]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 50:  # Skip tiny chunks
                chunks.append(chunk)
            start = end - overlap

        return chunks


# ──────────────────────────────────────────────────────────────────────────────
# RAG Knowledge Base Manager
# ──────────────────────────────────────────────────────────────────────────────

class EmpireRAG:
    """Main RAG system managing knowledge bases for all empire sites."""

    # Cross-empire collection for searching across all sites
    EMPIRE_COLLECTION = "empire_knowledge"

    def __init__(self):
        self.sites = load_sites()
        self.embedder = EmbeddingEngine()
        self.qdrant = QdrantStore()
        self.crawler = SiteCrawler()

    def init_collections(self):
        """Initialize all Qdrant collections."""
        # Per-site collections
        for site in self.sites.values():
            self.qdrant.create_collection(site.collection_name, EMBEDDING_DIM)

        # Cross-empire collection
        self.qdrant.create_collection(self.EMPIRE_COLLECTION, EMBEDDING_DIM)

        logger.info(f"Initialized {len(self.sites) + 1} collections")

    def crawl_and_index_site(self, site_id: str, force_reindex: bool = False):
        """Crawl a site and index all content into Qdrant."""
        site = self.sites.get(site_id)
        if not site:
            raise ValueError(f"Unknown site: {site_id}")

        logger.info(f"🕷️ Crawling {site.domain}...")

        if force_reindex:
            try:
                self.qdrant.delete_collection(site.collection_name)
            except Exception:
                pass
            self.qdrant.create_collection(site.collection_name, EMBEDDING_DIM)

        # Fetch all posts via WordPress API
        posts = self.crawler.get_all_posts(site)

        if not posts:
            logger.warning(f"No posts found for {site.domain}")
            return

        all_chunks = []
        all_embeddings = []
        points_site = []
        points_empire = []

        for post in posts:
            title = post.get("title", {}).get("rendered", "")
            content_html = post.get("content", {}).get("rendered", "")
            content_text = self.crawler.extract_text(content_html)
            post_url = post.get("link", f"https://{site.domain}/{post.get('slug', '')}")

            if not content_text or len(content_text) < 100:
                continue

            # Chunk the content
            chunks = self.crawler.chunk_text(content_text)

            for i, chunk_text in enumerate(chunks):
                chunk_id = hashlib.md5(f"{site_id}:{post_url}:{i}".encode()).hexdigest()

                # Prepend title for better embedding context
                embed_text = f"{title}. {chunk_text}"

                chunk = ContentChunk(
                    chunk_id=chunk_id,
                    site_id=site_id,
                    url=post_url,
                    title=title,
                    content=chunk_text,
                    word_count=len(chunk_text.split()),
                    chunk_index=i,
                    total_chunks=len(chunks),
                    post_id=post.get("id"),
                    categories=[str(c) for c in post.get("categories", [])],
                    tags=[str(t) for t in post.get("tags", [])],
                    published_date=post.get("date", ""),
                )

                all_chunks.append(chunk)
                all_embeddings.append(embed_text)

        if not all_chunks:
            logger.warning(f"No content chunks created for {site.domain}")
            return

        # Generate embeddings in batches
        logger.info(f"🧠 Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedder.embed_batch(
            all_embeddings, batch_size=5
        )

        # Prepare Qdrant points
        for chunk, embedding in zip(all_chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))

            point = {
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "site_id": chunk.site_id,
                    "site_name": site.name,
                    "domain": site.domain,
                    "url": chunk.url,
                    "title": chunk.title,
                    "content": chunk.content,
                    "word_count": chunk.word_count,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "post_id": chunk.post_id,
                    "published_date": chunk.published_date,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                },
            }

            points_site.append(point)
            points_empire.append(point)

        # Upsert to site-specific collection
        logger.info(f"📥 Indexing {len(points_site)} chunks to {site.collection_name}...")
        self.qdrant.upsert_points(site.collection_name, points_site)

        # Upsert to empire-wide collection
        self.qdrant.upsert_points(self.EMPIRE_COLLECTION, points_empire)

        logger.info(f"✅ Indexed {len(points_site)} chunks from {site.domain}")

    def crawl_all_sites(self, max_workers: int = 3):
        """Crawl and index all sites in parallel."""
        self.init_collections()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.crawl_and_index_site, sid): sid
                for sid in self.sites
            }

            for future in as_completed(futures):
                sid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to index {sid}: {e}")

    def search(self, query: str, site_id: Optional[str] = None,
               limit: int = 10) -> list[SearchResult]:
        """Search the knowledge base."""
        # Generate query embedding
        query_embedding = self.embedder.embed(query)

        # Determine collection
        if site_id:
            site = self.sites.get(site_id)
            if not site:
                raise ValueError(f"Unknown site: {site_id}")
            collection = site.collection_name
        else:
            collection = self.EMPIRE_COLLECTION

        # Search Qdrant
        results = self.qdrant.search(collection, query_embedding, limit=limit)

        search_results = []
        for r in results:
            payload = r.get("payload", {})
            chunk = ContentChunk(
                chunk_id=str(r.get("id", "")),
                site_id=payload.get("site_id", ""),
                url=payload.get("url", ""),
                title=payload.get("title", ""),
                content=payload.get("content", ""),
                word_count=payload.get("word_count", 0),
                chunk_index=payload.get("chunk_index", 0),
                total_chunks=payload.get("total_chunks", 1),
                post_id=payload.get("post_id"),
                published_date=payload.get("published_date"),
            )
            search_results.append(SearchResult(
                chunk=chunk,
                score=r.get("score", 0.0),
                site_id=payload.get("site_id", ""),
            ))

        return search_results

    def find_content_gaps(self, site_id: str, comparison_sites: Optional[list[str]] = None) -> list[dict]:
        """Find content gaps by comparing a site's knowledge base with others."""
        site = self.sites.get(site_id)
        if not site:
            raise ValueError(f"Unknown site: {site_id}")

        # Get all unique topics from the target site
        site_count = self.qdrant.count(site.collection_name)
        if site_count == 0:
            return [{"gap": "Site has no indexed content. Run crawl first."}]

        # Use LLM to analyze gaps
        # Fetch a sample of content from the site
        sample_query = self.embedder.embed(f"{site.name} main topics")
        site_content = self.qdrant.search(site.collection_name, sample_query, limit=20)

        titles = list(set(r["payload"].get("title", "") for r in site_content))

        # Compare with similar niche sites if specified
        comparison_topics = []
        if comparison_sites:
            for comp_id in comparison_sites:
                comp = self.sites.get(comp_id)
                if comp:
                    comp_content = self.qdrant.search(
                        comp.collection_name,
                        sample_query,
                        limit=20,
                    )
                    comparison_topics.extend([
                        r["payload"].get("title", "") for r in comp_content
                    ])

        return {
            "site": site_id,
            "existing_topics": titles[:20],
            "indexed_chunks": site_count,
            "comparison_topics": list(set(comparison_topics))[:20] if comparison_topics else [],
        }

    def build_context(self, query: str, site_id: Optional[str] = None,
                      max_tokens: int = 3000) -> str:
        """Build RAG context for LLM consumption."""
        results = self.search(query, site_id=site_id, limit=8)

        if not results:
            return "No relevant content found in the knowledge base."

        context_parts = []
        total_words = 0
        max_words = max_tokens // 1.3  # Rough token-to-word ratio

        for r in results:
            chunk_text = f"**{r.chunk.title}** ({r.chunk.url})\n{r.chunk.content}\n"
            words = len(chunk_text.split())

            if total_words + words > max_words:
                break

            context_parts.append(chunk_text)
            total_words += words

        return "\n---\n".join(context_parts)

    def get_stats(self) -> dict:
        """Get statistics for all collections."""
        stats = {}
        collections = self.qdrant.list_collections()

        for name in collections:
            count = self.qdrant.count(name)
            stats[name] = {"chunks": count}

        return stats

    def export_for_dify(self, site_id: str, output_dir: Optional[str] = None) -> str:
        """Export knowledge base in Dify-compatible format."""
        site = self.sites.get(site_id)
        if not site:
            raise ValueError(f"Unknown site: {site_id}")

        # Fetch all content from the site collection
        count = self.qdrant.count(site.collection_name)
        if count == 0:
            logger.warning(f"No content indexed for {site_id}")
            return ""

        # Get a large sample
        sample_vec = self.embedder.embed(f"{site.name} content")
        results = self.qdrant.search(site.collection_name, sample_vec, limit=min(count, 500))

        # Group by URL/post
        posts = {}
        for r in results:
            url = r["payload"].get("url", "")
            if url not in posts:
                posts[url] = {
                    "title": r["payload"].get("title", ""),
                    "url": url,
                    "chunks": [],
                }
            posts[url]["chunks"].append({
                "content": r["payload"].get("content", ""),
                "index": r["payload"].get("chunk_index", 0),
            })

        # Sort chunks within each post
        for post in posts.values():
            post["chunks"].sort(key=lambda c: c["index"])

        # Generate Dify-compatible documents
        documents = []
        for url, post in posts.items():
            full_content = "\n".join(c["content"] for c in post["chunks"])
            documents.append({
                "name": post["title"],
                "text": full_content,
                "metadata": {
                    "source": url,
                    "site": site.name,
                    "domain": site.domain,
                },
            })

        # Save to file
        out_dir = Path(output_dir) if output_dir else Path(f"data/dify-export")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{site_id}_knowledge.json"

        with open(out_file, "w") as f:
            json.dump(documents, f, indent=2)

        logger.info(f"Exported {len(documents)} documents to {out_file}")
        return str(out_file)


# ──────────────────────────────────────────────────────────────────────────────
# Dify App Builder
# ──────────────────────────────────────────────────────────────────────────────

class DifyAppBuilder:
    """Create Dify AI assistant apps for each site."""

    def __init__(self, rag: EmpireRAG):
        self.rag = rag
        self.dify_url = DIFY_URL

    def generate_app_config(self, site_id: str) -> dict:
        """Generate a Dify app configuration for a site."""
        site = self.rag.sites.get(site_id)
        if not site:
            raise ValueError(f"Unknown site: {site_id}")

        return {
            "name": f"{site.name} AI Assistant",
            "description": f"AI-powered assistant for {site.name} ({site.domain}). "
                          f"Answers questions about {site.niche} content with RAG-powered context.",
            "mode": "chat",
            "icon": "🤖",
            "icon_background": "#7C3AED",
            "model_config": {
                "provider": "openai_api_compatible",
                "model": "claude-sonnet",
                "completion_params": {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
            },
            "pre_prompt": f"""You are the AI assistant for {site.name} ({site.domain}).
Your expertise: {site.niche}

PERSONALITY:
- Helpful, knowledgeable, and engaging
- Match the site's brand voice: {site.niche}
- Reference specific articles when possible
- Suggest related topics the user might enjoy
- Keep responses focused and actionable

CONTEXT RULES:
- Use the provided context from the knowledge base to answer
- If the context doesn't contain the answer, say so honestly
- Always cite sources with article titles and links
- Never make up information not in the context

When suggesting products, use Amazon affiliate tag: {site.sites.get(site_id, SiteInfo(site_id, '', '', '', '', '')).site_id if hasattr(site, 'sites') else ''}
""",
            "retrieval_config": {
                "search_method": "semantic_search",
                "top_k": 5,
                "score_threshold": 0.5,
            },
            "suggested_questions": self._get_suggested_questions(site_id),
        }

    def _get_suggested_questions(self, site_id: str) -> list[str]:
        """Generate site-specific suggested questions."""
        question_map = {
            "witchcraftforbeginners": [
                "What crystals are best for beginners?",
                "How do I set up my first altar?",
                "What are the moon phases for rituals?",
            ],
            "smarthomewizards": [
                "What's the best smart home hub?",
                "How do I set up home automation?",
                "Zigbee vs Z-Wave: which is better?",
            ],
            "bulletjournals": [
                "What are the best spreads for beginners?",
                "How do I set up a habit tracker?",
                "What supplies do I need to start?",
            ],
            "wealthfromai": [
                "How can I make money with AI?",
                "What are the best AI side hustles?",
                "How do I start an AI consulting business?",
            ],
        }
        return question_map.get(site_id, [
            "What topics does this site cover?",
            "What's the most popular article?",
            "Can you recommend something for beginners?",
        ])


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Empire Arsenal RAG Knowledge Base v2.0")
    subparsers = parser.add_subparsers(dest="command")

    # Crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl and index a site")
    crawl_parser.add_argument("--site", help="Site ID to crawl")
    crawl_parser.add_argument("--all", action="store_true", help="Crawl all sites")
    crawl_parser.add_argument("--force", action="store_true", help="Force reindex")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the knowledge base")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--site", help="Limit to specific site")
    search_parser.add_argument("--limit", type=int, default=5, help="Max results")

    # Gaps command
    gaps_parser = subparsers.add_parser("gaps", help="Find content gaps")
    gaps_parser.add_argument("--site", required=True, help="Site to analyze")
    gaps_parser.add_argument("--compare", help="Comma-separated sites to compare")

    # Export command
    export_parser = subparsers.add_parser("export-dify", help="Export for Dify")
    export_parser.add_argument("--site", required=True, help="Site to export")
    export_parser.add_argument("--output", help="Output directory")

    # Stats command
    subparsers.add_parser("stats", help="Show collection statistics")

    # Init command
    subparsers.add_parser("init", help="Initialize all collections")

    args = parser.parse_args()
    rag = EmpireRAG()

    if args.command == "init":
        rag.init_collections()

    elif args.command == "crawl":
        if args.all:
            rag.crawl_all_sites()
        elif args.site:
            rag.init_collections()
            rag.crawl_and_index_site(args.site, force_reindex=args.force)
        else:
            print("Specify --site SITE_ID or --all")

    elif args.command == "search":
        results = rag.search(args.query, site_id=args.site, limit=args.limit)
        print(f"\n🔍 Search: '{args.query}' ({len(results)} results)\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.score:.3f}] {r.chunk.title}")
            print(f"     Site: {r.site_id} | {r.chunk.url}")
            print(f"     {r.chunk.content[:200]}...")
            print()

    elif args.command == "gaps":
        compare = args.compare.split(",") if args.compare else None
        gaps = rag.find_content_gaps(args.site, comparison_sites=compare)
        print(f"\n📊 Content Gap Analysis: {args.site}")
        print(json.dumps(gaps, indent=2))

    elif args.command == "export-dify":
        path = rag.export_for_dify(args.site, output_dir=args.output)
        print(f"✅ Exported to: {path}")

    elif args.command == "stats":
        stats = rag.get_stats()
        print("\n📊 RAG Knowledge Base Statistics:")
        for name, info in sorted(stats.items()):
            print(f"  {name:<35} {info['chunks']:>6} chunks")
        print(f"  {'─' * 45}")
        total = sum(s["chunks"] for s in stats.values())
        print(f"  {'TOTAL':<35} {total:>6} chunks")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
