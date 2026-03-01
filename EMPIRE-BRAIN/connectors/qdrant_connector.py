"""Qdrant Connector — Vector Database for Semantic Search

Stores embeddings of:
- Learnings (for "find what worked before" queries)
- Code solutions (for "I've solved this before" lookups)
- Skills (for "what tool does X" discovery)
- Session summaries (for context loading)

Uses Qdrant on Contabo VPS (port 6333).
"""
import json
import hashlib
from datetime import datetime
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QDRANT_HOST, QDRANT_PORT


class QdrantConnector:
    """Interface to Qdrant vector database."""

    COLLECTIONS = {
        "learnings": 384,   # Dimension for all-MiniLM-L6-v2
        "solutions": 384,
        "skills": 384,
        "sessions": 384,
    }

    def __init__(self, host: str = "", port: int = 0):
        self.host = host or QDRANT_HOST
        self.port = port or QDRANT_PORT
        self.client = None

    def connect(self) -> bool:
        """Connect to Qdrant."""
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=self.host, port=self.port, timeout=10)
            self.client.get_collections()
            return True
        except Exception as e:
            print(f"Qdrant connection failed: {e}")
            return False

    def init_collections(self):
        """Create collections if they don't exist."""
        if not self.client:
            if not self.connect():
                return False
        try:
            from qdrant_client.models import VectorParams, Distance
            existing = {c.name for c in self.client.get_collections().collections}
            for name, dim in self.COLLECTIONS.items():
                if name not in existing:
                    self.client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
            return True
        except Exception as e:
            print(f"Collection init failed: {e}")
            return False

    def upsert(self, collection: str, id: str, vector: list[float], payload: dict):
        """Upsert a vector with payload."""
        if not self.client:
            if not self.connect():
                return False
        try:
            from qdrant_client.models import PointStruct
            point_id = int(hashlib.md5(id.encode()).hexdigest()[:15], 16)
            self.client.upsert(
                collection_name=collection,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
            return True
        except Exception as e:
            print(f"Upsert failed: {e}")
            return False

    def search(self, collection: str, query_vector: list[float], limit: int = 10) -> list[dict]:
        """Search for similar vectors."""
        if not self.client:
            if not self.connect():
                return []
        try:
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
            )
            return [{"score": r.score, "payload": r.payload} for r in results]
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def count(self, collection: str) -> int:
        """Get count of vectors in collection."""
        if not self.client:
            if not self.connect():
                return 0
        try:
            info = self.client.get_collection(collection)
            return info.points_count
        except Exception:
            return 0

    def stats(self) -> dict:
        """Get stats for all collections."""
        result = {}
        for name in self.COLLECTIONS:
            result[name] = self.count(name)
        return result
