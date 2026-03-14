"""
KnowledgeGraph — Nodes and edges representing relationships between
coins, strategies, regimes, skills, and patterns.

Answers questions like:
  "What strategies work best for DOGE in BREAKOUT regime?"
  "Which coins correlate with BTC-USD?"
  "What patterns precede regime changes?"
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from moneyclaw.persistence.database import Database

logger = logging.getLogger(__name__)

# Standard node types
NODE_TYPES = ("coin", "strategy", "regime", "skill", "pattern", "indicator")

# Standard edge types
EDGE_TYPES = (
    "correlates_with",
    "performs_best_in",
    "precedes",
    "causes",
    "conflicts_with",
    "used_by",
)


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


class KnowledgeGraph:
    """In-database knowledge graph for trading intelligence.

    Parameters
    ----------
    db : Database
        Shared database instance.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        node_type: str,
        name: str,
        properties: dict | None = None,
        weight: float = 1.0,
    ) -> int:
        """Create or update a node. Returns node id."""
        now = _utcnow_iso()
        props_json = json.dumps(properties or {})

        with self.db._cursor() as cur:
            cur.execute(
                """
                INSERT INTO intel_kg_nodes (node_type, name, properties, weight, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_type, name) DO UPDATE SET
                    properties = ?,
                    weight = ?,
                    updated_at = ?
                """,
                (node_type, name, props_json, weight, now, now,
                 props_json, weight, now),
            )

            # Get the id
            cur.execute(
                "SELECT id FROM intel_kg_nodes WHERE node_type = ? AND name = ?",
                (node_type, name),
            )
            row = cur.fetchone()
            return row["id"] if row else 0

    def get_node(self, node_type: str, name: str) -> dict | None:
        """Get a node by type and name."""
        with self.db._cursor() as cur:
            cur.execute(
                "SELECT * FROM intel_kg_nodes WHERE node_type = ? AND name = ?",
                (node_type, name),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_nodes_by_type(self, node_type: str) -> list[dict]:
        """Get all nodes of a given type."""
        with self.db._cursor() as cur:
            cur.execute(
                "SELECT * FROM intel_kg_nodes WHERE node_type = ? ORDER BY weight DESC",
                (node_type,),
            )
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def upsert_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: str,
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> int:
        """Create or update an edge. Returns edge id."""
        now = _utcnow_iso()
        meta_json = json.dumps(metadata or {})

        with self.db._cursor() as cur:
            cur.execute(
                """
                INSERT INTO intel_kg_edges
                    (source_id, target_id, edge_type, weight, evidence, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                    weight = MIN(10.0, (weight + ?) / 2.0),
                    evidence = evidence + 1,
                    metadata = ?,
                    updated_at = ?
                """,
                (source_id, target_id, edge_type, weight, meta_json, now, now,
                 weight, meta_json, now),
            )

            cur.execute(
                """SELECT id FROM intel_kg_edges
                   WHERE source_id = ? AND target_id = ? AND edge_type = ?""",
                (source_id, target_id, edge_type),
            )
            row = cur.fetchone()
            return row["id"] if row else 0

    def strengthen_edge(self, edge_id: int, weight_boost: float = 0.1) -> None:
        """Increase edge weight and evidence count."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                UPDATE intel_kg_edges
                SET weight = MIN(10.0, weight + ?),
                    evidence = evidence + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (weight_boost, _utcnow_iso(), edge_id),
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_neighbors(
        self,
        node_type: str,
        name: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[dict]:
        """Find nodes connected to a given node.

        Parameters
        ----------
        direction : str
            "outgoing" = edges FROM this node, "incoming" = edges TO this node, "both".
        """
        node = self.get_node(node_type, name)
        if not node:
            return []

        node_id = node["id"]
        results: list[dict] = []

        if direction in ("outgoing", "both"):
            query = """
                SELECT e.*, n.node_type as target_type, n.name as target_name, n.properties as target_props
                FROM intel_kg_edges e
                JOIN intel_kg_nodes n ON e.target_id = n.id
                WHERE e.source_id = ?
            """
            params: list[Any] = [node_id]
            if edge_type:
                query += " AND e.edge_type = ?"
                params.append(edge_type)
            query += " ORDER BY e.weight DESC"

            with self.db._cursor() as cur:
                cur.execute(query, params)
                results.extend([dict(r) for r in cur.fetchall()])

        if direction in ("incoming", "both"):
            query = """
                SELECT e.*, n.node_type as source_type, n.name as source_name, n.properties as source_props
                FROM intel_kg_edges e
                JOIN intel_kg_nodes n ON e.source_id = n.id
                WHERE e.target_id = ?
            """
            params = [node_id]
            if edge_type:
                query += " AND e.edge_type = ?"
                params.append(edge_type)
            query += " ORDER BY e.weight DESC"

            with self.db._cursor() as cur:
                cur.execute(query, params)
                results.extend([dict(r) for r in cur.fetchall()])

        return results

    def query_path(
        self,
        strategy_name: str,
        regime: str,
        coin: str | None = None,
    ) -> list[dict]:
        """Find what the KG knows about a strategy+regime combination.

        Returns edges connecting strategy -> regime, strategy -> coin, etc.
        """
        results = []

        # Strategy -> Regime edges
        strat_node = self.get_node("strategy", strategy_name)
        regime_node = self.get_node("regime", regime)

        if strat_node and regime_node:
            with self.db._cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM intel_kg_edges
                    WHERE source_id = ? AND target_id = ?
                    """,
                    (strat_node["id"], regime_node["id"]),
                )
                results.extend([dict(r) for r in cur.fetchall()])

        # Strategy -> Coin edges (if coin specified)
        if coin and strat_node:
            coin_node = self.get_node("coin", coin)
            if coin_node:
                with self.db._cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM intel_kg_edges
                        WHERE source_id = ? AND target_id = ?
                        """,
                        (strat_node["id"], coin_node["id"]),
                    )
                    results.extend([dict(r) for r in cur.fetchall()])

        return results

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_base_nodes(self, coins: list[str], strategies: list[str]) -> int:
        """Create base nodes for all coins, strategies, and regimes.

        Returns total number of nodes created/updated.
        """
        count = 0

        for coin in coins:
            self.upsert_node("coin", coin)
            count += 1

        for strat in strategies:
            self.upsert_node("strategy", strat)
            count += 1

        regimes = [
            "trending_up", "trending_down", "ranging",
            "high_volatility", "low_volatility", "breakout",
        ]
        for regime in regimes:
            self.upsert_node("regime", regime)
            count += 1

        indicators = ["rsi", "macd", "bollinger", "ema", "atr", "volume"]
        for ind in indicators:
            self.upsert_node("indicator", ind)
            count += 1

        logger.info("KnowledgeGraph seeded with %d base nodes", count)
        return count

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return graph statistics."""
        with self.db._cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM intel_kg_nodes")
            nodes = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM intel_kg_edges")
            edges = cur.fetchone()["cnt"]

            cur.execute(
                """SELECT node_type, COUNT(*) as cnt
                   FROM intel_kg_nodes GROUP BY node_type"""
            )
            by_type = {r["node_type"]: r["cnt"] for r in cur.fetchall()}

        return {
            "total_nodes": nodes,
            "total_edges": edges,
            "nodes_by_type": by_type,
        }
