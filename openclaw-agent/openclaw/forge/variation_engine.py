"""VariationEngine -- prevents repetitive profile content across platforms.

Anti-repetition engine for template selection. Tracks which items from each
pool have been used recently and avoids re-selecting them until the pool
is exhausted. Lightweight, in-memory tracking (no persistence needed --
resets each session to keep profiles fresh across signup batches).

Part of the OpenClaw FORGE intelligence layer.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


class VariationEngine:
    """Pool selection with anti-repetition tracking.

    Maintains per-category sets of used indices. When all items in a pool
    have been selected, the tracking resets for that category so the cycle
    can start over with a fresh shuffle.

    Usage::

        ve = VariationEngine()
        tagline = ve.pick(tagline_pool, category="taglines")
        keywords = ve.pick_n(keyword_pool, 5, category="seo_keywords")
    """

    def __init__(self) -> None:
        self._used: dict[str, set[int]] = defaultdict(set)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def pick(self, pool: list[Any], category: str = "default") -> Any:
        """Pick one item from *pool* without repeating recent selections.

        If the pool is empty, returns ``None``. When all items have been
        used in the given category, resets tracking and picks from the
        full pool again.

        Args:
            pool: A list of items to choose from.
            category: Logical grouping for tracking (e.g., "taglines").

        Returns:
            A single item from the pool, or ``None`` if pool is empty.
        """
        if not pool:
            return None

        available = self._get_available_indices(pool, category)
        if not available:
            # All items exhausted -- reset and start fresh
            self._used[category].clear()
            available = list(range(len(pool)))

        idx = random.choice(available)
        self._used[category].add(idx)
        return pool[idx]

    def pick_n(self, pool: list[Any], n: int, category: str = "default") -> list[Any]:
        """Pick *n* unique items from *pool* without repeating recent selections.

        If *n* exceeds the pool size, returns as many unique items as
        possible (up to ``len(pool)``).

        Args:
            pool: A list of items to choose from.
            n: Number of unique items to select.
            category: Logical grouping for tracking.

        Returns:
            A list of up to *n* unique items from the pool.
        """
        if not pool:
            return []

        n = min(n, len(pool))
        results: list[Any] = []
        selected_indices: set[int] = set()

        for _ in range(n):
            available = [
                i for i in self._get_available_indices(pool, category)
                if i not in selected_indices
            ]
            if not available:
                # Reset category tracking, exclude already-selected this call
                self._used[category].clear()
                available = [
                    i for i in range(len(pool))
                    if i not in selected_indices
                ]
                if not available:
                    break

            idx = random.choice(available)
            self._used[category].add(idx)
            selected_indices.add(idx)
            results.append(pool[idx])

        return results

    def reset(self, category: str | None = None) -> None:
        """Reset tracking for a specific category, or all categories.

        Args:
            category: The category to reset, or ``None`` to reset all.
        """
        if category is None:
            self._used.clear()
        else:
            self._used.pop(category, None)

    def shuffle_pool(self, pool: list[Any]) -> list[Any]:
        """Return a shuffled copy of *pool* without modifying the original.

        Args:
            pool: The list to shuffle.

        Returns:
            A new list with the same elements in randomized order.
        """
        copy = list(pool)
        random.shuffle(copy)
        return copy

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_available_indices(self, pool: list[Any], category: str) -> list[int]:
        """Return indices in *pool* that have NOT been used in *category*.

        Args:
            pool: The full pool of items.
            category: The tracking category.

        Returns:
            A list of integer indices that are still available.
        """
        used = self._used[category]
        return [i for i in range(len(pool)) if i not in used]
