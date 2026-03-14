"""
PatternMiner — Statistical pattern discovery from trade history.

Discovers:
  - Strategy-regime patterns: "mean_reversion in breakout = 78% win rate"
  - Time patterns: "momentum works better in first 4 hours of trading"
  - Correlation patterns: "SOL tends to follow ETH moves with 15-min lag"
  - Sequence patterns: "3 consecutive losses often precede a regime change"

Uses scipy-free statistics (z-test approximation, binomial test).
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from typing import Any

from moneyclaw.persistence.database import Database

logger = logging.getLogger(__name__)

# Minimum samples for statistical significance
MIN_SAMPLES = 10
# Default null hypothesis win rate (random trading)
NULL_WIN_RATE = 0.5
# Significance threshold
P_VALUE_THRESHOLD = 0.05


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


def _binomial_z_test(successes: int, trials: int, null_prob: float = 0.5) -> float:
    """Approximate p-value for a two-sided binomial z-test.

    Uses normal approximation — valid for trials >= 10.
    Returns p-value (lower = more significant).
    """
    if trials < 5:
        return 1.0

    observed_rate = successes / trials
    se = math.sqrt(null_prob * (1 - null_prob) / trials)
    if se == 0:
        return 1.0

    z = abs(observed_rate - null_prob) / se

    # Approximate two-sided p-value from z-score using error function
    # P(Z > z) ≈ erfc(z / sqrt(2)) / 2
    p_two_sided = math.erfc(z / math.sqrt(2))
    return p_two_sided


class PatternMiner:
    """Mine statistical patterns from trade history.

    Parameters
    ----------
    db : Database
        Shared database instance.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Full mining pass
    # ------------------------------------------------------------------

    def mine_all(self, limit: int = 500) -> list[dict]:
        """Run all pattern mining algorithms. Returns list of discovered patterns."""
        trades = self._get_trades(limit)
        if len(trades) < MIN_SAMPLES:
            logger.debug("Not enough trades for mining (%d < %d)", len(trades), MIN_SAMPLES)
            return []

        patterns: list[dict] = []
        patterns.extend(self._mine_strategy_regime(trades))
        patterns.extend(self._mine_strategy_coin(trades))
        patterns.extend(self._mine_time_patterns(trades))
        patterns.extend(self._mine_sequence_patterns(trades))
        patterns.extend(self._mine_correlation_patterns())

        logger.info("PatternMiner found %d patterns from %d trades", len(patterns), len(trades))
        return patterns

    # ------------------------------------------------------------------
    # Strategy-Regime patterns
    # ------------------------------------------------------------------

    def _mine_strategy_regime(self, trades: list[dict]) -> list[dict]:
        """Find strategy+regime combinations with unusual win rates."""
        buckets: dict[str, list[bool]] = defaultdict(list)

        for t in trades:
            strategy = t.get("strategy", "")
            regime = self._get_regime_at_time(t.get("opened_at"))
            if not strategy or not regime:
                continue

            pnl = float(t.get("pnl", 0) or 0)
            key = f"{strategy}|{regime}"
            buckets[key].append(pnl > 0)

        return self._analyze_buckets(buckets, "strategy_regime")

    # ------------------------------------------------------------------
    # Strategy-Coin patterns
    # ------------------------------------------------------------------

    def _mine_strategy_coin(self, trades: list[dict]) -> list[dict]:
        """Find strategy+coin combinations with unusual win rates."""
        buckets: dict[str, list[bool]] = defaultdict(list)

        for t in trades:
            strategy = t.get("strategy", "")
            coin = t.get("product_id", "")
            if not strategy or not coin:
                continue

            pnl = float(t.get("pnl", 0) or 0)
            key = f"{strategy}|{coin}"
            buckets[key].append(pnl > 0)

        return self._analyze_buckets(buckets, "strategy_coin")

    # ------------------------------------------------------------------
    # Time patterns
    # ------------------------------------------------------------------

    def _mine_time_patterns(self, trades: list[dict]) -> list[dict]:
        """Find time-of-day patterns in trade outcomes."""
        buckets: dict[str, list[bool]] = defaultdict(list)

        for t in trades:
            opened_at = t.get("opened_at", "")
            if not opened_at:
                continue

            try:
                dt = datetime.fromisoformat(opened_at)
                hour_bucket = f"hour_{dt.hour // 4 * 4}-{(dt.hour // 4 + 1) * 4}"
                day_bucket = f"day_{dt.strftime('%A').lower()}"
            except (ValueError, TypeError):
                continue

            pnl = float(t.get("pnl", 0) or 0)
            won = pnl > 0
            buckets[hour_bucket].append(won)
            buckets[day_bucket].append(won)

        return self._analyze_buckets(buckets, "time")

    # ------------------------------------------------------------------
    # Sequence patterns
    # ------------------------------------------------------------------

    def _mine_sequence_patterns(self, trades: list[dict]) -> list[dict]:
        """Find streak patterns (e.g., 3 losses often followed by a win)."""
        patterns: list[dict] = []
        if len(trades) < 5:
            return patterns

        # Sort by time
        sorted_trades = sorted(trades, key=lambda t: t.get("opened_at", ""))

        # Track consecutive outcomes
        outcomes = [float(t.get("pnl", 0) or 0) > 0 for t in sorted_trades]

        for streak_len in (2, 3, 4):
            if len(outcomes) < streak_len + 1:
                continue

            # Count what follows N consecutive losses/wins
            for streak_type in (True, False):
                follow_wins = 0
                follow_total = 0

                for i in range(len(outcomes) - streak_len):
                    window = outcomes[i:i + streak_len]
                    if all(w == streak_type for w in window):
                        follow_total += 1
                        if outcomes[i + streak_len]:
                            follow_wins += 1

                if follow_total >= MIN_SAMPLES:
                    win_rate = follow_wins / follow_total
                    p_val = _binomial_z_test(follow_wins, follow_total)
                    is_sig = p_val < P_VALUE_THRESHOLD

                    streak_desc = "wins" if streak_type else "losses"
                    description = (
                        f"After {streak_len} consecutive {streak_desc}: "
                        f"{win_rate:.0%} next trade wins ({follow_total} samples)"
                    )

                    pattern = self._save_pattern(
                        pattern_type="sequence",
                        description=description,
                        key_fields={"streak_length": streak_len, "streak_type": streak_desc},
                        metric_value=win_rate,
                        sample_count=follow_total,
                        p_value=p_val,
                        is_significant=is_sig,
                    )
                    if pattern:
                        patterns.append(pattern)

        return patterns

    # ------------------------------------------------------------------
    # Correlation patterns
    # ------------------------------------------------------------------

    def _mine_correlation_patterns(self) -> list[dict]:
        """Find price correlation patterns between coin pairs.

        Uses Pearson correlation on recent close prices from candles.
        """
        patterns: list[dict] = []

        with self.db._cursor() as cur:
            cur.execute(
                "SELECT DISTINCT product_id FROM candles WHERE timeframe = 'FIVE_MINUTE'"
            )
            coins = [r["product_id"] for r in cur.fetchall()]

        if len(coins) < 2:
            return patterns

        # Gather recent close prices per coin (last 100 candles)
        coin_closes: dict[str, list[float]] = {}
        for coin in coins[:10]:  # limit to 10 coins for speed
            with self.db._cursor() as cur:
                cur.execute(
                    """
                    SELECT close FROM candles
                    WHERE product_id = ? AND timeframe = 'FIVE_MINUTE'
                    ORDER BY timestamp DESC LIMIT 100
                    """,
                    (coin,),
                )
                closes = [float(r["close"]) for r in cur.fetchall() if r["close"]]

            if len(closes) >= 50:
                coin_closes[coin] = list(reversed(closes))  # chronological order

        # Compute pairwise Pearson correlation
        coin_list = list(coin_closes.keys())
        for i in range(len(coin_list)):
            for j in range(i + 1, len(coin_list)):
                a_name, b_name = coin_list[i], coin_list[j]
                a_prices = coin_closes[a_name]
                b_prices = coin_closes[b_name]

                # Use returns (pct change) for correlation, not raw prices
                min_len = min(len(a_prices), len(b_prices))
                if min_len < 30:
                    continue

                a_returns = [(a_prices[k] - a_prices[k - 1]) / a_prices[k - 1]
                             for k in range(1, min_len) if a_prices[k - 1] > 0]
                b_returns = [(b_prices[k] - b_prices[k - 1]) / b_prices[k - 1]
                             for k in range(1, min_len) if b_prices[k - 1] > 0]

                n = min(len(a_returns), len(b_returns))
                if n < 20:
                    continue

                # Pearson correlation
                corr = self._pearson(a_returns[:n], b_returns[:n])
                if corr is None:
                    continue

                abs_corr = abs(corr)
                if abs_corr >= 0.7:  # only report strong correlations
                    direction = "positive" if corr > 0 else "negative"
                    description = (
                        f"correlation: {a_name} and {b_name} have "
                        f"{direction} correlation {corr:.2f} ({n} samples)"
                    )
                    pattern = self._save_pattern(
                        pattern_type="correlation",
                        description=description,
                        key_fields={"coin_a": a_name, "coin_b": b_name},
                        metric_value=corr,
                        sample_count=n,
                        p_value=None,
                        is_significant=abs_corr >= 0.8,
                    )
                    if pattern:
                        patterns.append(pattern)

        return patterns

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float | None:
        """Compute Pearson correlation coefficient between two lists."""
        n = len(x)
        if n < 5:
            return None

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        denom = math.sqrt(var_x * var_y)
        if denom == 0:
            return None
        return cov / denom

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_buckets(
        self,
        buckets: dict[str, list[bool]],
        pattern_type: str,
    ) -> list[dict]:
        """Analyze win-rate buckets and create patterns for significant ones."""
        patterns: list[dict] = []

        for key, outcomes in buckets.items():
            n = len(outcomes)
            if n < MIN_SAMPLES:
                continue

            wins = sum(outcomes)
            win_rate = wins / n
            p_val = _binomial_z_test(wins, n)
            is_sig = p_val < P_VALUE_THRESHOLD

            parts = key.split("|")
            key_fields: dict[str, str] = {}
            if pattern_type == "strategy_regime":
                key_fields = {"strategy": parts[0], "regime": parts[1] if len(parts) > 1 else ""}
            elif pattern_type == "strategy_coin":
                key_fields = {"strategy": parts[0], "coin": parts[1] if len(parts) > 1 else ""}
            elif pattern_type == "time":
                key_fields = {"time_bucket": parts[0]}

            description = f"{pattern_type}: {key} -> {win_rate:.0%} win rate ({n} samples, p={p_val:.3f})"

            pattern = self._save_pattern(
                pattern_type=pattern_type,
                description=description,
                key_fields=key_fields,
                metric_value=win_rate,
                sample_count=n,
                p_value=p_val,
                is_significant=is_sig,
            )
            if pattern:
                patterns.append(pattern)

        return patterns

    def _save_pattern(
        self,
        pattern_type: str,
        description: str,
        key_fields: dict,
        metric_value: float,
        sample_count: int,
        p_value: float | None,
        is_significant: bool,
    ) -> dict | None:
        """Save or update a pattern in the database."""
        now = _utcnow_iso()
        key_json = json.dumps(key_fields, sort_keys=True)

        with self.db._cursor() as cur:
            # Check if pattern exists
            cur.execute(
                """
                SELECT id FROM intel_patterns
                WHERE pattern_type = ? AND key_fields = ?
                """,
                (pattern_type, key_json),
            )
            existing = cur.fetchone()

            if existing:
                cur.execute(
                    """
                    UPDATE intel_patterns
                    SET description = ?, metric_value = ?, sample_count = ?,
                        p_value = ?, is_significant = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (description, metric_value, sample_count,
                     p_value, 1 if is_significant else 0, now,
                     existing["id"]),
                )
                pattern_id = existing["id"]
            else:
                cur.execute(
                    """
                    INSERT INTO intel_patterns
                        (pattern_type, description, key_fields, metric_value,
                         sample_count, p_value, is_significant, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (pattern_type, description, key_json, metric_value,
                     sample_count, p_value, 1 if is_significant else 0, now, now),
                )
                pattern_id = cur.lastrowid

        return {
            "id": pattern_id,
            "pattern_type": pattern_type,
            "description": description,
            "key_fields": key_fields,
            "metric_value": metric_value,
            "sample_count": sample_count,
            "p_value": p_value,
            "is_significant": is_significant,
        }

    def _get_trades(self, limit: int = 500) -> list[dict]:
        """Fetch recent closed trades from DB."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM trades
                WHERE closed_at IS NOT NULL
                ORDER BY closed_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def _get_regime_at_time(self, timestamp: str | None) -> str | None:
        """Find the market regime closest to a given timestamp."""
        if not timestamp:
            return None

        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT regime FROM market_regimes
                WHERE timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (timestamp,),
            )
            row = cur.fetchone()
            return row["regime"] if row else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_significant_patterns(self) -> list[dict]:
        """Return all statistically significant patterns."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM intel_patterns
                WHERE is_significant = 1
                ORDER BY sample_count DESC
                """
            )
            return [dict(r) for r in cur.fetchall()]

    def get_stats(self) -> dict:
        """Return pattern mining statistics."""
        with self.db._cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM intel_patterns")
            total = cur.fetchone()["total"]

            cur.execute(
                "SELECT COUNT(*) as sig FROM intel_patterns WHERE is_significant = 1"
            )
            sig = cur.fetchone()["sig"]

            cur.execute(
                """SELECT pattern_type, COUNT(*) as cnt
                   FROM intel_patterns GROUP BY pattern_type"""
            )
            by_type = {r["pattern_type"]: r["cnt"] for r in cur.fetchall()}

        return {
            "total_patterns": total,
            "significant_patterns": sig,
            "by_type": by_type,
        }
