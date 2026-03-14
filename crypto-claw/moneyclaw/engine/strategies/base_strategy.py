"""
BaseStrategy — Abstract base class for all MoneyClaw trading strategies.

Every strategy inherits from this class and implements:
  - default_params()   — sensible defaults that evolution can tune
  - evaluate()         — read indicators and produce a TradeSignal (or None)
  - should_exit()      — decide whether to close an existing position early
  - _regime_weights    — per-regime confidence multiplier map
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from moneyclaw.models import (
    Candle,
    Indicators,
    MarketRegime,
    StrategyName,
    TradeSignal,
)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    name: StrategyName

    def __init__(self, params: dict | None = None):
        self.params = params or self.default_params()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def default_params(self) -> dict:
        """Return default strategy parameters."""
        ...

    @abstractmethod
    def evaluate(
        self,
        product_id: str,
        candles: list[Candle],
        indicators: Indicators,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        """Evaluate market conditions and return a trade signal or None."""
        ...

    @abstractmethod
    def should_exit(
        self,
        product_id: str,
        entry_price: float,
        current_price: float,
        indicators: Indicators,
        regime: MarketRegime,
    ) -> bool:
        """Check if an existing position should be exited early (beyond stop/TP)."""
        ...

    @property
    @abstractmethod
    def _regime_weights(self) -> dict[MarketRegime, float]:
        """Map of regime to weight multiplier (0.0-2.0)."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def regime_weight(self, regime: MarketRegime) -> float:
        """Return weight multiplier for this strategy in the given regime.

        Falls back to 0.5 for regimes not explicitly listed.
        """
        return self._regime_weights.get(regime, 0.5)

    def update_params(self, new_params: dict) -> None:
        """Update strategy parameters (used by the evolution engine)."""
        self.params.update(new_params)
