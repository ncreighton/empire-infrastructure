"""
Tests for RiskManager — verifies every immutable safety rule is enforced.

These are the most critical tests in the entire system. If any of these
fail, the agent could lose real money. Every rule is non-negotiable.
"""

import pytest

from moneyclaw.engine.risk_manager import RiskManager
from moneyclaw.models import (
    TradeSignal,
    Position,
    TradingState,
    OrderSide,
    StrategyName,
    SignalStrength,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    entry_price: float = 50000.0,
    stop_loss: float = 49000.0,
    take_profit: float = 53000.0,
    **overrides,
) -> TradeSignal:
    """Create a valid TradeSignal with sensible defaults."""
    kwargs = dict(
        strategy=StrategyName.MOMENTUM,
        side=OrderSide.BUY,
        product_id="BTC-USD",
        strength=SignalStrength.STRONG,
        confidence=0.8,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reason="test",
    )
    kwargs.update(overrides)
    return TradeSignal(**kwargs)


def _make_position(product_id: str = "ETH-USD", **overrides) -> Position:
    """Create a Position with reasonable defaults."""
    kwargs = dict(
        product_id=product_id,
        side=OrderSide.BUY,
        entry_price=3000.0,
        quantity=0.01,
        stop_loss=2900.0,
        take_profit=3200.0,
        strategy=StrategyName.MOMENTUM,
    )
    kwargs.update(overrides)
    return Position(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRiskManager:
    """Tests for immutable safety rules."""

    def test_max_risk_per_trade(self):
        """Position size should never exceed 2% risk of portfolio."""
        rm = RiskManager(starting_capital=100.0)
        signal = _make_signal(
            entry_price=50000.0,
            stop_loss=49000.0,  # 2% stop distance
            take_profit=53000.0,
        )
        size = rm.calculate_position_size(signal, portfolio_value=100.0, cash_balance=100.0)
        # Max risk = 100 * 0.02 = $2. Stop distance = 2%. Size = 2 / 0.02 = $100.
        # But also capped by max_position_pct (12%) = $12, and cash reserve (80% available = $80).
        # So position size should be at most $12 (the tightest constraint).
        assert size <= 100.0 * 0.12, f"Position size ${size:.2f} exceeds 12% cap"
        # And the risk on that position must not exceed 2%
        stop_distance_frac = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        risk_amount = size * stop_distance_frac
        assert risk_amount <= 100.0 * 0.02 + 0.01, (
            f"Risk amount ${risk_amount:.4f} exceeds 2% of portfolio (${100.0 * 0.02:.2f})"
        )

    def test_circuit_breaker_daily_loss(self):
        """State becomes CIRCUIT_BREAKER when daily loss exceeds 10%."""
        rm = RiskManager(starting_capital=100.0)
        assert rm.state == TradingState.ACTIVE

        # Daily loss of $10 = exactly 10% of $100 starting capital
        state = rm.check_circuit_breaker(daily_pnl=-10.0, portfolio_value=90.0)
        assert state == TradingState.CIRCUIT_BREAKER

    def test_max_positions(self):
        """Rejects signal when 8 positions already open."""
        rm = RiskManager(starting_capital=100.0)
        signal = _make_signal()
        positions = [_make_position(product_id=f"COIN{i}-USD") for i in range(8)]

        approved, reason = rm.validate_signal(
            signal, portfolio_value=100.0, cash_balance=80.0, open_positions=positions
        )
        assert not approved
        assert "Max positions" in reason

    def test_cash_reserve(self):
        """Always maintains 20% cash reserve."""
        rm = RiskManager(starting_capital=100.0)
        # With $20 cash on a $100 portfolio, reserve = $20, available = $0
        available = rm.available_cash_for_trading(cash_balance=20.0, portfolio_value=100.0)
        assert available == 0.0

        # With $25 cash, only $5 is available for trading
        available = rm.available_cash_for_trading(cash_balance=25.0, portfolio_value=100.0)
        assert available == pytest.approx(5.0)

        # Signal should be rejected when no cash available after reserve
        signal = _make_signal()
        approved, reason = rm.validate_signal(
            signal, portfolio_value=100.0, cash_balance=20.0, open_positions=[]
        )
        assert not approved
        assert "reserve" in reason.lower() or "cash" in reason.lower()

    def test_stop_loss_required(self):
        """Rejects signals without stop_loss."""
        rm = RiskManager(starting_capital=100.0)
        signal = _make_signal(stop_loss=0.0)

        approved, reason = rm.validate_signal(
            signal, portfolio_value=100.0, cash_balance=100.0, open_positions=[]
        )
        assert not approved
        assert "stop" in reason.lower()

    def test_min_reward_risk_ratio(self):
        """Rejects signals with R:R < 1.5."""
        rm = RiskManager(starting_capital=100.0)
        # Entry=50000, SL=49000 (risk=1000), TP=50500 (reward=500) => R:R = 0.5
        signal = _make_signal(
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=50500.0,
        )

        approved, reason = rm.validate_signal(
            signal, portfolio_value=100.0, cash_balance=100.0, open_positions=[]
        )
        assert not approved
        assert "R:R" in reason or "ratio" in reason.lower()

    def test_drawdown_protection(self):
        """Circuit breaker at 25% drawdown from peak."""
        rm = RiskManager(starting_capital=100.0)
        rm.peak_value = 100.0

        # Portfolio drops to $75 = exactly 25% drawdown
        state = rm.check_circuit_breaker(daily_pnl=0.0, portfolio_value=75.0)
        assert state == TradingState.CIRCUIT_BREAKER

        # After circuit breaker, signals should be rejected
        signal = _make_signal()
        approved, reason = rm.validate_signal(
            signal, portfolio_value=75.0, cash_balance=75.0, open_positions=[]
        )
        assert not approved
        assert "paused" in reason.lower() or "circuit" in reason.lower()

    def test_reset_daily(self):
        """Daily reset clears circuit breaker and resets P&L."""
        rm = RiskManager(starting_capital=100.0)

        # Trip the circuit breaker via daily loss
        rm.check_circuit_breaker(daily_pnl=-15.0, portfolio_value=85.0)
        assert rm.state == TradingState.CIRCUIT_BREAKER

        # Daily reset should clear it
        rm.reset_daily()
        assert rm.state == TradingState.ACTIVE
        assert rm.daily_pnl == 0.0

    def test_pause_resume(self):
        """Pause/resume state transitions work correctly."""
        rm = RiskManager(starting_capital=100.0)
        assert rm.state == TradingState.ACTIVE

        rm.pause()
        assert rm.state == TradingState.PAUSED

        # Signals should be rejected while paused
        signal = _make_signal()
        approved, reason = rm.validate_signal(
            signal, portfolio_value=100.0, cash_balance=100.0, open_positions=[]
        )
        assert not approved

        rm.resume()
        assert rm.state == TradingState.ACTIVE

        # Signals should now be accepted (assuming they pass other rules)
        approved, reason = rm.validate_signal(
            signal, portfolio_value=100.0, cash_balance=100.0, open_positions=[]
        )
        assert approved
