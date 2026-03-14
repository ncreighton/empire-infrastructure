"""
Tests for PortfolioManager — position lifecycle, P&L, and summary accuracy.

Uses a temporary SQLite database for each test to ensure isolation.
"""

import os
import tempfile

import pytest

from moneyclaw.engine.portfolio_manager import PortfolioManager
from moneyclaw.persistence.database import Database
from moneyclaw.models import OrderSide, StrategyName


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db():
    """Create a temporary database for a single test."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_portfolio.db")
    db = Database(db_path)
    yield db
    # Cleanup handled by OS / pytest tmpdir management


@pytest.fixture
def portfolio(temp_db):
    """Create a PortfolioManager with $100 starting capital."""
    return PortfolioManager(db=temp_db, starting_capital=100.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPortfolioManager:

    def test_open_position(self, portfolio):
        """Opens position, cash decreases, position tracked."""
        initial_cash = portfolio.cash_balance

        pos = portfolio.open_position(
            product_id="BTC-USD",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.001,
            stop_loss=49000.0,
            take_profit=53000.0,
            strategy=StrategyName.MOMENTUM,
        )

        assert pos.product_id == "BTC-USD"
        assert pos.entry_price == 50000.0
        assert pos.quantity == 0.001

        # Cash should decrease by entry_price * quantity = $50
        expected_cash = initial_cash - (50000.0 * 0.001)
        assert portfolio.cash_balance == pytest.approx(expected_cash)

        # Position should be tracked
        assert portfolio.has_position("BTC-USD")
        assert portfolio.get_open_position_count() == 1

    def test_close_position_profit(self, portfolio):
        """Close at higher price => positive P&L, cash increases."""
        portfolio.open_position(
            product_id="BTC-USD",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.001,
            stop_loss=49000.0,
            take_profit=53000.0,
            strategy=StrategyName.MOMENTUM,
        )

        cash_after_open = portfolio.cash_balance

        trade = portfolio.close_position("BTC-USD", close_price=53000.0)

        assert trade is not None
        assert trade.pnl > 0, f"Expected positive P&L, got {trade.pnl}"
        # P&L = (53000 - 50000) * 0.001 = $3
        assert trade.pnl == pytest.approx(3.0)
        # Cash should increase by close_price * quantity = $53
        assert portfolio.cash_balance == pytest.approx(cash_after_open + 53000.0 * 0.001)
        assert not portfolio.has_position("BTC-USD")

    def test_close_position_loss(self, portfolio):
        """Close at lower price => negative P&L."""
        portfolio.open_position(
            product_id="ETH-USD",
            side=OrderSide.BUY,
            entry_price=3000.0,
            quantity=0.01,
            stop_loss=2800.0,
            take_profit=3500.0,
            strategy=StrategyName.MEAN_REVERSION,
        )

        trade = portfolio.close_position("ETH-USD", close_price=2800.0)

        assert trade is not None
        assert trade.pnl < 0, f"Expected negative P&L, got {trade.pnl}"
        # P&L = (2800 - 3000) * 0.01 = -$2
        assert trade.pnl == pytest.approx(-2.0)

    def test_portfolio_summary(self, portfolio):
        """total_value = cash + positions."""
        portfolio.open_position(
            product_id="BTC-USD",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.001,
            stop_loss=49000.0,
            take_profit=53000.0,
            strategy=StrategyName.MOMENTUM,
        )

        # Update the position price
        portfolio.update_prices({"BTC-USD": 51000.0})

        summary = portfolio.get_portfolio_summary()

        # Cash = 100 - 50 = 50. Position value = 51000 * 0.001 = 51.
        assert summary.cash_balance == pytest.approx(50.0)
        assert summary.positions_value == pytest.approx(51.0)
        assert summary.total_value == pytest.approx(101.0)
        assert summary.open_positions == 1

    def test_multiple_positions(self, portfolio):
        """Open multiple positions, summary reflects all of them."""
        portfolio.open_position(
            product_id="BTC-USD",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.0005,
            stop_loss=49000.0,
            take_profit=53000.0,
            strategy=StrategyName.MOMENTUM,
        )
        portfolio.open_position(
            product_id="ETH-USD",
            side=OrderSide.BUY,
            entry_price=3000.0,
            quantity=0.005,
            stop_loss=2800.0,
            take_profit=3500.0,
            strategy=StrategyName.MEAN_REVERSION,
        )

        assert portfolio.get_open_position_count() == 2
        assert portfolio.has_position("BTC-USD")
        assert portfolio.has_position("ETH-USD")

        summary = portfolio.get_portfolio_summary()
        # Cash = 100 - (50000*0.0005) - (3000*0.005) = 100 - 25 - 15 = 60
        assert summary.cash_balance == pytest.approx(60.0)
        assert summary.open_positions == 2

    def test_position_not_found(self, portfolio):
        """Closing non-existent position returns None."""
        result = portfolio.close_position("NONEXISTENT-USD", close_price=100.0)
        assert result is None

    def test_has_position(self, portfolio):
        """has_position returns correct bool."""
        assert not portfolio.has_position("BTC-USD")

        portfolio.open_position(
            product_id="BTC-USD",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.001,
            stop_loss=49000.0,
            take_profit=53000.0,
            strategy=StrategyName.MOMENTUM,
        )

        assert portfolio.has_position("BTC-USD")
        assert not portfolio.has_position("ETH-USD")

        portfolio.close_position("BTC-USD", close_price=51000.0)
        assert not portfolio.has_position("BTC-USD")
