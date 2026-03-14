"""
MoneyClaw Models — Foundation enums and dataclasses for the crypto trading agent.

All other modules import from here. This file defines the canonical types for
market regimes, orders, signals, trades, positions, and portfolio state.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MarketRegime(enum.Enum):
    """Detected market regime used to select and weight strategies."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"


class OrderSide(enum.Enum):
    """Direction of an order or position."""
    BUY = "buy"
    SELL = "sell"


class OrderType(enum.Enum):
    """Execution type for an order."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"


class OrderStatus(enum.Enum):
    """Lifecycle status of an order."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TradingState(enum.Enum):
    """Overall state of the trading engine."""
    ACTIVE = "active"
    PAUSED = "paused"
    CIRCUIT_BREAKER = "circuit_breaker"
    SHUTDOWN = "shutdown"


class StrategyName(enum.Enum):
    """Available trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"
    DCA_SMART = "dca_smart"
    MEME_MOMENTUM = "meme_momentum"
    VOLUME_SPIKE = "volume_spike"
    MEME_REVERSAL = "meme_reversal"


class SignalStrength(enum.Enum):
    """Qualitative strength of a trade signal."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    """A single OHLCV candlestick bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    product_id: str
    timeframe: str  # e.g. "1m", "5m", "1h", "1d"


@dataclass
class TradeSignal:
    """A signal emitted by a strategy recommending a trade."""
    strategy: StrategyName
    side: OrderSide
    product_id: str
    strength: SignalStrength
    confidence: float  # 0.0 – 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Trade:
    """A completed or in-progress trade with fill and P&L data."""
    id: str = field(default_factory=lambda: str(uuid4()))
    signal: Optional[TradeSignal] = None
    order_id: str = ""
    fill_price: float = 0.0
    quantity: float = 0.0
    side: Optional[OrderSide] = None
    product_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    close_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0


@dataclass
class Position:
    """A currently open position."""
    product_id: str
    side: OrderSide
    entry_price: float
    quantity: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    strategy: Optional[StrategyName] = None
    opened_at: datetime = field(default_factory=datetime.utcnow)
    unrealized_pnl: float = 0.0
    current_price: float = 0.0


@dataclass
class StrategyPerformance:
    """Rolling performance metrics for a single strategy."""
    strategy: StrategyName
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl_pct: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    weight: float = 0.0  # current allocation weight assigned by the meta-strategy
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PortfolioSummary:
    """Snapshot of the full portfolio state."""
    total_value: float = 0.0
    cash_balance: float = 0.0
    positions_value: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    open_positions: int = 0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    peak_value: float = 0.0


@dataclass
class Indicators:
    """Technical indicators computed from candle data."""
    # RSI
    rsi: float = 0.0

    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0

    # Exponential Moving Averages
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0

    # Average True Range
    atr: float = 0.0

    # Volume
    volume_sma: float = 0.0

    # Support / Resistance
    support: float = 0.0
    resistance: float = 0.0
