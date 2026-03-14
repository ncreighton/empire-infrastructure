"""Configuration loading for MoneyClaw crypto trading agent.

Loads settings from environment variables and JSON config files,
with sensible defaults for paper trading out of the box.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk upward from this file until we find the ``configs/`` directory."""
    current = Path(__file__).resolve().parent
    # moneyclaw/ sits one level below the project root
    candidate = current.parent
    if (candidate / "configs").is_dir():
        return candidate
    # Fallback: two levels up (in case of deeper nesting)
    candidate = candidate.parent
    if (candidate / "configs").is_dir():
        return candidate
    # Last resort: use the immediate parent of moneyclaw/
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = _find_project_root()


# ---------------------------------------------------------------------------
# Coinbase credentials
# ---------------------------------------------------------------------------

@dataclass
class CoinbaseConfig:
    """Coinbase Advanced Trade API credentials."""

    api_key: str = ""
    api_secret: str = ""
    paper_trade: bool = True


# ---------------------------------------------------------------------------
# Trading parameters
# ---------------------------------------------------------------------------

@dataclass
class TradingConfig:
    """Core trading and risk-management parameters."""

    starting_capital: float = 1000.0

    coins: list[str] = field(default_factory=lambda: [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "AVAX-USD",
        "LINK-USD",
        "DOT-USD",
        "POL-USD",
        "ADA-USD",
        "NEAR-USD",
        "ATOM-USD",
        "DOGE-USD",
        "SHIB-USD",
        "PEPE-USD",
        "FLOKI-USD",
        "BONK-USD",
        "WIF-USD",
        "DEGEN-USD",
        "TURBO-USD",
        "MOG-USD",
        "POPCAT-USD",
    ])

    tick_interval_seconds: int = 60

    candle_timeframes: list[str] = field(default_factory=lambda: [
        "ONE_MINUTE",
        "FIVE_MINUTE",
        "FIFTEEN_MINUTE",
        "ONE_HOUR",
        "FOUR_HOUR",
    ])

    # Risk management
    max_risk_per_trade_pct: float = 0.02
    max_position_pct: float = 0.12
    max_positions: int = 8
    daily_loss_limit_pct: float = 0.10
    max_drawdown_pct: float = 0.25
    min_reserve_pct: float = 0.20
    min_reward_risk_ratio: float = 1.5


# ---------------------------------------------------------------------------
# Strategy ensemble
# ---------------------------------------------------------------------------

_DEFAULT_STRATEGY_WEIGHTS: dict[str, float] = {
    "MOMENTUM": 0.15,
    "MEAN_REVERSION": 0.12,
    "BREAKOUT": 0.13,
    "VOLATILITY": 0.10,
    "DCA_SMART": 0.10,
    "MEME_MOMENTUM": 0.15,
    "VOLUME_SPIKE": 0.13,
    "MEME_REVERSAL": 0.12,
}

_DEFAULT_STRATEGY_PARAMS: dict[str, dict[str, Any]] = {
    "MOMENTUM": {
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "ema_short": 9,
        "ema_long": 21,
    },
    "MEAN_REVERSION": {
        "bb_period": 20,
        "bb_std_dev": 2.0,
        "z_score_entry": 2.0,
        "z_score_exit": 0.5,
        "lookback_period": 50,
    },
    "BREAKOUT": {
        "lookback_period": 20,
        "volume_multiplier": 1.5,
        "atr_period": 14,
        "atr_multiplier": 1.5,
        "confirmation_candles": 2,
    },
    "VOLATILITY": {
        "atr_period": 14,
        "vol_lookback": 30,
        "high_vol_threshold": 1.5,
        "low_vol_threshold": 0.5,
        "squeeze_lookback": 20,
    },
    "DCA_SMART": {
        "base_interval_hours": 24,
        "dip_threshold_pct": 0.05,
        "max_acceleration": 3.0,
        "rsi_weight": 0.4,
        "fear_greed_weight": 0.3,
        "volatility_weight": 0.3,
    },
    "MEME_MOMENTUM": {
        "rsi_entry_threshold": 55,
        "rsi_exit_threshold": 85,
        "volume_spike_multiplier": 3.0,
        "take_profit_pct": 0.03,
        "stop_loss_pct": 0.015,
        "min_confidence": 0.55,
        "volume_exit_ratio": 0.7,
    },
    "VOLUME_SPIKE": {
        "volume_spike_threshold": 5.0,
        "volume_spike_moderate": 3.0,
        "take_profit_pct": 0.02,
        "stop_loss_pct": 0.008,
        "rsi_min": 25,
        "rsi_max": 80,
        "min_confidence": 0.6,
        "meme_coin_bonus": 0.1,
    },
    "MEME_REVERSAL": {
        "rsi_oversold_threshold": 25,
        "rsi_exit_threshold": 50,
        "bb_entry_below": True,
        "take_profit_pct": 0.05,
        "stop_loss_pct": 0.02,
        "min_confidence": 0.55,
        "volume_decline_periods": 3,
    },
}


@dataclass
class StrategyConfig:
    """Strategy ensemble weights and per-strategy parameters."""

    weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_STRATEGY_WEIGHTS),
    )
    params: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            k: dict(v) for k, v in _DEFAULT_STRATEGY_PARAMS.items()
        },
    )


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

class Config:
    """Aggregated configuration loaded from env vars and JSON files.

    Usage::

        cfg = Config.load()
        print(cfg.coinbase.paper_trade)   # True
        print(cfg.trading.coins)          # ['BTC-USD', ...]
        print(cfg.db_path)               # 'data/moneyclaw.db'
    """

    def __init__(
        self,
        coinbase: CoinbaseConfig,
        trading: TradingConfig,
        strategy: StrategyConfig,
        *,
        base_dir: Path | None = None,
    ) -> None:
        self.coinbase = coinbase
        self.trading = trading
        self.strategy = strategy
        self._base_dir = base_dir or PROJECT_ROOT

    # -- derived paths / ports ---------------------------------------------

    @property
    def db_path(self) -> str:
        return str(self._base_dir / "data" / "moneyclaw.db")

    @property
    def api_port(self) -> int:
        raw = os.environ.get("API_PORT", "8110")
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 8110

    # -- factory -----------------------------------------------------------

    @classmethod
    def load(cls, base_dir: Path | None = None) -> "Config":
        """Create a ``Config`` from environment variables and JSON files.

        Parameters
        ----------
        base_dir:
            Override the auto-detected project root.  Useful for tests.
        """
        root = base_dir or PROJECT_ROOT

        coinbase = cls._load_coinbase()
        trading = cls._load_trading(root)
        strategy = cls._load_strategy(root)

        return cls(
            coinbase=coinbase,
            trading=trading,
            strategy=strategy,
            base_dir=root,
        )

    # -- private loaders ---------------------------------------------------

    @staticmethod
    def _load_coinbase() -> CoinbaseConfig:
        """Read Coinbase credentials from environment variables."""
        paper_raw = os.environ.get("PAPER_TRADE", "true").strip().lower()
        paper_trade = paper_raw not in ("0", "false", "no", "off")

        # The API secret may be an EC private key PEM with literal \n
        # escape sequences (common in .env files and Docker env vars).
        api_secret = os.environ.get("COINBASE_API_SECRET", "")
        if "\\n" in api_secret:
            api_secret = api_secret.replace("\\n", "\n")

        return CoinbaseConfig(
            api_key=os.environ.get("COINBASE_API_KEY", ""),
            api_secret=api_secret,
            paper_trade=paper_trade,
        )

    @staticmethod
    def _load_trading(root: Path) -> TradingConfig:
        """Load trading parameters from ``configs/default.json`` if it exists."""
        config_path = root / "configs" / "default.json"
        if not config_path.is_file():
            return TradingConfig()

        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return TradingConfig()

        kwargs: dict[str, Any] = {}
        # Map JSON keys to dataclass fields, accepting only known fields.
        field_names = {f.name for f in TradingConfig.__dataclass_fields__.values()}
        for key, value in raw.items():
            if key in field_names:
                kwargs[key] = value

        return TradingConfig(**kwargs)

    @staticmethod
    def _load_strategy(root: Path) -> StrategyConfig:
        """Load strategy config from ``configs/strategies.json`` if it exists."""
        config_path = root / "configs" / "strategies.json"
        if not config_path.is_file():
            return StrategyConfig()

        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return StrategyConfig()

        weights = raw.get("weights", dict(_DEFAULT_STRATEGY_WEIGHTS))
        params = raw.get("params", {
            k: dict(v) for k, v in _DEFAULT_STRATEGY_PARAMS.items()
        })

        return StrategyConfig(weights=weights, params=params)

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        paper = self.coinbase.paper_trade
        coins = len(self.trading.coins)
        strategies = len(self.strategy.weights)
        return (
            f"Config(paper_trade={paper}, coins={coins}, "
            f"strategies={strategies}, port={self.api_port})"
        )
