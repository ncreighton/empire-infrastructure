"""Full smoke test — engine init + API import."""
import os
os.environ["PAPER_TRADE"] = "true"

from moneyclaw.config import Config
from moneyclaw.engine.trading_engine import TradingEngine

config = Config.load()
engine = TradingEngine(config)

print(f"Paper: {config.coinbase.paper_trade}")
print(f"Capital: ${config.trading.starting_capital}")
print(f"Coins: {len(config.trading.coins)}")
print(f"State: {engine.risk_manager.state.value}")
print(f"Cash: ${engine.portfolio_manager.cash_balance}")
print(f"Strategies: {[s.value for s in engine.strategies.keys()]}")
print(f"Weights: {engine.strategy_weights}")

status = engine.get_status()
print(f"Regime: {status['regime']}")
print(f"Tick: {status['tick_count']}")

from api.app import app
print(f"API: {app.title}")
route_paths = sorted([r.path for r in app.routes if hasattr(r, 'path')])
print(f"Endpoints: {route_paths}")

print()
print("=== FULL SMOKE TEST PASSED ===")
