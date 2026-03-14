"""Test API endpoints using TestClient (no real server needed)."""
import os
os.environ["PAPER_TRADE"] = "true"

from fastapi.testclient import TestClient
from api.app import app
from moneyclaw.config import Config
from moneyclaw.engine.trading_engine import TradingEngine

# Manually init engine (the async on_event handler doesn't reliably
# fire under TestClient)
_config = Config.load()
_engine = TradingEngine(_config)
app.state.engine = _engine
app.state.config = _config

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["paper_trade"] is True
    print(f"  /health: {data}")


def test_balance():
    r = client.get("/balance")
    assert r.status_code == 200
    data = r.json()
    assert "total_value" in data or "cash_balance" in data or "detail" in data
    print(f"  /balance: {data}")


def test_trades():
    r = client.get("/trades")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    print(f"  /trades: {len(data)} trades")


def test_positions():
    r = client.get("/positions")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    print(f"  /positions: {len(data)} positions")


def test_status():
    r = client.get("/status")
    assert r.status_code == 200
    data = r.json()
    assert "risk" in data
    assert "strategy_weights" in data
    print(f"  /status: keys={list(data.keys())}")


def test_config():
    r = client.get("/config")
    assert r.status_code == 200
    data = r.json()
    # The raw credentials must not appear — only the sanitized "api_key_set" bool
    assert "api_secret" not in str(data)
    assert "api_key_set" in data          # sanitized flag is present
    assert isinstance(data["api_key_set"], bool)
    print(f"  /config: sanitized OK")


def test_pause_resume():
    r = client.post("/pause")
    assert r.status_code == 200
    assert r.json()["status"] == "paused"

    r = client.post("/resume")
    assert r.status_code == 200
    assert r.json()["status"] == "resumed"
    print(f"  /pause + /resume: OK")


if __name__ == "__main__":
    print("Testing API endpoints...")
    test_health()
    test_balance()
    test_trades()
    test_positions()
    test_status()
    test_config()
    test_pause_resume()
    print("\n=== ALL API TESTS PASSED ===")
