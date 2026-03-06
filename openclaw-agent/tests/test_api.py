"""Tests for api/app.py — FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_contains_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["service"] == "openclaw-agent"
        assert "version" in data
        assert "timestamp" in data

    def test_health_has_platform_count(self, client):
        data = client.get("/health").json()
        assert data["platforms_registered"] >= 30


class TestPlatformEndpoints:
    def test_list_platforms(self, client):
        resp = client.get("/platforms")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 30

    def test_platform_list_structure(self, client):
        data = client.get("/platforms").json()
        for p in data[:5]:
            assert "platform_id" in p
            assert "name" in p
            assert "category" in p
            assert "complexity" in p
            assert "status" in p

    def test_get_platform_valid(self, client):
        resp = client.get("/platform/gumroad")
        assert resp.status_code == 200
        data = resp.json()
        assert "platform" in data
        assert data["platform"]["id"] == "gumroad"

    def test_get_platform_invalid(self, client):
        resp = client.get("/platform/nonexistent_xyz_999")
        assert resp.status_code == 404


class TestProfileEndpoints:
    def test_generate_profile(self, client):
        resp = client.post(
            "/profile/generate",
            json={"platform_id": "gumroad"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform_id"] == "gumroad"
        assert "username" in data
        assert "bio" in data

    def test_generate_profile_invalid_platform(self, client):
        resp = client.post(
            "/profile/generate",
            json={"platform_id": "nonexistent_xyz"},
        )
        assert resp.status_code == 404


class TestAnalysisEndpoints:
    def test_analyze_valid_platform(self, client):
        resp = client.get("/analyze/gumroad")
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform_id"] == "gumroad"
        assert "complexity" in data
        assert "risks" in data

    def test_analyze_invalid_platform(self, client):
        resp = client.get("/analyze/nonexistent_xyz")
        assert resp.status_code == 404

    def test_prioritize_platforms(self, client):
        resp = client.get("/prioritize")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 10
        # Should be sorted by score
        for rec in data[:5]:
            assert "platform_id" in rec
            assert "score" in rec
            assert "priority" in rec
            assert "reasoning" in rec


class TestSyncEndpoints:
    def test_sync_preview(self, client):
        resp = client.post(
            "/sync/preview",
            json={"changes": {"bio": "Test bio"}, "platform_ids": ["gumroad"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["platform_id"] == "gumroad"

    def test_sync_status(self, client):
        resp = client.get("/sync/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_active" in data
        assert "consistent_fields" in data

    def test_sync_local(self, client):
        resp = client.post(
            "/sync",
            json={"changes": {"bio": "New bio"}, "platform_ids": ["gumroad"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_platforms" in data
        assert "succeeded" in data
        assert "results" in data


class TestPlatformDiscoveryEndpoints:
    def test_platforms_by_category(self, client):
        resp = client.get("/platforms/category/digital_product")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_platforms_by_invalid_category(self, client):
        resp = client.get("/platforms/category/nonexistent_category")
        assert resp.status_code == 400

    def test_easy_wins(self, client):
        resp = client.get("/platforms/easy-wins")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0


class TestAnalyticsEndpoints:
    def test_analytics_report(self, client):
        resp = client.get("/analytics/report")
        assert resp.status_code == 200

    def test_coverage_map(self, client):
        resp = client.get("/analytics/coverage")
        assert resp.status_code == 200

    def test_timeline(self, client):
        resp = client.get("/analytics/timeline")
        assert resp.status_code == 200

    def test_export_json(self, client):
        resp = client.post("/export", json={"format": "json"})
        assert resp.status_code == 200

    def test_export_csv(self, client):
        resp = client.post("/export", json={"format": "csv"})
        assert resp.status_code == 200


class TestInfrastructureEndpoints:
    def test_email_stats(self, client):
        resp = client.get("/email/stats")
        assert resp.status_code == 200

    def test_email_verified(self, client):
        resp = client.get("/email/verified")
        assert resp.status_code == 200

    def test_proxy_stats(self, client):
        resp = client.get("/proxies/stats")
        assert resp.status_code == 200

    def test_retry_stats(self, client):
        resp = client.get("/retry/stats")
        assert resp.status_code == 200

    def test_ratelimit_stats(self, client):
        resp = client.get("/ratelimit/stats")
        assert resp.status_code == 200

    def test_ratelimit_check(self, client):
        resp = client.get("/ratelimit/check/gumroad")
        assert resp.status_code == 200
        data = resp.json()
        assert "can_proceed" in data
        assert "platform_id" in data


class TestSchedulerEndpoints:
    def test_list_jobs(self, client):
        resp = client.get("/schedule/jobs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestWebSocketBroadcast:
    def test_broadcast_function_exists(self):
        from api.app import broadcast_ws
        assert callable(broadcast_ws)

    def test_websocket_connect(self, client):
        with client.websocket_connect("/ws/live") as ws:
            # Send a message and get ack
            ws.send_text("ping")
            data = ws.receive_json()
            assert data["type"] == "ack"


class TestDashboardEndpoint:
    def test_dashboard_returns_200(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200

    def test_dashboard_structure(self, client):
        data = client.get("/dashboard").json()
        assert "total_platforms" in data
        assert data["total_platforms"] >= 30
