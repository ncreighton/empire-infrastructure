"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestAPI:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "VideoForge" in data["service"]
        assert "endpoints" in data

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "videoforge"

    def test_analyze(self, client):
        response = client.post("/analyze", json={
            "topic": "moon rituals",
            "niche": "witchcraftforbeginners",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "analyze"
        assert data["scout_result"] is not None

    def test_create_dry_run(self, client):
        response = client.post("/create", json={
            "topic": "smart home tips",
            "niche": "smarthomewizards",
            "render": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "create"
        assert data["plan"] is not None
        assert data["plan"]["status"] == "assembled"

    def test_topics(self, client):
        response = client.post("/topics", json={
            "niche": "witchcraft",
            "count": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["topics"]) == 3

    def test_calendar(self, client):
        response = client.get("/calendar/mythicalarchives")
        assert response.status_code == 200
        data = response.json()
        assert len(data["calendar"]) == 7

    def test_insights(self, client):
        response = client.get("/insights")
        assert response.status_code == 200
        data = response.json()
        assert "total_videos" in data

    def test_cost_estimate(self, client):
        response = client.post("/cost-estimate", json={
            "topic": "test topic",
            "niche": "smarthomewizards",
        })
        assert response.status_code == 200
        data = response.json()
        assert "total_cost" in data
        assert data["total_cost"] > 0

    def test_knowledge_niches(self, client):
        response = client.get("/knowledge/niches")
        assert response.status_code == 200
        data = response.json()
        assert "witchcraftforbeginners" in data["niches"]

    def test_knowledge_hooks(self, client):
        response = client.get("/knowledge/hooks")
        assert response.status_code == 200
        assert "curiosity_gap" in response.json()["hooks"]

    def test_knowledge_platforms(self, client):
        response = client.get("/knowledge/platforms")
        assert response.status_code == 200
        assert "youtube_shorts" in response.json()["platforms"]

    def test_knowledge_moods(self, client):
        response = client.get("/knowledge/moods")
        assert response.status_code == 200
        assert len(response.json()["moods"]) >= 20

    def test_knowledge_subtitle_styles(self, client):
        response = client.get("/knowledge/subtitle-styles")
        assert response.status_code == 200
        assert "hormozi" in response.json()["styles"]

    def test_knowledge_trending(self, client):
        response = client.get("/knowledge/trending")
        assert response.status_code == 200
        assert len(response.json()["formats"]) >= 5

    def test_knowledge_shots(self, client):
        response = client.get("/knowledge/shots")
        assert response.status_code == 200
        assert len(response.json()["shots"]) >= 30

    def test_knowledge_voices(self, client):
        response = client.get("/knowledge/voices")
        assert response.status_code == 200
        assert "witchcraftforbeginners" in response.json()["voices"]

    def test_batch_create(self, client):
        response = client.post("/batch", json={
            "items": [
                {"topic": "moon rituals", "niche": "witchcraftforbeginners"},
                {"topic": "smart home", "niche": "smarthomewizards"},
            ],
            "render": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2
