"""
Tests for the FastAPI API endpoints.

Tests health endpoints, authentication gating, and rate limiting.
Uses FastAPI TestClient with mocked subsystems.

Auth is disabled via OPENCLAW_AUTH_DISABLED env var so tests can exercise
endpoint logic without needing real tokens.
"""

import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Disable auth BEFORE importing the API/auth modules so the module-level
# ``_auth_disabled`` flag in src.auth evaluates to True.
os.environ["OPENCLAW_AUTH_DISABLED"] = "true"

import pytest

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI_TEST = True
except ImportError:
    HAS_FASTAPI_TEST = False

try:
    from src.api import app, state, AppState
    HAS_API = True
except ImportError:
    HAS_API = False

pytestmark = pytest.mark.skipif(
    not (HAS_FASTAPI_TEST and HAS_API),
    reason="FastAPI or api module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def client():
    """FastAPI test client.

    The TestClient triggers the app lifespan which initialises all
    subsystems.  After the client is created we patch state attributes
    to inject mocks -- this avoids the lifespan overwriting our mocks.
    """
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_forge(client):
    """Replace state.forge with a MagicMock after the lifespan has run."""
    original = state.forge
    state.forge = MagicMock()
    state.forge.get_stats.return_value = {
        "forge_version": "1.0.0",
        "modules": ["scout", "sentinel", "oracle", "smith", "codex"],
    }
    state.forge.pre_flight = AsyncMock(return_value={
        "ready": True, "go_no_go": "GO",
        "scout": {}, "oracle": {}, "fixes": [],
    })
    state.forge.vision_prompt.return_value = "Test prompt"
    state.forge.record_outcome.return_value = {"task_id": "test", "outcome": "success"}
    state.forge.codex = MagicMock()
    state.forge.codex.get_app_history.return_value = {"total_tasks": 0}
    state.forge.codex.get_failure_patterns.return_value = []
    state.forge.codex.get_common_errors.return_value = {}
    yield state.forge
    state.forge = original


@pytest.fixture
def mock_amplify(client):
    """Replace state.amplify with a MagicMock after the lifespan has run."""
    original = state.amplify
    state.amplify = MagicMock()
    state.amplify.full_pipeline.return_value = {
        "_amplify": {"fully_processed": True, "stages_completed": []},
        "validation_summary": {"valid": True},
    }
    state.amplify.get_app_stats.return_value = {"total_records": 0}
    state.amplify.record_execution = MagicMock()
    yield state.amplify
    state.amplify = original


@pytest.fixture
def no_controller(client):
    """Set state.controller to None (phone disconnected)."""
    original = state.controller
    state.controller = None
    yield
    state.controller = original


@pytest.fixture
def no_screenpipe(client):
    """Set state.screenpipe to None (service unavailable)."""
    original = state.screenpipe
    state.screenpipe = None
    yield
    state.screenpipe = original


@pytest.fixture
def no_vision(client):
    """Set state.vision to None (service unavailable)."""
    original = state.vision
    state.vision = None
    yield
    state.vision = original


@pytest.fixture
def no_executor(client):
    """Set state.executor to None (task executor not initialised)."""
    original = state.executor
    state.executor = None
    yield
    state.executor = original


# ===================================================================
# TestHealthEndpoint
# ===================================================================

class TestHealthEndpoint:
    """Test /health endpoint."""

    @pytest.mark.unit
    def test_health_returns_200(self, client):
        """Health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    @pytest.mark.unit
    def test_health_contains_status(self, client):
        """Health response contains status field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    @pytest.mark.unit
    def test_health_contains_subsystems(self, client):
        """Health response contains subsystem status."""
        response = client.get("/health")
        data = response.json()
        assert "subsystems" in data
        subs = data["subsystems"]
        assert "forge" in subs
        assert "amplify" in subs

    @pytest.mark.unit
    def test_health_contains_timestamp(self, client):
        """Health response includes timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data

    @pytest.mark.unit
    def test_health_contains_version(self, client):
        """Health response includes version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"


# ===================================================================
# TestStatsEndpoint
# ===================================================================

class TestStatsEndpoint:
    """Test /stats endpoint."""

    @pytest.mark.unit
    def test_stats_returns_200(self, client):
        """Stats endpoint returns 200."""
        response = client.get("/stats")
        assert response.status_code == 200

    @pytest.mark.unit
    def test_stats_contains_forge(self, client, mock_forge):
        """Stats includes forge data when forge is mocked."""
        response = client.get("/stats")
        data = response.json()
        assert "forge" in data


# ===================================================================
# TestForgeEndpoints
# ===================================================================

class TestForgeEndpoints:
    """Test FORGE intelligence endpoints."""

    @pytest.mark.unit
    def test_forge_stats(self, client, mock_forge):
        """GET /forge/stats returns stats."""
        response = client.get("/forge/stats")
        assert response.status_code == 200

    @pytest.mark.unit
    def test_forge_preflight(self, client, mock_forge):
        """POST /forge/pre-flight runs analysis."""
        response = client.post("/forge/pre-flight", json={
            "phone_state": {
                "screen_on": True,
                "locked": False,
                "battery_percent": 80,
                "wifi_connected": True,
            },
            "task": {
                "app": "chrome",
                "steps": [{"type": "tap"}],
            },
        })
        assert response.status_code == 200

    @pytest.mark.unit
    def test_forge_vision_prompt(self, client, mock_forge):
        """POST /forge/vision-prompt returns prompt."""
        response = client.post("/forge/vision-prompt", json={
            "template": "find_element",
            "context": {"element_description": "Publish button"},
        })
        assert response.status_code == 200

    @pytest.mark.unit
    def test_forge_vision_prompt_invalid_template(self, client, mock_forge):
        """Invalid template returns 400."""
        mock_forge.vision_prompt.side_effect = ValueError("Unknown template")
        response = client.post("/forge/vision-prompt", json={
            "template": "nonexistent",
            "context": {},
        })
        assert response.status_code == 400

    @pytest.mark.unit
    def test_forge_codex_app(self, client, mock_forge):
        """GET /forge/codex/app/{name} returns app history."""
        response = client.get("/forge/codex/app/chrome")
        assert response.status_code == 200

    @pytest.mark.unit
    def test_forge_codex_patterns(self, client, mock_forge):
        """GET /forge/codex/patterns/{name} returns failure patterns."""
        response = client.get("/forge/codex/patterns/chrome")
        assert response.status_code == 200

    @pytest.mark.unit
    def test_forge_codex_learn(self, client, mock_forge):
        """POST /forge/codex/learn records outcome."""
        response = client.post("/forge/codex/learn", json={
            "task_id": "test-001",
            "outcome": "success",
            "duration": 5.0,
        })
        assert response.status_code == 200


# ===================================================================
# TestAmplifyEndpoints
# ===================================================================

class TestAmplifyEndpoints:
    """Test AMPLIFY pipeline endpoints."""

    @pytest.mark.unit
    def test_amplify_process(self, client, mock_amplify):
        """POST /amplify/process runs pipeline."""
        response = client.post("/amplify/process", json={
            "app": "chrome",
            "steps": [{"action": "tap_element", "target": "button"}],
        })
        assert response.status_code == 200

    @pytest.mark.unit
    def test_amplify_process_missing_app(self, client, mock_amplify):
        """AMPLIFY process with ValueError from pipeline returns 400."""
        mock_amplify.full_pipeline.side_effect = ValueError("must specify an app")
        response = client.post("/amplify/process", json={"steps": []})
        assert response.status_code == 400

    @pytest.mark.unit
    def test_amplify_stats(self, client, mock_amplify):
        """GET /amplify/stats/{app} returns stats."""
        response = client.get("/amplify/stats/chrome")
        assert response.status_code == 200

    @pytest.mark.unit
    def test_amplify_record(self, client, mock_amplify):
        """POST /amplify/record records timing."""
        response = client.post("/amplify/record", json={
            "action_type": "tap_element",
            "app_name": "chrome",
            "duration": 1.5,
            "success": True,
        })
        assert response.status_code == 200


# ===================================================================
# TestPhoneEndpoints
# ===================================================================

class TestPhoneEndpoints:
    """Test phone control endpoints when phone is disconnected."""

    @pytest.mark.unit
    def test_phone_state_without_controller(self, client, no_controller):
        """Phone state returns 503 when controller not available."""
        response = client.get("/phone/state")
        assert response.status_code == 503

    @pytest.mark.unit
    def test_phone_screenshot_without_controller(self, client, no_controller):
        """Screenshot returns 503 when controller not available."""
        response = client.post("/phone/screenshot")
        assert response.status_code == 503

    @pytest.mark.unit
    def test_phone_tap_without_controller(self, client, no_controller):
        """Tap returns 503 when controller not available."""
        response = client.post("/phone/tap", json={"x": 100, "y": 200})
        assert response.status_code == 503

    @pytest.mark.unit
    def test_phone_type_without_controller(self, client, no_controller):
        """Type returns 503 when controller not available."""
        response = client.post("/phone/type", json={"text": "hello"})
        assert response.status_code == 503


# ===================================================================
# TestTaskEndpoints
# ===================================================================

class TestTaskEndpoints:
    """Test task execution endpoints."""

    @pytest.mark.unit
    def test_task_execute_without_executor(self, client, no_executor):
        """Execute returns 503 when executor not initialized."""
        response = client.post("/task/execute", json={
            "task_description": "Open Chrome",
        })
        assert response.status_code == 503

    @pytest.mark.unit
    def test_task_status_not_found(self, client):
        """Unknown task ID returns 404."""
        response = client.get("/task/nonexistent-id/status")
        assert response.status_code == 404

    @pytest.mark.unit
    def test_task_complete(self, client, mock_forge):
        """Task complete endpoint records result."""
        # Register a fake task in running_tasks
        state.running_tasks["test-123"] = {"status": "running"}
        response = client.post("/task/test-123/complete", json={
            "success": True,
            "duration": 5.0,
        })
        assert response.status_code == 200
        # Clean up
        state.running_tasks.pop("test-123", None)


# ===================================================================
# TestScreenpipeEndpoints
# ===================================================================

class TestScreenpipeEndpoints:
    """Test Screenpipe endpoints when service unavailable."""

    @pytest.mark.unit
    def test_screenpipe_state_unavailable(self, client, no_screenpipe):
        """Screenpipe state returns 503 when not initialized."""
        response = client.get("/screenpipe/state")
        assert response.status_code == 503

    @pytest.mark.unit
    def test_screenpipe_errors_unavailable(self, client, no_screenpipe):
        """Screenpipe errors returns 503 when not initialized."""
        response = client.get("/screenpipe/errors")
        assert response.status_code == 503


# ===================================================================
# TestVisionEndpoints
# ===================================================================

class TestVisionEndpoints:
    """Test Vision endpoints when service unavailable."""

    @pytest.mark.unit
    def test_vision_analyze_unavailable(self, client, no_vision):
        """Vision analyze returns 503 when not initialized."""
        response = client.post("/vision/analyze", json={})
        assert response.status_code == 503

    @pytest.mark.unit
    def test_vision_find_element_unavailable(self, client, no_vision):
        """Vision find element returns 503 when not initialized."""
        response = client.post("/vision/find-element", json={"description": "button"})
        assert response.status_code == 503
