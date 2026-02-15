"""
Tests for the Autonomous Agent module.

Tests goal decomposition, GOAL_PATTERNS matching, module dispatch,
run loop, session management, and conversation mode. All external
dependencies are mocked.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.autonomous_agent import (
        ActionType,
        AgentSession,
        AgentStep,
        AutonomousAgent,
        Goal,
        GoalPriority,
        GoalStatus,
        ModuleName,
        SubGoal,
        get_autonomous_agent,
        GOAL_PATTERNS,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="autonomous_agent module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def agent_data_dir(tmp_path):
    """Isolated data directory for agent state."""
    d = tmp_path / "agent"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def agent(agent_data_dir):
    """Create a fresh AutonomousAgent with temp data dir."""
    return AutonomousAgent(data_dir=agent_data_dir)


# ===================================================================
# Enum / Dataclass Tests
# ===================================================================

class TestEnums:
    """Verify enum values exist and are sensible."""

    def test_goal_priority_ordering(self):
        # Verify all members exist
        assert GoalPriority.CRITICAL is not None
        assert GoalPriority.HIGH is not None
        assert GoalPriority.NORMAL is not None
        assert GoalPriority.LOW is not None
        assert GoalPriority.BACKGROUND is not None

    def test_goal_status_members(self):
        assert GoalStatus.PENDING is not None
        assert GoalStatus.ACTIVE is not None
        assert GoalStatus.IN_PROGRESS is not None
        assert GoalStatus.BLOCKED is not None
        assert GoalStatus.COMPLETED is not None
        assert GoalStatus.FAILED is not None
        assert GoalStatus.CANCELLED is not None

    def test_action_type_members(self):
        members = list(ActionType)
        assert len(members) >= 3

    def test_module_name_members(self):
        members = list(ModuleName)
        assert len(members) >= 5


class TestSubGoal:
    """Tests for SubGoal dataclass."""

    def test_create_subgoal(self):
        sg = SubGoal(
            description="Test sub goal",
            module="content_generator",
            method="generate",
        )
        assert sg.description == "Test sub goal"
        assert sg.module == "content_generator"
        assert sg.method == "generate"
        assert sg.status == GoalStatus.PENDING

    def test_subgoal_to_dict(self):
        sg = SubGoal(
            description="Write article",
            module="content_generator",
            method="generate",
        )
        d = sg.to_dict()
        assert "description" in d
        assert "module" in d
        assert d["status"] == GoalStatus.PENDING.value


class TestGoal:
    """Tests for Goal dataclass."""

    def test_create_goal(self):
        g = Goal(
            text="Publish article to witchcraft site",
            priority=GoalPriority.HIGH,
        )
        assert g.text == "Publish article to witchcraft site"
        assert g.priority == GoalPriority.HIGH
        assert g.status == GoalStatus.PENDING

    def test_goal_has_id(self):
        g = Goal(
            text="Test",
            priority=GoalPriority.NORMAL,
        )
        assert g.id is not None and len(g.id) > 0


class TestAgentStep:
    """Tests for AgentStep tracking."""

    def test_step_creation(self):
        step = AgentStep(
            action_type=ActionType.ACT,
            module="content_generator",
            method="generate",
            description="Generate article",
            success=True,
        )
        assert step.success is True
        assert step.module == "content_generator"
        assert step.action_type == ActionType.ACT


class TestAgentSession:
    """Tests for session tracking."""

    def test_session_creation(self):
        session = AgentSession()
        assert session.id is not None
        assert session.active is True
        assert session.goals_completed == 0


# ===================================================================
# GOAL_PATTERNS Tests
# ===================================================================

class TestGoalPatterns:
    """Verify GOAL_PATTERNS dictionary maps properly."""

    def test_patterns_is_dict(self):
        assert isinstance(GOAL_PATTERNS, dict)
        assert len(GOAL_PATTERNS) > 0

    def test_common_patterns_exist(self):
        keys = [k.lower() for k in GOAL_PATTERNS]
        # Check at least some expected patterns
        found_publish = any("publish" in k for k in keys)
        found_content = any("content" in k for k in keys)
        assert found_publish or found_content, (
            f"Expected publish or content patterns, got: {keys[:5]}"
        )

    def test_pattern_values_are_lists(self):
        for key, value in GOAL_PATTERNS.items():
            assert isinstance(value, list), f"Pattern '{key}' value should be a list"


# ===================================================================
# Agent Core Tests
# ===================================================================

class TestAutonomousAgent:
    """Tests for the main AutonomousAgent class."""

    def test_init(self, agent, agent_data_dir):
        assert agent is not None

    def test_set_goal(self, agent):
        """set_goal is async; use set_goal_sync which returns a Goal object."""
        with patch("src.autonomous_agent.anthropic", create=True):
            goal = agent.set_goal_sync(
                text="Publish content to witchcraft site",
                priority=GoalPriority.HIGH,
            )
        assert goal is not None
        assert isinstance(goal, Goal)
        assert isinstance(goal.id, str)

    def test_set_goal_returns_unique_ids(self, agent):
        with patch("src.autonomous_agent.anthropic", create=True):
            g1 = agent.set_goal_sync("Goal A", priority=GoalPriority.NORMAL)
            g2 = agent.set_goal_sync("Goal B", priority=GoalPriority.NORMAL)
        assert g1.id != g2.id

    def test_list_goals(self, agent):
        with patch("src.autonomous_agent.anthropic", create=True):
            agent.set_goal_sync("Goal 1", priority=GoalPriority.NORMAL)
            agent.set_goal_sync("Goal 2", priority=GoalPriority.HIGH)
        goals = agent.list_goals()
        assert len(goals) >= 2
        # list_goals returns list of dicts
        assert isinstance(goals[0], dict)

    def test_cancel_goal(self, agent):
        with patch("src.autonomous_agent.anthropic", create=True):
            goal = agent.set_goal_sync("Cancel me", priority=GoalPriority.LOW)
        result = agent.cancel_goal(goal.id)
        assert isinstance(result, dict)
        assert result["success"] is True
        goals = agent.list_goals()
        cancelled = [g for g in goals if g["id"] == goal.id]
        if cancelled:
            assert cancelled[0]["status"] == GoalStatus.CANCELLED.value

    def test_cancel_nonexistent_goal(self, agent):
        result = agent.cancel_goal("nonexistent_id_xyz")
        assert isinstance(result, dict)
        assert result["success"] is False

    def test_status(self, agent):
        with patch("src.autonomous_agent.anthropic", create=True):
            agent.set_goal_sync("Status check", priority=GoalPriority.NORMAL)
        status = agent.status()
        assert isinstance(status, dict)
        assert "running" in status
        assert "pending_goals" in status

    def test_list_sessions(self, agent):
        sessions = agent.list_sessions()
        assert isinstance(sessions, list)


# ===================================================================
# Decompose and Dispatch Tests
# ===================================================================

class TestDecomposition:
    """Test goal decomposition into sub-goals."""

    @pytest.mark.asyncio
    async def test_decompose_goal_pattern_match(self, agent):
        """_decompose_goal takes (text, context) and matches GOAL_PATTERNS."""
        sub_goals = await agent._decompose_goal("manage instagram for witchcraft", {})
        assert isinstance(sub_goals, list)
        assert len(sub_goals) > 0
        assert isinstance(sub_goals[0], SubGoal)

    @pytest.mark.asyncio
    async def test_execute_subgoal(self, agent):
        sg = SubGoal(
            description="Generate article",
            module="content_generator",
            method="generate",
        )
        goal = Goal(text="Test goal", priority=GoalPriority.NORMAL, sub_goals=[sg])
        with patch.object(agent, "_dispatch", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = {"status": "ok", "success": True}
            step = await agent._execute_subgoal(sg, goal)
            assert isinstance(step, AgentStep)
            assert step.success is True


# ===================================================================
# Run Loop Tests
# ===================================================================

class TestRunLoop:
    """Test the main agent run loop."""

    @pytest.mark.asyncio
    async def test_run_processes_goals(self, agent):
        # Use pattern-based decomposition to avoid needing anthropic
        with patch.object(agent, "_dispatch", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = {"status": "ok", "success": True}
            await agent.set_goal("manage instagram for witchcraft", GoalPriority.HIGH)
            result = await agent.run(max_steps=5, timeout_minutes=0.1)
            assert isinstance(result, dict)
            assert "steps_taken" in result

    @pytest.mark.asyncio
    async def test_run_handles_empty_goals(self, agent):
        # Should not raise with no goals
        result = await agent.run(max_steps=1, timeout_minutes=0.1)
        assert isinstance(result, dict)
        assert result["steps_taken"] == 0


# ===================================================================
# Conversation Mode Tests
# ===================================================================

class TestConversation:
    """Test interactive conversation mode."""

    @pytest.mark.asyncio
    async def test_converse_returns_response(self, agent):
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "I can help with that. Let me check the sites."
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        # anthropic is imported locally inside converse(), so patch at module level
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = await agent.converse("What sites need content?")
            # converse returns a dict with 'response' key
            assert isinstance(result, dict)
            assert "response" in result
            assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_quick_action(self, agent):
        """quick_action takes (module, method, **kwargs), dispatches directly."""
        with patch.object(agent, "_dispatch", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = {"status": "done"}
            result = await agent.quick_action("content_generator", "generate", title="Test")
            assert result is not None
            assert result["status"] == "done"


# ===================================================================
# Session Management Tests
# ===================================================================

class TestSessionManagement:
    """Test session create/retrieve/list."""

    def test_get_session_nonexistent(self, agent):
        result = agent.get_session("nonexistent_session_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_run_creates_session(self, agent):
        # Use a pattern-matched goal so no anthropic import needed
        await agent.set_goal("manage instagram for test", GoalPriority.LOW)
        with patch.object(agent, "_dispatch", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = {"status": "ok", "success": True}
            await agent.run(max_steps=5, timeout_minutes=0.1)
        sessions = agent.list_sessions()
        assert isinstance(sessions, list)
        assert len(sessions) >= 1


# ===================================================================
# Persistence Tests
# ===================================================================

class TestPersistence:
    """Test state saving and loading."""

    def test_state_saved_to_disk(self, agent, agent_data_dir):
        with patch("src.autonomous_agent.anthropic", create=True):
            agent.set_goal_sync("Persistent goal", priority=GoalPriority.NORMAL)
        agent._save_state()
        state_file = agent_data_dir / "state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "goals" in data

    def test_load_state_from_disk(self, agent_data_dir):
        # Pre-seed a state file with correct field names and lowercase enum values
        state = {
            "goals": [
                {
                    "id": "g1",
                    "text": "Seeded goal",
                    "priority": "normal",
                    "status": "pending",
                    "sub_goals": [],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "started_at": "",
                    "completed_at": "",
                    "error": "",
                    "context": {},
                }
            ],
            "sessions": [],
        }
        (agent_data_dir / "state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        agent2 = AutonomousAgent(data_dir=agent_data_dir)
        goals = agent2.list_goals()
        assert len(goals) >= 1


# ===================================================================
# Singleton Tests
# ===================================================================

class TestSingleton:
    """Test the factory/singleton function."""

    def test_get_autonomous_agent_returns_instance(self, tmp_path):
        with patch("src.autonomous_agent.DATA_DIR", tmp_path / "agent"), \
             patch("src.autonomous_agent._instance", None):
            (tmp_path / "agent").mkdir(exist_ok=True)
            a = get_autonomous_agent()
            assert isinstance(a, AutonomousAgent)


# ===================================================================
# Delegate to Orchestrator Tests
# ===================================================================

class TestDelegation:
    """Test delegating complex goals to the orchestrator."""

    @pytest.mark.asyncio
    async def test_delegate_to_orchestrator(self, agent):
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_mission = AsyncMock(return_value={"success": True, "status": "completed"})

        with patch.object(agent, "_get_module", return_value=mock_orchestrator):
            result = await agent.delegate_to_orchestrator("content_pipeline", {"site_id": "witchcraft"})
            assert result is not None
            assert isinstance(result, dict)
            assert result["success"] is True
