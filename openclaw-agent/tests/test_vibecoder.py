"""Comprehensive test suite for the VibeCoder agent system.

Tests: models, project_scout, code_sentinel, mission_oracle,
       vibe_planner, vibe_executor, vibe_reviewer, code_amplify,
       vibecoder_engine, mission_daemon, model_router.
"""

import asyncio
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Test: Models ────────────────────────────────────────────────────────────

def test_models():
    """Test all dataclasses and enums."""
    from openclaw.vibecoder.models import (
        MissionStatus, MissionScope, EngineType, StepType, StepStatus,
        ReviewVerdict, QualityGrade, DeployTarget,
        ProjectInfo, MissionStep, Mission, SentinelResult, OracleEstimate,
        ReviewResult, CodeChange, AmplifyResult, VibeDashboard,
    )

    # Enums
    assert MissionStatus.QUEUED.value == "queued"
    assert MissionScope.BUGFIX.value == "bugfix"
    assert EngineType.ALGORITHMIC.value == "algorithmic"
    assert StepType.CREATE_FILE.value == "create_file"
    assert StepStatus.PENDING.value == "pending"
    assert ReviewVerdict.APPROVED.value == "approved"
    assert QualityGrade.S.value == "S"
    assert DeployTarget.VPS_DOCKER.value == "vps_docker"

    # Dataclasses with defaults
    step = MissionStep(step_number=1, step_type=StepType.CREATE_FILE, description="test")
    assert step.status == StepStatus.PENDING
    assert step.engine == EngineType.ALGORITHMIC
    assert step.tokens_used == 0

    mission = Mission(mission_id="test", project_id="proj", title="T", description="D")
    assert mission.status == MissionStatus.QUEUED
    assert mission.scope == MissionScope.UNKNOWN
    assert mission.total_cost_usd == 0.0
    assert mission.steps == []

    # SentinelResult scoring
    sr = SentinelResult(lint_score=20, security_score=20, test_score=20,
                        convention_score=15, complexity_score=15, coverage_score=10)
    sr.calculate()
    assert sr.total_score == 100.0
    assert sr.grade == QualityGrade.S

    sr2 = SentinelResult(lint_score=15, security_score=15, test_score=10,
                         convention_score=10, complexity_score=10, coverage_score=5)
    sr2.calculate()
    assert sr2.grade == QualityGrade.C  # 65 points

    # ProjectInfo
    pi = ProjectInfo(project_id="test", root_path="/tmp")
    assert pi.language == "python"
    assert pi.has_tests is False

    print("  Models: PASSED")


# ─── Test: ProjectScout ──────────────────────────────────────────────────────

def test_project_scout():
    """Test codebase analysis."""
    from openclaw.vibecoder.forge.project_scout import ProjectScout

    scout = ProjectScout()

    # Analyze the openclaw-agent project itself
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    info = scout.analyze("openclaw-agent", root)

    assert info.project_id == "openclaw-agent"
    assert info.language == "python"
    # has_git may be False if .git is in parent monorepo root
    assert info.has_tests is True
    assert info.total_files > 0
    assert info.total_lines > 0
    assert len(info.entry_points) > 0
    assert len(info.dependencies) > 0

    # Test with a temp empty directory
    with tempfile.TemporaryDirectory() as tmpdir:
        info2 = scout.analyze("empty-project", tmpdir)
        assert info2.total_files == 0

    print("  ProjectScout: PASSED")


# ─── Test: CodeSentinel ──────────────────────────────────────────────────────

def test_code_sentinel():
    """Test code quality scoring and secret detection."""
    from openclaw.vibecoder.forge.code_sentinel import CodeSentinel
    from openclaw.vibecoder.models import CodeChange, QualityGrade

    sentinel = CodeSentinel()

    # Clean changes should score high
    clean = [CodeChange(file_path="app.py", change_type="modified",
                        diff="+def hello():\n+    return 'world'\n")]
    result = sentinel.score(clean)
    assert result.total_score >= 70, f"Clean code scored {result.total_score}"
    assert not result.blockers

    # Secret detection
    secret_changes = [CodeChange(
        file_path="config.py",
        diff='+api_key = "sk-proj-abc123def456ghi789jkl012mno345"',
    )]
    result2 = sentinel.score(secret_changes)
    assert result2.security_score == 0.0, "Should detect API key"

    # DB connection string
    db_change = [CodeChange(
        file_path="db.py",
        diff='+postgresql://admin:secretpass@host.com/mydb',
    )]
    result3 = sentinel.score(db_change)
    assert result3.security_score < 20, "Should detect DB connection string"

    # Quick check
    issues = sentinel.quick_check("test.py", 'sk-abc123def456ghi789jkl012mno')
    assert len(issues) > 0, "Should find API key in quick check"

    # Get blockers
    blockers = sentinel.get_blockers(secret_changes)
    assert len(blockers) > 0

    # Empty changes
    empty_result = sentinel.score([])
    assert empty_result.total_score == 100.0

    print("  CodeSentinel: PASSED")


# ─── Test: MissionOracle ─────────────────────────────────────────────────────

def test_mission_oracle():
    """Test scope classification and engine routing."""
    from openclaw.vibecoder.forge.mission_oracle import MissionOracle
    from openclaw.vibecoder.models import (
        MissionScope, EngineType, MissionStep, StepType, Mission,
    )

    oracle = MissionOracle()

    # Scope classification
    assert oracle.classify_scope("Fix login bug", "login page crashes") == MissionScope.BUGFIX
    assert oracle.classify_scope("Add dark mode", "implement dark theme") == MissionScope.FEATURE
    assert oracle.classify_scope("Refactor auth", "restructure authentication") == MissionScope.REFACTOR
    assert oracle.classify_scope("Update README", "add documentation") == MissionScope.DOCS
    assert oracle.classify_scope("Add pytest", "increase test coverage") == MissionScope.TEST
    assert oracle.classify_scope("Setup Docker", "deploy pipeline") == MissionScope.DEPLOY

    # Engine routing
    git_step = MissionStep(step_number=1, step_type=StepType.GIT_OPERATION, description="commit")
    assert oracle.route_step(git_step) == EngineType.ALGORITHMIC

    test_step = MissionStep(step_number=1, step_type=StepType.RUN_TESTS, description="run tests")
    assert oracle.route_step(test_step) == EngineType.ALGORITHMIC

    edit_step = MissionStep(step_number=1, step_type=StepType.EDIT_FILE, description="edit file")
    assert oracle.route_step(edit_step) == EngineType.API_SONNET

    review_step = MissionStep(step_number=1, step_type=StepType.REVIEW, description="review code")
    assert oracle.route_step(review_step) == EngineType.API_HAIKU

    refactor_step = MissionStep(step_number=1, step_type=StepType.REFACTOR, description="refactor")
    assert oracle.route_step(refactor_step) == EngineType.CLI_CLAUDE

    # Cost estimation
    mission = Mission(mission_id="est", project_id="test", title="Fix bug",
                      description="Fix login issue", scope=MissionScope.BUGFIX)
    estimate = oracle.estimate_mission(mission)
    assert estimate.estimated_cost_usd > 0
    assert estimate.confidence > 0
    assert estimate.reasoning

    print("  MissionOracle: PASSED")


# ─── Test: CodeSmith ─────────────────────────────────────────────────────────

def test_code_smith():
    """Test code template rendering."""
    from openclaw.vibecoder.forge.code_smith import CodeSmith

    smith = CodeSmith()

    # Render Python module template
    result = smith.render("python_module", module_name="utils", description="Utility functions")
    assert "Utility functions" in result
    assert '"""' in result

    # Render Python class
    result2 = smith.render("python_class", class_name="UserManager",
                           description="Manages user operations")
    assert "class UserManager" in result2

    # Template suggestion
    suggestion = smith.suggest_template("create a new python function")
    assert suggestion is not None
    assert smith.has_template(suggestion)

    # Render with required params
    result3 = smith.render("init_py", package_name="mypackage")
    assert result3  # Should not be empty

    # List templates
    templates = smith.list_templates()
    assert len(templates) >= 10

    print("  CodeSmith: PASSED")


# ─── Test: VibeCodex ─────────────────────────────────────────────────────────

def test_vibe_codex():
    """Test SQLite persistence for missions and projects."""
    from openclaw.vibecoder.forge.vibe_codex import VibeCodex
    from openclaw.vibecoder.models import (
        Mission, MissionStatus, MissionScope, MissionStep, StepType,
        StepStatus, EngineType, CodeChange, ProjectInfo, DeployTarget,
    )
    from datetime import datetime

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        codex = VibeCodex(db_path=db_path)

        # Create mission
        mission = Mission(
            mission_id="test-001", project_id="myproj", title="Test Mission",
            description="A test", scope=MissionScope.FEATURE,
            status=MissionStatus.QUEUED, created_at=datetime.now(),
        )
        codex.create_mission(mission)

        # Get mission
        row = codex.get_mission("test-001")
        assert row is not None
        assert row["title"] == "Test Mission"
        assert row["scope"] == "feature"

        # Update status
        codex.update_mission_status(
            "test-001", MissionStatus.EXECUTING,
            branch_name="vibe/test", total_tokens=100,
        )
        row2 = codex.get_mission("test-001")
        assert row2["status"] == "executing"
        assert row2["branch_name"] == "vibe/test"
        assert row2["total_tokens"] == 100

        # Pause
        codex.update_mission_status("test-001", MissionStatus.PAUSED)
        row3 = codex.get_mission("test-001")
        assert row3["status"] == "paused"

        # Log step
        step = MissionStep(
            step_number=1, step_type=StepType.EDIT_FILE,
            description="Edit app.py", target_file="app.py",
            engine=EngineType.API_SONNET, status=StepStatus.COMPLETED,
            tokens_used=500, cost_usd=0.005,
            started_at=datetime.now(), completed_at=datetime.now(),
        )
        codex.log_step("test-001", step)
        steps = codex.get_steps("test-001")
        assert len(steps) == 1
        assert steps[0]["step_type"] == "edit_file"

        # Log code change
        change = CodeChange(file_path="app.py", change_type="modified",
                            lines_added=10, lines_removed=2)
        codex.log_change("test-001", change)
        changes = codex.get_changes("test-001")
        assert len(changes) == 1

        # Register project
        info = ProjectInfo(
            project_id="myproj", root_path="/tmp/myproj",
            language="python", framework="fastapi",
            has_tests=True, has_git=True,
            deploy_target=DeployTarget.VPS_DOCKER,
        )
        codex.register_project(info)
        proj = codex.get_project("myproj")
        assert proj is not None
        assert proj["language"] == "python"

        # project_to_info
        info_back = codex.project_to_info(proj)
        assert info_back.project_id == "myproj"
        assert info_back.framework == "fastapi"
        assert info_back.deploy_target == DeployTarget.VPS_DOCKER

        # List missions
        missions = codex.get_missions()
        assert len(missions) >= 1

        # Queued missions
        codex.update_mission_status("test-001", MissionStatus.QUEUED)
        queued = codex.get_queued_missions()
        assert len(queued) >= 1

        # Dashboard
        dash = codex.get_dashboard()
        assert dash.total_missions >= 1
        assert dash.registered_projects >= 1

        # Delete mission
        ok = codex.delete_mission("test-001")
        # Must be queued to cancel
        assert ok is True
        row4 = codex.get_mission("test-001")
        assert row4["status"] == "cancelled"

        print("  VibeCodex: PASSED")

    finally:
        try:
            os.unlink(db_path)
        except PermissionError:
            pass  # Windows SQLite file lock


# ─── Test: VibePlannerAgent ──────────────────────────────────────────────────

def test_vibe_planner():
    """Test mission decomposition into steps."""
    from openclaw.vibecoder.agents.vibe_planner_agent import VibePlannerAgent
    from openclaw.vibecoder.models import (
        Mission, MissionScope, MissionStep, StepType, ProjectInfo, EngineType,
    )

    planner = VibePlannerAgent()

    # Test bugfix planning
    mission = Mission(
        mission_id="plan-1", project_id="test",
        title="Fix login bug", description="Fix bug in auth.py",
        scope=MissionScope.BUGFIX,
    )
    steps = planner.plan(mission)
    assert len(steps) >= 2  # At least edit + commit
    assert any(s.step_type == StepType.GIT_OPERATION for s in steps)

    # Test with project context
    proj = ProjectInfo(
        project_id="test", root_path="/tmp",
        language="python", has_tests=True, has_git=True,
        dependencies=["pytest"],
    )
    mission2 = Mission(
        mission_id="plan-2", project_id="test",
        title="Add feature", description="Add new util function in utils.py",
        scope=MissionScope.FEATURE,
    )
    steps2 = planner.plan(mission2, proj)
    assert len(steps2) >= 3

    # Each step should have engine assigned
    for s in steps2:
        assert s.engine is not None

    # Target file extraction
    result = VibePlannerAgent._extract_target_file("Fix bug in openclaw/models.py")
    assert result == "openclaw/models.py"

    result2 = VibePlannerAgent._extract_target_file("Update `src/app.ts`")
    assert result2 == "src/app.ts"

    result3 = VibePlannerAgent._extract_target_file("Do something")
    assert result3 is None

    print("  VibePlannerAgent: PASSED")


# ─── Test: VibeReviewerAgent ─────────────────────────────────────────────────

def test_vibe_reviewer():
    """Test algorithmic code review."""
    from openclaw.vibecoder.agents.vibe_reviewer_agent import VibeReviewerAgent
    from openclaw.vibecoder.models import (
        CodeChange, Mission, MissionScope, ReviewVerdict,
    )

    reviewer = VibeReviewerAgent()

    # Clean changes should be approved
    changes = [CodeChange(
        file_path="utils.py", change_type="modified",
        diff="+def helper():\n+    return True\n",
        lines_added=2, lines_removed=0,
    )]
    mission = Mission(mission_id="rev-1", project_id="test",
                      title="Add helper", description="Add helper function")

    result = reviewer.review(changes, mission)
    assert result.verdict in (ReviewVerdict.APPROVED, ReviewVerdict.NEEDS_CHANGES)
    assert result.score >= 0
    assert result.files_reviewed == 1

    # Changes with debug artifacts should get lower scores
    debug_changes = [CodeChange(
        file_path="app.py", change_type="modified",
        diff="+import pdb\n+breakpoint()\n+print('debug here')\n",
        lines_added=3, lines_removed=0,
    )]
    result2 = reviewer.review(debug_changes, mission)
    assert result2.score < result.score or len(result2.issues) > len(result.issues)

    print("  VibeReviewerAgent: PASSED")


# ─── Test: CodeAmplify ───────────────────────────────────────────────────────

def test_code_amplify():
    """Test 6-stage plan quality pipeline."""
    from openclaw.vibecoder.amplify.code_amplify import CodeAmplify
    from openclaw.vibecoder.models import (
        Mission, MissionScope, MissionStep, StepType, EngineType, ProjectInfo,
    )

    amplify = CodeAmplify()

    # Create a simple mission with steps
    mission = Mission(
        mission_id="amp-1", project_id="test",
        title="Add feature", description="Add utility function",
        scope=MissionScope.FEATURE,
        steps=[
            MissionStep(step_number=1, step_type=StepType.CREATE_FILE,
                        description="Create util.py", engine=EngineType.API_SONNET),
            MissionStep(step_number=2, step_type=StepType.GIT_OPERATION,
                        description="Commit new feature", engine=EngineType.ALGORITHMIC),
        ],
    )
    proj = ProjectInfo(
        project_id="test", root_path="/tmp",
        language="python", has_tests=True, has_git=True,
    )

    result = amplify.amplify(mission, proj)
    assert result.stages_completed == 6, f"Only {result.stages_completed}/6 stages completed"
    assert result.quality_score > 0
    assert "enrich" in result.stage_details
    assert "expand" in result.stage_details
    assert "fortify" in result.stage_details
    assert "anticipate" in result.stage_details
    assert "optimize" in result.stage_details
    assert "validate" in result.stage_details

    # AMPLIFY should have added branch step and test step
    has_branch = any(
        s.step_type == StepType.GIT_OPERATION and "branch" in s.description.lower()
        for s in mission.steps
    )
    has_test = any(s.step_type == StepType.RUN_TESTS for s in mission.steps)

    # expand should have added these
    assert has_branch, "AMPLIFY should add branch step"
    assert has_test, "AMPLIFY should add test step"

    print("  CodeAmplify: PASSED")


# ─── Test: VibeCoderEngine ───────────────────────────────────────────────────

def test_vibecoder_engine():
    """Test the master orchestrator."""
    from openclaw.vibecoder.vibecoder_engine import VibeCoderEngine

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        engine = VibeCoderEngine(db_path=db_path)

        # Verify all modules initialized
        assert engine.scout is not None
        assert engine.sentinel is not None
        assert engine.oracle is not None
        assert engine.smith is not None
        assert engine.codex is not None
        assert engine.amplify is not None
        assert engine.planner is not None
        assert engine.executor is not None
        assert engine.reviewer is not None
        assert engine.model_router is not None

        # Submit mission
        mission = engine.submit_mission(
            project_id="test-proj", title="Add docstring",
            description="Add docstring to models.py",
        )
        assert mission.mission_id
        assert mission.status.value == "queued"
        assert mission.scope.value != "unknown"

        # List missions
        missions = engine.list_missions()
        assert len(missions) >= 1

        # Get mission
        m = engine.get_mission(mission.mission_id)
        assert m is not None

        # Estimate
        est = engine.estimate("test-proj", "Fix bug", "Fix login issue")
        assert est.estimated_cost_usd >= 0
        assert est.engine is not None

        # Pause
        ok = engine.pause_mission(mission.mission_id)
        assert ok is True
        m2 = engine.get_mission(mission.mission_id)
        assert m2["status"] == "paused"

        # Resume
        ok2 = engine.resume_mission(mission.mission_id)
        assert ok2 is True

        # Cancel
        ok3 = engine.cancel_mission(mission.mission_id)
        assert ok3 is True

        # Dashboard
        dash = engine.get_dashboard()
        assert dash.total_missions >= 1

        print("  VibeCoderEngine: PASSED")

    finally:
        try:
            os.unlink(db_path)
        except PermissionError:
            pass  # Windows SQLite file lock


# ─── Test: MissionDaemon ─────────────────────────────────────────────────────

def test_mission_daemon():
    """Test the background queue processor."""
    from openclaw.vibecoder.daemon.mission_daemon import MissionDaemon
    from openclaw.vibecoder.vibecoder_engine import VibeCoderEngine

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        engine = VibeCoderEngine(db_path=db_path)
        daemon = MissionDaemon(engine, poll_interval=1, max_concurrent=1)

        assert daemon.poll_interval == 1
        assert daemon.max_concurrent == 1
        assert daemon._running is False

        print("  MissionDaemon: PASSED")

    finally:
        try:
            os.unlink(db_path)
        except PermissionError:
            pass  # Windows SQLite file lock


# ─── Test: ModelRouter (extended) ────────────────────────────────────────────

def test_model_router_integration():
    """Test ModelRouter integration with executor."""
    from openclaw.vibecoder.forge.model_router import ModelRouter, ModelTier
    from openclaw.vibecoder.agents.vibe_executor_agent import VibeExecutorAgent

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        router = ModelRouter(db_path=db_path, monthly_budget=50.0)
        executor = VibeExecutorAgent(model_router=router)

        assert executor._model_router is router

        # Test diverse task routing
        tasks = {
            "Classify intent": ModelTier.HAIKU,
            "Extract data from text": ModelTier.HAIKU,
            "Format JSON output": ModelTier.HAIKU,
            "Review code quality": ModelTier.SONNET,
            "Write blog post about technology": ModelTier.SONNET,
            "Debug error in stack trace": ModelTier.SONNET,
        }
        for task, expected_tier in tasks.items():
            d = router.route(task)
            assert d.model_spec.tier == expected_tier, (
                f"Task '{task}': expected {expected_tier.value}, got {d.model_spec.tier.value}"
            )

        # Test budget pressure affects routing
        # (simulate high spend by recording many outcomes)
        for i in range(20):
            d = router.route("Write comprehensive analysis")
            router.record_outcome(d, 5000, 3000, quality_score=0.9)

        pressure = router._get_budget_pressure()
        assert pressure > 0, "Budget pressure should increase with spend"

        print("  ModelRouter Integration: PASSED")

    finally:
        try:
            os.unlink(db_path)
        except PermissionError:
            pass  # Windows SQLite file lock


# ─── Run all ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("VibeCoder Comprehensive Test Suite")
    print("=" * 60)

    test_models()
    test_project_scout()
    test_code_sentinel()
    test_mission_oracle()
    test_code_smith()
    test_vibe_codex()
    test_vibe_planner()
    test_vibe_reviewer()
    test_code_amplify()
    test_vibecoder_engine()
    test_mission_daemon()
    test_model_router_integration()

    print("=" * 60)
    print("ALL 12 TEST SUITES PASSED")
