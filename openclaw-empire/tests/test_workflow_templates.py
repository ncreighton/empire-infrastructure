"""Test workflow_templates -- OpenClaw Empire."""
from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch data directories before imports
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_template_dirs(tmp_path, monkeypatch):
    """Redirect all workflow template I/O to temp directory."""
    data_dir = tmp_path / "workflow_templates"
    data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("src.workflow_templates.DATA_DIR", data_dir)
    monkeypatch.setattr("src.workflow_templates.TEMPLATES_FILE", data_dir / "templates.json")
    monkeypatch.setattr("src.workflow_templates.EXECUTION_FILE", data_dir / "execution_records.json")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.workflow_templates import (
    ConditionOperator,
    ExecutionRecord,
    ExecutionStatus,
    MissionType,
    PARAM_RE,
    StepDefinition,
    StepType,
    VALID_SITE_IDS,
    WorkflowTemplate,
    WorkflowTemplateManager,
    get_templates,
)


# ===========================================================================
# MISSION TYPE ENUM
# ===========================================================================


class TestMissionType:

    def test_nine_types(self):
        assert len(list(MissionType)) == 9

    def test_values(self):
        expected = {
            "content_publish", "social_growth", "account_creation",
            "app_exploration", "monetization", "site_maintenance",
            "revenue_check", "device_maintenance", "substack_daily",
        }
        assert {mt.value for mt in MissionType} == expected


# ===========================================================================
# STEP TYPE ENUM
# ===========================================================================


class TestStepType:

    def test_six_step_types(self):
        assert len(list(StepType)) == 6

    def test_values(self):
        expected = {"dispatch", "condition", "parallel", "delay", "notify", "log"}
        assert {st.value for st in StepType} == expected


# ===========================================================================
# STEP DEFINITION DATACLASS
# ===========================================================================


class TestStepDefinition:

    def test_defaults(self):
        sd = StepDefinition(name="Test Step")
        assert sd.step_type == StepType.DISPATCH.value
        assert sd.on_failure == "fail"
        assert sd.max_retries == 0
        assert sd.timeout_seconds == 300

    def test_to_dict_roundtrip(self):
        sd = StepDefinition(
            step_id="step-abc",
            name="Generate Content",
            module="src.content_generator",
            method="generate_article",
            kwargs_template={"site_id": "{param.site_id}", "title": "{param.title}"},
            on_failure="retry",
            max_retries=2,
        )
        d = sd.to_dict()
        restored = StepDefinition.from_dict(d)
        assert restored.step_id == "step-abc"
        assert restored.on_failure == "retry"
        assert restored.kwargs_template["site_id"] == "{param.site_id}"

    def test_from_dict_ignores_unknown(self):
        data = {"name": "Test", "module": "src.test", "unknown_key": "ignored"}
        sd = StepDefinition.from_dict(data)
        assert sd.name == "Test"


# ===========================================================================
# WORKFLOW TEMPLATE DATACLASS
# ===========================================================================


class TestWorkflowTemplate:

    def test_defaults(self):
        wt = WorkflowTemplate()
        assert wt.mission_type == MissionType.CONTENT_PUBLISH.value
        assert wt.version == "1.0.0"
        assert wt.is_builtin is False
        assert wt.steps == []

    def test_to_dict_roundtrip(self):
        wt = WorkflowTemplate(
            template_id="tmpl-test",
            mission_type=MissionType.SOCIAL_GROWTH.value,
            name="Social Growth",
            required_params=["site_id", "platform"],
            tags=["social"],
        )
        d = wt.to_dict()
        restored = WorkflowTemplate.from_dict(d)
        assert restored.template_id == "tmpl-test"
        assert restored.mission_type == "social_growth"
        assert restored.required_params == ["site_id", "platform"]

    def test_get_steps_parses_dicts(self):
        wt = WorkflowTemplate(steps=[
            StepDefinition(name="Step 1", module="mod1").to_dict(),
            StepDefinition(name="Step 2", module="mod2").to_dict(),
        ])
        parsed = wt.get_steps()
        assert len(parsed) == 2
        assert all(isinstance(s, StepDefinition) for s in parsed)
        assert parsed[0].name == "Step 1"


# ===========================================================================
# EXECUTION RECORD DATACLASS
# ===========================================================================


class TestExecutionRecord:

    def test_defaults(self):
        er = ExecutionRecord()
        assert er.status == ExecutionStatus.PENDING.value
        assert er.steps_completed == 0

    def test_roundtrip(self):
        er = ExecutionRecord(
            template_id="tmpl-1",
            mission_type="content_publish",
            status=ExecutionStatus.COMPLETED.value,
            steps_completed=5,
            steps_total=5,
            duration_seconds=45.2,
        )
        d = er.to_dict()
        restored = ExecutionRecord.from_dict(d)
        assert restored.steps_completed == 5
        assert restored.duration_seconds == 45.2


# ===========================================================================
# BUILT-IN TEMPLATES
# ===========================================================================


class TestBuiltinTemplates:
    """Verify all 9 built-in mission templates are created on init."""

    @pytest.fixture
    def mgr(self):
        return WorkflowTemplateManager()

    def test_nine_builtin_templates(self, mgr):
        builtins = mgr.list_templates(builtin_only=True)
        assert len(builtins) == 9

    def test_all_mission_types_have_templates(self, mgr):
        for mt in MissionType:
            template = mgr.get_template(mt)
            assert template is not None, f"No template for {mt.value}"
            assert template.mission_type == mt.value

    def test_content_publish_template_has_steps(self, mgr):
        template = mgr.get_template(MissionType.CONTENT_PUBLISH)
        assert len(template.steps) >= 4
        assert template.required_params == ["site_id", "title"]

    def test_social_growth_template(self, mgr):
        template = mgr.get_template(MissionType.SOCIAL_GROWTH)
        assert "social" in template.tags
        assert template.required_params == ["site_id", "platform"]

    def test_substack_daily_template(self, mgr):
        template = mgr.get_template(MissionType.SUBSTACK_DAILY)
        assert template.is_builtin is True


# ===========================================================================
# PARAMETER SUBSTITUTION
# ===========================================================================


class TestParameterSubstitution:
    """Test {param.xxx} placeholder rendering."""

    @pytest.fixture
    def mgr(self):
        return WorkflowTemplateManager()

    def test_simple_param_substitution(self, mgr):
        template = WorkflowTemplate(
            steps=[
                StepDefinition(
                    name="Test",
                    kwargs_template={"site_id": "{param.site_id}", "title": "{param.title}"},
                ).to_dict(),
            ],
        )
        rendered = mgr.render_steps(template, {"site_id": "witchcraft", "title": "Moon Water"})
        assert rendered[0]["kwargs_template"]["site_id"] == "witchcraft"
        assert rendered[0]["kwargs_template"]["title"] == "Moon Water"

    def test_unresolved_param_stays_as_placeholder(self, mgr):
        template = WorkflowTemplate(
            steps=[
                StepDefinition(
                    name="Test",
                    kwargs_template={"missing": "{param.nonexistent}"},
                ).to_dict(),
            ],
        )
        rendered = mgr.render_steps(template, {})
        assert rendered[0]["kwargs_template"]["missing"] == "{param.nonexistent}"

    def test_inline_substitution(self, mgr):
        template = WorkflowTemplate(
            steps=[
                StepDefinition(
                    name="Test",
                    kwargs_template={
                        "message": "Published '{param.title}' to {param.site_id}",
                    },
                ).to_dict(),
            ],
        )
        rendered = mgr.render_steps(template, {"site_id": "witchcraft", "title": "Moon Water"})
        assert rendered[0]["kwargs_template"]["message"] == "Published 'Moon Water' to witchcraft"

    def test_optional_params_merged(self, mgr):
        template = WorkflowTemplate(
            steps=[
                StepDefinition(
                    name="Test",
                    kwargs_template={"count": "{param.word_count}"},
                ).to_dict(),
            ],
            optional_params={"word_count": 2000},
        )
        rendered = mgr.render_steps(template, {})
        assert rendered[0]["kwargs_template"]["count"] == 2000

    def test_user_params_override_defaults(self, mgr):
        template = WorkflowTemplate(
            steps=[
                StepDefinition(
                    name="Test",
                    kwargs_template={"count": "{param.word_count}"},
                ).to_dict(),
            ],
            optional_params={"word_count": 2000},
        )
        rendered = mgr.render_steps(template, {"word_count": 3500})
        assert rendered[0]["kwargs_template"]["count"] == 3500


# ===========================================================================
# TEMPLATE VALIDATION
# ===========================================================================


class TestTemplateValidation:

    @pytest.fixture
    def mgr(self):
        return WorkflowTemplateManager()

    def test_valid_params_return_no_errors(self, mgr):
        template = mgr.get_template(MissionType.CONTENT_PUBLISH)
        errors = mgr.validate_params(template, {"site_id": "witchcraft", "title": "Moon Water"})
        assert errors == []

    def test_missing_required_param(self, mgr):
        template = mgr.get_template(MissionType.CONTENT_PUBLISH)
        errors = mgr.validate_params(template, {"title": "Moon Water"})
        # Should report site_id as missing
        assert any("site_id" in e for e in errors)

    def test_empty_required_param(self, mgr):
        template = mgr.get_template(MissionType.CONTENT_PUBLISH)
        errors = mgr.validate_params(template, {"site_id": "", "title": "Moon Water"})
        assert any("site_id" in e for e in errors)

    def test_invalid_site_id(self, mgr):
        template = mgr.get_template(MissionType.CONTENT_PUBLISH)
        errors = mgr.validate_params(template, {"site_id": "INVALID", "title": "Test"})
        assert any("Unknown site_id" in e for e in errors)


# ===========================================================================
# EXECUTION TRACKING
# ===========================================================================


class TestExecutionTracking:

    @pytest.fixture
    def mgr(self):
        return WorkflowTemplateManager()

    def test_record_execution(self, mgr):
        record = mgr.record_execution(
            template_id="tmpl-1",
            mission_type="content_publish",
            params={"site_id": "witchcraft"},
            status=ExecutionStatus.COMPLETED.value,
            steps_completed=5,
            steps_total=5,
            duration_seconds=30.0,
        )
        assert isinstance(record, ExecutionRecord)
        assert record.status == "completed"

    def test_get_execution_history(self, mgr):
        # Record a few executions
        for i in range(3):
            mgr.record_execution(
                template_id=f"tmpl-{i}",
                mission_type="content_publish",
                params={},
                status="completed",
                duration_seconds=float(i * 10),
            )
        history = mgr.get_execution_history()
        assert len(history) == 3

    def test_get_execution_history_with_filter(self, mgr):
        mgr.record_execution(
            template_id="tmpl-1", mission_type="content_publish",
            params={}, status="completed",
        )
        mgr.record_execution(
            template_id="tmpl-2", mission_type="social_growth",
            params={}, status="failed",
        )
        filtered = mgr.get_execution_history(mission_type="social_growth")
        assert len(filtered) == 1
        assert filtered[0].mission_type == "social_growth"

    def test_get_execution_stats(self, mgr):
        mgr.record_execution(
            template_id="tmpl-1", mission_type="content_publish",
            params={}, status="completed", duration_seconds=10.0,
        )
        stats = mgr.get_execution_stats()
        assert stats["total_executions"] == 1
        assert stats["templates_count"] >= 9  # 9 builtins


# ===========================================================================
# CUSTOM TEMPLATE CRUD
# ===========================================================================


class TestCustomTemplateCRUD:

    @pytest.fixture
    def mgr(self):
        return WorkflowTemplateManager()

    def test_create_custom(self, mgr):
        template = mgr.create_custom(
            mission_type="custom_workflow",
            name="My Custom Workflow",
            description="A test workflow",
            required_params=["target"],
            tags=["custom"],
        )
        assert template.is_builtin is False
        assert template.name == "My Custom Workflow"

    def test_create_custom_empty_name_raises(self, mgr):
        with pytest.raises(ValueError, match="must not be empty"):
            mgr.create_custom(mission_type="test", name="")

    def test_delete_custom(self, mgr):
        template = mgr.create_custom(mission_type="test", name="Deletable")
        deleted = mgr.delete_custom(template.template_id)
        assert deleted is True
        assert mgr.get_template_by_id(template.template_id) is None

    def test_delete_nonexistent_returns_false(self, mgr):
        assert mgr.delete_custom("nonexistent-id") is False

    def test_delete_builtin_raises(self, mgr):
        builtin = mgr.get_template(MissionType.CONTENT_PUBLISH)
        with pytest.raises(ValueError, match="Cannot delete"):
            mgr.delete_custom(builtin.template_id)


# ===========================================================================
# EXPORT / IMPORT
# ===========================================================================


class TestExportImport:

    @pytest.fixture
    def mgr(self):
        return WorkflowTemplateManager()

    def test_export_template(self, mgr):
        builtin = mgr.get_template(MissionType.CONTENT_PUBLISH)
        data = mgr.export_template(builtin.template_id)
        assert isinstance(data, dict)
        assert data["name"] == builtin.name

    def test_export_nonexistent_raises(self, mgr):
        with pytest.raises(KeyError):
            mgr.export_template("nonexistent")

    def test_import_template(self, mgr):
        data = {
            "mission_type": "custom",
            "name": "Imported Workflow",
            "steps": [],
            "is_builtin": False,
        }
        imported = mgr.import_template(data)
        assert imported.name == "Imported Workflow"
        assert imported.is_builtin is False

    def test_import_never_overwrites_builtin(self, mgr):
        builtin = mgr.get_template(MissionType.CONTENT_PUBLISH)
        data = builtin.to_dict()
        data["name"] = "Modified Builtin"
        imported = mgr.import_template(data)
        # Should have been assigned a new ID
        assert imported.template_id != builtin.template_id
        assert imported.is_builtin is False


# ===========================================================================
# SERIALIZATION / DESERIALIZATION
# ===========================================================================


class TestSerialization:

    @pytest.fixture
    def mgr(self):
        return WorkflowTemplateManager()

    def test_template_json_serializable(self, mgr):
        template = mgr.get_template(MissionType.CONTENT_PUBLISH)
        d = template.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        restored = WorkflowTemplate.from_dict(deserialized)
        assert restored.name == template.name

    def test_execution_record_json_serializable(self):
        record = ExecutionRecord(
            template_id="test",
            mission_type="content_publish",
            params={"site_id": "witchcraft"},
            status="completed",
        )
        serialized = json.dumps(record.to_dict())
        assert isinstance(serialized, str)


# ===========================================================================
# SINGLETON
# ===========================================================================


class TestGetTemplates:

    def test_returns_manager_instance(self):
        mgr = get_templates()
        assert isinstance(mgr, WorkflowTemplateManager)
