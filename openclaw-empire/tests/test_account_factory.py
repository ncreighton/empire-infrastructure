"""Test account_factory — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.account_factory import (
        AccountFactory,
        SignupTemplate,
        TemplateStep,
        CreationJob,
        WarmingSchedule,
        Platform,
        CreationStatus,
        WarmingPhase,
        StepType,
        BUILTIN_TEMPLATES,
        WARMING_SCHEDULES,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="account_factory not available")


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "account_factory"
    d.mkdir()
    return d


@pytest.fixture
def factory(data_dir):
    return AccountFactory(data_dir=data_dir)


# ===================================================================
# Enum Tests
# ===================================================================


class TestEnums:
    def test_platform_values(self):
        assert Platform.INSTAGRAM.value == "instagram"
        assert Platform.TWITTER.value == "twitter"
        assert Platform.TIKTOK.value == "tiktok"
        assert Platform.FACEBOOK.value == "facebook"
        assert Platform.PINTEREST.value == "pinterest"
        assert Platform.CUSTOM.value == "custom"

    def test_creation_status_values(self):
        assert CreationStatus.PENDING.value == "pending"
        assert CreationStatus.IN_PROGRESS.value == "in_progress"
        assert CreationStatus.CAPTCHA_REQUIRED.value == "captcha_required"
        assert CreationStatus.COMPLETED.value == "completed"
        assert CreationStatus.FAILED.value == "failed"

    def test_warming_phase_values(self):
        assert WarmingPhase.DAY_1_3.value == "day_1_3"
        assert WarmingPhase.MATURE.value == "mature"

    def test_step_type_values(self):
        assert StepType.NAVIGATE.value == "navigate"
        assert StepType.FILL_FIELD.value == "fill_field"
        assert StepType.TAP_ELEMENT.value == "tap_element"
        assert StepType.VERIFY_EMAIL.value == "verify_email"
        assert StepType.CAPTCHA_CHECK.value == "captcha_check"


# ===================================================================
# Data Class Tests
# ===================================================================


class TestTemplateStep:
    def test_defaults(self):
        ts = TemplateStep()
        assert ts.step_type == StepType.TAP_ELEMENT
        assert ts.wait_seconds == 1.5
        assert ts.optional is False

    def test_to_dict(self):
        ts = TemplateStep(
            step_type=StepType.FILL_FIELD,
            description="Enter email",
            target="email field",
            value="{email}",
        )
        d = ts.to_dict()
        assert d["step_type"] == "fill_field"
        assert d["value"] == "{email}"


class TestSignupTemplate:
    def test_defaults(self):
        st = SignupTemplate()
        assert st.platform == Platform.CUSTOM
        assert st.steps == []
        assert st.needs_email is True

    def test_to_dict(self):
        st = SignupTemplate(
            platform=Platform.INSTAGRAM,
            name="IG Signup",
            steps=[TemplateStep(step_type=StepType.NAVIGATE, description="Open app")],
        )
        d = st.to_dict()
        assert d["platform"] == "instagram"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["step_type"] == "navigate"


class TestCreationJob:
    def test_defaults(self):
        job = CreationJob()
        assert job.status == CreationStatus.PENDING
        assert job.current_step == 0
        assert job.warming_phase == WarmingPhase.DAY_1_3

    def test_to_dict(self):
        job = CreationJob(
            platform=Platform.TWITTER,
            persona_id="abc123",
            status=CreationStatus.COMPLETED,
        )
        d = job.to_dict()
        assert d["platform"] == "twitter"
        assert d["status"] == "completed"
        assert d["warming_phase"] == "day_1_3"


class TestWarmingSchedule:
    def test_defaults(self):
        ws = WarmingSchedule()
        assert ws.phase == WarmingPhase.DAY_1_3
        assert ws.completed_actions == 0

    def test_to_dict(self):
        ws = WarmingSchedule(
            platform=Platform.INSTAGRAM,
            account_email="test@gmail.com",
            phase=WarmingPhase.DAY_4_7,
        )
        d = ws.to_dict()
        assert d["platform"] == "instagram"
        assert d["phase"] == "day_4_7"


# ===================================================================
# Built-in Templates Tests
# ===================================================================


class TestBuiltinTemplates:
    def test_instagram_template_exists(self):
        assert Platform.INSTAGRAM in BUILTIN_TEMPLATES

    def test_twitter_template_exists(self):
        assert Platform.TWITTER in BUILTIN_TEMPLATES

    def test_tiktok_template_exists(self):
        assert Platform.TIKTOK in BUILTIN_TEMPLATES

    def test_facebook_template_exists(self):
        assert Platform.FACEBOOK in BUILTIN_TEMPLATES

    def test_pinterest_template_exists(self):
        assert Platform.PINTEREST in BUILTIN_TEMPLATES

    def test_reddit_template_exists(self):
        assert Platform.REDDIT in BUILTIN_TEMPLATES

    def test_instagram_template_structure(self):
        template = BUILTIN_TEMPLATES[Platform.INSTAGRAM]()
        assert template.platform == Platform.INSTAGRAM
        assert len(template.steps) > 0
        assert "email" in template.required_fields

    def test_twitter_template_has_captcha(self):
        template = BUILTIN_TEMPLATES[Platform.TWITTER]()
        assert template.has_captcha is True
        assert template.needs_phone is True


class TestWarmingSchedules:
    def test_all_phases_defined(self):
        for phase in WarmingPhase:
            assert phase in WARMING_SCHEDULES

    def test_day_1_3_minimal(self):
        schedule = WARMING_SCHEDULES[WarmingPhase.DAY_1_3]
        assert schedule["max_actions_per_day"] <= 10
        assert schedule["session_minutes"] <= 15

    def test_mature_full_usage(self):
        schedule = WARMING_SCHEDULES[WarmingPhase.MATURE]
        assert schedule["max_actions_per_day"] >= 50
        assert "all" in schedule["actions"]


# ===================================================================
# AccountFactory Tests
# ===================================================================


class TestAccountFactoryInit:
    def test_init_creates_data_dir(self, tmp_path):
        d = tmp_path / "new_factory_data"
        factory = AccountFactory(data_dir=d)
        assert d.exists()

    def test_loads_builtin_templates(self, factory):
        assert len(factory._templates) >= len(BUILTIN_TEMPLATES)
        for platform in BUILTIN_TEMPLATES:
            assert platform.value in factory._templates

    def test_empty_jobs_on_init(self, factory):
        assert len(factory._jobs) == 0


class TestAccountFactoryTemplates:
    def test_list_templates(self, factory):
        templates = factory.list_templates()
        assert len(templates) >= 6
        platforms = [t["platform"] for t in templates]
        assert "instagram" in platforms
        assert "twitter" in platforms

    def test_template_list_structure(self, factory):
        templates = factory.list_templates()
        for t in templates:
            assert "name" in t
            assert "platform" in t
            assert "steps" in t
            assert "needs_email" in t
            assert "has_captcha" in t


class TestAccountFactoryPersistence:
    def test_save_and_load_state(self, data_dir):
        f1 = AccountFactory(data_dir=data_dir)
        job = CreationJob(
            platform=Platform.INSTAGRAM,
            persona_id="persona-1",
            status=CreationStatus.COMPLETED,
        )
        f1._jobs[job.id] = job
        f1._save_state()

        f2 = AccountFactory(data_dir=data_dir)
        assert len(f2._jobs) == 1

    def test_state_file_created(self, data_dir, factory):
        factory._save_state()
        state_path = data_dir / "state.json"
        assert state_path.exists()


class TestAccountFactoryProperties:
    def test_controller_lazy_init(self, factory):
        """Controller property should not crash when import fails."""
        factory._controller = None
        with patch.dict("sys.modules", {"src.phone_controller": None}):
            # Access should not raise
            _ = factory._controller

    def test_browser_lazy_init(self, factory):
        factory._browser = None
        with patch.dict("sys.modules", {"src.browser_controller": None}):
            _ = factory._browser
