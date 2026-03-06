"""Tests for openclaw/automation/profile_sync.py — cross-platform profile sync."""

import pytest

from openclaw.automation.profile_sync import ProfileSync, SyncChange, SyncResult, SyncPlan
from openclaw.forge.platform_codex import PlatformCodex
from openclaw.forge.profile_sentinel import ProfileSentinel
from openclaw.models import AccountStatus, ProfileContent, QualityGrade, SentinelScore


@pytest.fixture
def codex(tmp_path):
    db_path = str(tmp_path / "test_sync.db")
    return PlatformCodex(db_path=db_path)


@pytest.fixture
def sentinel():
    return ProfileSentinel()


@pytest.fixture
def sync(codex, sentinel):
    return ProfileSync(codex=codex, sentinel=sentinel)


class TestPlanSync:
    def test_plan_sync_empty_no_active(self, sync):
        plan = sync.plan_sync({"bio": "New bio text"})
        assert isinstance(plan, SyncPlan)
        assert len(plan.target_platforms) == 0

    def test_plan_sync_with_active_account(self, codex, sync):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        content = ProfileContent(platform_id="gumroad", username="test", bio="Old bio")
        score = SentinelScore(platform_id="gumroad", total_score=50.0, grade=QualityGrade.D)
        codex.store_profile(content, score)

        plan = sync.plan_sync({"bio": "New bio text"})
        assert "gumroad" in plan.target_platforms
        assert len(plan.changes) == 1
        assert plan.changes[0].field_name == "bio"
        assert plan.changes[0].new_value == "New bio text"

    def test_plan_sync_ignores_invalid_fields(self, codex, sync):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        plan = sync.plan_sync({"bio": "New bio", "invalid_field": "ignored"})
        field_names = [c.field_name for c in plan.changes]
        assert "bio" in field_names
        assert "invalid_field" not in field_names


class TestPreviewSync:
    def test_preview_returns_list(self, codex, sync):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        content = ProfileContent(platform_id="gumroad", username="test", bio="Old bio")
        score = SentinelScore(platform_id="gumroad", total_score=50.0, grade=QualityGrade.D)
        codex.store_profile(content, score)

        plan = sync.plan_sync({"bio": "Updated bio"})
        preview = sync.preview_sync(plan)
        assert isinstance(preview, list)
        assert len(preview) == 1
        assert preview[0]["platform_id"] == "gumroad"
        assert "changes" in preview[0]

    def test_preview_shows_diffs(self, codex, sync):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        content = ProfileContent(platform_id="gumroad", username="test", bio="Old bio")
        score = SentinelScore(platform_id="gumroad", total_score=50.0, grade=QualityGrade.D)
        codex.store_profile(content, score)

        plan = sync.plan_sync({"bio": "Brand new bio text"})
        preview = sync.preview_sync(plan)
        assert preview[0]["has_changes"] is True
        assert len(preview[0]["changes"]) > 0


class TestUpdateLocal:
    def test_update_local_updates_codex(self, codex, sync):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        content = ProfileContent(platform_id="gumroad", username="test", bio="Old bio")
        score = SentinelScore(platform_id="gumroad", total_score=50.0, grade=QualityGrade.D)
        codex.store_profile(content, score)

        plan = sync.plan_sync({"bio": "Updated bio"})
        sync.update_local(plan)

        # Verify the profile was updated in the codex
        profile = codex.get_profile("gumroad")
        assert profile is not None
        assert profile["content"]["bio"] == "Updated bio"


class TestGetSyncStatus:
    def test_sync_status_empty(self, sync):
        status = sync.get_sync_status()
        assert isinstance(status, dict)
        assert status["total_active"] == 0
        assert "consistent_fields" in status
        assert "mismatched_fields" in status

    def test_sync_status_with_accounts(self, codex, sync):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        content = ProfileContent(platform_id="gumroad", username="test", bio="Bio")
        score = SentinelScore(platform_id="gumroad", total_score=50.0, grade=QualityGrade.D)
        codex.store_profile(content, score)

        status = sync.get_sync_status()
        assert status["total_active"] == 1


class TestSyncPlanProperties:
    def test_succeeded_count(self):
        plan = SyncPlan(changes=[], target_platforms=["a", "b"])
        plan.results = [
            SyncResult(platform_id="a", platform_name="A", success=True),
            SyncResult(platform_id="b", platform_name="B", success=False),
        ]
        assert plan.succeeded == 1

    def test_failed_count(self):
        plan = SyncPlan(changes=[], target_platforms=["a", "b"])
        plan.results = [
            SyncResult(platform_id="a", platform_name="A", success=True),
            SyncResult(platform_id="b", platform_name="B", success=False),
        ]
        assert plan.failed == 1

    def test_empty_results(self):
        plan = SyncPlan(changes=[], target_platforms=[])
        assert plan.succeeded == 0
        assert plan.failed == 0
