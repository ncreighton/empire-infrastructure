"""Test prompt_library — OpenClaw Empire."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_data(tmp_path, monkeypatch):
    """Redirect prompt library data to temp dir."""
    monkeypatch.setattr("src.prompt_library.PROMPT_DATA_DIR", tmp_path / "prompts")
    monkeypatch.setattr("src.prompt_library.TEMPLATES_FILE", tmp_path / "prompts" / "templates.json")
    monkeypatch.setattr("src.prompt_library.USAGE_LOG_FILE", tmp_path / "prompts" / "usage_log.json")
    (tmp_path / "prompts").mkdir(parents=True, exist_ok=True)
    # Reset singleton
    import src.prompt_library as pl_mod
    pl_mod._prompt_library = None
    yield


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from src.prompt_library import (
    PromptCategory,
    PromptLibrary,
    PromptModel,
    PromptTemplate,
    PromptVariant,
    VariantStatus,
    get_prompt_library,
    _extract_variables,
    _render_template,
    _validate_variables,
)


# ===================================================================
# Enum tests
# ===================================================================

class TestPromptEnums:
    """Test all enum values."""

    def test_prompt_category_has_10_values(self):
        categories = list(PromptCategory)
        assert len(categories) == 10
        assert PromptCategory.CONTENT.value == "content"
        assert PromptCategory.SEO.value == "seo"
        assert PromptCategory.SOCIAL.value == "social"
        assert PromptCategory.VOICE.value == "voice"
        assert PromptCategory.RESEARCH.value == "research"
        assert PromptCategory.CLASSIFICATION.value == "classification"
        assert PromptCategory.NEWSLETTER.value == "newsletter"
        assert PromptCategory.VISION.value == "vision"
        assert PromptCategory.CONVERSATION.value == "conversation"
        assert PromptCategory.SYSTEM.value == "system"

    def test_prompt_model_values(self):
        assert PromptModel.HAIKU.value == "claude-haiku-4-5-20251001"
        assert PromptModel.SONNET.value == "claude-sonnet-4-20250514"
        assert PromptModel.OPUS.value == "claude-opus-4-20250514"

    def test_variant_status_values(self):
        assert VariantStatus.ACTIVE.value == "active"
        assert VariantStatus.CHAMPION.value == "champion"
        assert VariantStatus.CHALLENGER.value == "challenger"
        assert VariantStatus.RETIRED.value == "retired"


# ===================================================================
# PromptVariant
# ===================================================================

class TestPromptVariant:
    def test_create_variant(self):
        v = PromptVariant(
            variant_id="v1",
            template="Write about {topic}",
            status=VariantStatus.CHAMPION,
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert v.variant_id == "v1"
        assert v.usage_count == 0
        assert v.success_rate == 0.0

    def test_update_metrics(self):
        v = PromptVariant(
            variant_id="v1", template="t", status=VariantStatus.ACTIVE,
            created_at="2026-01-01",
        )
        v.update_metrics(success=True, quality_score=0.9, latency_ms=1000, token_cost=0.01)
        assert v.usage_count == 1
        assert v.success_count == 1
        assert v.avg_quality_score == 0.9
        assert v.avg_latency_ms == 1000
        assert v.avg_token_cost == 0.01

        v.update_metrics(success=False, quality_score=0.5, latency_ms=2000, token_cost=0.02)
        assert v.usage_count == 2
        assert v.success_count == 1
        assert v.success_rate == 0.5

    def test_to_dict_and_from_dict_roundtrip(self):
        v = PromptVariant(
            variant_id="v2", template="Test {var}",
            status=VariantStatus.CHALLENGER, created_at="2026-02-01",
            notes="Testing variant",
        )
        d = v.to_dict()
        assert d["status"] == "challenger"
        restored = PromptVariant.from_dict(d)
        assert restored.variant_id == "v2"
        assert restored.status == VariantStatus.CHALLENGER


# ===================================================================
# Variable extraction and rendering
# ===================================================================

class TestVariableHelpers:
    def test_extract_variables(self):
        text = "Hello {name}, your topic is {topic}. Count: {count}"
        variables = _extract_variables(text)
        assert variables == ["count", "name", "topic"]

    def test_extract_variables_ignores_escaped_braces(self):
        text = "{{escaped}} but {real_var} here"
        variables = _extract_variables(text)
        assert variables == ["real_var"]

    def test_render_template_substitutes_variables(self):
        text = "Article about {topic} for {site}"
        result = _render_template(text, {"topic": "Moon Rituals", "site": "WitchcraftForBeginners"})
        assert result == "Article about Moon Rituals for WitchcraftForBeginners"

    def test_render_template_missing_var_raises(self):
        text = "Article about {topic} for {site}"
        with pytest.raises(KeyError, match="site"):
            _render_template(text, {"topic": "only-topic"})

    def test_validate_variables_missing_raises(self):
        text = "{title} and {keyword}"
        with pytest.raises(ValueError, match="keyword"):
            _validate_variables(text, {"title": "Moon"}, "test_template")

    def test_validate_variables_all_present(self):
        text = "{title} and {keyword}"
        # Should not raise
        _validate_variables(text, {"title": "Moon", "keyword": "ritual"}, "test")


# ===================================================================
# PromptTemplate
# ===================================================================

class TestPromptTemplate:
    def test_to_dict_and_from_dict_roundtrip(self):
        v1 = PromptVariant(
            variant_id="v1", template="Write {topic}",
            status=VariantStatus.CHAMPION, created_at="2026-01-01",
        )
        t = PromptTemplate(
            template_id="content.test",
            name="Test Template",
            category=PromptCategory.CONTENT,
            description="A test prompt",
            model=PromptModel.SONNET,
            max_tokens=2000,
            variables=["topic"],
            variants={"v1": v1},
            active_variant="v1",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        d = t.to_dict()
        assert d["category"] == "content"
        assert d["model"] == "claude-sonnet-4-20250514"

        restored = PromptTemplate.from_dict(d)
        assert restored.template_id == "content.test"
        assert restored.category == PromptCategory.CONTENT
        assert restored.model == PromptModel.SONNET

    def test_champion_property(self):
        v1 = PromptVariant(
            variant_id="v1", template="t1",
            status=VariantStatus.CHAMPION, created_at="",
        )
        v2 = PromptVariant(
            variant_id="v2", template="t2",
            status=VariantStatus.CHALLENGER, created_at="",
        )
        t = PromptTemplate(
            template_id="test", name="test", category=PromptCategory.SEO,
            description="", model=PromptModel.HAIKU, max_tokens=100,
            variables=[], variants={"v1": v1, "v2": v2},
            active_variant="v1",
        )
        assert t.champion.variant_id == "v1"
        assert t.challenger.variant_id == "v2"

    def test_total_usage(self):
        v1 = PromptVariant(
            variant_id="v1", template="t", status=VariantStatus.CHAMPION,
            created_at="", usage_count=10,
        )
        v2 = PromptVariant(
            variant_id="v2", template="t", status=VariantStatus.ACTIVE,
            created_at="", usage_count=5,
        )
        t = PromptTemplate(
            template_id="test", name="test", category=PromptCategory.CONTENT,
            description="", model=PromptModel.SONNET, max_tokens=100,
            variables=[], variants={"v1": v1, "v2": v2},
            active_variant="v1",
        )
        assert t.total_usage == 15


# ===================================================================
# PromptLibrary — registration and rendering
# ===================================================================

class TestPromptLibrary:
    def test_register_template(self):
        lib = PromptLibrary()
        t = lib.register(
            template_id="content.test",
            name="Test Prompt",
            category=PromptCategory.CONTENT,
            template_text="Write an article about {topic} targeting {keyword}.",
            model=PromptModel.SONNET,
            max_tokens=2000,
            description="Test description",
        )
        assert t.template_id == "content.test"
        assert t.variables == ["keyword", "topic"]
        assert "v1" in t.variants
        assert t.variants["v1"].status == VariantStatus.CHAMPION

    def test_register_duplicate_raises(self):
        lib = PromptLibrary()
        lib.register(
            template_id="dup.test", name="D", category=PromptCategory.SEO,
            template_text="t", model=PromptModel.HAIKU, max_tokens=100,
        )
        with pytest.raises(ValueError, match="already exists"):
            lib.register(
                template_id="dup.test", name="D2", category=PromptCategory.SEO,
                template_text="t2", model=PromptModel.HAIKU, max_tokens=100,
            )

    def test_get_template(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.get", name="G", category=PromptCategory.CONTENT,
            template_text="t", model=PromptModel.SONNET, max_tokens=100,
        )
        t = lib.get_template("content.get")
        assert t.name == "G"

    def test_get_template_not_found_raises(self):
        lib = PromptLibrary()
        with pytest.raises(KeyError, match="not found"):
            lib.get_template("nonexistent")

    def test_list_templates_by_category(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.a", name="A", category=PromptCategory.CONTENT,
            template_text="t", model=PromptModel.SONNET, max_tokens=100,
        )
        lib.register(
            template_id="seo.b", name="B", category=PromptCategory.SEO,
            template_text="t", model=PromptModel.HAIKU, max_tokens=100,
        )
        content_only = lib.list_templates(category=PromptCategory.CONTENT)
        assert len(content_only) == 1
        assert content_only[0].template_id == "content.a"

    def test_delete_template(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.del", name="D", category=PromptCategory.CONTENT,
            template_text="t", model=PromptModel.SONNET, max_tokens=100,
        )
        lib.delete_template("content.del")
        with pytest.raises(KeyError):
            lib.get_template("content.del")

    def test_render_sync(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.render",
            name="Render Test",
            category=PromptCategory.CONTENT,
            template_text="Write about {topic} for keyword {keyword}.",
            model=PromptModel.SONNET,
            max_tokens=2000,
        )
        rendered, variant_id = lib.render_sync(
            "content.render", topic="Moon Water", keyword="moon water ritual",
        )
        assert "Moon Water" in rendered
        assert "moon water ritual" in rendered
        assert variant_id == "v1"

    def test_render_missing_variable_raises(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.vars",
            name="Vars Test",
            category=PromptCategory.CONTENT,
            template_text="Article: {title} about {keyword}.",
            model=PromptModel.SONNET,
            max_tokens=1000,
        )
        with pytest.raises((ValueError, KeyError)):
            lib.render_sync("content.vars", title="Only Title")

    def test_search_templates(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.moon", name="Moon Article",
            category=PromptCategory.CONTENT, template_text="t",
            model=PromptModel.SONNET, max_tokens=100,
            description="Generate moon phase content",
        )
        results = lib.search("moon")
        assert len(results) == 1
        assert results[0].template_id == "content.moon"


# ===================================================================
# Variant management
# ===================================================================

class TestVariantManagement:
    def test_add_variant(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.ab",
            name="AB Test",
            category=PromptCategory.CONTENT,
            template_text="Original: {topic}",
            model=PromptModel.SONNET,
            max_tokens=1000,
        )
        v2 = lib.add_variant(
            "content.ab", "v2", "Improved: {topic}", notes="Better intro",
        )
        assert v2.variant_id == "v2"
        t = lib.get_template("content.ab")
        assert "v2" in t.variants

    def test_add_duplicate_variant_raises(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.dup", name="D", category=PromptCategory.CONTENT,
            template_text="t", model=PromptModel.SONNET, max_tokens=100,
        )
        with pytest.raises(ValueError, match="already exists"):
            lib.add_variant("content.dup", "v1", "duplicate")


# ===================================================================
# A/B Testing
# ===================================================================

class TestABTesting:
    def _setup_ab_lib(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.ab",
            name="AB",
            category=PromptCategory.CONTENT,
            template_text="Champion: {topic}",
            model=PromptModel.SONNET,
            max_tokens=1000,
        )
        lib.add_variant("content.ab", "v2", "Challenger: {topic}")
        return lib

    def test_start_ab_test(self):
        lib = self._setup_ab_lib()
        lib.start_ab_test("content.ab", "v2", split=0.3)
        t = lib.get_template("content.ab")
        assert t.ab_test_active is True
        assert t.ab_test_split == 0.3
        assert t.variants["v1"].status == VariantStatus.CHAMPION
        assert t.variants["v2"].status == VariantStatus.CHALLENGER

    def test_start_ab_test_invalid_split_raises(self):
        lib = self._setup_ab_lib()
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            lib.start_ab_test("content.ab", "v2", split=0.0)

    def test_start_ab_test_same_variant_raises(self):
        lib = self._setup_ab_lib()
        with pytest.raises(ValueError, match="already the active"):
            lib.start_ab_test("content.ab", "v1", split=0.5)

    def test_stop_ab_test_winner_champion(self):
        lib = self._setup_ab_lib()
        lib.start_ab_test("content.ab", "v2", split=0.5)
        lib.stop_ab_test("content.ab", "v1")
        t = lib.get_template("content.ab")
        assert t.ab_test_active is False
        assert t.active_variant == "v1"
        assert t.variants["v1"].status == VariantStatus.CHAMPION
        assert t.variants["v2"].status == VariantStatus.RETIRED

    def test_stop_ab_test_winner_challenger(self):
        lib = self._setup_ab_lib()
        lib.start_ab_test("content.ab", "v2", split=0.5)
        lib.stop_ab_test("content.ab", "v2")
        t = lib.get_template("content.ab")
        assert t.ab_test_active is False
        assert t.active_variant == "v2"
        assert t.variants["v2"].status == VariantStatus.CHAMPION
        assert t.variants["v1"].status == VariantStatus.RETIRED

    def test_get_ab_results(self):
        lib = self._setup_ab_lib()
        lib.start_ab_test("content.ab", "v2", split=0.5)
        results = lib.get_ab_results("content.ab")
        assert results["ab_test_active"] is True
        assert results["champion"] is not None
        assert results["challenger"] is not None
        assert results["recommendation"] == "insufficient_data"

    def test_ab_test_variant_selection_respects_split(self):
        lib = self._setup_ab_lib()
        lib.start_ab_test("content.ab", "v2", split=0.5)
        t = lib.get_template("content.ab")
        # Run selection many times and check both variants are selected
        selections = set()
        for _ in range(100):
            variant = lib._select_variant(t)
            selections.add(variant.variant_id)
        assert "v1" in selections
        assert "v2" in selections


# ===================================================================
# Result tracking
# ===================================================================

class TestResultTracking:
    def test_record_result_updates_metrics(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.track",
            name="Tracking",
            category=PromptCategory.CONTENT,
            template_text="Write {topic}",
            model=PromptModel.SONNET,
            max_tokens=1000,
        )
        lib.record_result(
            "content.track", "v1",
            success=True, quality_score=0.92, latency_ms=3400, token_cost=0.012,
        )
        t = lib.get_template("content.track")
        v = t.variants["v1"]
        assert v.usage_count == 1
        assert v.success_count == 1
        assert v.avg_quality_score == pytest.approx(0.92, abs=0.01)


# ===================================================================
# Prompt versioning
# ===================================================================

class TestPromptVersioning:
    def test_version_increments_on_variant_add(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.ver",
            name="Ver",
            category=PromptCategory.CONTENT,
            template_text="t",
            model=PromptModel.SONNET,
            max_tokens=100,
        )
        t = lib.get_template("content.ver")
        assert t.version == 1
        lib.add_variant("content.ver", "v2", "t2")
        assert t.version == 2


# ===================================================================
# Export / Import
# ===================================================================

class TestExportImport:
    def test_export_template(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.exp",
            name="Export",
            category=PromptCategory.CONTENT,
            template_text="Write {topic}",
            model=PromptModel.SONNET,
            max_tokens=2000,
        )
        data = lib.export_template("content.exp")
        assert data["template_id"] == "content.exp"
        assert "_export_meta" in data

    def test_import_template(self):
        lib = PromptLibrary()
        data = {
            "template_id": "imported.test",
            "name": "Imported",
            "category": "seo",
            "description": "imported prompt",
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 500,
            "variables": ["keyword"],
            "variants": {
                "v1": {
                    "variant_id": "v1",
                    "template": "SEO for {keyword}",
                    "status": "champion",
                    "created_at": "2026-01-01",
                }
            },
            "active_variant": "v1",
        }
        t = lib.import_template(data)
        assert t.template_id == "imported.test"
        assert t.category == PromptCategory.SEO


# ===================================================================
# Seed defaults and all categories
# ===================================================================

class TestSeedDefaults:
    def test_seed_defaults_creates_templates(self):
        lib = PromptLibrary()
        created = lib.seed_defaults()
        assert created > 0
        templates = lib.list_templates()
        assert len(templates) >= 10

    def test_seed_defaults_idempotent(self):
        lib = PromptLibrary()
        first = lib.seed_defaults()
        second = lib.seed_defaults()
        assert second == 0  # no new ones created

    def test_all_10_categories_have_templates_after_seed(self):
        lib = PromptLibrary()
        lib.seed_defaults()
        categories_seen = set()
        for t in lib.list_templates():
            categories_seen.add(t.category)
        for cat in PromptCategory:
            assert cat in categories_seen, f"Category {cat.value} missing from seeded defaults"


# ===================================================================
# Statistics
# ===================================================================

class TestStats:
    def test_get_stats(self):
        lib = PromptLibrary()
        lib.register(
            template_id="content.stat",
            name="Stat",
            category=PromptCategory.CONTENT,
            template_text="t",
            model=PromptModel.SONNET,
            max_tokens=100,
        )
        stats = lib.get_stats()
        assert stats["total_templates"] == 1
        assert stats["total_variants"] == 1
        assert "by_category" in stats
        assert "by_model" in stats


# ===================================================================
# Singleton
# ===================================================================

class TestSingleton:
    def test_get_prompt_library_returns_same_instance(self):
        lib1 = get_prompt_library()
        lib2 = get_prompt_library()
        assert lib1 is lib2
