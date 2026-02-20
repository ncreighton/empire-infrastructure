"""
Tests for ZimmWriter Controller.
Run: python -m pytest tests/ -v
Note: Most tests require ZimmWriter to be running.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.site_presets import SITE_PRESETS, get_preset, get_all_domains, get_presets_by_niche


class TestSitePresets:
    def test_all_14_sites_configured(self):
        assert len(SITE_PRESETS) >= 14, f"Only {len(SITE_PRESETS)} sites configured"

    def test_each_preset_has_required_fields(self):
        required = ["domain", "niche", "h2_count", "section_length", "voice", "ai_model"]
        for domain, config in SITE_PRESETS.items():
            for field in required:
                assert field in config, f"{domain} missing field: {field}"

    def test_get_preset_returns_config(self):
        config = get_preset("smarthomewizards.com")
        assert config is not None
        assert config["niche"] == "Smart Home Automation"

    def test_get_preset_returns_none_for_unknown(self):
        assert get_preset("nonexistent.com") is None

    def test_get_all_domains(self):
        domains = get_all_domains()
        assert "witchcraftforbeginners.com" in domains
        assert "smarthomewizards.com" in domains
        assert len(domains) >= 14

    def test_get_presets_by_niche(self):
        ai_sites = get_presets_by_niche("AI")
        assert len(ai_sites) >= 3  # aiinactionhub, aidiscoverydigest, clearainews, etc.

    def test_witchcraft_preserves_voice(self):
        config = get_preset("witchcraftforbeginners.com")
        assert config["serp_scraping"] == True  # SERP enabled on all sites
        assert config["literary_devices"] == True

    def test_all_sites_use_claude_model(self):
        for domain, config in SITE_PRESETS.items():
            assert "Claude" in config["ai_model"], f"{domain} not using Claude"


class TestArticleTypes:
    def test_classify_how_to(self):
        from src.article_types import classify_title
        assert classify_title("How to Set Up Your Smart Home") == "how_to"

    def test_classify_listicle(self):
        from src.article_types import classify_title
        assert classify_title("10 Best Smart Plugs for 2026") == "listicle"

    def test_classify_review(self):
        from src.article_types import classify_title
        assert classify_title("Ring vs Nest Doorbell Comparison") == "review"

    def test_classify_guide(self):
        from src.article_types import classify_title
        assert classify_title("Complete Guide to Home Automation") == "guide"

    def test_classify_news(self):
        from src.article_types import classify_title
        assert classify_title("Google Announces New Nest Hub Features") == "news"

    def test_classify_informational_default(self):
        from src.article_types import classify_title
        assert classify_title("What Is a Smart Thermostat") == "informational"

    def test_classify_titles_batch(self):
        from src.article_types import classify_titles
        results = classify_titles(["How to Cook Rice", "10 Best Laptops"])
        assert results["How to Cook Rice"] == "how_to"
        assert results["10 Best Laptops"] == "listicle"

    def test_get_settings_overrides(self):
        from src.article_types import get_settings_overrides
        overrides = get_settings_overrides("how_to")
        assert "h2_lower_limit" in overrides or "section_length" in overrides

    def test_get_dominant_type(self):
        from src.article_types import get_dominant_type
        titles = ["How to A", "How to B", "10 Best C"]
        assert get_dominant_type(titles) == "how_to"


class TestOutlineTemplates:
    def test_all_types_have_templates(self):
        from src.outline_templates import OUTLINE_TEMPLATES
        expected_types = ["how_to", "listicle", "review", "guide", "news", "informational"]
        for t in expected_types:
            assert t in OUTLINE_TEMPLATES, f"Missing template for: {t}"
            assert len(OUTLINE_TEMPLATES[t]) >= 1

    def test_get_random_template(self):
        from src.outline_templates import get_random_template
        template = get_random_template("how_to")
        assert isinstance(template, str)
        assert len(template) > 20

    def test_rotate_template_deterministic(self):
        from src.outline_templates import rotate_template
        t1 = rotate_template("listicle", 42)
        t2 = rotate_template("listicle", 42)
        assert t1 == t2  # Same index = same template


class TestCampaignEngine:
    def test_plan_campaign(self):
        from src.campaign_engine import CampaignEngine
        engine = CampaignEngine()
        plan = engine.plan_campaign("smarthomewizards.com", [
            "How to Set Up Alexa",
            "10 Best Smart Plugs",
        ])
        assert plan.domain == "smarthomewizards.com"
        assert len(plan.titles) == 2
        assert plan.dominant_type in ("how_to", "listicle")
        assert plan.outline_template  # non-empty

    def test_plan_and_generate_csv(self):
        import os, tempfile
        from src.campaign_engine import CampaignEngine
        engine = CampaignEngine()
        plan, csv_path = engine.plan_and_generate("smarthomewizards.com", [
            "How to Set Up a Smart Speaker",
        ])
        assert os.path.exists(csv_path)
        assert csv_path.endswith(".csv")
        # Clean up
        os.remove(csv_path)

    def test_get_campaign_summary(self):
        from src.campaign_engine import CampaignEngine
        engine = CampaignEngine()
        plan = engine.plan_campaign("smarthomewizards.com", ["How to X"])
        summary = engine.get_campaign_summary(plan)
        assert "title_count" in summary
        assert summary["title_count"] == 1


class TestScreenNavigator:
    def test_screen_enum(self):
        from src.screen_navigator import Screen
        assert Screen.MENU.value == "menu"
        assert Screen.BULK_WRITER.value == "bulk_writer"

    def test_detect_screen_from_title(self):
        from src.screen_navigator import detect_screen_from_title, Screen
        assert detect_screen_from_title("ZimmWriter v10.869: Menu") == Screen.MENU
        assert detect_screen_from_title("ZimmWriter v10.869: Bulk Blog Writer") == Screen.BULK_WRITER
        assert detect_screen_from_title("ZimmWriter v10.869: SEO Writer") == Screen.SEO_WRITER
        assert detect_screen_from_title("") == Screen.UNKNOWN

    def test_menu_buttons_complete(self):
        from src.screen_navigator import MENU_BUTTONS, Screen
        assert Screen.BULK_WRITER in MENU_BUTTONS
        assert Screen.SEO_WRITER in MENU_BUTTONS
        assert Screen.AI_VAULT in MENU_BUTTONS
        assert len(MENU_BUTTONS) == 12

    def test_get_available_screens(self):
        # Can't instantiate ScreenNavigator without controller, but test constants
        from src.screen_navigator import MENU_BUTTONS
        assert all("auto_id" in info for info in MENU_BUTTONS.values())


class TestSitePresetsFeatures:
    def test_all_sites_have_serp(self):
        for domain, config in SITE_PRESETS.items():
            assert config.get("serp_scraping") == True, f"{domain} missing serp_scraping"
            assert "serp_settings" in config, f"{domain} missing serp_settings"

    def test_deep_research_sites(self):
        dr_sites = [d for d, c in SITE_PRESETS.items() if c.get("deep_research")]
        assert len(dr_sites) == 5

    def test_all_sites_have_style_mimic(self):
        for domain, config in SITE_PRESETS.items():
            assert config.get("style_mimic") == True, f"{domain} missing style_mimic"
            sm = config.get("style_mimic_settings", {})
            assert sm.get("style_text"), f"{domain} has empty style_mimic text"

    def test_all_sites_have_custom_prompt(self):
        for domain, config in SITE_PRESETS.items():
            assert config.get("custom_prompt") == True, f"{domain} missing custom_prompt"
            cp = config.get("custom_prompt_settings", {})
            assert cp.get("prompt_text"), f"{domain} has empty custom_prompt text"
            assert cp.get("prompt_name"), f"{domain} has empty prompt_name"

    def test_link_pack_sites_have_settings(self):
        for domain, config in SITE_PRESETS.items():
            if config.get("link_pack"):
                assert "link_pack_settings" in config, f"{domain} has link_pack but no settings"

    def test_image_prompts_are_topic_adaptive(self):
        from src.image_prompts import FEATURED_IMAGE_PROMPTS, SUBHEADING_IMAGE_PROMPTS
        for domain, prompt in FEATURED_IMAGE_PROMPTS.items():
            assert "{title}" in prompt, f"{domain} featured prompt missing {{title}}"
            assert "text" in prompt.lower(), f"{domain} featured prompt missing no-text rule"
            assert len(prompt) > 200, f"{domain} featured prompt too short ({len(prompt)})"
        for domain, prompt in SUBHEADING_IMAGE_PROMPTS.items():
            assert "{title}" in prompt, f"{domain} subheading prompt missing {{title}}"
            assert "{subheading}" in prompt, f"{domain} subheading prompt missing {{subheading}}"


class TestCSVGenerator:
    def test_import(self):
        from src.csv_generator import generate_csv_from_titles, generate_bulk_csv
        assert callable(generate_csv_from_titles)
        assert callable(generate_bulk_csv)


class TestUtils:
    def test_timestamp(self):
        from src.utils import timestamp
        ts = timestamp()
        assert len(ts) == 15  # YYYYMMDD_HHMMSS

    def test_find_exe_returns_string_or_none(self):
        from src.utils import find_zimmwriter_exe
        result = find_zimmwriter_exe()
        assert result is None or isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
