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
    def test_all_16_sites_configured(self):
        assert len(SITE_PRESETS) >= 16, f"Only {len(SITE_PRESETS)} sites configured"

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
        assert len(domains) >= 16

    def test_get_presets_by_niche(self):
        ai_sites = get_presets_by_niche("AI")
        assert len(ai_sites) >= 3  # aiinactionhub, aidiscoverydigest, clearainews, etc.

    def test_witchcraft_preserves_voice(self):
        config = get_preset("witchcraftforbeginners.com")
        assert config["serp_scraping"] == False  # Preserve unique voice
        assert config["literary_devices"] == True

    def test_all_sites_use_claude_model(self):
        for domain, config in SITE_PRESETS.items():
            assert "Claude" in config["ai_model"], f"{domain} not using Claude"


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
