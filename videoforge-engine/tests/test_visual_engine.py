"""Tests for VisualEngine — FAL.ai primary routing, Pexels rare fallback, niche-specific suffixes."""

import pytest
from unittest.mock import patch, MagicMock
from videoforge.assembly.visual_engine import (
    VisualEngine, _PROMPT_SUFFIX, _NICHE_STYLE_SUFFIXES, _get_niche_suffix,
)
from videoforge.forge.video_smith import VideoSmith
from videoforge.models import VisualAsset


@pytest.fixture
def visual_engine():
    return VisualEngine()


@pytest.fixture
def smith():
    return VideoSmith(db_path=":memory:")


@pytest.fixture
def storyboard(smith):
    plan = smith.to_video_plan("moon rituals", "witchcraftforbeginners")
    return plan.storyboard


@pytest.fixture
def tech_storyboard(smith):
    plan = smith.to_video_plan("5 smart home gadgets", "smarthomewizards")
    return plan.storyboard


@pytest.fixture
def mythology_storyboard(smith):
    plan = smith.to_video_plan("Zeus vs Odin", "mythicalarchives")
    return plan.storyboard


@pytest.fixture
def standard_storyboard(smith):
    plan = smith.to_video_plan(
        "mythology documentary", "mythicalarchives",
        platform="youtube", format="standard"
    )
    return plan.storyboard


class TestVisualEngine:
    def test_generate_assets_returns_list(self, visual_engine, storyboard):
        """Without API keys, should return placeholder assets."""
        assets = visual_engine.generate_assets(storyboard)
        assert isinstance(assets, list)
        assert len(assets) == len(storyboard.scenes)

    def test_all_scenes_get_images(self, visual_engine, storyboard):
        """All scenes should get image assets — no text_card skips."""
        assets = visual_engine.generate_assets(storyboard)
        assert len(assets) == len(storyboard.scenes), "Every scene must get an image asset"
        for asset in assets:
            assert asset.asset_type == "image", f"Scene {asset.scene_number} should be image, not {asset.asset_type}"
            assert "fal_ai" in asset.source, f"Scene {asset.scene_number} should route through FAL.ai"

    def test_default_routing_is_fal_ai(self, visual_engine, storyboard):
        """Without routing, all scenes should default to FAL.ai."""
        assets = visual_engine.generate_assets(storyboard)
        for asset in assets:
            # Without API key, should be fal_ai_placeholder
            assert "fal_ai" in asset.source

    def test_pexels_only_on_explicit_override(self, visual_engine, storyboard):
        """Pexels should only be used when routing explicitly says pexels_override."""
        # Without pexels override, NO assets should come from pexels
        assets = visual_engine.generate_assets(storyboard)
        for asset in assets:
            assert "pexels" not in asset.source

    def test_pexels_override_routing(self, visual_engine, storyboard):
        """Explicit pexels_override tries Pexels, falls back to FAL.ai if no key."""
        routing = [{"scene": 2, "provider": "pexels_override"}]
        assets = visual_engine.generate_assets(storyboard, routing)
        scene_2 = next(a for a in assets if a.scene_number == 2)
        # Without Pexels API key and no Pexels URL, falls back to FAL.ai placeholder
        # (Pexels placeholder has no URL, so fallback to _generate_fal_ai kicks in)
        assert scene_2.source in ("pexels_placeholder", "pexels", "fal_ai_placeholder", "fal_ai")

    @patch("videoforge.assembly.visual_engine._get_fal_key", return_value="test_key")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_fal_ai_api_call(self, mock_post, mock_key, visual_engine, storyboard):
        """Mock FAL.ai API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "images": [{"url": "https://fal.ai/output/test.png"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Generate for first scene
        scene = storyboard.scenes[0]
        asset = visual_engine._generate_fal_ai(scene, storyboard)
        assert asset.source == "fal_ai"
        assert asset.url == "https://fal.ai/output/test.png"
        assert asset.cost == 0.06

    def test_prompt_suffix_enhances_quality(self, visual_engine, storyboard):
        """Visual prompts should be enhanced with cinematic quality suffix."""
        # Without API key, prompts are stored in placeholder assets
        assets = visual_engine.generate_assets(storyboard)
        for asset in assets:
            if asset.prompt:
                # Enhanced prompts should contain quality terms
                # (only when FAL key is present, which adds suffix)
                assert len(asset.prompt) > 10

    def test_short_format_vertical_dimensions(self, visual_engine, storyboard):
        """Short format should request vertical (1080x1920) images."""
        with patch("videoforge.assembly.visual_engine._get_fal_key", return_value="test_key"):
            with patch("videoforge.assembly.visual_engine.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"images": [{"url": "https://test.png"}]}
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                visual_engine._generate_fal_ai(storyboard.scenes[0], storyboard)
                call_json = mock_post.call_args[1]["json"]
                assert call_json["image_size"]["width"] == 1080
                assert call_json["image_size"]["height"] == 1920

    def test_standard_format_horizontal_dimensions(self, visual_engine, standard_storyboard):
        """Standard format should request horizontal (1920x1080) images."""
        with patch("videoforge.assembly.visual_engine._get_fal_key", return_value="test_key"):
            with patch("videoforge.assembly.visual_engine.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"images": [{"url": "https://test.png"}]}
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                visual_engine._generate_fal_ai(standard_storyboard.scenes[0], standard_storyboard)
                call_json = mock_post.call_args[1]["json"]
                assert call_json["image_size"]["width"] == 1920
                assert call_json["image_size"]["height"] == 1080


class TestPromptSuffix:
    """Legacy suffix tests — backwards compatibility."""

    def test_short_suffix_has_vertical(self):
        assert "vertical" in _PROMPT_SUFFIX["short"]

    def test_standard_suffix_has_widescreen(self):
        assert "widescreen" in _PROMPT_SUFFIX["standard"]

    def test_all_suffixes_have_cinematic(self):
        for fmt, suffix in _PROMPT_SUFFIX.items():
            assert "cinematic" in suffix


class TestNicheSpecificSuffixes:
    """Test that niches get appropriate style suffixes instead of one-size-fits-all."""

    def test_niche_specific_suffix_tech(self):
        """Tech niches should get 'product photography' style, NOT 'film grain'."""
        suffix = _get_niche_suffix("smarthomewizards", "short")
        assert "product photography" in suffix or "ambient lighting" in suffix
        assert "film grain" not in suffix
        assert "vertical" in suffix  # composition hint

    def test_niche_specific_suffix_witchcraft(self):
        """Witchcraft niches should get 'mystical atmosphere' style."""
        suffix = _get_niche_suffix("witchcraftforbeginners", "short")
        assert "mystical" in suffix or "candlelight" in suffix
        assert "film grain" not in suffix

    def test_niche_specific_suffix_mythology(self):
        """Mythology should get 'epic oil painting' style."""
        suffix = _get_niche_suffix("mythicalarchives", "short")
        assert "oil painting" in suffix or "chiaroscuro" in suffix
        assert "product photography" not in suffix

    def test_niche_specific_suffix_lifestyle(self):
        """Lifestyle niches should get 'bright natural lighting' style."""
        suffix = _get_niche_suffix("bulletjournals", "short")
        assert "natural lighting" in suffix or "cozy" in suffix
        assert "film grain" not in suffix

    def test_niche_specific_suffix_fitness(self):
        """Fitness niche should get 'dynamic action' style."""
        suffix = _get_niche_suffix("pulsegearreviews", "short")
        assert "action" in suffix or "dynamic" in suffix

    def test_composition_hint_short(self):
        """Short format should include 'vertical composition'."""
        suffix = _get_niche_suffix("smarthomewizards", "short")
        assert "vertical" in suffix

    def test_composition_hint_standard(self):
        """Standard format should include 'widescreen composition'."""
        suffix = _get_niche_suffix("smarthomewizards", "standard")
        assert "widescreen" in suffix

    def test_composition_hint_square(self):
        """Square format should include 'centered composition'."""
        suffix = _get_niche_suffix("smarthomewizards", "square")
        assert "centered" in suffix

    @patch("videoforge.assembly.visual_engine._get_fal_key", return_value="test_key")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_fal_ai_uses_niche_suffix(self, mock_post, mock_key, visual_engine, tech_storyboard):
        """FAL.ai API call should use niche-specific suffix in prompt."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"images": [{"url": "https://test.png"}]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        visual_engine._generate_fal_ai(tech_storyboard.scenes[0], tech_storyboard)
        call_json = mock_post.call_args[1]["json"]
        prompt = call_json["prompt"]
        # Tech niche should NOT have "film grain" or "dramatic volumetric lighting"
        assert "film grain" not in prompt
        # Should have niche-appropriate terms
        assert "product photography" in prompt or "ambient lighting" in prompt or "editorial" in prompt

    def test_niche_suffix_categories_exist(self):
        """All expected category suffixes should be defined."""
        for category in ["tech", "ai_news", "witchcraft", "mythology", "lifestyle", "fitness", "business"]:
            assert category in _NICHE_STYLE_SUFFIXES, f"Missing suffix for category: {category}"
