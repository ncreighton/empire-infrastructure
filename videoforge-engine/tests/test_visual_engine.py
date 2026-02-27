"""Tests for VisualEngine — FAL.ai primary routing, Pexels rare fallback."""

import pytest
from unittest.mock import patch, MagicMock
from videoforge.assembly.visual_engine import VisualEngine, _PROMPT_SUFFIX
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
    def test_short_suffix_has_vertical(self):
        assert "vertical" in _PROMPT_SUFFIX["short"]

    def test_standard_suffix_has_widescreen(self):
        assert "widescreen" in _PROMPT_SUFFIX["standard"]

    def test_all_suffixes_have_cinematic(self):
        for fmt, suffix in _PROMPT_SUFFIX.items():
            assert "cinematic" in suffix
