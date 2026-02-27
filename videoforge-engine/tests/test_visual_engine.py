"""Tests for VisualEngine — multi-provider routing, niche-specific suffixes."""

import pytest
from unittest.mock import patch, MagicMock
from videoforge.assembly.visual_engine import (
    VisualEngine, _PROMPT_SUFFIX, _NICHE_STYLE_SUFFIXES, _get_niche_suffix,
    _PROVIDER_CHAIN, _PROVIDER_COSTS, _OPENAI_SIZE_MAP,
    _get_runware_key, _get_openai_key, _get_fal_key,
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


# ── No-key patches (block all providers to isolate tests) ─────────────

def _patch_no_keys():
    """Patch all key loaders to return empty (no API keys)."""
    return [
        patch("videoforge.assembly.visual_engine._get_runware_key", return_value=""),
        patch("videoforge.assembly.visual_engine._get_openai_key", return_value=""),
        patch("videoforge.assembly.visual_engine._get_fal_key", return_value=""),
    ]


class TestVisualEngine:
    def test_generate_assets_returns_list(self, visual_engine, storyboard):
        """Without API keys, should return placeholder assets."""
        patches = _patch_no_keys()
        for p in patches:
            p.start()
        try:
            assets = visual_engine.generate_assets(storyboard)
            assert isinstance(assets, list)
            assert len(assets) == len(storyboard.scenes)
        finally:
            for p in patches:
                p.stop()

    def test_all_scenes_get_images(self, visual_engine, storyboard):
        """All scenes should get image assets — no text_card skips."""
        patches = _patch_no_keys()
        for p in patches:
            p.start()
        try:
            assets = visual_engine.generate_assets(storyboard)
            assert len(assets) == len(storyboard.scenes), "Every scene must get an image asset"
            for asset in assets:
                assert asset.asset_type == "image", f"Scene {asset.scene_number} should be image, not {asset.asset_type}"
        finally:
            for p in patches:
                p.stop()

    def test_no_keys_returns_all_providers_failed(self, visual_engine, storyboard):
        """Without any API keys, all scenes should get all_providers_failed."""
        patches = _patch_no_keys()
        for p in patches:
            p.start()
        try:
            assets = visual_engine.generate_assets(storyboard)
            for asset in assets:
                assert asset.source == "all_providers_failed"
        finally:
            for p in patches:
                p.stop()

    def test_pexels_only_on_explicit_override(self, visual_engine, storyboard):
        """Pexels should only be used when routing explicitly says pexels_override."""
        patches = _patch_no_keys()
        for p in patches:
            p.start()
        try:
            assets = visual_engine.generate_assets(storyboard)
            for asset in assets:
                assert "pexels" not in asset.source
        finally:
            for p in patches:
                p.stop()

    def test_pexels_override_routing(self, visual_engine, storyboard):
        """Explicit pexels_override tries Pexels, falls back to fallback chain."""
        patches = _patch_no_keys() + [
            patch("videoforge.assembly.visual_engine._get_pexels_key", return_value=""),
        ]
        for p in patches:
            p.start()
        try:
            routing = [{"scene": 2, "provider": "pexels_override"}]
            assets = visual_engine.generate_assets(storyboard, routing)
            scene_2 = next(a for a in assets if a.scene_number == 2)
            # Without Pexels API key and no URL, falls back to fallback chain
            valid_sources = ("pexels_placeholder", "all_providers_failed")
            assert scene_2.source in valid_sources
        finally:
            for p in patches:
                p.stop()

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

        scene = storyboard.scenes[0]
        asset = visual_engine._generate_fal_ai(scene, storyboard)
        assert asset.source == "fal_ai"
        assert asset.url == "https://fal.ai/output/test.png"
        assert asset.cost == 0.06

    def test_prompt_suffix_enhances_quality(self, visual_engine, storyboard):
        """Visual prompts should be enhanced with cinematic quality suffix."""
        patches = _patch_no_keys()
        for p in patches:
            p.start()
        try:
            assets = visual_engine.generate_assets(storyboard)
            for asset in assets:
                if asset.prompt:
                    assert len(asset.prompt) > 10
        finally:
            for p in patches:
                p.stop()

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
        assert "film grain" not in prompt
        assert "product photography" in prompt or "ambient lighting" in prompt or "editorial" in prompt

    def test_niche_suffix_categories_exist(self):
        """All expected category suffixes should be defined."""
        for category in ["tech", "ai_news", "witchcraft", "mythology", "lifestyle", "fitness", "business"]:
            assert category in _NICHE_STYLE_SUFFIXES, f"Missing suffix for category: {category}"


# ── Multi-Provider Routing Tests ──────────────────────────────────────

class TestMultiProviderRouting:
    """Test niche-based provider chain and fallback logic."""

    def test_provider_chain_mythology(self):
        """Mythology niche should prioritize OpenAI."""
        chain = _PROVIDER_CHAIN["mythology"]
        assert chain[0] == "openai"
        assert "runware" in chain
        assert "fal_ai" in chain

    def test_provider_chain_witchcraft(self):
        """Witchcraft niche should prioritize OpenAI."""
        chain = _PROVIDER_CHAIN["witchcraft"]
        assert chain[0] == "openai"

    def test_provider_chain_tech(self):
        """Tech niche should prioritize Runware."""
        chain = _PROVIDER_CHAIN["tech"]
        assert chain[0] == "runware"
        assert "openai" in chain

    def test_provider_chain_all_categories_defined(self):
        """All niche categories should have a provider chain."""
        for cat in ["mythology", "witchcraft", "tech", "ai_news", "lifestyle", "fitness", "business"]:
            assert cat in _PROVIDER_CHAIN

    def test_provider_costs_defined(self):
        """All providers should have cost entries."""
        assert _PROVIDER_COSTS["runware"] == 0.02
        assert _PROVIDER_COSTS["openai"] == 0.04
        assert _PROVIDER_COSTS["fal_ai"] == 0.06

    @patch("videoforge.assembly.visual_engine._get_runware_key", return_value="test_rw_key")
    @patch("videoforge.assembly.visual_engine._get_openai_key", return_value="")
    @patch("videoforge.assembly.visual_engine._get_fal_key", return_value="")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_fallback_runware_success(self, mock_post, mock_fal, mock_oai, mock_rw,
                                      visual_engine, tech_storyboard):
        """When Runware has a key and succeeds, use it for tech niche."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"imageURL": "https://cdn.runware.ai/test.png", "taskUUID": "vf_1"}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        assets = visual_engine.generate_assets(tech_storyboard)
        # All scenes should use Runware (tech niche primary)
        for asset in assets:
            assert asset.source == "runware"
            assert asset.url.startswith("https://")
            assert asset.cost == 0.02

    @patch("videoforge.assembly.visual_engine._get_openai_key", return_value="test_oai_key")
    @patch("videoforge.assembly.visual_engine._get_runware_key", return_value="")
    @patch("videoforge.assembly.visual_engine._get_fal_key", return_value="")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_fallback_openai_for_mythology(self, mock_post, mock_fal, mock_rw, mock_oai,
                                            visual_engine, mythology_storyboard):
        """Mythology niche should use OpenAI as primary."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"url": "https://oaidalleapiprodscus.blob.core.windows.net/test.png"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        assets = visual_engine.generate_assets(mythology_storyboard)
        for asset in assets:
            assert asset.source == "openai"
            assert asset.cost == 0.04

    @patch("videoforge.assembly.visual_engine._get_runware_key", return_value="test_key")
    @patch("videoforge.assembly.visual_engine._get_openai_key", return_value="test_key")
    @patch("videoforge.assembly.visual_engine._get_fal_key", return_value="")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_fallback_on_primary_failure(self, mock_post, mock_fal, mock_oai, mock_rw,
                                          visual_engine, tech_storyboard):
        """When primary provider fails, should fall back to secondary."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            # First call (Runware) fails, second (OpenAI) succeeds
            if "runware" in args[0]:
                mock_resp.raise_for_status.side_effect = Exception("Runware API error")
                return mock_resp
            else:
                mock_resp.json.return_value = {
                    "data": [{"url": "https://openai.test/image.png"}]
                }
                mock_resp.raise_for_status = MagicMock()
                return mock_resp

        mock_post.side_effect = side_effect

        scene = tech_storyboard.scenes[0]
        asset = visual_engine._generate_with_fallback(scene, tech_storyboard)
        assert asset.source == "openai"
        assert asset.url.startswith("https://")

    def test_all_providers_failed(self, visual_engine, storyboard):
        """When all providers fail, returns all_providers_failed placeholder."""
        patches = _patch_no_keys()
        for p in patches:
            p.start()
        try:
            scene = storyboard.scenes[0]
            asset = visual_engine._generate_with_fallback(scene, storyboard)
            assert asset.source == "all_providers_failed"
            assert asset.cost == 0.0
        finally:
            for p in patches:
                p.stop()

    @patch("videoforge.assembly.visual_engine._get_runware_key", return_value="test_key")
    @patch("videoforge.assembly.visual_engine._get_openai_key", return_value="")
    @patch("videoforge.assembly.visual_engine._get_fal_key", return_value="")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_explicit_provider_routing(self, mock_post, mock_fal, mock_oai, mock_rw,
                                        visual_engine, storyboard):
        """Routing override forces specific provider."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"imageURL": "https://cdn.runware.ai/explicit.png", "taskUUID": "vf_2"}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        routing = [{"scene": 1, "provider": "runware"}]
        assets = visual_engine.generate_assets(storyboard, routing)
        scene_1 = next(a for a in assets if a.scene_number == 1)
        assert scene_1.source == "runware"
        assert scene_1.url == "https://cdn.runware.ai/explicit.png"


class TestRunwareProvider:
    """Test Runware API integration."""

    @patch("videoforge.assembly.visual_engine._get_runware_key", return_value="test_rw_key")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_runware_api_call(self, mock_post, mock_key, visual_engine, storyboard):
        """Mock Runware API call — verify request format and response parsing."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"imageURL": "https://cdn.runware.ai/output/abc123.png", "taskUUID": "vf_1_1234"}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        scene = storyboard.scenes[0]
        asset = visual_engine._generate_runware(scene, storyboard)

        assert asset.source == "runware"
        assert asset.url == "https://cdn.runware.ai/output/abc123.png"
        assert asset.cost == 0.02

        # Verify request format
        call_args = mock_post.call_args
        assert "runware.ai" in call_args[0][0]
        assert "Bearer test_rw_key" in call_args[1]["headers"]["Authorization"]
        payload = call_args[1]["json"][0]
        assert payload["taskType"] == "imageInference"
        assert "negativePrompt" in payload
        assert payload["outputFormat"] == "PNG"

    def test_runware_no_key(self, visual_engine, storyboard):
        """Without API key, Runware returns no-key placeholder."""
        with patch("videoforge.assembly.visual_engine._get_runware_key", return_value=""):
            scene = storyboard.scenes[0]
            asset = visual_engine._generate_runware(scene, storyboard)
            assert asset.source == "runware_no_key"
            assert not asset.url
            assert asset.cost == 0.0

    @patch("videoforge.assembly.visual_engine._get_runware_key", return_value="test_key")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_runware_dimensions_short(self, mock_post, mock_key, visual_engine, storyboard):
        """Short format should request 1088x1920 from Runware (multiples of 64)."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"imageURL": "https://test.png"}]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        visual_engine._generate_runware(storyboard.scenes[0], storyboard)
        payload = mock_post.call_args[1]["json"][0]
        assert payload["width"] == 1088
        assert payload["height"] == 1920

    @patch("videoforge.assembly.visual_engine._get_runware_key", return_value="test_key")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_runware_dimensions_standard(self, mock_post, mock_key, visual_engine, standard_storyboard):
        """Standard format should request 1920x1088 from Runware (multiples of 64)."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"imageURL": "https://test.png"}]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        visual_engine._generate_runware(standard_storyboard.scenes[0], standard_storyboard)
        payload = mock_post.call_args[1]["json"][0]
        assert payload["width"] == 1920
        assert payload["height"] == 1088

    @patch("videoforge.assembly.visual_engine._get_runware_key", return_value="test_key")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_runware_uses_niche_suffix(self, mock_post, mock_key, visual_engine, tech_storyboard):
        """Runware prompt should include niche-specific suffix."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"imageURL": "https://test.png"}]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        visual_engine._generate_runware(tech_storyboard.scenes[0], tech_storyboard)
        payload = mock_post.call_args[1]["json"][0]
        prompt = payload["positivePrompt"]
        assert "product photography" in prompt or "ambient lighting" in prompt or "editorial" in prompt


class TestOpenAIProvider:
    """Test OpenAI DALL-E 3 integration."""

    @patch("videoforge.assembly.visual_engine._get_openai_key", return_value="test_oai_key")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_openai_api_call(self, mock_post, mock_key, visual_engine, storyboard):
        """Mock OpenAI DALL-E 3 API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"url": "https://oaidalleapiprodscus.blob.core.windows.net/test.png"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        scene = storyboard.scenes[0]
        asset = visual_engine._generate_openai(scene, storyboard)

        assert asset.source == "openai"
        assert "oaidalleapiprodscus" in asset.url
        assert asset.cost == 0.04

        # Verify request format
        call_args = mock_post.call_args
        assert "openai.com" in call_args[0][0]
        assert "Bearer test_oai_key" in call_args[1]["headers"]["Authorization"]
        payload = call_args[1]["json"]
        assert payload["model"] == "dall-e-3"
        assert payload["quality"] == "standard"
        assert payload["n"] == 1

    def test_openai_no_key(self, visual_engine, storyboard):
        """Without API key, OpenAI returns no-key placeholder."""
        with patch("videoforge.assembly.visual_engine._get_openai_key", return_value=""):
            scene = storyboard.scenes[0]
            asset = visual_engine._generate_openai(scene, storyboard)
            assert asset.source == "openai_no_key"
            assert not asset.url
            assert asset.cost == 0.0

    def test_openai_size_mapping_short(self):
        """Short format maps to 1024x1792."""
        assert _OPENAI_SIZE_MAP["short"] == "1024x1792"

    def test_openai_size_mapping_standard(self):
        """Standard format maps to 1792x1024."""
        assert _OPENAI_SIZE_MAP["standard"] == "1792x1024"

    def test_openai_size_mapping_square(self):
        """Square format maps to 1024x1024."""
        assert _OPENAI_SIZE_MAP["square"] == "1024x1024"

    @patch("videoforge.assembly.visual_engine._get_openai_key", return_value="test_key")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_openai_short_format_size(self, mock_post, mock_key, visual_engine, storyboard):
        """Short storyboard should request 1024x1792 from DALL-E 3."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"url": "https://test.png"}]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        visual_engine._generate_openai(storyboard.scenes[0], storyboard)
        payload = mock_post.call_args[1]["json"]
        assert payload["size"] == "1024x1792"

    @patch("videoforge.assembly.visual_engine._get_openai_key", return_value="test_key")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_openai_standard_format_size(self, mock_post, mock_key, visual_engine, standard_storyboard):
        """Standard storyboard should request 1792x1024 from DALL-E 3."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"url": "https://test.png"}]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        visual_engine._generate_openai(standard_storyboard.scenes[0], standard_storyboard)
        payload = mock_post.call_args[1]["json"]
        assert payload["size"] == "1792x1024"

    @patch("videoforge.assembly.visual_engine._get_openai_key", return_value="test_key")
    @patch("videoforge.assembly.visual_engine.requests.post")
    def test_openai_uses_niche_suffix(self, mock_post, mock_key, visual_engine, mythology_storyboard):
        """OpenAI prompt should include niche-specific suffix for mythology."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"url": "https://test.png"}]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        visual_engine._generate_openai(mythology_storyboard.scenes[0], mythology_storyboard)
        payload = mock_post.call_args[1]["json"]
        prompt = payload["prompt"]
        assert "oil painting" in prompt or "chiaroscuro" in prompt or "ancient" in prompt
