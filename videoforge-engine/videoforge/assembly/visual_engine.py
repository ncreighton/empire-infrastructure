"""VisualEngine — Multi-provider AI image generation with niche-based routing.

Provider priority by niche category:
- mythology/witchcraft: OpenAI DALL-E 3 (epic artistic scenes) -> Runware -> FAL.ai
- tech/ai_news/lifestyle/fitness/business: Runware (fast, cheap) -> OpenAI -> FAL.ai
- Pexels: stock footage, only as explicit override via routing
"""

import os
import uuid
import logging
import requests
from ..models import VisualAsset, Storyboard
from ..knowledge.niche_profiles import get_niche_profile
from ..knowledge.domain_expertise import get_style_suffix

logger = logging.getLogger(__name__)

# Shared ai_gen_client disabled — module lacks generate_image attribute.
# Use direct API calls instead.
_ai_gen_client = None


PEXELS_BASE = "https://api.pexels.com/videos/search"

# Niche-category-based style suffixes — replaces one-size-fits-all cinematic
_NICHE_STYLE_SUFFIXES = {
    "tech": ", clean product photography, soft ambient lighting, modern minimalist interior, shallow depth of field, editorial style, sharp focus, 4K, professional",
    "ai_news": ", futuristic digital environment, holographic displays, neon accents, clean tech aesthetic, sharp focus, 4K, professional photography",
    "witchcraft": ", mystical atmosphere, candlelight, soft ethereal glow, dark moody background, sacred space aesthetic, shallow depth of field, fine art photography",
    "mythology": ", epic oil painting style, dramatic chiaroscuro lighting, ancient grandeur, detailed textures, museum quality, masterwork illustration",
    "lifestyle": ", bright natural lighting, cozy warm interior, lifestyle photography, inviting atmosphere, editorial style, shallow depth of field, 4K",
    "fitness": ", dynamic action photography, high contrast, gym or outdoor setting, motivational energy, sharp focus, sports photography, 4K",
    "business": ", professional corporate aesthetic, clean modern workspace, data visualization, confident atmosphere, editorial photography, 4K",
}

# Composition hint by format (appended after niche style suffix)
_COMPOSITION_HINT = {
    "short": ", vertical composition",
    "standard": ", widescreen composition",
    "square": ", centered composition",
}

# Legacy _PROMPT_SUFFIX kept for backwards compatibility with existing tests
_PROMPT_SUFFIX = {
    "short": ", ultra realistic, cinematic film still, 8K UHD, sharp focus, dramatic volumetric lighting, depth of field, film grain, color graded, vertical composition, professional cinematography, photorealistic, award-winning photography",
    "standard": ", ultra realistic, cinematic film still, 8K UHD, sharp focus, dramatic volumetric lighting, depth of field, film grain, color graded, widescreen composition, professional cinematography, photorealistic, award-winning photography",
    "square": ", ultra realistic, cinematic film still, 8K UHD, sharp focus, dramatic volumetric lighting, depth of field, film grain, color graded, centered composition, professional cinematography, photorealistic, award-winning photography",
}

# ── Provider Routing ──────────────────────────────────────────────────

# Provider chain by niche category — tried in order until one succeeds
_PROVIDER_CHAIN = {
    "mythology": ["openai", "runware", "fal_ai"],
    "witchcraft": ["openai", "runware", "fal_ai"],
    "tech": ["runware", "openai", "fal_ai"],
    "ai_news": ["runware", "openai", "fal_ai"],
    "lifestyle": ["runware", "openai", "fal_ai"],
    "fitness": ["runware", "openai", "fal_ai"],
    "business": ["runware", "openai", "fal_ai"],
}

_DEFAULT_CHAIN = ["runware", "openai", "fal_ai"]

_PROVIDER_COSTS = {
    "runware": 0.02,
    "openai": 0.04,
    "fal_ai": 0.06,
}

# DALL-E 3 size mapping (closest to target aspect ratios)
_OPENAI_SIZE_MAP = {
    "short": "1024x1792",
    "standard": "1792x1024",
    "square": "1024x1024",
}


def _brighten_prompt(prompt: str) -> str:
    """Strip dark/shadow terms from prompts and ensure brightness keywords.

    AI-generated visual directions often default to dark/moody imagery.
    This ensures images are vivid and well-lit for video content.
    """
    import re
    # Remove terms that cause dark image generation
    dark_terms = [
        r'\bdark\s+(?:atmospheric|moody|background|shadows?|tones?)\b',
        r'\bchiaroscuro\b', r'\bsilhouette\b', r'\bdarkening\b',
        r'\bdimly[- ]lit\b', r'\bshadowy\b', r'\bmurky\b',
        r'\bdramatic shadows?\b', r'\bdark atmospheric\b',
    ]
    cleaned = prompt
    for pattern in dark_terms:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    # Clean up double commas/spaces from removals
    cleaned = re.sub(r',\s*,', ',', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip().strip(',').strip()

    # Append brightness keywords if not already present
    brightness_terms = ["bright", "vivid", "well-lit", "vibrant"]
    has_brightness = any(t in cleaned.lower() for t in brightness_terms)
    if not has_brightness:
        cleaned += ", bright vivid colors, well-lit"

    return cleaned


def _get_niche_suffix(niche: str, fmt: str) -> str:
    """Get the niche-specific prompt suffix with composition hint.

    Uses per-niche style from domain expertise if available,
    falls back to category-level suffix, then generic cinematic.
    """
    # Try niche-specific suffix from domain expertise
    suffix = get_style_suffix(niche)

    # Add composition hint
    composition = _COMPOSITION_HINT.get(fmt, _COMPOSITION_HINT["short"])
    return suffix + composition


def _load_key_from_file(env_path: str, key_name: str) -> str:
    """Load a key from a .env file by key name."""
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key_name}="):
                    val = line.split("=", 1)[1]
                    if val:
                        return val
    return ""


def _get_pexels_key() -> str:
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key:
        key = _load_key_from_file(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "api_keys.env"),
            "PEXELS_API_KEY",
        )
    if not key:
        key = _load_key_from_file(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", ".env"),
            "PEXELS_API_KEY",
        )
    return key


def _get_fal_key() -> str:
    key = os.environ.get("FAL_KEY", "")
    if not key:
        key = _load_key_from_file(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "api_keys.env"),
            "FAL_KEY",
        )
    if not key:
        key = _load_key_from_file(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", ".env"),
            "FAL_KEY",
        )
    return key


def _get_runware_key() -> str:
    """Load Runware API key from env var or config files."""
    key = os.environ.get("RUNWARE_API_KEY", "")
    if not key:
        key = _load_key_from_file(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "api_keys.env"),
            "RUNWARE_API_KEY",
        )
    if not key:
        key = _load_key_from_file(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", ".env"),
            "RUNWARE_API_KEY",
        )
    return key


def _get_openai_key() -> str:
    """Load OpenAI API key from env var or config files."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        key = _load_key_from_file(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "api_keys.env"),
            "OPENAI_API_KEY",
        )
    if not key:
        key = _load_key_from_file(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", ".env"),
            "OPENAI_API_KEY",
        )
    return key


class VisualEngine:
    """Generates and sources visual assets for video scenes.

    Multi-provider routing with niche-based priority:
    - mythology/witchcraft: OpenAI DALL-E 3 -> Runware -> FAL.ai
    - tech/lifestyle/etc: Runware -> OpenAI -> FAL.ai
    - Pexels: only as explicit override via routing
    """

    def generate_assets(self, storyboard: Storyboard,
                        routing: list = None) -> list:
        """Generate visual assets for all scenes in a storyboard.

        Uses niche-based provider routing with automatic fallback.
        Pexels only when routing explicitly says 'pexels_override'.
        """
        assets = []
        routing_map = {}
        if routing:
            routing_map = {r["scene"]: r for r in routing}

        for scene in storyboard.scenes:
            route = routing_map.get(scene.scene_number, {})
            provider = route.get("provider", None)

            if provider == "pexels_override":
                asset = self._search_pexels(scene, storyboard)
                if not asset.url:
                    asset = self._generate_with_fallback(scene, storyboard)
                assets.append(asset)
            elif provider:
                # Explicit provider specified in routing
                asset = self._generate_single_provider(provider, scene, storyboard)
                if not asset.url or not asset.url.startswith("http"):
                    # Explicit provider failed, try fallback chain
                    asset = self._generate_with_fallback(scene, storyboard)
                assets.append(asset)
            else:
                # Smart routing based on niche
                asset = self._generate_with_fallback(scene, storyboard)
                assets.append(asset)

        return assets

    def _generate_single_provider(self, provider: str, scene, storyboard: Storyboard) -> VisualAsset:
        """Try a single specific provider."""
        dispatch = {
            "runware": self._generate_runware,
            "openai": self._generate_openai,
            "fal_ai": self._generate_fal_ai,
        }
        gen_fn = dispatch.get(provider)
        if not gen_fn:
            logger.warning(f"Unknown provider: {provider}")
            return VisualAsset(
                scene_number=scene.scene_number,
                asset_type="image",
                source=f"{provider}_unknown",
                prompt=scene.visual_prompt,
                cost=0.0,
                duration=scene.duration_seconds,
            )
        try:
            return gen_fn(scene, storyboard)
        except Exception as e:
            logger.warning(f"{provider} failed for scene {scene.scene_number}: {e}")
            return VisualAsset(
                scene_number=scene.scene_number,
                asset_type="image",
                source=f"{provider}_failed",
                prompt=scene.visual_prompt,
                cost=0.0,
                duration=scene.duration_seconds,
            )

    def _generate_with_fallback(self, scene, storyboard: Storyboard) -> VisualAsset:
        """Try providers in niche-priority order until one succeeds."""
        profile = get_niche_profile(storyboard.niche)
        category = profile.get("category", "tech") if profile else "tech"
        chain = _PROVIDER_CHAIN.get(category, _DEFAULT_CHAIN)

        for provider_name in chain:
            try:
                if provider_name == "runware":
                    asset = self._generate_runware(scene, storyboard)
                elif provider_name == "openai":
                    asset = self._generate_openai(scene, storyboard)
                elif provider_name == "fal_ai":
                    asset = self._generate_fal_ai(scene, storyboard)
                else:
                    continue

                if asset.url and asset.url.startswith("http"):
                    return asset
            except Exception as e:
                logger.warning(f"{provider_name} failed for scene {scene.scene_number}: {e}")
                continue

        # All providers failed
        return VisualAsset(
            scene_number=scene.scene_number,
            asset_type="image",
            source="all_providers_failed",
            prompt=scene.visual_prompt,
            cost=0.0,
            duration=scene.duration_seconds,
        )

    def _generate_runware(self, scene, storyboard: Storyboard) -> VisualAsset:
        """Generate an image using Runware API."""
        runware_key = _get_runware_key()
        if not runware_key:
            logger.info("No RUNWARE_API_KEY — skipping Runware")
            return VisualAsset(
                scene_number=scene.scene_number,
                asset_type="image",
                source="runware_no_key",
                prompt=scene.visual_prompt,
                cost=0.0,
                duration=scene.duration_seconds,
            )

        fmt = storyboard.format if storyboard.format in _COMPOSITION_HINT else "short"
        niche_suffix = _get_niche_suffix(storyboard.niche, fmt)
        enhanced_prompt = _brighten_prompt(scene.visual_prompt) + niche_suffix

        # Runware requires dimensions in multiples of 64
        width = 1088 if storyboard.format == "short" else 1920
        height = 1920 if storyboard.format == "short" else 1088

        task_uuid = str(uuid.uuid4())

        try:
            response = requests.post(
                "https://api.runware.ai/v1",
                headers={
                    "Authorization": f"Bearer {runware_key}",
                    "Content-Type": "application/json",
                },
                json=[{
                    "taskType": "imageInference",
                    "taskUUID": task_uuid,
                    "positivePrompt": enhanced_prompt,
                    "negativePrompt": "blurry, low quality, text, watermark, deformed, disfigured",
                    "width": width,
                    "height": height,
                    "model": "runware:100@1",
                    "numberResults": 1,
                    "outputFormat": "PNG",
                }],
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            # Response is a list of result items
            items = data if isinstance(data, list) else data.get("data", [])
            image_url = ""
            for item in items:
                if item.get("imageURL"):
                    image_url = item["imageURL"]
                    break

            if image_url:
                return VisualAsset(
                    scene_number=scene.scene_number,
                    asset_type="image",
                    source="runware",
                    prompt=enhanced_prompt,
                    url=image_url,
                    cost=_PROVIDER_COSTS["runware"],
                    duration=scene.duration_seconds,
                )
        except Exception as e:
            logger.warning(f"Runware generation failed for scene {scene.scene_number}: {e}")

        return VisualAsset(
            scene_number=scene.scene_number,
            asset_type="image",
            source="runware_failed",
            prompt=enhanced_prompt,
            cost=0.0,
            duration=scene.duration_seconds,
        )

    def _generate_openai(self, scene, storyboard: Storyboard) -> VisualAsset:
        """Generate an image using OpenAI DALL-E 3."""
        openai_key = _get_openai_key()
        if not openai_key:
            logger.info("No OPENAI_API_KEY — skipping OpenAI")
            return VisualAsset(
                scene_number=scene.scene_number,
                asset_type="image",
                source="openai_no_key",
                prompt=scene.visual_prompt,
                cost=0.0,
                duration=scene.duration_seconds,
            )

        fmt = storyboard.format if storyboard.format in _COMPOSITION_HINT else "short"
        niche_suffix = _get_niche_suffix(storyboard.niche, fmt)
        enhanced_prompt = _brighten_prompt(scene.visual_prompt) + niche_suffix

        size = _OPENAI_SIZE_MAP.get(fmt, "1024x1792")

        try:
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "dall-e-3",
                    "prompt": enhanced_prompt,
                    "size": size,
                    "quality": "standard",
                    "n": 1,
                },
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()

            image_url = ""
            items = data.get("data", [])
            if items:
                image_url = items[0].get("url", "")

            if image_url:
                return VisualAsset(
                    scene_number=scene.scene_number,
                    asset_type="image",
                    source="openai",
                    prompt=enhanced_prompt,
                    url=image_url,
                    cost=_PROVIDER_COSTS["openai"],
                    duration=scene.duration_seconds,
                )
        except Exception as e:
            logger.warning(f"OpenAI generation failed for scene {scene.scene_number}: {e}")

        return VisualAsset(
            scene_number=scene.scene_number,
            asset_type="image",
            source="openai_failed",
            prompt=enhanced_prompt,
            cost=0.0,
            duration=scene.duration_seconds,
        )

    def _generate_fal_ai(self, scene, storyboard: Storyboard) -> VisualAsset:
        """Generate an image using FAL.ai FLUX Pro."""
        fal_key = _get_fal_key()
        if not fal_key:
            logger.info("No FAL_KEY — returning placeholder asset")
            return VisualAsset(
                scene_number=scene.scene_number,
                asset_type="image",
                source="fal_ai_placeholder",
                prompt=scene.visual_prompt,
                cost=0.0,
                duration=scene.duration_seconds,
            )

        # Enhance prompt with niche-specific quality suffix
        fmt = storyboard.format if storyboard.format in _COMPOSITION_HINT else "short"
        niche_suffix = _get_niche_suffix(storyboard.niche, fmt)
        enhanced_prompt = _brighten_prompt(scene.visual_prompt) + niche_suffix

        # Use shared ai_gen_client if available
        if _ai_gen_client:
            try:
                result = _ai_gen_client.generate_image(
                    prompt=enhanced_prompt,
                    model="flux_pro",
                    width=1080 if storyboard.format == "short" else 1920,
                    height=1920 if storyboard.format == "short" else 1080,
                )
                return VisualAsset(
                    scene_number=scene.scene_number,
                    asset_type="image",
                    source="fal_ai",
                    prompt=enhanced_prompt,
                    url=result.get("url", ""),
                    cost=0.05,
                    duration=scene.duration_seconds,
                )
            except Exception as e:
                logger.warning(f"ai_gen_client failed: {e}")

        # Direct FAL.ai API call with retry on 422
        width = 1080 if storyboard.format == "short" else 1920
        height = 1920 if storyboard.format == "short" else 1080

        prompts_to_try = [enhanced_prompt]
        # Simplified fallback prompt (shorter, fewer modifiers)
        simple_prompt = _brighten_prompt(scene.visual_prompt) + ", cinematic, high quality, 4K"
        prompts_to_try.append(simple_prompt)
        # Generic fallback (strips topic to avoid content filters)
        shot = scene.shot_type or "cinematic"
        generic_prompt = f"{shot} shot, bright natural lighting, vivid colors, epic atmosphere, 4K, cinematic, depth of field, professional photography"
        prompts_to_try.append(generic_prompt)

        for attempt, prompt in enumerate(prompts_to_try):
            try:
                response = requests.post(
                    "https://fal.run/fal-ai/flux-pro/v1.1",
                    headers={
                        "Authorization": f"Key {fal_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "prompt": prompt,
                        "image_size": {"width": width, "height": height},
                        "num_images": 1,
                        "num_inference_steps": 40,
                        "guidance_scale": 3.5,
                        "safety_tolerance": "5",
                    },
                    timeout=90,
                )
                response.raise_for_status()
                data = response.json()
                image_url = data.get("images", [{}])[0].get("url", "")

                return VisualAsset(
                    scene_number=scene.scene_number,
                    asset_type="image",
                    source="fal_ai",
                    prompt=prompt,
                    url=image_url,
                    cost=0.06,
                    duration=scene.duration_seconds,
                )
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 422 and attempt < len(prompts_to_try) - 1:
                    logger.warning(f"FAL.ai 422 on scene {scene.scene_number} attempt {attempt+1}, retrying with simpler prompt")
                    continue
                logger.warning(f"FAL.ai generation failed: {e}")
            except Exception as e:
                logger.warning(f"FAL.ai generation failed: {e}")
                break

        return VisualAsset(
            scene_number=scene.scene_number,
            asset_type="image",
            source="fal_ai_failed",
            prompt=enhanced_prompt,
            cost=0.0,
            duration=scene.duration_seconds,
        )

    def _search_pexels(self, scene, storyboard: Storyboard) -> VisualAsset:
        """Search Pexels for stock video clips (rare fallback only)."""
        pexels_key = _get_pexels_key()
        if not pexels_key:
            return VisualAsset(
                scene_number=scene.scene_number,
                asset_type="stock_footage",
                source="pexels_placeholder",
                prompt=scene.visual_prompt,
                cost=0.0,
                duration=scene.duration_seconds,
            )

        # Better query extraction — use key nouns from visual prompt
        words = scene.visual_prompt.split()
        # Skip common style words, grab content words
        skip = {"style", "shot", "close-up", "dramatic", "cinematic", "composition",
                "aesthetic", "establishing", "dynamic", "clean", "bold", "atmospheric",
                "and", "the", "a", "with", "for", "of", "in", "on"}
        content_words = [w.strip(",.") for w in words if w.lower().strip(",.") not in skip]
        query = " ".join(content_words[:6])

        try:
            response = requests.get(
                PEXELS_BASE,
                headers={"Authorization": pexels_key},
                params={
                    "query": query,
                    "per_page": 3,
                    "orientation": "portrait" if storyboard.format == "short" else "landscape",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            videos = data.get("videos", [])

            if videos:
                video = videos[0]
                files = video.get("video_files", [])
                best = max(files, key=lambda f: f.get("width", 0)) if files else {}

                return VisualAsset(
                    scene_number=scene.scene_number,
                    asset_type="stock_footage",
                    source="pexels",
                    prompt=query,
                    url=best.get("link", ""),
                    cost=0.0,
                    duration=scene.duration_seconds,
                )
        except Exception as e:
            logger.warning(f"Pexels search failed: {e}")

        return VisualAsset(
            scene_number=scene.scene_number,
            asset_type="stock_footage",
            source="pexels_no_results",
            prompt=query,
            cost=0.0,
            duration=scene.duration_seconds,
        )
