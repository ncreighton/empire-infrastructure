"""
Per-model image option configurations for ZimmWriter's O button windows.

O button options are PER-MODEL (not per-site). Each unique image model used
across the 14 sites needs its options configured once.

Discovered control IDs (ZimmWriter v10.869):
  All models:
    111 = Enable Compression (CheckBox)
    113 = aspect_ratio (ComboBox)
  Non-ideogram: 115 = seed (Edit)
  Ideogram: 115 = Magic Prompt (ComboBox: OFF/AUTO/ON)
            117 = Style (ComboBox: AUTO/GENERAL/REALISTIC/DESIGN)
            119 = seed (Edit)
  Subheading ideogram also has: 121 = Activate Similarity (ComboBox: no/yes)

Aspect ratio values vary by model family:
  Non-ideogram: "16:9", "1:1", "9:16", "3:4", "4:3"
  Ideogram: "landscape_16_9", "square_hd", "square", "portrait_4_3",
            "portrait_16_9", "landscape_4_3"
"""

from typing import Dict, Any, Optional


# Per-model option configs.
# Keys are the model strings as they appear in ZimmWriter dropdowns.
# Values are dicts of option settings to apply in the O window.

IMAGE_MODEL_OPTIONS: Dict[str, Dict[str, Any]] = {

    # ── Google Imagen ──
    "imagegen-4 $.050/img (F)": {
        "is_ideogram": False,
        "enable_compression": True,
        "aspect_ratio": "16:9",
    },

    # ── Ideogram models ──
    # Note: ideogram uses "landscape_16_9" not "16:9"
    # Magic Prompt: "AUTO" lets model enhance prompts when helpful
    # Style: "REALISTIC" for photo-like results
    "ideogram 3t $.030/img (F)": {
        "is_ideogram": True,
        "enable_compression": True,
        "aspect_ratio": "landscape_16_9",
        "magic_prompt": "AUTO",
        "style": "REALISTIC",
        "activate_similarity": "no",
    },

    "ideogram 3b $.060/img (F)": {
        "is_ideogram": True,
        "enable_compression": True,
        "aspect_ratio": "landscape_16_9",
        "magic_prompt": "AUTO",
        "style": "REALISTIC",
        "activate_similarity": "no",
    },

    "ideogram 3q $.090/img (F)": {
        "is_ideogram": True,
        "enable_compression": True,
        "aspect_ratio": "landscape_16_9",
        "magic_prompt": "AUTO",
        "style": "REALISTIC",
        "activate_similarity": "no",
    },

    # ── Flux models ──
    "flux dev $.025/img (F)": {
        "is_ideogram": False,
        "enable_compression": True,
        "aspect_ratio": "16:9",
    },

    "flux pro $.040/img (F)": {
        "is_ideogram": False,
        "enable_compression": True,
        "aspect_ratio": "16:9",
    },

    "flux schnell $.003/img (F)": {
        "is_ideogram": False,
        "enable_compression": True,
        "aspect_ratio": "16:9",
    },

    # ── OpenAI GPT Image ──
    "gpt-image-1 med $.063/img (OA)": {
        "is_ideogram": False,
        "enable_compression": True,
        "aspect_ratio": "16:9",
    },

    "gpt-image-1 low $.016/img (OA)": {
        "is_ideogram": False,
        "enable_compression": True,
        "aspect_ratio": "16:9",
    },
}


def get_model_options(model_name: str) -> Optional[Dict[str, Any]]:
    """Get option config for a specific model."""
    return IMAGE_MODEL_OPTIONS.get(model_name)


def get_unique_models_from_presets(presets: dict) -> set:
    """Extract unique image model names from all site presets."""
    models = set()
    for domain, config in presets.items():
        feat = config.get("featured_image")
        sub = config.get("subheading_images_model")
        if feat and feat != "None":
            models.add(feat)
        if sub and sub != "None":
            models.add(sub)
    return models


def get_all_model_names() -> list:
    """Get all configured model names."""
    return list(IMAGE_MODEL_OPTIONS.keys())
