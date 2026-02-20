"""
Site preset configurations for 14 active websites.
Each preset maps the site-configs.json data into ZimmWriter controller parameters.

Fields cover ALL Bulk Writer dropdowns and checkboxes so profiles can be saved
with complete, exact settings. Also includes image meta-prompts, model options,
and feature toggle config window settings (SERP, Deep Research, Style Mimic,
Custom Prompts, Link Packs, etc.).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from .image_prompts import get_featured_prompt, get_subheading_prompt

# ── Config file loaders ──

_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def _load_style_sample(domain: str) -> str:
    """Load brand voice style sample text for a domain."""
    path = _CONFIGS_DIR / "style_samples" / f"{domain}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _load_custom_prompt(niche_key: str) -> str:
    """Load custom editorial prompt text for a niche group."""
    path = _CONFIGS_DIR / "custom_prompts" / f"{niche_key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _load_section_prompt(filename: str) -> str:
    """Load a section-specific prompt from configs/custom_prompts/sections/."""
    path = _CONFIGS_DIR / "custom_prompts" / "sections" / f"{filename}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


# Domain -> custom prompt niche key mapping
_DOMAIN_PROMPT_MAP = {
    "aiinactionhub.com": "ai_technology",
    "aidiscoverydigest.com": "ai_technology",
    "clearainews.com": "ai_technology",
    "wealthfromai.com": "ai_technology",
    "smarthomewizards.com": "smart_home",
    "smarthomegearreviews.com": "reviews",
    "theconnectedhaven.com": "smart_home",
    "witchcraftforbeginners.com": "spiritual",
    "manifestandalign.com": "spiritual",
    "family-flourish.com": "family",
    "mythicalarchives.com": "mythology",
    "wearablegearreviews.com": "reviews",
    "pulsegearreviews.com": "reviews",
    "bulletjournals.net": "creative",
}

# Domain -> short cp name (used in {cp_NAME} format for ZimmWriter)
_DOMAIN_CP_PREFIX = {
    "aiinactionhub.com": "aiinactionhub",
    "aidiscoverydigest.com": "aidiscoverydigest",
    "clearainews.com": "clearainews",
    "wealthfromai.com": "wealthfromai",
    "smarthomewizards.com": "smarthomewizards",
    "smarthomegearreviews.com": "smarthomegearreviews",
    "theconnectedhaven.com": "theconnectedhaven",
    "witchcraftforbeginners.com": "witchcraft",
    "manifestandalign.com": "manifestandalign",
    "family-flourish.com": "familyflourish",
    "mythicalarchives.com": "mythicalarchives",
    "wearablegearreviews.com": "wearablegear",
    "pulsegearreviews.com": "pulsegear",
    "bulletjournals.net": "bulletjournals",
}

# Niches that should have product_layout section prompts
_PRODUCT_LAYOUT_NICHES = {"reviews"}


def _build_prompt_config(domain: str) -> dict:
    """Build the full custom_prompt_settings for a domain.

    Returns dict with 'prompts' (list of name+text to save) and
    'section_assignments' (section name -> prompt name mapping).
    """
    niche = _DOMAIN_PROMPT_MAP.get(domain, "")
    cp_prefix = _DOMAIN_CP_PREFIX.get(domain, "")
    if not niche or not cp_prefix:
        return {}

    # Build list of prompts to save
    prompts = []

    # Niche-specific section prompts
    for section in ("intro", "conclusion", "faq"):
        text = _load_section_prompt(f"{section}_{niche}")
        if text:
            prompts.append({
                "name": f"{{cp_{niche}_{section}}}",
                "text": text,
            })

    # Main editorial voice prompt (per-site, used for subheadings)
    subheading_text = _load_custom_prompt(niche)
    if subheading_text:
        prompts.append({
            "name": f"{{cp_{cp_prefix}}}",
            "text": subheading_text,
        })

    # Universal prompts (shared across all sites)
    for section, filename in (
        ("transitions", "transitions_universal"),
        ("key_takeaways", "key_takeaways_universal"),
        ("meta_description", "meta_description_universal"),
    ):
        text = _load_section_prompt(filename)
        if text:
            prompts.append({
                "name": f"{{cp_{section}}}",
                "text": text,
            })

    # Product layout (only for review niches)
    if niche in _PRODUCT_LAYOUT_NICHES:
        text = _load_section_prompt("product_layout_reviews")
        if text:
            prompts.append({
                "name": "{cp_product_layout}",
                "text": text,
            })

    # Build section assignments
    assignments = {
        "intro": f"{{cp_{niche}_intro}}",
        "conclusion": f"{{cp_{niche}_conclusion}}",
        "subheadings": f"{{cp_{cp_prefix}}}",
        "transitions": "{cp_transitions}",
        "key_takeaways": "{cp_key_takeaways}",
        "faq": f"{{cp_{niche}_faq}}",
        "meta_description": "{cp_meta_description}",
    }

    # Product layout only for review niches
    if niche in _PRODUCT_LAYOUT_NICHES:
        assignments["product_layout"] = "{cp_product_layout}"

    return {
        "prompts": prompts,
        "section_assignments": assignments,
    }

SITE_PRESETS: Dict[str, Dict[str, Any]] = {

    # ═══════════════════════════════════════
    # AI & TECHNOLOGY SITES
    # ═══════════════════════════════════════

    "aiinactionhub.com": {
        "domain": "aiinactionhub.com",
        "niche": "AI & Technology",
        # Dropdowns
        "h2_count": "Automatic",
        "h2_auto_limit": 10,
        "h2_lower_limit": 5,
        "ai_outline_quality": "High $$",
        "section_length": "Medium",
        "voice": "Second Person (You, Your, Yours)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Explorer",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "ideogram 3t $.030/img (F)",
        "subheading_image_quantity": "Every Other H2 (Max 3)",
        "subheading_images_model": "flux schnell $.003/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Haiku (ANT)",
        "ai_model_translation": "None",
        # Checkboxes
        "lists": True,
        "tables": True,
        "blockquotes": True,
        "literary_devices": False,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        # Feature toggles
        "serp_scraping": True,
        "deep_research": True,
        # Feature toggle config windows
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "deep_research_settings": {"ai_model": "Sonar Online (OR)", "links_per_article": "3", "links_per_subheading": "1"},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("aiinactionhub.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("aiinactionhub.com"),
        "link_pack": True,
        "link_pack_settings": {"pack_name": "aiinactionhub_internal", "insertion_limit": "3"},
        # WordPress
        "wordpress_category": "AI Technology",
        "wordpress_settings": {
            "site_url": "https://aiinactionhub.com",
            "user_name": "AIinActionEditor",
            "category": "AI Technology",
            "article_status": "draft",
        },
        # Image prompts & options
        "featured_image_prompt": get_featured_prompt("aiinactionhub.com"),
        "subheading_image_prompt": get_subheading_prompt("aiinactionhub.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "landscape_16_9",
                         "magic_prompt": "AUTO", "style": "REALISTIC", "activate_similarity": "no"},
            "subheading": {"enable_compression": True, "aspect_ratio": "16:9"},
        },
    },

    "aidiscoverydigest.com": {
        "domain": "aidiscoverydigest.com",
        "niche": "AI Research & Analysis",
        "h2_count": "Automatic",
        "h2_auto_limit": 12,
        "h2_lower_limit": 6,
        "ai_outline_quality": "High $$",
        "section_length": "Long",
        "voice": "Third Person (He, She, It, They)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Sage",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "flux dev $.025/img (F)",
        "subheading_image_quantity": "Every 3rd H2 (Max 3)",
        "subheading_images_model": "flux schnell $.003/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Haiku (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": True,
        "blockquotes": True,
        "literary_devices": False,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "deep_research": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "deep_research_settings": {"ai_model": "Sonar Online (OR)", "links_per_article": "5", "links_per_subheading": "2"},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("aidiscoverydigest.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("aidiscoverydigest.com"),
        "link_pack": True,
        "link_pack_settings": {"pack_name": "aidiscoverydigest_internal", "insertion_limit": "3"},
        "wordpress_category": "AI Research",
        "wordpress_settings": {
            "site_url": "https://aidiscoverydigest.com",
            "user_name": "AITrendCurator",
            "category": "AI Research",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("aidiscoverydigest.com"),
        "subheading_image_prompt": get_subheading_prompt("aidiscoverydigest.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "16:9"},
            "subheading": {"enable_compression": True, "aspect_ratio": "16:9"},
        },
    },

    "clearainews.com": {
        "domain": "clearainews.com",
        "niche": "AI News & Tech Journalism",
        "h2_count": "Automatic",
        "h2_auto_limit": 8,
        "h2_lower_limit": 4,
        "ai_outline_quality": "Normal $",
        "section_length": "Short",
        "voice": "Third Person (He, She, It, They)",
        "intro": "Standard Intro",
        "faq": "No FAQ",
        "audience_personality": "Ruler",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "ideogram 3b $.060/img (F)",
        "subheading_image_quantity": "Every 3rd H2 (Max 3)",
        "subheading_images_model": "flux schnell $.003/img (F)",
        "ai_model_image_prompts": "GPT-4o Mini (OA)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": False,
        "blockquotes": True,
        "literary_devices": False,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "deep_research": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "deep_research_settings": {"ai_model": "Sonar Online (OR)", "links_per_article": "5", "links_per_subheading": "2"},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("clearainews.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("clearainews.com"),
        "link_pack": True,
        "link_pack_settings": {"pack_name": "clearainews_internal", "insertion_limit": "3"},
        "wordpress_category": "AI News",
        "wordpress_settings": {
            "site_url": "https://clearainews.com",
            "user_name": "ClearAIEditor",
            "category": "AI News",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("clearainews.com"),
        "subheading_image_prompt": get_subheading_prompt("clearainews.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "landscape_16_9",
                         "magic_prompt": "AUTO", "style": "REALISTIC", "activate_similarity": "no"},
            "subheading": {"enable_compression": True, "aspect_ratio": "16:9"},
        },
    },

    "wealthfromai.com": {
        "domain": "wealthfromai.com",
        "niche": "AI-Powered Income Strategies",
        "h2_count": "Automatic",
        "h2_auto_limit": 10,
        "h2_lower_limit": 5,
        "ai_outline_quality": "High $$",
        "section_length": "Medium",
        "voice": "Second Person (You, Your, Yours)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Hero",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "ideogram 3t $.030/img (F)",
        "subheading_image_quantity": "Every Other H2 (Max 5)",
        "subheading_images_model": "flux dev $.025/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Haiku (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": True,
        "blockquotes": True,
        "literary_devices": False,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "deep_research": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "deep_research_settings": {"ai_model": "Sonar Online (OR)", "links_per_article": "3", "links_per_subheading": "1"},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("wealthfromai.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("wealthfromai.com"),
        "link_pack": True,
        "link_pack_settings": {"pack_name": "wealthfromai_internal", "insertion_limit": "3"},
        "wordpress_category": "AI Income Strategies",
        "wordpress_settings": {
            "site_url": "https://wealthfromai.com",
            "user_name": "AIWealthGuide",
            "category": "AI Income Strategies",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("wealthfromai.com"),
        "subheading_image_prompt": get_subheading_prompt("wealthfromai.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "landscape_16_9",
                         "magic_prompt": "AUTO", "style": "REALISTIC", "activate_similarity": "no"},
            "subheading": {"enable_compression": True, "aspect_ratio": "16:9"},
        },
    },

    # ═══════════════════════════════════════
    # SMART HOME SITES
    # ═══════════════════════════════════════

    "smarthomewizards.com": {
        "domain": "smarthomewizards.com",
        "niche": "Smart Home Automation",
        "h2_count": "Automatic",
        "h2_auto_limit": 10,
        "h2_lower_limit": 5,
        "ai_outline_quality": "High $$",
        "section_length": "Medium",
        "voice": "Second Person (You, Your, Yours)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Magician",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "imagegen-4 $.050/img (F)",
        "subheading_image_quantity": "Every Other H2 (Max 5)",
        "subheading_images_model": "ideogram 3t $.030/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Haiku (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": True,
        "blockquotes": False,
        "literary_devices": False,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "link_pack": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "link_pack_settings": {"pack_name": "smarthomewizards_internal", "insertion_limit": "3"},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("smarthomewizards.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("smarthomewizards.com"),
        "wordpress_category": "Smart Home Guides",
        "wordpress_settings": {
            "site_url": "https://smarthomewizards.com",
            "user_name": "SmartHomeGuru",
            "category": "Smart Home Guides",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("smarthomewizards.com"),
        "subheading_image_prompt": get_subheading_prompt("smarthomewizards.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "16:9"},
            "subheading": {"enable_compression": True, "aspect_ratio": "landscape_16_9",
                           "magic_prompt": "AUTO", "style": "REALISTIC", "activate_similarity": "no"},
        },
    },

    "smarthomegearreviews.com": {
        "domain": "smarthomegearreviews.com",
        "niche": "Smart Home Product Reviews",
        "h2_count": "Automatic",
        "h2_auto_limit": 10,
        "h2_lower_limit": 5,
        "ai_outline_quality": "High $$",
        "section_length": "Medium",
        "voice": "Second Person (You, Your, Yours)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Everyman",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "gpt-image-1 med $.063/img (OA)",
        "subheading_image_quantity": "Every H2 (Max 5)",
        "subheading_images_model": "ideogram 3t $.030/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Haiku (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": True,
        "blockquotes": False,
        "literary_devices": False,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "link_pack": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "link_pack_settings": {"pack_name": "smarthomegearreviews_internal", "insertion_limit": "3"},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("smarthomegearreviews.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("smarthomegearreviews.com"),
        "wordpress_category": "Product Reviews",
        "wordpress_settings": {
            "site_url": "https://smarthomegearreviews.com",
            "user_name": "SmartHomeEditor",
            "category": "Product Reviews",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("smarthomegearreviews.com"),
        "subheading_image_prompt": get_subheading_prompt("smarthomegearreviews.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "16:9"},
            "subheading": {"enable_compression": True, "aspect_ratio": "landscape_16_9",
                           "magic_prompt": "AUTO", "style": "REALISTIC", "activate_similarity": "no"},
        },
    },

    "theconnectedhaven.com": {
        "domain": "theconnectedhaven.com",
        "niche": "Smart Home Lifestyle & Wellness",
        "h2_count": "Automatic",
        "h2_auto_limit": 8,
        "h2_lower_limit": 4,
        "ai_outline_quality": "High $$",
        "section_length": "Medium",
        "voice": "Second Person (You, Your, Yours)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Caregiver",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "flux pro $.040/img (F)",
        "subheading_image_quantity": "Every Other H2 (Max 3)",
        "subheading_images_model": "flux schnell $.003/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Haiku (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": False,
        "blockquotes": True,
        "literary_devices": True,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("theconnectedhaven.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("theconnectedhaven.com"),
        "link_pack": True,
        "link_pack_settings": {"pack_name": "theconnectedhaven_internal", "insertion_limit": "3"},
        "wordpress_category": "Connected Living",
        "wordpress_settings": {
            "site_url": "https://theconnectedhaven.com",
            "user_name": "TheSmartHomeGuide",
            "category": "Connected Living",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("theconnectedhaven.com"),
        "subheading_image_prompt": get_subheading_prompt("theconnectedhaven.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "16:9"},
            "subheading": {"enable_compression": True, "aspect_ratio": "16:9"},
        },
    },

    # ═══════════════════════════════════════
    # SPIRITUALITY & WITCHCRAFT
    # ═══════════════════════════════════════

    "witchcraftforbeginners.com": {
        "domain": "witchcraftforbeginners.com",
        "niche": "Witchcraft & Spirituality",
        "h2_count": "Automatic",
        "h2_auto_limit": 8,
        "h2_lower_limit": 4,
        "ai_outline_quality": "High $$",
        "section_length": "Medium",
        "voice": "Second Person (You, Your, Yours)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Creator",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "ideogram 3q $.090/img (F)",
        "subheading_image_quantity": "Every Other H2 (Max 5)",
        "subheading_images_model": "ideogram 3t $.030/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Sonnet (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": False,
        "blockquotes": True,
        "literary_devices": True,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("witchcraftforbeginners.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("witchcraftforbeginners.com"),
        "link_pack": True,
        "link_pack_settings": {"pack_name": "witchcraft_internal", "insertion_limit": "3"},
        "wordpress_category": "Witchcraft Basics",
        "wordpress_settings": {
            "site_url": "https://witchcraftforbeginners.com",
            "user_name": "MoonlightMystic",
            "category": "Witchcraft Basics",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("witchcraftforbeginners.com"),
        "subheading_image_prompt": get_subheading_prompt("witchcraftforbeginners.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "landscape_16_9",
                         "magic_prompt": "AUTO", "style": "REALISTIC", "activate_similarity": "no"},
            "subheading": {"enable_compression": True, "aspect_ratio": "landscape_16_9",
                           "magic_prompt": "AUTO", "style": "REALISTIC", "activate_similarity": "no"},
        },
    },

    "manifestandalign.com": {
        "domain": "manifestandalign.com",
        "niche": "Manifestation & Spiritual Growth",
        "h2_count": "Automatic",
        "h2_auto_limit": 8,
        "h2_lower_limit": 4,
        "ai_outline_quality": "High $$",
        "section_length": "Medium",
        "voice": "Second Person (You, Your, Yours)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Innocent",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "flux pro $.040/img (F)",
        "subheading_image_quantity": "Every 3rd H2 (Max 3)",
        "subheading_images_model": "flux schnell $.003/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Haiku (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": False,
        "blockquotes": True,
        "literary_devices": True,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("manifestandalign.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("manifestandalign.com"),
        "link_pack": True,
        "link_pack_settings": {"pack_name": "manifestandalign_internal", "insertion_limit": "3"},
        "wordpress_category": "Manifestation Guides",
        "wordpress_settings": {
            "site_url": "https://manifestandalign.com",
            "user_name": "ManifestMaster",
            "category": "Manifestation Guides",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("manifestandalign.com"),
        "subheading_image_prompt": get_subheading_prompt("manifestandalign.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "16:9"},
            "subheading": {"enable_compression": True, "aspect_ratio": "16:9"},
        },
    },

    # ═══════════════════════════════════════
    # LIFESTYLE & FAMILY
    # ═══════════════════════════════════════

    "family-flourish.com": {
        "domain": "family-flourish.com",
        "niche": "Family & Parenting",
        "h2_count": "Automatic",
        "h2_auto_limit": 8,
        "h2_lower_limit": 4,
        "ai_outline_quality": "High $$",
        "section_length": "Medium",
        "voice": "Second Person (You, Your, Yours)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Lover",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "flux dev $.025/img (F)",
        "subheading_image_quantity": "Every Other H2 (Max 3)",
        "subheading_images_model": "flux schnell $.003/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Haiku (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": False,
        "blockquotes": True,
        "literary_devices": False,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("family-flourish.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("family-flourish.com"),
        "link_pack": True,
        "link_pack_settings": {"pack_name": "family_flourish_internal", "insertion_limit": "3"},
        "wordpress_category": "Family Life",
        "wordpress_settings": {
            "site_url": "https://family-flourish.com",
            "user_name": "FamilyGrowthGuide",
            "category": "Family Life",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("family-flourish.com"),
        "subheading_image_prompt": get_subheading_prompt("family-flourish.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "16:9"},
            "subheading": {"enable_compression": True, "aspect_ratio": "16:9"},
        },
    },

    # ═══════════════════════════════════════
    # KNOWLEDGE & EDUCATION
    # ═══════════════════════════════════════

    "mythicalarchives.com": {
        "domain": "mythicalarchives.com",
        "niche": "Mythology & Folklore",
        "h2_count": "Automatic",
        "h2_auto_limit": 10,
        "h2_lower_limit": 5,
        "ai_outline_quality": "High $$",
        "section_length": "Long",
        "voice": "Third Person (He, She, It, They)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Sage",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "ideogram 3b $.060/img (F)",
        "subheading_image_quantity": "Every Other H2 (Max 5)",
        "subheading_images_model": "ideogram 3t $.030/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Sonnet (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": True,
        "blockquotes": True,
        "literary_devices": True,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "deep_research": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "deep_research_settings": {"ai_model": "Sonar Online (OR)", "links_per_article": "3", "links_per_subheading": "1"},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("mythicalarchives.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("mythicalarchives.com"),
        "link_pack": True,
        "link_pack_settings": {"pack_name": "mythicalarchives_internal", "insertion_limit": "3"},
        "wordpress_category": "Mythology",
        "wordpress_settings": {
            "site_url": "https://mythicalarchives.com",
            "user_name": "ArcaneArchivist",
            "category": "Mythology",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("mythicalarchives.com"),
        "subheading_image_prompt": get_subheading_prompt("mythicalarchives.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "landscape_16_9",
                         "magic_prompt": "AUTO", "style": "REALISTIC", "activate_similarity": "no"},
            "subheading": {"enable_compression": True, "aspect_ratio": "landscape_16_9",
                           "magic_prompt": "AUTO", "style": "REALISTIC", "activate_similarity": "no"},
        },
    },

    # ═══════════════════════════════════════
    # PRODUCT REVIEW SITES
    # ═══════════════════════════════════════

    "wearablegearreviews.com": {
        "domain": "wearablegearreviews.com",
        "niche": "Wearable Tech Reviews",
        "h2_count": "Automatic",
        "h2_auto_limit": 10,
        "h2_lower_limit": 5,
        "ai_outline_quality": "High $$",
        "section_length": "Medium",
        "voice": "Second Person (You, Your, Yours)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Explorer",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "gpt-image-1 med $.063/img (OA)",
        "subheading_image_quantity": "Every H2 (Max 5)",
        "subheading_images_model": "ideogram 3t $.030/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Haiku (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": True,
        "blockquotes": False,
        "literary_devices": False,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "link_pack": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "link_pack_settings": {"pack_name": "wearablegearreviews_internal", "insertion_limit": "3"},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("wearablegearreviews.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("wearablegearreviews.com"),
        "wordpress_category": "Wearable Reviews",
        "wordpress_settings": {
            "site_url": "https://wearablegearreviews.com",
            "user_name": "WearableReviewPro",
            "category": "Wearable Reviews",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("wearablegearreviews.com"),
        "subheading_image_prompt": get_subheading_prompt("wearablegearreviews.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "16:9"},
            "subheading": {"enable_compression": True, "aspect_ratio": "landscape_16_9",
                           "magic_prompt": "AUTO", "style": "REALISTIC", "activate_similarity": "no"},
        },
    },

    "pulsegearreviews.com": {
        "domain": "pulsegearreviews.com",
        "niche": "EDC & Tactical Gear Reviews",
        "h2_count": "Automatic",
        "h2_auto_limit": 10,
        "h2_lower_limit": 5,
        "ai_outline_quality": "High $$",
        "section_length": "Medium",
        "voice": "Second Person (You, Your, Yours)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Outlaw",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "gpt-image-1 low $.016/img (OA)",
        "subheading_image_quantity": "Every H2 (Max 5)",
        "subheading_images_model": "flux dev $.025/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Haiku (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": True,
        "blockquotes": False,
        "literary_devices": False,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "link_pack": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "link_pack_settings": {"pack_name": "pulsegearreviews_internal", "insertion_limit": "3"},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("pulsegearreviews.com")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("pulsegearreviews.com"),
        "wordpress_category": "EDC Reviews",
        "wordpress_settings": {
            "site_url": "https://pulsegearreviews.com",
            "user_name": "PulseGearEditor",
            "category": "EDC Reviews",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("pulsegearreviews.com"),
        "subheading_image_prompt": get_subheading_prompt("pulsegearreviews.com"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "16:9"},
            "subheading": {"enable_compression": True, "aspect_ratio": "16:9"},
        },
    },

    # ═══════════════════════════════════════
    # HOBBY & NICHE SITES
    # ═══════════════════════════════════════

    "bulletjournals.net": {
        "domain": "bulletjournals.net",
        "niche": "Bullet Journaling & Productivity",
        "h2_count": "Automatic",
        "h2_auto_limit": 8,
        "h2_lower_limit": 4,
        "ai_outline_quality": "High $$",
        "section_length": "Medium",
        "voice": "Second Person (You, Your, Yours)",
        "intro": "Standard Intro",
        "faq": "FAQ + Long Answers",
        "audience_personality": "Creator",
        "ai_model": "Claude-4.5 Sonnet (ANT)",
        "featured_image": "flux pro $.040/img (F)",
        "subheading_image_quantity": "Every Other H2 (Max 5)",
        "subheading_images_model": "flux schnell $.003/img (F)",
        "ai_model_image_prompts": "Claude-4.5 Haiku (ANT)",
        "ai_model_translation": "None",
        "lists": True,
        "tables": False,
        "blockquotes": True,
        "literary_devices": False,
        "enable_h3": True,
        "key_takeaways": True,
        "nuke_ai_words": True,
        "bold_readability": True,
        "disable_skinny_paragraphs": False,
        "disable_active_voice": False,
        "disable_conclusion": False,
        "auto_style": False,
        "automatic_keywords": False,
        "image_prompt_per_h2": False,
        "progress_indicator": True,
        "overwrite_url_cache": False,
        "serp_scraping": True,
        "serp_settings": {"country": "United States", "language": "English", "enable": True},
        "style_mimic": True,
        "style_mimic_settings": {"style_text": _load_style_sample("bulletjournals.net")},
        "custom_prompt": True,
        "custom_prompt_settings": _build_prompt_config("bulletjournals.net"),
        "link_pack": True,
        "link_pack_settings": {"pack_name": "bulletjournals_internal", "insertion_limit": "3"},
        "wordpress_category": "Bullet Journal Ideas",
        "wordpress_settings": {
            "site_url": "https://bulletjournals.net",
            "user_name": "BulletJournalPro",
            "category": "Bullet Journal Ideas",
            "article_status": "draft",
        },
        "featured_image_prompt": get_featured_prompt("bulletjournals.net"),
        "subheading_image_prompt": get_subheading_prompt("bulletjournals.net"),
        "image_options": {
            "featured": {"enable_compression": True, "aspect_ratio": "16:9"},
            "subheading": {"enable_compression": True, "aspect_ratio": "16:9"},
        },
    },
}


def get_preset(domain: str) -> Optional[Dict[str, Any]]:
    """Get preset config for a domain."""
    return SITE_PRESETS.get(domain)

def get_all_domains() -> list:
    """Get list of all configured domains."""
    return list(SITE_PRESETS.keys())

def get_presets_by_niche(niche_keyword: str) -> Dict[str, Dict[str, Any]]:
    """Find presets matching a niche keyword."""
    return {
        domain: config
        for domain, config in SITE_PRESETS.items()
        if niche_keyword.lower() in config.get("niche", "").lower()
    }
