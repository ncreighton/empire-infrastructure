"""
Image meta-prompts for all 14 active sites.

These are META-PROMPTS — instructions for the AI text model (selected in
"AI Model for Image Prompts") that tell it HOW to write the actual image
prompt sent to the image generation model (Ideogram, Flux, GPT Image, etc.).

The meta-prompt is NOT a direct image prompt. It must explicitly instruct
the AI to "create/write an image prompt" based on the article content.

Architecture:
  - Lead with "Read and analyze..." to force the AI to parse the actual topic
  - Encode brand identity as photography STYLE (lighting, color, mood)
  - Include composition and framing direction (close-up, wide shot, etc.)
  - Specify the output medium ("editorial photograph", "product shot", etc.)
  - Include negative instructions ("no text, no watermarks, no logos")
  - Set word count for the generated prompt (40-60 words)
  - Featured prompts use {title} placeholder
  - Subheading prompts use BOTH {title} AND {subheading} placeholders

Placeholders (replaced by ZimmWriter at runtime):
  {title}      = the article's title
  {subheading} = the current H2/H3 heading (subheading prompts only)
"""

from typing import Dict


# ═══════════════════════════════════════════
# BRAND STYLE DEFINITIONS
# ═══════════════════════════════════════════
# Style describes HOW the image looks (lighting, color, mood, aesthetic)
# NOT what objects or scenes appear. The subject comes from the article topic.

_BRAND_STYLES = {
    # AI & Technology (4 sites)
    "ai_professional": {
        "palette": "cool blue and silver tones with white accent lighting",
        "lighting": "clean professional studio lighting with soft diffusion",
        "mood": "modern, innovative, trustworthy",
        "medium": "editorial technology photograph",
        "composition": "sharp focus on the main subject with a clean blurred background",
    },
    "ai_research": {
        "palette": "deep navy blue with warm amber accent highlights",
        "lighting": "dramatic side lighting with analytical precision",
        "mood": "intellectual, cutting-edge, authoritative",
        "medium": "editorial research photograph",
        "composition": "shallow depth of field, the key subject fills the frame",
    },
    "ai_news": {
        "palette": "high-contrast black and white with vivid accent color pops",
        "lighting": "bold directional lighting with sharp defined shadows",
        "mood": "dynamic, urgent, journalistic energy",
        "medium": "photojournalistic editorial shot",
        "composition": "dynamic angle, decisive moment framing",
    },
    "ai_finance": {
        "palette": "rich gold, deep navy, and polished chrome tones",
        "lighting": "warm overhead key light with cool fill, professional grade",
        "mood": "sophisticated, premium, authoritative",
        "medium": "executive business photograph",
        "composition": "centered subject with balanced negative space",
    },

    # Smart Home (3 sites)
    "smart_home_tutorial": {
        "palette": "warm whites and soft blues with cozy ambient glow",
        "lighting": "warm ambient room lighting with soft LED accent glow",
        "mood": "inviting, approachable, practical",
        "medium": "lifestyle home technology photograph",
        "composition": "environmental shot showing the device or setup in its natural home context",
    },
    "smart_home_product": {
        "palette": "clean white and light gray with subtle color from the product",
        "lighting": "controlled studio lighting, soft box diffused, minimal shadows",
        "mood": "precise, informative, polished",
        "medium": "product photography shot",
        "composition": "product in sharp focus against clean minimal background, three-quarter angle",
    },
    "smart_home_lifestyle": {
        "palette": "warm earthy tones with natural green and wood accents",
        "lighting": "soft golden hour natural light streaming through windows",
        "mood": "cozy, comfortable, lived-in warmth",
        "medium": "lifestyle interior photograph",
        "composition": "wide environmental shot with the subject naturally placed in a home setting",
    },

    # Spiritual (2 sites)
    "witchcraft": {
        "palette": "deep purple, burnished gold, and midnight blue",
        "lighting": "warm flickering candlelight mixed with ethereal moonlight glow",
        "mood": "mystical, atmospheric, reverent",
        "medium": "atmospheric still-life photograph",
        "composition": "close-up or tabletop arrangement with rich bokeh background",
    },
    "manifestation": {
        "palette": "soft lavender, rose gold, and celestial white",
        "lighting": "dreamy soft-focus backlighting with gentle lens flare",
        "mood": "ethereal, hopeful, transcendent",
        "medium": "dreamy lifestyle photograph",
        "composition": "centered ethereal composition with abundant soft bokeh",
    },

    # Family
    "family": {
        "palette": "bright warm tones, cheerful yellows, natural greens",
        "lighting": "bright natural sunlight, golden hour warmth",
        "mood": "joyful, genuine, heartfelt",
        "medium": "candid family lifestyle photograph",
        "composition": "natural candid framing, real moments, genuine expressions",
    },

    # Mythology
    "mythology": {
        "palette": "rich jewel tones — emerald, ruby, sapphire, and aged gold",
        "lighting": "dramatic volumetric lighting with god rays and atmospheric haze",
        "mood": "epic, ancient, awe-inspiring",
        "medium": "cinematic fantasy photograph",
        "composition": "epic wide-angle establishing shot with dramatic scale",
    },

    # Reviews (2 sites)
    "wearable_reviews": {
        "palette": "natural outdoor tones with the device color as accent",
        "lighting": "bright natural daylight with active energy",
        "mood": "energetic, authentic, performance-focused",
        "medium": "active lifestyle product photograph",
        "composition": "the wearable device in sharp focus, active context blurred behind",
    },
    "tactical_reviews": {
        "palette": "dark tactical tones with bold orange or red accents",
        "lighting": "harsh directional light, strong contrast, defined edges",
        "mood": "rugged, reliable, no-nonsense",
        "medium": "product action photograph",
        "composition": "dramatic close-up detail shot, texture and build quality visible",
    },

    # Bullet Journals
    "bullet_journal": {
        "palette": "warm pastels, colorful washi tape tones, natural paper whites",
        "lighting": "warm overhead natural window light, soft and even",
        "mood": "creative, organized, inspiring",
        "medium": "flat-lay creative photography",
        "composition": "overhead flat-lay with artful arrangement of journaling supplies and pages",
    },
}

# Map each domain to its brand style key
_DOMAIN_STYLE_MAP = {
    "aiinactionhub.com": "ai_professional",
    "aidiscoverydigest.com": "ai_research",
    "clearainews.com": "ai_news",
    "wealthfromai.com": "ai_finance",
    "smarthomewizards.com": "smart_home_tutorial",
    "smarthomegearreviews.com": "smart_home_product",
    "theconnectedhaven.com": "smart_home_lifestyle",
    "witchcraftforbeginners.com": "witchcraft",
    "manifestandalign.com": "manifestation",
    "family-flourish.com": "family",
    "mythicalarchives.com": "mythology",
    "wearablegearreviews.com": "wearable_reviews",
    "pulsegearreviews.com": "tactical_reviews",
    "bulletjournals.net": "bullet_journal",
}


def _build_featured_prompt(domain: str) -> str:
    """Build a detailed topic-adaptive featured image meta-prompt."""
    style_key = _DOMAIN_STYLE_MAP.get(domain)
    if not style_key:
        return ""
    s = _BRAND_STYLES[style_key]
    return (
        'Read and deeply analyze the article title "{title}". '
        "Identify the specific real-world subject, activity, product, or concept "
        "that the article is actually about. "
        f"Then write a 40-60 word {s['medium']} prompt that directly depicts "
        "that specific subject as the hero of the image. "
        f"Color palette: {s['palette']}. "
        f"Lighting: {s['lighting']}. "
        f"Mood: {s['mood']}. "
        f"Composition: {s['composition']}. "
        "The image must be visually distinct and uniquely tied to this "
        "particular article topic — a viewer should immediately understand "
        "what the article is about from the image alone. "
        "Do not include any text, words, watermarks, or logos in the image."
    )


def _build_subheading_prompt(domain: str) -> str:
    """Build a detailed topic-adaptive subheading image meta-prompt."""
    style_key = _DOMAIN_STYLE_MAP.get(domain)
    if not style_key:
        return ""
    s = _BRAND_STYLES[style_key]
    return (
        'Read the section heading "{subheading}" in the context of the article '
        '"{title}". Identify the one specific concept, step, product, or detail '
        "that this particular section covers. "
        f"Write a 35-45 word {s['medium']} prompt showing that exact concept. "
        f"Color palette: {s['palette']}. "
        f"Lighting: {s['lighting']}. "
        "This subheading image must look noticeably different from the article's "
        "featured image and from images for other subheadings — focus on the "
        "unique aspect this section discusses. "
        "Do not include any text, words, watermarks, or logos in the image."
    )


# ═══════════════════════════════════════════
# FEATURED IMAGE META-PROMPTS
# ═══════════════════════════════════════════

FEATURED_IMAGE_PROMPTS: Dict[str, str] = {
    domain: _build_featured_prompt(domain)
    for domain in _DOMAIN_STYLE_MAP
}


# ═══════════════════════════════════════════
# SUBHEADING IMAGE META-PROMPTS
# ═══════════════════════════════════════════

SUBHEADING_IMAGE_PROMPTS: Dict[str, str] = {
    domain: _build_subheading_prompt(domain)
    for domain in _DOMAIN_STYLE_MAP
}


def get_featured_prompt(domain: str) -> str:
    """Get the featured image meta-prompt for a domain."""
    return FEATURED_IMAGE_PROMPTS.get(domain, "")


def get_subheading_prompt(domain: str) -> str:
    """Get the subheading image meta-prompt for a domain."""
    return SUBHEADING_IMAGE_PROMPTS.get(domain, "")


def get_all_prompts(domain: str) -> dict:
    """Get both prompts for a domain."""
    return {
        "featured": get_featured_prompt(domain),
        "subheading": get_subheading_prompt(domain),
    }


def get_brand_style(domain: str) -> str:
    """Get the brand style description for a domain."""
    style_key = _DOMAIN_STYLE_MAP.get(domain)
    if not style_key:
        return ""
    s = _BRAND_STYLES[style_key]
    return f"{s['palette']}; {s['lighting']}; {s['mood']}"
