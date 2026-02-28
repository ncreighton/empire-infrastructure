#!/usr/bin/env python3
"""
ForgeFiles Product Profiles
==============================
Product category knowledge base for intelligent script generation.
Classifies 3D print models by keyword matching into 8 categories,
each with hooks, retention anchors, CTAs, music mood, transition style,
and voice tuning parameters.
"""

import random
from pathlib import Path


# ============================================================================
# CATEGORY → MATERIAL & LIGHTING DEFAULTS
# ============================================================================

CATEGORY_MATERIALS = {
    "display_model":    "silk_silver_pla",   # dramatic showcase
    "articulated_toy":  "blue_pla",          # playful color
    "functional_print": "gray_pla",          # clean utility look
    "planter":          "green_pla",         # natural earth tone
    "lamp_lighting":    "resin_clear",       # translucent for light
    "collectible":      "silk_gold_pla",     # premium collector feel
    "home_decor":       "matte_white",       # modern design
    "pet_accessory":    "orange_pla",        # warm, fun color
}

CATEGORY_LIGHTING = {
    "display_model":    "dramatic",
    "articulated_toy":  "studio",
    "functional_print": "product",
    "planter":          "studio",
    "lamp_lighting":    "dramatic",
    "collectible":      "dramatic",
    "home_decor":       "product",
    "pet_accessory":    "studio",
}

CATEGORY_CAMERA_STYLE = {
    "display_model":    "hero_spin",
    "collectible":      "hero_spin",
    "articulated_toy":  "orbital",
    "functional_print": "dolly_in",
    "planter":          "standard",
    "lamp_lighting":    "standard",
    "home_decor":       "standard",
    "pet_accessory":    "standard",
}


# ============================================================================
# PRODUCT CATEGORY PROFILES
# ============================================================================

PROFILES = {
    "articulated_toy": {
        "keywords": [
            "articulated", "flexi", "flexible", "fidget", "toy",
            "octopus", "gecko", "lizard", "snake", "caterpillar",
            "print-in-place", "print_in_place", "movable", "jointed",
        ],
        "tone": "playful, energetic",
        "hooks": [
            "This {name} moves right off the build plate.",
            "No assembly. No glue. Just print and play.",
            "Watch every joint on this {name} come alive.",
            "You won't believe this printed in one piece.",
            "The most satisfying print you'll run this week.",
        ],
        "mid_roll_anchors": [
            "And every single joint moves freely — straight off the bed.",
            "The detail only gets better up close.",
            "This is the kind of print people pick up and can't put down.",
        ],
        "cta_variants": [
            "Grab the STL and print your own — link in bio.",
            "File's ready to download. Link below.",
            "Print this tonight. STL link in the description.",
        ],
        "music_mood": "upbeat",
        "transition_style": "dynamic",
        "energy_level": "high",
        "voice_settings": {
            "stability": 0.40,
            "similarity_boost": 0.80,
            "style": 0.35,
        },
    },
    "display_model": {
        "keywords": [
            "dragon", "skull", "statue", "bust", "figure", "figurine",
            "sculpture", "guardian", "phoenix", "wolf", "lion", "eagle",
            "knight", "warrior", "demon", "angel", "mythical", "beast",
        ],
        "tone": "dramatic, cinematic",
        "hooks": [
            "Some prints are functional. This one is art.",
            "This {name} demands a spot on your shelf.",
            "Every angle reveals something new.",
            "You're going to want to print this twice.",
            "The kind of detail that stops people mid-scroll.",
        ],
        "mid_roll_anchors": [
            "And the detail only gets better up close.",
            "Look at how the light catches every surface.",
            "This is what high-detail printing looks like.",
        ],
        "cta_variants": [
            "The STL is ready. Link in the description.",
            "Download and print — link below.",
            "Make it yours. STL link in bio.",
        ],
        "music_mood": "epic",
        "transition_style": "cinematic",
        "energy_level": "medium",
        "voice_settings": {
            "stability": 0.55,
            "similarity_boost": 0.75,
            "style": 0.25,
        },
    },
    "functional_print": {
        "keywords": [
            "phone", "stand", "holder", "mount", "bracket", "clip",
            "organizer", "tool", "hook", "hanger", "shelf", "rack",
            "adapter", "case", "enclosure", "box", "tray", "dock",
            "functional", "utility", "practical",
        ],
        "tone": "practical, problem-solving",
        "hooks": [
            "You didn't know you needed this until now.",
            "Simple design. Solves a real problem.",
            "This is why you own a 3D printer.",
            "Stop buying what you can print.",
            "The print that actually earns its place.",
        ],
        "mid_roll_anchors": [
            "And it fits perfectly — every time.",
            "Designed for real-world use, not just looks.",
            "No supports needed. Just slice and go.",
        ],
        "cta_variants": [
            "STL's free to download — link below.",
            "Grab the file and print it tonight. Link in bio.",
            "Download link in the description.",
        ],
        "music_mood": "tech",
        "transition_style": "clean",
        "energy_level": "medium",
        "voice_settings": {
            "stability": 0.55,
            "similarity_boost": 0.70,
            "style": 0.15,
        },
    },
    "planter": {
        "keywords": [
            "planter", "pot", "vase", "succulent", "flower", "garden",
            "plant", "herb", "botanical", "terrarium",
        ],
        "tone": "warm, natural",
        "hooks": [
            "Your plants deserve better than a plastic pot.",
            "This {name} looks even better with something growing in it.",
            "Print it. Plant it. Love it.",
            "The planter your windowsill has been waiting for.",
            "Where design meets nature.",
        ],
        "mid_roll_anchors": [
            "And it has a built-in drainage system.",
            "The texture really comes alive in matte PLA.",
            "It prints beautifully in any earth tone.",
        ],
        "cta_variants": [
            "STL link in the description. Happy planting.",
            "Grab the file — link below.",
            "Download and start printing. Link in bio.",
        ],
        "music_mood": "chill",
        "transition_style": "organic",
        "energy_level": "low",
        "voice_settings": {
            "stability": 0.60,
            "similarity_boost": 0.70,
            "style": 0.15,
        },
    },
    "lamp_lighting": {
        "keywords": [
            "lamp", "light", "lantern", "lithophane", "shade",
            "nightlight", "led", "glow", "candle", "luminary",
        ],
        "tone": "atmospheric, moody",
        "hooks": [
            "Wait until you see this with the light on.",
            "This {name} transforms the whole room.",
            "Printed by day. Glowing by night.",
            "The glow on this one is unreal.",
            "Ambient lighting, straight off the build plate.",
        ],
        "mid_roll_anchors": [
            "And the way light passes through the layers is mesmerizing.",
            "It looks completely different with the lights off.",
            "The wall shadows alone are worth the print.",
        ],
        "cta_variants": [
            "Print your own — STL link below.",
            "File's ready. Link in the description.",
            "Download the STL and light it up. Link in bio.",
        ],
        "music_mood": "dark",
        "transition_style": "atmospheric",
        "energy_level": "low",
        "voice_settings": {
            "stability": 0.60,
            "similarity_boost": 0.75,
            "style": 0.20,
        },
    },
    "collectible": {
        "keywords": [
            "mini", "miniature", "tabletop", "dnd", "warhammer",
            "rpg", "chess", "token", "dice", "tower", "terrain",
            "collectible", "collector", "series", "set",
        ],
        "tone": "bold, commanding",
        "hooks": [
            "Add this {name} to your collection.",
            "Your tabletop just got an upgrade.",
            "This level of detail at this scale — insane.",
            "Built for collectors who notice the details.",
            "The centerpiece your shelf needs.",
        ],
        "mid_roll_anchors": [
            "And the scale detail is wild — zoom in.",
            "Every millimeter is intentional.",
            "It paints beautifully too.",
        ],
        "cta_variants": [
            "Download the STL — link in the description.",
            "Grab the file and start printing. Link below.",
            "STL available now. Link in bio.",
        ],
        "music_mood": "epic",
        "transition_style": "dramatic",
        "energy_level": "high",
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.80,
            "style": 0.30,
        },
    },
    "home_decor": {
        "keywords": [
            "decor", "decoration", "geometric", "wall", "art",
            "shelf", "bookend", "coaster", "frame", "modern",
            "abstract", "ornament", "sculpture",
        ],
        "tone": "stylish, design-focused",
        "hooks": [
            "Printed decor that looks like it belongs in a design store.",
            "This {name} ties the whole room together.",
            "Your friends will never guess this was 3D printed.",
            "Modern design, made at home.",
            "When your printer doubles as an interior designer.",
        ],
        "mid_roll_anchors": [
            "And it looks even better in matte black.",
            "The geometry catches light beautifully.",
            "It fits right into any modern space.",
        ],
        "cta_variants": [
            "STL ready to download — link below.",
            "Grab the file. Link in the description.",
            "Make it yours. Download link in bio.",
        ],
        "music_mood": "chill",
        "transition_style": "elegant",
        "energy_level": "low",
        "voice_settings": {
            "stability": 0.55,
            "similarity_boost": 0.75,
            "style": 0.20,
        },
    },
    "pet_accessory": {
        "keywords": [
            "pet", "dog", "cat", "bowl", "feeder", "toy",
            "collar", "tag", "leash", "treat", "animal",
        ],
        "tone": "warm, fun",
        "hooks": [
            "Your pet deserves custom gear.",
            "Printed with love — for the furry one.",
            "This {name} is a hit with every dog that sees it.",
            "Custom pet accessories, fresh off the printer.",
            "Because store-bought is boring.",
        ],
        "mid_roll_anchors": [
            "And it's tough enough for daily use.",
            "Pet-safe PLA — no sharp edges.",
            "You can customize the size for any breed.",
        ],
        "cta_variants": [
            "STL link in the description. Print one for your pet.",
            "Download the file — link below.",
            "Grab the STL. Link in bio.",
        ],
        "music_mood": "upbeat",
        "transition_style": "fun",
        "energy_level": "high",
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.75,
            "style": 0.30,
        },
    },
}

# Default profile for unclassified models
DEFAULT_PROFILE = {
    "category": "display_model",
    "tone": "confident, professional",
    "hooks": [
        "Check this out.",
        "You need to see this {name}.",
        "New design just dropped.",
        "Fresh off the build plate.",
        "Here's something worth printing.",
    ],
    "mid_roll_anchors": [
        "And the detail speaks for itself.",
        "Every surface is intentional.",
        "It prints cleaner than you'd expect.",
    ],
    "cta_variants": [
        "STL link in the description.",
        "Download link below.",
        "Grab the file — link in bio.",
    ],
    "music_mood": "chill",
    "transition_style": "cinematic",
    "energy_level": "medium",
    "voice_settings": {
        "stability": 0.50,
        "similarity_boost": 0.75,
        "style": 0.20,
    },
}


# ============================================================================
# CLASSIFICATION
# ============================================================================

def classify_product(model_name):
    """Classify a product by keyword matching on the model name.

    Args:
        model_name: STL filename stem (e.g. 'crystal_dragon_with_spread_wings')

    Returns:
        dict — full profile with 'category' key added
    """
    name_lower = model_name.lower().replace("-", "_")
    tokens = set(name_lower.split("_"))
    # Also match substrings for compound keywords
    name_str = name_lower.replace("_", " ")

    best_category = None
    best_score = 0

    for category, profile in PROFILES.items():
        score = 0
        for keyword in profile["keywords"]:
            kw = keyword.lower().replace("-", "_")
            if kw in tokens:
                score += 2  # exact token match
            elif kw in name_str:
                score += 1  # substring match

        if score > best_score:
            best_score = score
            best_category = category

    if best_category and best_score > 0:
        result = dict(PROFILES[best_category])
        result["category"] = best_category
        result["match_score"] = best_score
        return result

    # No match — return default
    result = dict(DEFAULT_PROFILE)
    result["match_score"] = 0
    return result


# ============================================================================
# MUSIC MOOD MAPPING
# ============================================================================

MOOD_TO_FILENAME = {
    "chill": "chill_ambient_loop",
    "epic": "epic_cinematic_swell",
    "dark": "dark_moody_atmosphere",
    "tech": "tech_electronic_pulse",
    "upbeat": "upbeat_positive_energy",
}


def get_music_for_mood(mood, music_tracks=None):
    """Map a mood string to the corresponding music file.

    Args:
        mood: One of chill/epic/dark/tech/upbeat
        music_tracks: List of available music file paths (optional)

    Returns:
        Path string to the best matching music file, or None
    """
    target = MOOD_TO_FILENAME.get(mood, "chill_ambient_loop")

    if music_tracks:
        for track in music_tracks:
            if target in Path(track).stem.lower():
                return track
        # Fallback: return first available
        return music_tracks[0] if music_tracks else None

    # Try default location
    default_path = Path(__file__).resolve().parent.parent / "brand_assets" / "music" / f"{target}.mp3"
    if default_path.exists():
        return str(default_path)
    return None


# ============================================================================
# AUTO-SELECTION HELPERS
# ============================================================================

def get_material_for_model(model_name):
    """Return the best material name for a model based on its category."""
    profile = classify_product(model_name)
    return CATEGORY_MATERIALS.get(profile["category"], "gray_pla")


def get_lighting_for_model(model_name):
    """Return the best lighting setup name for a model."""
    profile = classify_product(model_name)
    return CATEGORY_LIGHTING.get(profile["category"], "studio")


def get_camera_style_for_model(model_name):
    """Return the best camera style for a model based on its category."""
    profile = classify_product(model_name)
    return CATEGORY_CAMERA_STYLE.get(profile["category"], "standard")


# ============================================================================
# VOICE TUNING
# ============================================================================

def get_voice_for_category(category):
    """Get ElevenLabs voice settings tuned for the product category.

    All categories use the same voice ID (George) but with different
    stability/style parameters to vary energy and delivery.

    Returns:
        dict with voice_id, stability, similarity_boost, style, use_speaker_boost
    """
    profile = PROFILES.get(category, DEFAULT_PROFILE)
    settings = profile.get("voice_settings", DEFAULT_PROFILE["voice_settings"])

    return {
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",  # George
        "stability": settings.get("stability", 0.50),
        "similarity_boost": settings.get("similarity_boost", 0.75),
        "style": settings.get("style", 0.20),
        "use_speaker_boost": True,
    }


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    test_names = sys.argv[1:] or [
        "crystal_dragon_with_spread_wings",
        "articulated_octopus",
        "phone_stand_minimal",
        "geometric_vase",
        "dragon_guardian",
        "flexi_gecko",
    ]

    for name in test_names:
        profile = classify_product(name)
        print(f"\n{'=' * 50}")
        print(f"  Model: {name}")
        print(f"  Category: {profile['category']}")
        print(f"  Score: {profile['match_score']}")
        print(f"  Tone: {profile['tone']}")
        print(f"  Music: {profile['music_mood']}")
        print(f"  Transitions: {profile['transition_style']}")
        print(f"  Energy: {profile['energy_level']}")
        print(f"  Hook: {random.choice(profile['hooks']).format(name=name.replace('_', ' ').title())}")
        voice = get_voice_for_category(profile['category'])
        print(f"  Voice: stability={voice['stability']}, style={voice['style']}")
