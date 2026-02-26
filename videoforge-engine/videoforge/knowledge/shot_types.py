"""30+ shot types with mood mappings and visual descriptions."""

SHOT_TYPES = {
    "wide_establishing": {
        "name": "Wide Establishing",
        "description": "Full environment context, sets the scene",
        "mood": ["epic", "cinematic", "calm"],
        "best_for": ["intro", "transition", "location_reveal"],
        "duration_range": (3, 6),
    },
    "medium": {
        "name": "Medium Shot",
        "description": "Subject from waist up, balanced framing",
        "mood": ["neutral", "conversational", "educational"],
        "best_for": ["narration", "tutorial", "explanation"],
        "duration_range": (3, 8),
    },
    "close_up": {
        "name": "Close Up",
        "description": "Face or detail focus, emotional connection",
        "mood": ["intimate", "dramatic", "emotional"],
        "best_for": ["reaction", "detail", "emphasis"],
        "duration_range": (2, 5),
    },
    "extreme_close_up": {
        "name": "Extreme Close Up",
        "description": "Tight detail — eyes, texture, ingredient",
        "mood": ["intense", "mysterious", "dramatic"],
        "best_for": ["detail_reveal", "tension", "product_focus"],
        "duration_range": (1, 3),
    },
    "overhead_flat_lay": {
        "name": "Overhead / Flat Lay",
        "description": "Bird's eye view, perfect for arrangements",
        "mood": ["organized", "aesthetic", "calm"],
        "best_for": ["tutorial", "product_display", "altar_setup"],
        "duration_range": (3, 6),
    },
    "low_angle": {
        "name": "Low Angle",
        "description": "Looking up at subject, conveys power",
        "mood": ["epic", "powerful", "dramatic"],
        "best_for": ["hero_moment", "authority", "reveal"],
        "duration_range": (2, 5),
    },
    "high_angle": {
        "name": "High Angle",
        "description": "Looking down, vulnerability or overview",
        "mood": ["gentle", "overview", "reflective"],
        "best_for": ["overview", "vulnerability", "workspace"],
        "duration_range": (2, 5),
    },
    "dutch_angle": {
        "name": "Dutch Angle",
        "description": "Tilted frame, creates unease or energy",
        "mood": ["edgy", "dynamic", "unsettling"],
        "best_for": ["tension", "surprise", "pattern_interrupt"],
        "duration_range": (1, 3),
    },
    "slow_zoom_in": {
        "name": "Slow Zoom In",
        "description": "Gradual focus pull, builds intensity",
        "mood": ["building", "dramatic", "emotional"],
        "best_for": ["emphasis", "revelation", "tension_build"],
        "duration_range": (3, 7),
    },
    "slow_zoom_out": {
        "name": "Slow Zoom Out",
        "description": "Reveal wider context, creates perspective",
        "mood": ["epic", "reflective", "resolution"],
        "best_for": ["reveal", "conclusion", "perspective"],
        "duration_range": (3, 7),
    },
    "pan_left": {
        "name": "Pan Left",
        "description": "Horizontal sweep left, exploration",
        "mood": ["flowing", "exploratory", "cinematic"],
        "best_for": ["scene_scan", "transition", "reveal"],
        "duration_range": (3, 6),
    },
    "pan_right": {
        "name": "Pan Right",
        "description": "Horizontal sweep right, progression",
        "mood": ["flowing", "progressive", "cinematic"],
        "best_for": ["scene_scan", "timeline", "comparison"],
        "duration_range": (3, 6),
    },
    "tilt_up": {
        "name": "Tilt Up",
        "description": "Vertical sweep upward, aspiration",
        "mood": ["uplifting", "epic", "aspirational"],
        "best_for": ["reveal", "hero_moment", "scale"],
        "duration_range": (2, 5),
    },
    "tilt_down": {
        "name": "Tilt Down",
        "description": "Vertical sweep downward, grounding",
        "mood": ["grounding", "revealing", "focused"],
        "best_for": ["detail_reveal", "grounding", "approach"],
        "duration_range": (2, 5),
    },
    "tracking_forward": {
        "name": "Tracking Forward",
        "description": "Moving toward subject, engagement pull",
        "mood": ["immersive", "dynamic", "engaging"],
        "best_for": ["approach", "engagement", "journey"],
        "duration_range": (3, 6),
    },
    "tracking_side": {
        "name": "Tracking Side",
        "description": "Moving alongside subject, journey feel",
        "mood": ["dynamic", "narrative", "flowing"],
        "best_for": ["process", "journey", "comparison"],
        "duration_range": (3, 6),
    },
    "static_locked": {
        "name": "Static Locked",
        "description": "Fixed camera, clean and professional",
        "mood": ["clean", "professional", "stable"],
        "best_for": ["text_overlay", "data", "talking_head"],
        "duration_range": (3, 8),
    },
    "whip_pan": {
        "name": "Whip Pan",
        "description": "Fast pan blur, high energy transition",
        "mood": ["energetic", "chaotic", "exciting"],
        "best_for": ["pattern_interrupt", "transition", "surprise"],
        "duration_range": (0.5, 1.5),
    },
    "dolly_zoom": {
        "name": "Dolly Zoom (Vertigo)",
        "description": "Push-pull effect, disorientation",
        "mood": ["unsettling", "dramatic", "surreal"],
        "best_for": ["revelation", "shock", "realization"],
        "duration_range": (2, 4),
    },
    "aerial_drone": {
        "name": "Aerial / Drone",
        "description": "High altitude sweeping view",
        "mood": ["epic", "majestic", "establishing"],
        "best_for": ["intro", "scale", "travel"],
        "duration_range": (4, 8),
    },
    "split_screen": {
        "name": "Split Screen",
        "description": "Two visuals side by side",
        "mood": ["comparative", "dynamic", "informational"],
        "best_for": ["comparison", "before_after", "parallel"],
        "duration_range": (3, 6),
    },
    "text_card": {
        "name": "Text Card",
        "description": "Full screen text on gradient/image",
        "mood": ["clean", "impactful", "minimal"],
        "best_for": ["hook", "stat", "quote", "cta"],
        "duration_range": (2, 4),
    },
    "b_roll_montage": {
        "name": "B-Roll Montage",
        "description": "Quick cuts of related footage",
        "mood": ["dynamic", "energetic", "informational"],
        "best_for": ["process", "overview", "energy_boost"],
        "duration_range": (3, 8),
    },
    "product_hero": {
        "name": "Product Hero",
        "description": "Glamour shot of product, clean background",
        "mood": ["premium", "focused", "aspirational"],
        "best_for": ["product_reveal", "review", "recommendation"],
        "duration_range": (3, 5),
    },
    "screen_recording": {
        "name": "Screen Recording",
        "description": "Software/app demo capture",
        "mood": ["tutorial", "practical", "educational"],
        "best_for": ["demo", "tutorial", "walkthrough"],
        "duration_range": (5, 15),
    },
    "particle_overlay": {
        "name": "Particle Overlay",
        "description": "Sparkles, dust, embers over footage",
        "mood": ["magical", "ethereal", "dreamy"],
        "best_for": ["witchcraft", "spiritual", "ambient"],
        "duration_range": (3, 6),
    },
    "kinetic_text": {
        "name": "Kinetic Typography",
        "description": "Animated text movement and emphasis",
        "mood": ["energetic", "modern", "informational"],
        "best_for": ["quote", "stat", "listicle", "key_point"],
        "duration_range": (2, 5),
    },
    "time_lapse": {
        "name": "Time Lapse",
        "description": "Compressed time showing process/change",
        "mood": ["dynamic", "satisfying", "productive"],
        "best_for": ["process", "transformation", "setup"],
        "duration_range": (3, 6),
    },
    "slow_motion": {
        "name": "Slow Motion",
        "description": "Slowed footage for emphasis and beauty",
        "mood": ["dramatic", "beautiful", "emphatic"],
        "best_for": ["detail", "emphasis", "aesthetic"],
        "duration_range": (2, 5),
    },
    "animated_diagram": {
        "name": "Animated Diagram",
        "description": "Motion graphics explaining a concept",
        "mood": ["educational", "clean", "professional"],
        "best_for": ["explanation", "process", "data"],
        "duration_range": (4, 8),
    },
}


def get_shot_type(key: str) -> dict:
    """Get shot type data by key."""
    return SHOT_TYPES.get(key, SHOT_TYPES["medium"])


def get_shots_for_mood(mood: str) -> list:
    """Get all shot types that match a given mood."""
    return [
        {"key": k, **v}
        for k, v in SHOT_TYPES.items()
        if mood.lower() in v.get("mood", [])
    ]
