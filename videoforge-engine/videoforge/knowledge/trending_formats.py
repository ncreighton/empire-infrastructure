"""Current viral video formats — updated periodically."""

TRENDING_FORMATS = {
    "faceless_narrator": {
        "name": "Faceless Narrator",
        "description": "AI voiceover + stock/AI visuals, no on-camera talent needed",
        "platforms": ["youtube_shorts", "tiktok", "youtube"],
        "popularity": 9,
        "effort_level": "low",
        "structure": ["hook_text_card", "narration_over_visuals", "cta"],
        "best_niches": ["mythology", "witchcraft", "ai_news", "business"],
    },
    "split_screen_reaction": {
        "name": "Split Screen + Reaction",
        "description": "Content on one side, reaction/commentary on other",
        "platforms": ["tiktok", "youtube_shorts"],
        "popularity": 8,
        "effort_level": "medium",
        "structure": ["original_clip", "reaction_layer", "commentary"],
        "best_niches": ["tech", "news", "entertainment"],
    },
    "text_story_scroll": {
        "name": "Text Story Scroll",
        "description": "Text scrolls up like Reddit/Twitter stories with voice reading",
        "platforms": ["tiktok", "youtube_shorts"],
        "popularity": 7,
        "effort_level": "low",
        "structure": ["text_card", "scroll_animation", "voice_reading"],
        "best_niches": ["story", "advice", "horror"],
    },
    "countdown_list": {
        "name": "Countdown List",
        "description": "Numbered items from highest to #1, each with visual",
        "platforms": ["youtube_shorts", "tiktok", "youtube"],
        "popularity": 9,
        "effort_level": "medium",
        "structure": ["hook", "numbered_items_descending", "reveal_#1"],
        "best_niches": ["all"],
    },
    "day_in_life": {
        "name": "Day in the Life / Routine",
        "description": "Show a routine or process from start to finish",
        "platforms": ["tiktok", "instagram_reels", "youtube_shorts"],
        "popularity": 8,
        "effort_level": "medium",
        "structure": ["morning_hook", "process_steps", "satisfying_end"],
        "best_niches": ["lifestyle", "journal", "witchcraft"],
    },
    "quiz_poll": {
        "name": "Quiz / Poll",
        "description": "Ask questions, reveal answers, boost engagement",
        "platforms": ["tiktok", "youtube_shorts"],
        "popularity": 7,
        "effort_level": "low",
        "structure": ["question_card", "pause_for_guess", "reveal_answer"],
        "best_niches": ["education", "mythology", "tech"],
    },
    "before_after_transformation": {
        "name": "Before/After Transformation",
        "description": "Show dramatic change, very satisfying format",
        "platforms": ["tiktok", "instagram_reels", "youtube_shorts"],
        "popularity": 9,
        "effort_level": "medium",
        "structure": ["before_state", "process_montage", "reveal_after"],
        "best_niches": ["diy", "fitness", "tech", "journal"],
    },
    "asmr_satisfying": {
        "name": "ASMR / Satisfying",
        "description": "Close-up visuals with satisfying sounds, oddly addictive",
        "platforms": ["tiktok", "instagram_reels"],
        "popularity": 8,
        "effort_level": "low",
        "structure": ["close_up_process", "satisfying_audio", "loop"],
        "best_niches": ["journal", "cooking", "crafts", "witchcraft"],
    },
    "myth_fact": {
        "name": "Myth vs Fact",
        "description": "Debunk myths, show real facts, educational and engaging",
        "platforms": ["youtube_shorts", "tiktok", "youtube"],
        "popularity": 8,
        "effort_level": "low",
        "structure": ["state_myth", "dramatic_pause", "reveal_fact"],
        "best_niches": ["tech", "health", "witchcraft", "mythology"],
    },
    "cinematic_montage": {
        "name": "Cinematic Montage",
        "description": "Beautiful visuals, epic music, minimal narration",
        "platforms": ["youtube", "instagram_reels"],
        "popularity": 7,
        "effort_level": "high",
        "structure": ["atmospheric_intro", "visual_sequence", "emotional_peak"],
        "best_niches": ["mythology", "travel", "witchcraft"],
    },
    "tutorial_speedrun": {
        "name": "Tutorial Speedrun",
        "description": "Full tutorial compressed into 30-60 seconds",
        "platforms": ["tiktok", "youtube_shorts"],
        "popularity": 8,
        "effort_level": "medium",
        "structure": ["result_preview", "fast_steps", "final_result"],
        "best_niches": ["tech", "diy", "journal", "cooking"],
    },
    "hot_take_rant": {
        "name": "Hot Take / Rant",
        "description": "Bold opinion delivered with passion, drives comments",
        "platforms": ["tiktok", "youtube_shorts"],
        "popularity": 7,
        "effort_level": "low",
        "structure": ["bold_statement", "supporting_points", "challenge_audience"],
        "best_niches": ["business", "tech", "lifestyle"],
    },
}


def get_trending_formats(niche: str = None, platform: str = None) -> list:
    """Get trending formats, optionally filtered by niche or platform."""
    results = []
    for key, fmt in TRENDING_FORMATS.items():
        if niche and niche not in fmt.get("best_niches", []) and "all" not in fmt.get("best_niches", []):
            continue
        if platform and platform not in fmt.get("platforms", []):
            continue
        results.append({"key": key, **fmt})
    results.sort(key=lambda x: x.get("popularity", 0), reverse=True)
    return results
