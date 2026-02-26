"""Retention strategies — mid-roll hooks, loop structures, engagement triggers."""

RETENTION_PATTERNS = {
    "pattern_interrupt_visual": {
        "name": "Visual Pattern Interrupt",
        "description": "Sudden visual change every 8-15 seconds to re-capture attention",
        "trigger_interval_seconds": (8, 15),
        "techniques": [
            "Quick zoom cut",
            "Color grade shift",
            "Split screen flash",
            "Text overlay pop",
            "Camera angle change",
        ],
        "platforms": ["youtube_shorts", "tiktok"],
        "effectiveness": 9,
    },
    "pattern_interrupt_audio": {
        "name": "Audio Pattern Interrupt",
        "description": "Sound effect or music change to re-engage listener",
        "trigger_interval_seconds": (10, 20),
        "techniques": [
            "Whoosh SFX",
            "Bass drop",
            "Record scratch",
            "Notification ding",
            "Music tempo change",
        ],
        "platforms": ["all"],
        "effectiveness": 7,
    },
    "open_loop": {
        "name": "Open Loop",
        "description": "Promise upcoming revelation to prevent drop-off",
        "trigger_interval_seconds": (15, 30),
        "techniques": [
            "\"But first...\"",
            "\"Wait till you see number {n}...\"",
            "\"The best part is coming...\"",
            "\"But that's not even the crazy part...\"",
            "\"Stay to the end for the bonus...\"",
        ],
        "platforms": ["youtube", "youtube_shorts"],
        "effectiveness": 8,
    },
    "loop_structure": {
        "name": "Loop Structure",
        "description": "Video ending connects seamlessly to the beginning for rewatches",
        "trigger_interval_seconds": None,
        "techniques": [
            "End mid-sentence, start completes it",
            "Visual match cut (last frame = first frame)",
            "\"Wait, what?\" moment at end triggers rewatch",
            "Countdown that resets",
        ],
        "platforms": ["tiktok", "youtube_shorts", "instagram_reels"],
        "effectiveness": 9,
    },
    "engagement_question": {
        "name": "Engagement Question",
        "description": "Ask viewer a direct question to boost comments",
        "trigger_interval_seconds": (20, 45),
        "techniques": [
            "\"Comment which one you'd choose\"",
            "\"Do you agree? Let me know\"",
            "\"What would you add to this list?\"",
            "\"Tag someone who needs to see this\"",
        ],
        "platforms": ["all"],
        "effectiveness": 7,
    },
    "curiosity_escalation": {
        "name": "Curiosity Escalation",
        "description": "Each segment raises stakes higher than the last",
        "trigger_interval_seconds": None,
        "techniques": [
            "\"Good → Better → Mind-blowing\" structure",
            "\"That was weird, but THIS is insane\"",
            "Escalating numbers / stakes",
            "Each reveal bigger than the last",
        ],
        "platforms": ["youtube_shorts", "tiktok", "youtube"],
        "effectiveness": 9,
    },
    "social_proof": {
        "name": "Social Proof Anchor",
        "description": "Reference community, views, or shared experience",
        "trigger_interval_seconds": (30, 60),
        "techniques": [
            "\"Thousands of you asked about this\"",
            "\"This blew up last time\"",
            "\"Everyone's been talking about {topic}\"",
            "Comment screenshot overlay",
        ],
        "platforms": ["all"],
        "effectiveness": 6,
    },
    "micro_payoff": {
        "name": "Micro Payoff",
        "description": "Small satisfying moments throughout to maintain dopamine",
        "trigger_interval_seconds": (5, 10),
        "techniques": [
            "Satisfying transition",
            "Text animation pop",
            "Sound effect sync",
            "Visual reveal",
            "Number counter tick",
        ],
        "platforms": ["tiktok", "youtube_shorts"],
        "effectiveness": 8,
    },
}

# Platform -> recommended retention strategy keys
PLATFORM_RETENTION_MAP = {
    "youtube_shorts": [
        "pattern_interrupt_visual", "loop_structure",
        "curiosity_escalation", "micro_payoff",
    ],
    "tiktok": [
        "loop_structure", "micro_payoff",
        "pattern_interrupt_visual", "engagement_question",
    ],
    "youtube": [
        "open_loop", "curiosity_escalation",
        "pattern_interrupt_visual", "engagement_question",
    ],
    "instagram_reels": [
        "loop_structure", "micro_payoff",
        "pattern_interrupt_visual", "social_proof",
    ],
    "facebook_reels": [
        "pattern_interrupt_visual", "engagement_question",
        "social_proof", "open_loop",
    ],
}


def get_retention_strategy(key: str) -> dict:
    """Get a retention pattern by key."""
    return RETENTION_PATTERNS.get(key, RETENTION_PATTERNS["pattern_interrupt_visual"])


def get_retention_for_platform(platform: str) -> list:
    """Get recommended retention strategies for a platform."""
    keys = PLATFORM_RETENTION_MAP.get(platform, ["pattern_interrupt_visual"])
    return [{"key": k, **RETENTION_PATTERNS[k]} for k in keys if k in RETENTION_PATTERNS]
