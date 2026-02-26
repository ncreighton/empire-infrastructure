"""10+ viral hook formulas with templates, power scores, and niche affinity."""

HOOK_FORMULAS = {
    "pattern_interrupt": {
        "name": "Pattern Interrupt",
        "power": 9,
        "description": "Breaks viewer's scroll pattern with unexpected visual or statement",
        "templates": [
            "STOP scrolling — {topic} will change everything",
            "Wait... did you know {surprising_fact}?",
            "This is NOT what you think — {topic}",
            "POV: You just discovered {topic}",
        ],
        "best_for": ["tiktok", "youtube_shorts", "instagram_reels"],
        "niches": ["all"],
        "retention_anchor": "Immediate visual disruption + promise",
    },
    "curiosity_gap": {
        "name": "Curiosity Gap",
        "power": 9,
        "description": "Creates an information gap the viewer must close",
        "templates": [
            "The {niche} secret that nobody talks about...",
            "I found out why {common_problem} and it's terrifying",
            "There's a reason {thing} works and it's not what you think",
            "{Number} people know this {topic} trick — here's why",
        ],
        "best_for": ["youtube_shorts", "tiktok", "youtube"],
        "niches": ["all"],
        "retention_anchor": "Delayed payoff — answer comes mid-video",
    },
    "contrarian": {
        "name": "Contrarian Take",
        "power": 8,
        "description": "Challenges accepted wisdom, creates controversy",
        "templates": [
            "Everything you know about {topic} is WRONG",
            "Stop doing {common_practice} — here's what works instead",
            "{Popular_opinion} is actually harmful — here's proof",
            "Unpopular opinion: {contrarian_take}",
        ],
        "best_for": ["youtube_shorts", "tiktok", "youtube"],
        "niches": ["all"],
        "retention_anchor": "Viewer stays to validate or dispute",
    },
    "story_hook": {
        "name": "Story Hook",
        "power": 8,
        "description": "Starts a narrative that demands completion",
        "templates": [
            "A {person} once {did_thing} and what happened next...",
            "Last {time_period} I tried {topic} and...",
            "The story of {subject} will blow your mind",
            "In {year}, {dramatic_event} — nobody saw this coming",
        ],
        "best_for": ["youtube", "youtube_shorts", "tiktok"],
        "niches": ["mythology", "witchcraft", "lifestyle", "documentary"],
        "retention_anchor": "Narrative arc — must see the ending",
    },
    "list_authority": {
        "name": "List / Authority",
        "power": 7,
        "description": "Numbered list creates clear expectation and value promise",
        "templates": [
            "{Number} {topic} hacks that actually work",
            "Top {number} {items} you NEED to know about",
            "{Number} things I wish I knew about {topic}",
            "The only {number} {items} you'll ever need",
        ],
        "best_for": ["youtube_shorts", "tiktok", "instagram_reels"],
        "niches": ["all"],
        "retention_anchor": "Viewers count along, stay for next item",
    },
    "fear_of_missing": {
        "name": "Fear of Missing Out",
        "power": 8,
        "description": "Creates urgency around exclusive or timely information",
        "templates": [
            "You're missing out on {topic} — here's why",
            "If you're not using {thing} you're falling behind",
            "This {topic} trend is about to EXPLODE",
            "{Thing} is changing FAST — you need to see this",
        ],
        "best_for": ["tiktok", "youtube_shorts"],
        "niches": ["tech", "ai_news", "hustle", "trends"],
        "retention_anchor": "Urgency drives watch-through",
    },
    "relatable_pain": {
        "name": "Relatable Pain Point",
        "power": 9,
        "description": "Identifies a shared frustration, promises solution",
        "templates": [
            "Tired of {pain_point}? Watch this",
            "Why does {annoying_thing} always happen? Here's the fix",
            "If {relatable_situation} this is for you",
            "The {topic} struggle is REAL — but not anymore",
        ],
        "best_for": ["youtube_shorts", "tiktok", "youtube"],
        "niches": ["all"],
        "retention_anchor": "Personal identification keeps attention",
    },
    "shocking_stat": {
        "name": "Shocking Statistic",
        "power": 8,
        "description": "Leads with a surprising number or data point",
        "templates": [
            "{Percentage}% of people don't know this about {topic}",
            "Only {number} in {total} get this right about {topic}",
            "{Topic} costs {amount} per year — here's how to fix it",
            "In the last {time}, {big_number} {dramatic_metric}",
        ],
        "best_for": ["youtube", "youtube_shorts", "facebook_reels"],
        "niches": ["tech", "business", "health", "news"],
        "retention_anchor": "Stat creates intellectual investment",
    },
    "direct_challenge": {
        "name": "Direct Challenge",
        "power": 7,
        "description": "Challenges the viewer directly, creating personal stakes",
        "templates": [
            "I bet you can't {challenge} — prove me wrong",
            "Only {type_of_person} will understand this",
            "Can you guess what {thing} is? Most people get it wrong",
            "Test yourself: do you know {topic}?",
        ],
        "best_for": ["tiktok", "youtube_shorts"],
        "niches": ["education", "trivia", "fitness", "gaming"],
        "retention_anchor": "Personal challenge demands engagement",
    },
    "before_after": {
        "name": "Before / After",
        "power": 8,
        "description": "Shows dramatic transformation, visual proof",
        "templates": [
            "Before vs After {topic} — the difference is insane",
            "I tried {thing} for {duration} — here's what happened",
            "Day 1 vs Day {number} of {practice}",
            "{Topic}: expectation vs reality",
        ],
        "best_for": ["tiktok", "youtube_shorts", "instagram_reels"],
        "niches": ["fitness", "diy", "tech", "beauty", "witchcraft"],
        "retention_anchor": "Visual payoff at the end",
    },
}

# Niche -> ranked hook formulas (best first)
NICHE_HOOK_RANKING = {
    "witchcraft": ["story_hook", "curiosity_gap", "before_after", "relatable_pain"],
    "mythology": ["story_hook", "curiosity_gap", "shocking_stat", "pattern_interrupt"],
    "tech": ["curiosity_gap", "list_authority", "fear_of_missing", "shocking_stat"],
    "ai_news": ["fear_of_missing", "shocking_stat", "curiosity_gap", "contrarian"],
    "lifestyle": ["relatable_pain", "list_authority", "before_after", "story_hook"],
    "fitness": ["before_after", "direct_challenge", "relatable_pain", "list_authority"],
    "business": ["shocking_stat", "fear_of_missing", "contrarian", "curiosity_gap"],
    "journal": ["relatable_pain", "list_authority", "before_after", "story_hook"],
    "review": ["list_authority", "shocking_stat", "contrarian", "curiosity_gap"],
}

# Map niche IDs to hook ranking keys
_NICHE_TO_CATEGORY = {
    "witchcraftforbeginners": "witchcraft",
    "moonrituallibrary": "witchcraft",
    "manifestandalign": "witchcraft",
    "mythicalarchives": "mythology",
    "smarthomewizards": "tech",
    "smarthomegearreviews": "review",
    "pulsegearreviews": "review",
    "wearablegearreviews": "review",
    "aidiscoverydigest": "ai_news",
    "aiinactionhub": "tech",
    "clearainews": "ai_news",
    "wealthfromai": "business",
    "bulletjournals": "journal",
    "theconnectedhaven": "lifestyle",
    "familyflourish": "lifestyle",
    "celebrationseason": "lifestyle",
}


def get_hook_formula(key: str) -> dict:
    """Get a hook formula by key."""
    return HOOK_FORMULAS.get(key, HOOK_FORMULAS["curiosity_gap"])


def get_best_hook(niche: str) -> str:
    """Get the best hook formula key for a niche."""
    category = _NICHE_TO_CATEGORY.get(niche, "tech")
    ranking = NICHE_HOOK_RANKING.get(category, ["curiosity_gap"])
    return ranking[0]


def get_hooks_ranked(niche: str) -> list:
    """Get hook formula keys ranked by niche affinity."""
    category = _NICHE_TO_CATEGORY.get(niche, "tech")
    return NICHE_HOOK_RANKING.get(category, ["curiosity_gap", "list_authority"])
