"""ForgeFiles Reddit profile, subreddits, and persona configuration."""

# --- Account Identity ---
FORGEFILES_PROFILE = {
    "reddit_username": "StillLabelingCables",
    "etsy_url": "https://www.etsy.com/shop/forgefile/",
    "etsy_shop_name": "ForgeFile",
    "niche": "3d_printing",
    "persona": (
        "Fellow maker who designs STL files, prints daily, and gives specific "
        "advice (settings, materials, slicer tips). Not salesy — shares genuine "
        "enthusiasm for functional prints and problem-solving. Occasionally "
        "mentions own designs when directly relevant, never pushes."
    ),
}

# --- Subreddit Tiers ---
# Tier 1: post + engage — our primary communities
TIER1_SUBREDDITS = [
    {"name": "3Dprinting", "min_karma_to_post": 50, "flair_required": False},
    {"name": "functionalprint", "min_karma_to_post": 0, "flair_required": False},
    {"name": "ender3", "min_karma_to_post": 0, "flair_required": False},
    {"name": "BambuLab", "min_karma_to_post": 0, "flair_required": False},
    {"name": "resinprinting", "min_karma_to_post": 0, "flair_required": False},
]

# Tier 2: engage only (comments + votes)
TIER2_SUBREDDITS = [
    {"name": "3Dmodeling"},
    {"name": "blender"},
    {"name": "stlfiles"},
    {"name": "PrintedMinis"},
    {"name": "cosplayprops"},
]

# Tier 3: browse for diversity (votes only, rare comments)
TIER3_SUBREDDITS = [
    {"name": "DIY"},
    {"name": "maker"},
    {"name": "MechanicalKeyboards"},
    {"name": "homelab"},
    {"name": "prusa3d"},
    {"name": "Elegoo"},
]

ALL_SUBREDDITS = TIER1_SUBREDDITS + TIER2_SUBREDDITS + TIER3_SUBREDDITS

# --- Self-Promotion Rules ---
PROMO_RULES = {
    "max_promo_ratio": 0.10,        # Max 10% of all posts/comments can be promo
    "min_karma_for_promo": 100,      # Must have 100+ karma before any promo
    "min_account_age_days": 14,      # 14-day minimum account age
    "promo_cooldown_hours": 48,      # 48h between promo posts
    "max_promo_per_week": 1,         # Hard cap: 1 promo post per week
    "max_same_sub_comments_day": 3,  # Max 3 comments in same sub per day
}

# --- Content Anchors ---
# Topics we can speak authentically about
EXPERTISE_TOPICS = [
    "functional prints", "organizer designs", "cable management",
    "print settings", "bed adhesion", "PLA vs PETG vs TPU",
    "layer height", "infill patterns", "slicer settings",
    "support removal", "post-processing", "Cura", "PrusaSlicer",
    "OrcaSlicer", "Bambu Studio", "first layer calibration",
    "nozzle clogs", "stringing", "retraction settings",
    "STL design tips", "Fusion 360", "OpenSCAD",
    "articulated prints", "gear mechanisms", "snap-fit joints",
]

# Phrases that signal a promo opportunity (someone asking for STLs, designs)
PROMO_TRIGGER_PHRASES = [
    "where can i find", "anyone have an stl", "stl for this",
    "where did you get", "is there a file", "looking for a design",
    "need a print for", "recommend a model", "where to buy stl",
]

# --- Reddit App Package ---
REDDIT_PACKAGE = "com.reddit.frontpage"
REDDIT_ACTIVITY = "com.reddit.frontpage/.MainActivity"
