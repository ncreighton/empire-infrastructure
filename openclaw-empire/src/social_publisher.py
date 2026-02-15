"""
Social Media Auto-Publisher -- OpenClaw Empire Edition
======================================================

Automated social media content generation and publishing pipeline for
Nick Creighton's 16-site WordPress publishing empire. When a new article
is published on any site, this module generates platform-specific social
posts (voice-matched via BrandVoiceEngine) and queues them for publishing.

Supported platforms: Pinterest, Instagram, Facebook, Twitter/X, LinkedIn.

Data storage:
    data/social/campaigns.json  -- all campaigns (bounded at 2000)
    data/social/queue.json      -- pending posts
    data/social/posted.json     -- history (bounded at 5000)

Usage:
    from src.social_publisher import get_publisher
    pub = get_publisher()
    campaign = pub.create_campaign("witchcraft", "Moon Water Guide",
        "Learn to charge moon water...",
        "https://witchcraftforbeginners.com/moon-water/", ["moon water"])
    campaign = pub.create_campaign_from_article("witchcraft", wp_post_id=1234)
    results = pub.process_queue()

CLI:
    python -m src.social_publisher campaign --site witchcraft --title "Moon Water" --url "https://..."
    python -m src.social_publisher queue
    python -m src.social_publisher process
    python -m src.social_publisher stats --platform pinterest --days 30
    python -m src.social_publisher hashtags --site witchcraft --count 20
"""
from __future__ import annotations

import argparse, asyncio, json, logging, os, random, re, sys, textwrap, uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("social_publisher")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
SITE_REGISTRY_PATH = BASE_DIR / "configs" / "site-registry.json"
SOCIAL_DATA_DIR = BASE_DIR / "data" / "social"
CAMPAIGNS_FILE = SOCIAL_DATA_DIR / "campaigns.json"
QUEUE_FILE = SOCIAL_DATA_DIR / "queue.json"
POSTED_FILE = SOCIAL_DATA_DIR / "posted.json"
HASHTAG_DIR = SOCIAL_DATA_DIR / "hashtags"

SOCIAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
HASHTAG_DIR.mkdir(parents=True, exist_ok=True)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_CAPTION_TOKENS = 500
MAX_CAMPAIGNS = 2000
MAX_POSTED_HISTORY = 5000

ALL_SITE_IDS = [
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
]
WITCHCRAFT_SITES = [
    "witchcraft", "crystalwitchcraft", "herbalwitchery", "moonphasewitch",
    "tarotbeginners", "spellsrituals", "paganpathways", "witchyhomedecor",
    "seasonalwitchcraft",
]
AI_SITES = ["aiaction", "aidiscovery", "wealthai"]
SMARTHOME_SITES = ["smarthome"]
FAMILY_SITES = ["family"]
MYTHOLOGY_SITES = ["mythical"]
BULLETJOURNAL_SITES = ["bulletjournals"]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Platform(str, Enum):
    PINTEREST = "pinterest"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"

class PostStatus(str, Enum):
    QUEUED = "queued"
    POSTED = "posted"
    FAILED = "failed"
    SKIPPED = "skipped"

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class PlatformConfig:
    """Configuration and constraints for a social media platform."""
    platform: str
    enabled: bool
    api_key_env: str
    api_secret_env: str
    access_token_env: str
    default_hashtag_count: int
    max_caption_length: int
    image_required: bool

    def has_credentials(self) -> bool:
        token = os.environ.get(self.access_token_env)
        return token is not None and len(token) > 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SocialPost:
    """A single social media post targeting one platform."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    site_id: str = ""
    platform: str = ""
    title: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    url: str = ""
    image_path: Optional[str] = None
    status: str = PostStatus.QUEUED.value
    scheduled_time: Optional[str] = None
    posted_time: Optional[str] = None
    post_id_external: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SocialPost:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def full_caption(self) -> str:
        if not self.hashtags:
            return self.caption
        tag_str = " ".join(f"#{t.lstrip('#')}" for t in self.hashtags)
        return f"{self.caption}\n\n{tag_str}"

    def caption_length(self) -> int:
        return len(self.full_caption())


@dataclass
class SocialCampaign:
    """A collection of social posts generated for one article across platforms."""
    campaign_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    site_id: str = ""
    article_title: str = ""
    article_url: str = ""
    posts: List[SocialPost] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id, "site_id": self.site_id,
            "article_title": self.article_title, "article_url": self.article_url,
            "posts": [p.to_dict() for p in self.posts],
            "created_at": self.created_at, "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SocialCampaign:
        posts_raw = data.get("posts", [])
        posts = [SocialPost.from_dict(p) for p in posts_raw]
        scalar_keys = {"campaign_id", "site_id", "article_title", "article_url", "created_at", "status"}
        scalars = {k: v for k, v in data.items() if k in scalar_keys}
        return cls(posts=posts, **scalars)

    def summary(self) -> str:
        platforms = [p.platform for p in self.posts]
        return (f"Campaign {self.campaign_id[:8]}... | {self.site_id} | "
                f"'{self.article_title}' | {len(self.posts)} posts ({', '.join(platforms)})")

# ---------------------------------------------------------------------------
# Platform Specifications
# ---------------------------------------------------------------------------
def _build_platform_configs() -> Dict[str, PlatformConfig]:
    cfgs = [
        ("pinterest", True, "PINTEREST_APP_ID", "PINTEREST_APP_SECRET", "PINTEREST_TOKEN", 5, 500, True),
        ("instagram", True, "INSTAGRAM_APP_ID", "INSTAGRAM_APP_SECRET", "INSTAGRAM_TOKEN", 20, 2200, True),
        ("facebook", True, "FACEBOOK_APP_ID", "FACEBOOK_APP_SECRET", "FACEBOOK_TOKEN", 5, 63206, False),
        ("twitter", True, "TWITTER_API_KEY", "TWITTER_API_SECRET", "TWITTER_BEARER_TOKEN", 3, 280, False),
        ("linkedin", True, "LINKEDIN_CLIENT_ID", "LINKEDIN_CLIENT_SECRET", "LINKEDIN_TOKEN", 5, 3000, False),
    ]
    return {c[0]: PlatformConfig(*c) for c in cfgs}

# Optimal posting times in Eastern Time (ET) as (hour, minute) tuples
OPTIMAL_TIMES: Dict[str, List[Tuple[int, int]]] = {
    "pinterest": [(20, 0), (21, 0), (22, 0)],   # 8-11 PM ET
    "instagram": [(11, 0), (14, 0)],              # 11 AM, 2 PM ET
    "facebook": [(9, 0), (13, 0)],                # 9 AM, 1 PM ET
    "twitter": [(9, 0), (12, 0)],                 # 9 AM, 12 PM ET
    "linkedin": [(8, 0), (12, 0)],                # 8 AM, 12 PM ET
}

# ---------------------------------------------------------------------------
# Hashtag Banks (per niche, ~20 tags each)
# ---------------------------------------------------------------------------
HASHTAG_BANKS: Dict[str, List[str]] = {
    "witchcraft": [
        "witchcraft", "witchesofinstagram", "pagansoftiktok", "witchyvibes", "bruja",
        "modernwitch", "spells", "crystalwitch", "moonmagic", "witchythings",
        "witchlife", "babywitch", "witchesofig", "witchcrafttok", "witchcommunity",
        "eclecticwitch", "greenwitch", "hedgewitch", "kitchenwitch", "cottagewitchcraft",
        "witchywoman", "witchaesthetic", "sacredspace", "divinefeminine",
        "magicalpractice", "metaphysical", "occult", "paganism", "wicca", "spellwork",
    ],
    "crystal-magic": [
        "crystalhealing", "crystals", "crystalcollection", "amethyst", "rosequartz",
        "citrine", "selenite", "crystalgrid", "healingcrystals", "crystalmagic",
        "gemstones", "crystalshop", "crystalenergy", "crystalvibes", "tumbledstones",
        "chakrahealing", "quartzcrystal", "crystalwitch", "stonelover", "mineralcollection",
    ],
    "herbal-magic": [
        "herbalism", "herbalmedicine", "greenwitchcraft", "kitchenwitchery",
        "herbalremedies", "plantmagic", "apothecary", "herbgarden", "wildcrafting",
        "foraging", "greenwitch", "herbalwitch", "herbalist", "plantmedicine",
        "naturalmedicine", "essentialoils", "tinctures", "herbaltea", "botanicalwitch",
        "earthmagic",
    ],
    "lunar-magic": [
        "moonmagic", "fullmoon", "newmoon", "moonritual", "lunarwitch", "moonphases",
        "moonwater", "lunarmagic", "moonchild", "moongoddess", "moonlight",
        "moonvibes", "moonenergy", "celestialwitch", "moonmanifesting", "lunarenergy",
        "crescentmoon", "waningmoon", "waxingmoon", "darkmoon",
    ],
    "tarot-divination": [
        "tarot", "tarotreading", "tarotcards", "tarotcommunity", "dailytarot",
        "tarotspread", "tarotdeck", "tarotwitch", "divination", "oraclecards",
        "tarotlover", "cardreading", "tarotofinstagram", "intuitivereading",
        "spiritualguidance", "majorarcana", "minorarcana", "tarottips",
        "tarotbeginner", "cartomancy",
    ],
    "spells-rituals": [
        "spellwork", "spellcasting", "magicspells", "ritualmagic", "candlemagic",
        "protectionspell", "lovespell", "moonspell", "spelljar", "witchyrituals",
        "sacredritual", "sigils", "banishingspell", "manifestation", "spellcraft",
        "ceremonialmagic", "magicpractice", "dailyritual", "fullmoonritual",
        "newmoonritual",
    ],
    "pagan-spirituality": [
        "pagan", "paganism", "paganpath", "neopagan", "heathen", "druid", "druidry",
        "polytheism", "ancestorwork", "devotional", "sacredspace", "paganwitch",
        "pagancommunity", "wheeloftheyear", "sabbat", "esbat", "paganlife",
        "oldways", "folkpractice", "spiritualpath",
    ],
    "witchy-decor": [
        "witchydecor", "witchyhome", "altaraesthetic", "darkacademia", "gothichome",
        "cottagecore", "witchyaesthetic", "altarspace", "crystaldisplay",
        "candleaesthetic", "moodyinteriors", "vintagewitch", "apothecaryaesthetic",
        "tapestry", "cauldron", "witchyroomtour", "sacredcorner", "altarinspiration",
        "darkinteriors", "witchyvibes",
    ],
    "seasonal-wheel-of-year": [
        "wheeloftheyear", "samhain", "yule", "imbolc", "ostara", "beltane", "litha",
        "lughnasadh", "mabon", "sabbat", "sabbatcelebration", "seasonalwitch",
        "seasonalliving", "solstice", "equinox", "harvestseason", "wintersolstice",
        "summersolstice", "springequinox", "autumnequinox",
    ],
    "smart-home-tech": [
        "smarthome", "homeautomation", "iot", "alexatips", "googlehome",
        "smartlighting", "smarthometech", "homeassistant", "smartspeaker",
        "smartlock", "smartplug", "smarthomegadgets", "techreview", "automationlife",
        "connectedhome", "homesecurity", "smartdevices", "voicecontrol", "zigbee",
        "matter", "thread", "wifi6", "smarthometips", "techenthusiast",
    ],
    "ai-technology": [
        "artificialintelligence", "machinelearning", "aitools", "techtrends",
        "futureofwork", "deeplearning", "llm", "generativeai", "chatgpt", "claudeai",
        "techinnovation", "automation", "datascience", "nlp", "aistartup",
        "techblog", "airesearch", "promptengineering", "agentic", "aiagents",
    ],
    "ai-discovery": [
        "aitools", "newaitools", "aitoolsoftiktok", "aiapps", "techfinds",
        "hiddengemtools", "productivitytools", "bestaitools", "aiweekly", "techtok",
        "aiupdates", "aicommunity", "opensourceai", "aitips", "techcurator",
        "airesources", "toolreview", "saastools", "underratedai", "aifinds",
    ],
    "ai-money": [
        "aimoney", "makemoneyonline", "sidehustle", "passiveincome", "aifreelance",
        "aibusiness", "onlinebusiness", "digitalproducts", "aiautomation",
        "aisidehustle", "monetizeai", "techentrepreneur", "onlineincome",
        "workfromhome", "aicashflow", "contentcreator", "saas", "solopreneur",
        "aiplaybook", "microSaaS",
    ],
    "family-wellness": [
        "parenting", "familylife", "momlife", "dadlife", "parentingtips", "familyfun",
        "familytime", "positiveparenting", "gentleparenting", "parentingadvice",
        "familyactivities", "kidsactivities", "parentlife", "raisingkids",
        "familyfirst", "parentingjourney", "familygoals", "familywellness",
        "mentalhealth", "selfcare",
    ],
    "mythology": [
        "mythology", "folklore", "ancienthistory", "greekmythology", "norsemythology",
        "egyptianmythology", "celticmythology", "myths", "legends",
        "mythsandlegends", "ancientworld", "mythologyfacts", "pantheon", "heros",
        "historybuff", "ancientcivilization", "worldmythology", "folklorefriday",
        "trickster", "epic",
    ],
    "productivity-journaling": [
        "bulletjournal", "bujo", "journaling", "plannercommunity", "bujoinspiration",
        "bujolove", "bulletjournaling", "bujosetup", "journalart", "studygram",
        "planneraddict", "bujoideas", "journalspread", "plannerlife", "productivity",
        "goalsetting", "habittracker", "weeklyspread", "monthlysetup", "stationery",
    ],
}

# Map site_id -> niche key in HASHTAG_BANKS
_SITE_NICHE_MAP: Dict[str, str] = {
    "witchcraft": "witchcraft", "smarthome": "smart-home-tech",
    "aiaction": "ai-technology", "aidiscovery": "ai-discovery",
    "wealthai": "ai-money", "family": "family-wellness",
    "mythical": "mythology", "bulletjournals": "productivity-journaling",
    "crystalwitchcraft": "crystal-magic", "herbalwitchery": "herbal-magic",
    "moonphasewitch": "lunar-magic", "tarotbeginners": "tarot-divination",
    "spellsrituals": "spells-rituals", "paganpathways": "pagan-spirituality",
    "witchyhomedecor": "witchy-decor", "seasonalwitchcraft": "seasonal-wheel-of-year",
}

# ---------------------------------------------------------------------------
# Site Registry & Voice Helpers
# ---------------------------------------------------------------------------
def _load_site_registry() -> List[Dict[str, Any]]:
    try:
        with open(SITE_REGISTRY_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh).get("sites", [])
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Could not load site registry: %s", exc)
        return []

def _get_site_metadata(site_id: str) -> Dict[str, Any]:
    for site in _load_site_registry():
        if site.get("id") == site_id:
            return site
    raise KeyError(f"Site '{site_id}' not found in registry.")

def _get_voice_instructions(site_id: str) -> str:
    """Load compact voice instructions from BrandVoiceEngine (fallback if unavailable)."""
    try:
        from src.brand_voice_engine import get_engine
        return get_engine().get_voice_instructions(site_id)
    except Exception:
        metadata = _get_site_metadata(site_id)
        return f"Write in a {metadata.get('voice', 'neutral')} voice for the {metadata.get('niche', 'general')} niche."

def _get_site_domain(site_id: str) -> str:
    try:
        return _get_site_metadata(site_id).get("domain", f"{site_id}.com")
    except KeyError:
        return f"{site_id}.com"

# ---------------------------------------------------------------------------
# Anthropic API Helpers
# ---------------------------------------------------------------------------
def _get_async_anthropic_client():
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic -- required for caption generation")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    return anthropic.AsyncAnthropic(api_key=api_key)

async def _call_haiku(system_prompt: str, user_message: str, max_tokens: int = MAX_CAPTION_TOKENS) -> str:
    """Call Claude Haiku with optional prompt caching."""
    client = _get_async_anthropic_client()
    sys_block: Dict[str, Any] = {"type": "text", "text": system_prompt}
    if len(system_prompt) > 4000:
        sys_block["cache_control"] = {"type": "ephemeral"}
    response = await client.messages.create(
        model=HAIKU_MODEL, max_tokens=max_tokens,
        system=[sys_block], messages=[{"role": "user", "content": user_message}],
    )
    return "\n".join(b.text for b in response.content if hasattr(b, "text"))

def _call_haiku_sync(system_prompt: str, user_message: str, max_tokens: int = MAX_CAPTION_TOKENS) -> str:
    return _run_sync(_call_haiku(system_prompt, user_message, max_tokens))

# ---------------------------------------------------------------------------
# JSON Persistence
# ---------------------------------------------------------------------------
def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default if default is not None else []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default if default is not None else []

def _save_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str, ensure_ascii=False)
        if path.exists():
            path.unlink()
        tmp.rename(path)
    except OSError as exc:
        logger.error("Failed to save %s: %s", path, exc)
        if tmp.exists():
            tmp.unlink()

def _bounded_append(path: Path, new_entries: List[Dict[str, Any]], max_entries: int) -> None:
    existing = _load_json(path, default=[])
    if not isinstance(existing, list):
        existing = []
    existing.extend(new_entries)
    if len(existing) > max_entries:
        existing = existing[-max_entries:]
    _save_json(path, existing)

# ---------------------------------------------------------------------------
# Caption Prompts
# ---------------------------------------------------------------------------
_PINTEREST_SYSTEM = textwrap.dedent("""\
    You are a Pinterest marketing expert writing pin descriptions.
    RULES: 1) First 50 chars MUST contain primary keyword (Pinterest SEO)
    2) Rich description (3-4 sentences) with natural keywords 3) Clear CTA
    4) End with exactly 5 hashtags 5) Under 500 chars total 6) Match brand voice
    OUTPUT (no markdown/labels): [Description with CTA]\\n\\n#tag1 #tag2 #tag3 #tag4 #tag5
""")

_INSTAGRAM_SYSTEM = textwrap.dedent("""\
    You are an Instagram content strategist writing captions.
    RULES: 1) First line is a HOOK 2) 3-5 value sentences 3) End with question/CTA
    4) After blank line, 15-20 niche hashtags 5) Under 2200 chars 6) Match brand voice
    OUTPUT (no markdown/labels): [Hook]\\n\\n[Value]\\n\\n[CTA]\\n\\n#tag1 ... #tag20
""")

_FACEBOOK_SYSTEM = textwrap.dedent("""\
    You are a Facebook content creator writing posts.
    RULES: 1) Conversational hook (first 2 lines grab before "See more")
    2) 2-3 sentence teaser 3) Include article URL 4) 3-5 hashtags 5) Match brand voice
    OUTPUT (no markdown/labels): [Hook]\\n\\n[Teaser]\\n\\n[URL]\\n\\n#tag1 #tag2 #tag3
""")

_TWITTER_SYSTEM = textwrap.dedent("""\
    You are a Twitter/X strategist writing tweets.
    RULES: 1) ENTIRE tweet including URL and hashtags MUST be under 280 chars
    2) Lead with most interesting point 3) Include URL 4) 2-3 hashtags 5) Match brand voice
    OUTPUT (no markdown/labels): [Tweet text] [URL] #tag1 #tag2
""")

_LINKEDIN_SYSTEM = textwrap.dedent("""\
    You are a LinkedIn content strategist writing posts.
    RULES: 1) Professional hook or industry observation 2) Frame as solving challenge
    3) 2-3 insight sentences 4) Include URL 5) 3-5 professional hashtags 6) Match brand voice
    OUTPUT (no markdown/labels): [Hook]\\n\\n[Insight]\\n\\n[URL]\\n\\n#tag1 #tag2 #tag3
""")

# ---------------------------------------------------------------------------
# Caption Parsing
# ---------------------------------------------------------------------------
def _extract_hashtags(text: str) -> Tuple[str, List[str]]:
    """Split caption into (body, list_of_hashtags_without_#)."""
    hashtags = re.findall(r"#(\w+)", text, re.UNICODE)
    lines = text.rstrip().split("\n")
    body_lines: List[str] = []
    hashtag_block = False
    for line in reversed(lines):
        stripped = line.strip()
        if not hashtag_block:
            if stripped and all(w.startswith("#") for w in stripped.split() if w):
                hashtag_block = True
                continue
            elif stripped == "":
                continue
        body_lines.insert(0, line)
    body = "\n".join(body_lines).strip() or text.strip()
    return body, hashtags

def _truncate_caption(caption: str, max_length: int) -> str:
    if len(caption) <= max_length:
        return caption
    truncated = caption[:max_length - 3]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]
    return truncated + "..."

def _strip_html_simple(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    for old, new in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                     ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()

def _run_sync(coro):
    """Run async coroutine synchronously, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)

# ---------------------------------------------------------------------------
# SocialPublisher
# ---------------------------------------------------------------------------
class SocialPublisher:
    """Central engine for social media publishing across the empire."""

    def __init__(self) -> None:
        self._platform_configs = _build_platform_configs()
        self._sites = _load_site_registry()
        self._site_map: Dict[str, Dict[str, Any]] = {s["id"]: s for s in self._sites if "id" in s}
        logger.info("SocialPublisher: %d platforms, %d sites", len(self._platform_configs), len(self._site_map))

    # -- Platform Config --------------------------------------------------
    def get_platform_config(self, platform: str) -> PlatformConfig:
        platform = platform.lower()
        if platform not in self._platform_configs:
            raise KeyError(f"Platform '{platform}' not found. Available: {', '.join(sorted(self._platform_configs))}")
        return self._platform_configs[platform]

    def get_enabled_platforms(self, site_id: str) -> List[str]:
        """Return enabled platforms in priority order for a site's niche."""
        all_enabled = [n for n, c in self._platform_configs.items() if c.enabled]
        if site_id in WITCHCRAFT_SITES:
            priority = ["pinterest", "instagram", "facebook", "twitter"]
        elif site_id in AI_SITES:
            priority = ["twitter", "linkedin", "facebook", "instagram"]
        elif site_id in SMARTHOME_SITES:
            priority = ["twitter", "facebook", "pinterest", "instagram", "linkedin"]
        elif site_id in FAMILY_SITES:
            priority = ["pinterest", "facebook", "instagram", "twitter"]
        elif site_id in MYTHOLOGY_SITES:
            priority = ["instagram", "pinterest", "twitter", "facebook"]
        elif site_id in BULLETJOURNAL_SITES:
            priority = ["pinterest", "instagram", "facebook", "twitter"]
        else:
            priority = ["pinterest", "instagram", "facebook", "twitter", "linkedin"]
        ordered = [p for p in priority if p in all_enabled]
        for p in all_enabled:
            if p not in ordered:
                ordered.append(p)
        return ordered

    # -- Hashtag Engine ---------------------------------------------------
    def get_niche_hashtags(self, site_id: str, count: int = 20) -> List[str]:
        """Get randomized niche hashtags. Witchcraft sub-niches mix parent bank."""
        niche = _SITE_NICHE_MAP.get(site_id, "witchcraft")
        bank = HASHTAG_BANKS.get(niche, [])
        if site_id in WITCHCRAFT_SITES and niche != "witchcraft":
            parent = HASHTAG_BANKS.get("witchcraft", [])
            nc = min(int(count * 0.6), len(bank))
            pc = min(count - nc, len(parent))
            result = (random.sample(bank, nc) if bank else []) + (random.sample(parent, pc) if parent else [])
        else:
            result = random.sample(bank, min(count, len(bank))) if bank else []
        seen: set = set()
        return [t for t in result if not (t.lower() in seen or seen.add(t.lower()))][:count]  # type: ignore[func-returns-value]

    def get_trending_hashtags(self, niche: str, count: int = 5) -> List[str]:
        """Trending hashtags (future: API integration). Returns top evergreen tags."""
        return HASHTAG_BANKS.get(niche, [])[:count]

    def mix_hashtags(self, niche_tags: List[str], trending_tags: List[str], count: int = 20) -> List[str]:
        """Mix trending (first) and niche hashtags, deduplicated."""
        seen: set = set()
        result: List[str] = []
        for tag in trending_tags + niche_tags:
            if len(result) >= count:
                break
            if tag.lower() not in seen:
                seen.add(tag.lower())
                result.append(tag)
        return result

    # -- Caption Generation -----------------------------------------------
    async def generate_pinterest_pin_async(self, site_id: str, title: str,
                                            description: str, keywords: Optional[List[str]] = None) -> SocialPost:
        voice = _get_voice_instructions(site_id)
        kw = ", ".join(keywords) if keywords else "none"
        raw = await _call_haiku(_PINTEREST_SYSTEM, f"VOICE:\n{voice}\n\nTITLE: {title}\nDESC: {description}\nKEYWORDS: {kw}\n\nWrite a Pinterest pin description.")
        body, hashtags = _extract_hashtags(raw)
        body = _truncate_caption(body, 500)
        if len(hashtags) < 5:
            hashtags.extend(self.get_niche_hashtags(site_id, 5 - len(hashtags)))
        return SocialPost(site_id=site_id, platform="pinterest", title=title,
                          caption=body, hashtags=hashtags[:5],
                          metadata={"keywords": keywords or [], "generator": "haiku"})

    def generate_pinterest_pin(self, site_id: str, title: str, description: str,
                                keywords: Optional[List[str]] = None) -> SocialPost:
        return _run_sync(self.generate_pinterest_pin_async(site_id, title, description, keywords))

    async def generate_instagram_post_async(self, site_id: str, title: str,
                                             description: str, keywords: Optional[List[str]] = None) -> SocialPost:
        voice = _get_voice_instructions(site_id)
        domain = _get_site_domain(site_id)
        kw = ", ".join(keywords) if keywords else "none"
        raw = await _call_haiku(_INSTAGRAM_SYSTEM,
            f"VOICE:\n{voice}\n\nSITE: {domain}\nTITLE: {title}\nDESC: {description}\nKEYWORDS: {kw}\n\nWrite an Instagram caption. Include 'link in bio'.")
        body, hashtags = _extract_hashtags(raw)
        body = _truncate_caption(body, 2200)
        if len(hashtags) < 15:
            hashtags.extend(self.get_niche_hashtags(site_id, 20 - len(hashtags)))
        return SocialPost(site_id=site_id, platform="instagram", title=title,
                          caption=body, hashtags=hashtags[:20],
                          metadata={"keywords": keywords or [], "generator": "haiku"})

    def generate_instagram_post(self, site_id: str, title: str, description: str,
                                 keywords: Optional[List[str]] = None) -> SocialPost:
        return _run_sync(self.generate_instagram_post_async(site_id, title, description, keywords))

    async def generate_facebook_post_async(self, site_id: str, title: str,
                                            description: str, url: str) -> SocialPost:
        voice = _get_voice_instructions(site_id)
        raw = await _call_haiku(_FACEBOOK_SYSTEM,
            f"VOICE:\n{voice}\n\nTITLE: {title}\nDESC: {description}\nURL: {url}\n\nWrite a Facebook post.")
        body, hashtags = _extract_hashtags(raw)
        if len(hashtags) < 3:
            hashtags.extend(self.get_niche_hashtags(site_id, 5 - len(hashtags)))
        return SocialPost(site_id=site_id, platform="facebook", title=title,
                          caption=body, hashtags=hashtags[:5], url=url,
                          metadata={"generator": "haiku"})

    def generate_facebook_post(self, site_id: str, title: str, description: str, url: str) -> SocialPost:
        return _run_sync(self.generate_facebook_post_async(site_id, title, description, url))

    async def generate_twitter_post_async(self, site_id: str, title: str, url: str) -> SocialPost:
        voice = _get_voice_instructions(site_id)
        raw = await _call_haiku(_TWITTER_SYSTEM,
            f"VOICE:\n{voice}\n\nTITLE: {title}\nURL: {url}\n\nWrite a tweet. URLs = 23 chars on Twitter. Total must be under 280.")
        body, hashtags = _extract_hashtags(raw)
        if len(hashtags) < 2:
            hashtags.extend(self.get_niche_hashtags(site_id, 3 - len(hashtags)))
        hashtags = hashtags[:3]
        # Enforce 280 char limit
        full = body if url in body else f"{body} {url}"
        tag_str = " ".join(f"#{t}" for t in hashtags)
        if len(f"{full} {tag_str}") > 280:
            available = 280 - len(url) - len(tag_str) - 4
            body = _truncate_caption(body, max(available, 50))
        return SocialPost(site_id=site_id, platform="twitter", title=title,
                          caption=body, hashtags=hashtags, url=url,
                          metadata={"generator": "haiku"})

    def generate_twitter_post(self, site_id: str, title: str, url: str) -> SocialPost:
        return _run_sync(self.generate_twitter_post_async(site_id, title, url))

    async def generate_linkedin_post_async(self, site_id: str, title: str,
                                            description: str, url: str) -> SocialPost:
        voice = _get_voice_instructions(site_id)
        raw = await _call_haiku(_LINKEDIN_SYSTEM,
            f"VOICE (professional audience):\n{voice}\n\nTITLE: {title}\nDESC: {description}\nURL: {url}\n\nWrite a LinkedIn post.")
        body, hashtags = _extract_hashtags(raw)
        body = _truncate_caption(body, 3000)
        if len(hashtags) < 3:
            hashtags.extend(self.get_niche_hashtags(site_id, 5 - len(hashtags)))
        return SocialPost(site_id=site_id, platform="linkedin", title=title,
                          caption=body, hashtags=hashtags[:5], url=url,
                          metadata={"generator": "haiku"})

    def generate_linkedin_post(self, site_id: str, title: str, description: str, url: str) -> SocialPost:
        return _run_sync(self.generate_linkedin_post_async(site_id, title, description, url))

    # -- Campaign Generation ----------------------------------------------
    async def create_campaign_async(self, site_id: str, title: str, description: str,
                                     url: str, keywords: Optional[List[str]] = None,
                                     platforms: Optional[List[str]] = None) -> SocialCampaign:
        """Generate social posts for all enabled platforms in one call."""
        if site_id not in self._site_map:
            raise KeyError(f"Site '{site_id}' not found. Available: {', '.join(sorted(self._site_map))}")
        target = platforms or self.get_enabled_platforms(site_id)
        campaign = SocialCampaign(site_id=site_id, article_title=title, article_url=url)

        gen_map = {
            "pinterest": lambda: self.generate_pinterest_pin_async(site_id, title, description, keywords),
            "instagram": lambda: self.generate_instagram_post_async(site_id, title, description, keywords),
            "facebook": lambda: self.generate_facebook_post_async(site_id, title, description, url),
            "twitter": lambda: self.generate_twitter_post_async(site_id, title, url),
            "linkedin": lambda: self.generate_linkedin_post_async(site_id, title, description, url),
        }
        tasks = [gen_map[p]() for p in target if p in gen_map]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, SocialPost):
                    result.url = url
                    campaign.posts.append(result)
                elif isinstance(result, Exception):
                    logger.error("Campaign post generation failed: %s", result)
                    campaign.posts.append(SocialPost(
                        site_id=site_id, platform="unknown", title=title, url=url,
                        status=PostStatus.FAILED.value, error=str(result)))

        self._save_campaign(campaign)
        logger.info("Campaign %s: %d posts for %s", campaign.campaign_id[:8], len(campaign.posts), site_id)
        return campaign

    def create_campaign(self, site_id: str, title: str, description: str, url: str,
                         keywords: Optional[List[str]] = None,
                         platforms: Optional[List[str]] = None) -> SocialCampaign:
        return _run_sync(self.create_campaign_async(site_id, title, description, url, keywords, platforms))

    async def create_campaign_from_article_async(self, site_id: str, wp_post_id: int) -> SocialCampaign:
        """Fetch article from WordPress and generate campaign for all platforms."""
        try:
            from src.wordpress_client import get_site_client
        except ImportError:
            raise ImportError("WordPress client not available (src/wordpress_client.py)")

        client = get_site_client(site_id)
        post_data = client.get_post_sync(wp_post_id)

        # Extract title
        raw_title = post_data.get("title", {})
        title = _strip_html_simple(raw_title.get("rendered", str(raw_title)) if isinstance(raw_title, dict) else str(raw_title))

        # Extract description from excerpt or content
        raw_exc = post_data.get("excerpt", {})
        description = _strip_html_simple(raw_exc.get("rendered", str(raw_exc)) if isinstance(raw_exc, dict) else str(raw_exc))
        if not description:
            raw_c = post_data.get("content", {})
            description = _strip_html_simple((raw_c.get("rendered", "") if isinstance(raw_c, dict) else str(raw_c)))[:200]

        url = post_data.get("link", "")

        # Extract RankMath focus keyword
        meta = post_data.get("meta", {})
        focus_kw = meta.get("rank_math_focus_keyword", "") if isinstance(meta, dict) else ""
        keywords = [k.strip() for k in focus_kw.split(",") if k.strip()] if focus_kw else []

        return await self.create_campaign_async(site_id, title, description, url, keywords)

    def create_campaign_from_article(self, site_id: str, wp_post_id: int) -> SocialCampaign:
        return _run_sync(self.create_campaign_from_article_async(site_id, wp_post_id))

    # -- Publishing Queue -------------------------------------------------
    def queue_post(self, post: SocialPost) -> SocialPost:
        post.status = PostStatus.QUEUED.value
        queue = _load_json(QUEUE_FILE, default=[])
        queue.append(post.to_dict())
        _save_json(QUEUE_FILE, queue)
        logger.info("Queued %s for %s on %s", post.id[:8], post.site_id, post.platform)
        return post

    def queue_campaign(self, campaign: SocialCampaign) -> List[SocialPost]:
        queue = _load_json(QUEUE_FILE, default=[])
        queued = []
        for post in campaign.posts:
            if post.status == PostStatus.FAILED.value:
                continue
            post.status = PostStatus.QUEUED.value
            queue.append(post.to_dict())
            queued.append(post)
        _save_json(QUEUE_FILE, queue)
        logger.info("Queued %d posts from campaign %s", len(queued), campaign.campaign_id[:8])
        return queued

    def get_queue(self, platform: Optional[str] = None, site_id: Optional[str] = None) -> List[SocialPost]:
        posts = [SocialPost.from_dict(p) for p in _load_json(QUEUE_FILE, default=[])]
        if platform:
            posts = [p for p in posts if p.platform == platform.lower()]
        if site_id:
            posts = [p for p in posts if p.site_id == site_id]
        return posts

    def remove_from_queue(self, post_id: str) -> bool:
        raw = _load_json(QUEUE_FILE, default=[])
        filtered = [p for p in raw if p.get("id") != post_id]
        if len(filtered) < len(raw):
            _save_json(QUEUE_FILE, filtered)
            return True
        return False

    def clear_queue(self, platform: Optional[str] = None, site_id: Optional[str] = None) -> int:
        raw = _load_json(QUEUE_FILE, default=[])
        original = len(raw)
        if platform or site_id:
            raw = [p for p in raw if not (
                (platform and p.get("platform") == platform.lower()) or
                (site_id and p.get("site_id") == site_id)
            )]
        else:
            raw = []
        _save_json(QUEUE_FILE, raw)
        return original - len(raw)

    # -- Scheduling -------------------------------------------------------
    def schedule_post(self, post: SocialPost, time: str) -> SocialPost:
        post.scheduled_time = time
        post.status = PostStatus.QUEUED.value
        raw = _load_json(QUEUE_FILE, default=[])
        updated = False
        for i, p in enumerate(raw):
            if p.get("id") == post.id:
                raw[i] = post.to_dict()
                updated = True
                break
        if not updated:
            raw.append(post.to_dict())
        _save_json(QUEUE_FILE, raw)
        logger.info("Scheduled %s for %s at %s", post.id[:8], post.platform, time)
        return post

    def get_optimal_time(self, platform: str, site_id: str) -> str:
        """Calculate next optimal posting time in UTC (from ET best-time windows)."""
        slots = OPTIMAL_TIMES.get(platform.lower(), [(12, 0)])
        now = datetime.now(timezone.utc)
        candidates = []
        for hour, minute in slots:
            utc_hour = hour + 5  # EST -> UTC
            if utc_hour >= 24:
                candidate = now.replace(hour=utc_hour - 24, minute=minute, second=0, microsecond=0) + timedelta(days=1)
            else:
                candidate = now.replace(hour=utc_hour, minute=minute, second=0, microsecond=0)
            if candidate <= now:
                candidate += timedelta(days=1)
            candidates.append(candidate)
        return min(candidates).isoformat()

    # -- Platform Publishing ----------------------------------------------
    async def process_queue_async(self) -> List[SocialPost]:
        """Attempt to publish all queued posts whose scheduled time has passed."""
        raw = _load_json(QUEUE_FILE, default=[])
        if not raw:
            logger.info("Queue empty")
            return []
        now = datetime.now(timezone.utc)
        processed, remaining = [], []

        for pd in raw:
            post = SocialPost.from_dict(pd)
            if post.scheduled_time:
                try:
                    sched = datetime.fromisoformat(post.scheduled_time)
                    if sched.tzinfo is None:
                        sched = sched.replace(tzinfo=timezone.utc)
                    if sched > now:
                        remaining.append(pd)
                        continue
                except (ValueError, TypeError):
                    pass
            success = await self._dispatch_publish(post)
            post.status = PostStatus.POSTED.value if success else PostStatus.FAILED.value
            if success:
                post.posted_time = now.isoformat()
            processed.append(post)

        _save_json(QUEUE_FILE, remaining)
        if processed:
            _bounded_append(POSTED_FILE, [p.to_dict() for p in processed], MAX_POSTED_HISTORY)
        logger.info("Processed %d, %d remaining", len(processed), len(remaining))
        return processed

    def process_queue(self) -> List[SocialPost]:
        return _run_sync(self.process_queue_async())

    async def _dispatch_publish(self, post: SocialPost) -> bool:
        dispatch = {
            "pinterest": self._publish_pinterest,
            "instagram": self._publish_instagram,
            "facebook": self._publish_facebook,
            "twitter": self._publish_twitter,
            "linkedin": self._publish_linkedin,
        }
        handler = dispatch.get(post.platform.lower())
        if not handler:
            post.error = f"Unknown platform: {post.platform}"
            return False
        try:
            return await handler(post)
        except Exception as exc:
            post.error = str(exc)
            logger.error("Publish to %s failed: %s", post.platform, exc)
            return False

    async def _publish_pinterest(self, post: SocialPost) -> bool:
        """Publish pin via Pinterest API v5."""
        token = os.environ.get("PINTEREST_TOKEN")
        if not token:
            post.error = "PINTEREST_TOKEN not set"
            post.status = PostStatus.SKIPPED.value
            return False
        try:
            import aiohttp
        except ImportError:
            post.error = "aiohttp not installed"
            return False
        pin_data: Dict[str, Any] = {
            "title": post.title[:100],
            "description": post.full_caption()[:500],
            "link": post.url,
        }
        if post.image_path:
            pin_data["media_source"] = {"source_type": "image_url", "url": post.image_path}
        board_id = post.metadata.get("board_id") or os.environ.get("PINTEREST_BOARD_ID", "")
        if board_id:
            pin_data["board_id"] = board_id
        async with aiohttp.ClientSession() as s:
            async with s.post("https://api.pinterest.com/v5/pins",
                              headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                              json=pin_data) as r:
                if r.status in (200, 201):
                    post.post_id_external = (await r.json()).get("id", "")
                    return True
                post.error = f"Pinterest {r.status}: {(await r.text())[:200]}"
                return False

    async def _publish_instagram(self, post: SocialPost) -> bool:
        """Publish via Instagram Graph API (two-step: container then publish)."""
        token = os.environ.get("INSTAGRAM_TOKEN")
        biz_id = os.environ.get("INSTAGRAM_BUSINESS_ID")
        if not token or not biz_id:
            post.error = "INSTAGRAM_TOKEN/INSTAGRAM_BUSINESS_ID not set"
            post.status = PostStatus.SKIPPED.value
            return False
        try:
            import aiohttp
        except ImportError:
            post.error = "aiohttp not installed"
            return False
        async with aiohttp.ClientSession() as s:
            # Step 1: create container
            params: Dict[str, Any] = {"caption": post.full_caption()[:2200], "access_token": token}
            if post.image_path:
                params["image_url"] = post.image_path
            async with s.post(f"https://graph.facebook.com/v19.0/{biz_id}/media", params=params) as r:
                if r.status != 200:
                    post.error = f"IG container {r.status}: {(await r.text())[:200]}"
                    return False
                cid = (await r.json()).get("id")
                if not cid:
                    post.error = "No container ID"
                    return False
            # Step 2: publish
            async with s.post(f"https://graph.facebook.com/v19.0/{biz_id}/media_publish",
                              params={"creation_id": cid, "access_token": token}) as r:
                if r.status == 200:
                    post.post_id_external = (await r.json()).get("id", "")
                    return True
                post.error = f"IG publish {r.status}: {(await r.text())[:200]}"
                return False

    async def _publish_facebook(self, post: SocialPost) -> bool:
        """Publish to Facebook Page via Graph API."""
        token = os.environ.get("FACEBOOK_TOKEN")
        page_id = os.environ.get("FACEBOOK_PAGE_ID")
        if not token or not page_id:
            post.error = "FACEBOOK_TOKEN/FACEBOOK_PAGE_ID not set"
            post.status = PostStatus.SKIPPED.value
            return False
        try:
            import aiohttp
        except ImportError:
            post.error = "aiohttp not installed"
            return False
        async with aiohttp.ClientSession() as s:
            async with s.post(f"https://graph.facebook.com/v19.0/{page_id}/feed",
                              params={"message": post.full_caption(), "link": post.url,
                                      "access_token": token}) as r:
                if r.status == 200:
                    post.post_id_external = (await r.json()).get("id", "")
                    return True
                post.error = f"FB {r.status}: {(await r.text())[:200]}"
                return False

    async def _publish_twitter(self, post: SocialPost) -> bool:
        """Publish tweet via X API v2."""
        bearer = os.environ.get("TWITTER_BEARER_TOKEN")
        if not bearer:
            post.error = "TWITTER_BEARER_TOKEN not set"
            post.status = PostStatus.SKIPPED.value
            return False
        try:
            import aiohttp
        except ImportError:
            post.error = "aiohttp not installed"
            return False
        tweet = post.caption
        if post.url and post.url not in tweet:
            tweet = f"{tweet} {post.url}"
        tag_str = " ".join(f"#{t}" for t in post.hashtags)
        if tag_str:
            tweet = f"{tweet} {tag_str}"
        async with aiohttp.ClientSession() as s:
            async with s.post("https://api.twitter.com/2/tweets",
                              headers={"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"},
                              json={"text": tweet[:280]}) as r:
                if r.status in (200, 201):
                    post.post_id_external = (await r.json()).get("data", {}).get("id", "")
                    return True
                post.error = f"Twitter {r.status}: {(await r.text())[:200]}"
                return False

    async def _publish_linkedin(self, post: SocialPost) -> bool:
        """Publish via LinkedIn UGC Posts API v2."""
        token = os.environ.get("LINKEDIN_TOKEN")
        person_id = os.environ.get("LINKEDIN_PERSON_ID")
        if not token or not person_id:
            post.error = "LINKEDIN_TOKEN/LINKEDIN_PERSON_ID not set"
            post.status = PostStatus.SKIPPED.value
            return False
        try:
            import aiohttp
        except ImportError:
            post.error = "aiohttp not installed"
            return False
        payload = {
            "author": f"urn:li:person:{person_id}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {"com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": post.full_caption()[:3000]},
                "shareMediaCategory": "ARTICLE",
                "media": [{"status": "READY", "originalUrl": post.url,
                           "title": {"text": post.title[:200]}}],
            }},
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }
        async with aiohttp.ClientSession() as s:
            async with s.post("https://api.linkedin.com/v2/ugcPosts",
                              headers={"Authorization": f"Bearer {token}",
                                       "Content-Type": "application/json",
                                       "X-Restli-Protocol-Version": "2.0.0"},
                              json=payload) as r:
                if r.status in (200, 201):
                    post.post_id_external = (await r.json()).get("id", "")
                    return True
                post.error = f"LinkedIn {r.status}: {(await r.text())[:200]}"
                return False

    # -- Analytics --------------------------------------------------------
    def get_campaign_stats(self, campaign_id: str) -> Dict[str, Any]:
        for c in _load_json(CAMPAIGNS_FILE, default=[]):
            if c.get("campaign_id") == campaign_id:
                camp = SocialCampaign.from_dict(c)
                statuses: Dict[str, int] = {}
                plat_status: Dict[str, str] = {}
                for p in camp.posts:
                    statuses[p.status] = statuses.get(p.status, 0) + 1
                    plat_status[p.platform] = p.status
                return {"campaign_id": camp.campaign_id, "site_id": camp.site_id,
                        "article_title": camp.article_title, "article_url": camp.article_url,
                        "created_at": camp.created_at, "total_posts": len(camp.posts),
                        "status_breakdown": statuses, "platform_status": plat_status}
        return {"error": f"Campaign {campaign_id} not found"}

    def get_platform_stats(self, platform: str, days: int = 30) -> Dict[str, Any]:
        platform = platform.lower()
        posted = _load_json(POSTED_FILE, default=[])
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        total = success = failed = skipped = 0
        recent: List[Dict[str, Any]] = []
        for pd in posted:
            if pd.get("platform") != platform:
                continue
            pt_str = pd.get("posted_time")
            if pt_str:
                try:
                    pt = datetime.fromisoformat(pt_str)
                    if pt.tzinfo is None:
                        pt = pt.replace(tzinfo=timezone.utc)
                    if pt < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            total += 1
            s = pd.get("status", "")
            if s == "posted": success += 1
            elif s == "failed": failed += 1
            elif s == "skipped": skipped += 1
            if len(recent) < 10:
                recent.append({"id": pd.get("id", "")[:8], "title": pd.get("title", ""),
                               "site_id": pd.get("site_id", ""), "status": s,
                               "posted_time": pt_str})
        return {"platform": platform, "period_days": days, "total_posts": total,
                "successful": success, "failed": failed, "skipped": skipped,
                "success_rate": round(success / total * 100, 1) if total else 0.0,
                "recent_posts": recent}

    def get_site_social_summary(self, site_id: str, days: int = 30) -> Dict[str, Any]:
        posted = _load_json(POSTED_FILE, default=[])
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        plat_counts: Dict[str, Dict[str, int]] = {}
        total = 0
        for pd in posted:
            if pd.get("site_id") != site_id:
                continue
            pt_str = pd.get("posted_time")
            if pt_str:
                try:
                    pt = datetime.fromisoformat(pt_str)
                    if pt.tzinfo is None:
                        pt = pt.replace(tzinfo=timezone.utc)
                    if pt < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            plat = pd.get("platform", "unknown")
            if plat not in plat_counts:
                plat_counts[plat] = {"total": 0, "posted": 0, "failed": 0, "skipped": 0}
            plat_counts[plat]["total"] += 1
            s = pd.get("status", "")
            if s in plat_counts[plat]:
                plat_counts[plat][s] += 1
            total += 1
        return {"site_id": site_id, "period_days": days, "total_posts": total, "platforms": plat_counts}

    def best_performing_posts(self, platform: str, count: int = 10) -> List[Dict[str, Any]]:
        platform = platform.lower()
        posted = _load_json(POSTED_FILE, default=[])
        ok = [p for p in posted if p.get("platform") == platform and p.get("status") == "posted"]
        ok.sort(key=lambda x: x.get("posted_time", ""), reverse=True)
        return [{"id": p.get("id", "")[:8], "site_id": p.get("site_id", ""),
                 "title": p.get("title", ""), "caption_preview": p.get("caption", "")[:100],
                 "hashtags": p.get("hashtags", []), "url": p.get("url", ""),
                 "posted_time": p.get("posted_time", ""),
                 "external_id": p.get("post_id_external", "")} for p in ok[:count]]

    # -- Campaign Persistence ---------------------------------------------
    def _save_campaign(self, campaign: SocialCampaign) -> None:
        _bounded_append(CAMPAIGNS_FILE, [campaign.to_dict()], MAX_CAMPAIGNS)

    def get_campaign(self, campaign_id: str) -> Optional[SocialCampaign]:
        for c in _load_json(CAMPAIGNS_FILE, default=[]):
            if c.get("campaign_id") == campaign_id:
                return SocialCampaign.from_dict(c)
        return None

    def list_campaigns(self, site_id: Optional[str] = None, limit: int = 20) -> List[SocialCampaign]:
        raw = _load_json(CAMPAIGNS_FILE, default=[])
        if site_id:
            raw = [c for c in raw if c.get("site_id") == site_id]
        raw = raw[-limit:]
        raw.reverse()
        return [SocialCampaign.from_dict(c) for c in raw]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_publisher_instance: Optional[SocialPublisher] = None

def get_publisher() -> SocialPublisher:
    """Get or create the singleton SocialPublisher instance."""
    global _publisher_instance
    if _publisher_instance is None:
        _publisher_instance = SocialPublisher()
    return _publisher_instance

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli_campaign(args: argparse.Namespace) -> None:
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] if args.keywords else []
    pub = get_publisher()
    print(f"\nGenerating campaign for '{args.site}': {args.title}")
    try:
        campaign = pub.create_campaign(
            args.site, args.title, args.description or args.title, args.url,
            keywords or None, args.platforms.split(",") if args.platforms else None)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr); sys.exit(1)
    if args.queue:
        pub.queue_campaign(campaign)
        print("[+] Posts queued\n")
    print(f"{'='*70}\n  CAMPAIGN {campaign.campaign_id[:12]}... | {campaign.site_id}\n{'='*70}")
    for i, p in enumerate(campaign.posts, 1):
        preview = p.caption[:100].replace("\n", " ") if p.caption else "(none)"
        tags = " ".join(f"#{t}" for t in p.hashtags[:5])
        print(f"\n  [{i}] {p.platform.upper()} | {p.status}\n      {preview}...\n      {tags}")
        if p.error: print(f"      Error: {p.error}")
    print(f"\n{'='*70}\n")

def _cli_queue(args: argparse.Namespace) -> None:
    posts = get_publisher().get_queue(platform=args.platform, site_id=args.site)
    if not posts:
        print("\nQueue empty.\n"); return
    print(f"\n{'='*70}\n  QUEUE ({len(posts)} posts)\n{'='*70}")
    for i, p in enumerate(posts, 1):
        sched = p.scheduled_time or "immediate"
        print(f"\n  [{i}] {p.platform.upper()} | {p.site_id} | {sched}\n      {p.title}")
    print(f"\n{'='*70}\n")

def _cli_process(args: argparse.Namespace) -> None:
    print("\nProcessing queue...")
    results = get_publisher().process_queue()
    if not results:
        print("Nothing to process.\n"); return
    posted = sum(1 for r in results if r.status == "posted")
    failed = sum(1 for r in results if r.status == "failed")
    print(f"\nProcessed {len(results)}: {posted} posted, {failed} failed\n")
    for r in results:
        icon = "[+]" if r.status == "posted" else "[-]"
        print(f"  {icon} {r.platform.upper()} | {r.site_id} | {r.status}")
        if r.error: print(f"      {r.error}")
    print()

def _cli_stats(args: argparse.Namespace) -> None:
    pub = get_publisher()
    if args.platform:
        s = pub.get_platform_stats(args.platform, args.days)
        print(f"\n{'='*70}\n  {s['platform'].upper()} ({s['period_days']}d)\n{'='*70}")
        print(f"  Total: {s['total_posts']} | OK: {s['successful']} | Fail: {s['failed']} | Skip: {s['skipped']} | Rate: {s['success_rate']}%")
        for rp in s["recent_posts"]:
            print(f"    {rp['id']} {rp['site_id']} {rp['title'][:35]} {rp['status']}")
    elif args.site:
        s = pub.get_site_social_summary(args.site, args.days)
        print(f"\n{'='*70}\n  {s['site_id']} ({s['period_days']}d) -- {s['total_posts']} posts\n{'='*70}")
        for plat, c in s["platforms"].items():
            print(f"    {plat:<12} total={c['total']} ok={c['posted']} fail={c['failed']}")
    else:
        print(f"\n{'='*70}\n  ALL PLATFORMS ({args.days}d)\n{'='*70}")
        for plat in [p.value for p in Platform]:
            s = pub.get_platform_stats(plat, args.days)
            print(f"  {plat:<12} total={s['total_posts']:<4} ok={s['successful']:<4} fail={s['failed']:<4} rate={s['success_rate']}%")
    print(f"\n{'='*70}\n")

def _cli_hashtags(args: argparse.Namespace) -> None:
    tags = get_publisher().get_niche_hashtags(args.site, args.count)
    niche = _SITE_NICHE_MAP.get(args.site, "unknown")
    print(f"\n{'='*70}\n  HASHTAGS: {args.site} ({niche})\n{'='*70}\n")
    for i in range(0, len(tags), 3):
        print("  " + "".join(f"#{t}".ljust(28) for t in tags[i:i+3]))
    print(f"\n  Copy: {' '.join('#' + t for t in tags)}\n{'='*70}\n")

def _cli_best(args: argparse.Namespace) -> None:
    posts = get_publisher().best_performing_posts(args.platform, args.count)
    if not posts:
        print(f"\nNo successful posts for {args.platform}.\n"); return
    print(f"\n{'='*70}\n  BEST: {args.platform.upper()} (top {len(posts)})\n{'='*70}")
    for i, p in enumerate(posts, 1):
        print(f"\n  [{i}] {p['title'][:50]}\n      {p['site_id']} | {p['posted_time']}\n      {p['caption_preview']}...")
    print(f"\n{'='*70}\n")

def _cli_campaigns(args: argparse.Namespace) -> None:
    campaigns = get_publisher().list_campaigns(site_id=args.site, limit=args.limit)
    if not campaigns:
        print("\nNo campaigns found.\n"); return
    print(f"\n{'='*70}\n  CAMPAIGNS ({len(campaigns)})\n{'='*70}")
    for c in campaigns:
        plats = ", ".join(p.platform for p in c.posts)
        print(f"\n  {c.campaign_id[:12]}... | {c.site_id}\n    {c.article_title[:50]}\n    {c.article_url}\n    {len(c.posts)} posts ({plats})")
    print(f"\n{'='*70}\n")

def main() -> None:
    parser = argparse.ArgumentParser(prog="social_publisher",
        description="Social Media Auto-Publisher for the OpenClaw Empire (16 WordPress sites)")
    sub = parser.add_subparsers(dest="command", help="Commands")

    p = sub.add_parser("campaign", help="Generate social campaign")
    p.add_argument("--site", required=True); p.add_argument("--title", required=True)
    p.add_argument("--url", required=True); p.add_argument("--description", default=None)
    p.add_argument("--keywords", default=None); p.add_argument("--platforms", default=None)
    p.add_argument("--queue", action="store_true")
    p.set_defaults(func=_cli_campaign)

    p = sub.add_parser("queue", help="Show queue")
    p.add_argument("--platform", default=None); p.add_argument("--site", default=None)
    p.set_defaults(func=_cli_queue)

    p = sub.add_parser("process", help="Publish queued posts")
    p.set_defaults(func=_cli_process)

    p = sub.add_parser("stats", help="Publishing statistics")
    p.add_argument("--platform", default=None); p.add_argument("--site", default=None)
    p.add_argument("--days", type=int, default=30)
    p.set_defaults(func=_cli_stats)

    p = sub.add_parser("hashtags", help="Show niche hashtags")
    p.add_argument("--site", required=True); p.add_argument("--count", type=int, default=20)
    p.set_defaults(func=_cli_hashtags)

    p = sub.add_parser("best", help="Best-performing posts")
    p.add_argument("--platform", required=True); p.add_argument("--count", type=int, default=10)
    p.set_defaults(func=_cli_best)

    p = sub.add_parser("campaigns", help="List campaigns")
    p.add_argument("--site", default=None); p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=_cli_campaigns)

    args = parser.parse_args()
    if not args.command:
        parser.print_help(); sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
