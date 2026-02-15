"""
Identity Manager — Digital Identity Generator & Persona Management (Phase 5)
=============================================================================

Generate and manage realistic digital identities (personas) for use across
platforms. Each identity has a name, backstory, interests, demographics,
platform-specific bios, avatar descriptions, linked email chains, and
account references.

Identities are generated via Haiku for realism but fall back to
deterministic generation if the API is unavailable.

Data stored under: data/identities/
    personas.json           — all generated personas
    persona_templates.json  — reusable templates
    platform_profiles.json  — platform-specific profile adaptations
    avatar_prompts.json     — AI image generation prompts for avatars
    identity_groups.json    — groups of related identities

Usage:
    from src.identity_manager import get_identity_manager

    mgr = get_identity_manager()
    persona = mgr.generate_persona_sync(niche="witchcraft")
    profile = mgr.create_platform_profile_sync(persona.id, Platform.INSTAGRAM)

CLI:
    python -m src.identity_manager generate --niche witchcraft --count 3
    python -m src.identity_manager list --status active
    python -m src.identity_manager show --id <persona-id>
    python -m src.identity_manager profile --persona-id <id> --platform instagram
    python -m src.identity_manager search --query "witch"
    python -m src.identity_manager group create --name "Witchcraft Ring"
    python -m src.identity_manager export --output /tmp/personas.json
    python -m src.identity_manager import --input /tmp/personas.json
    python -m src.identity_manager stats
    python -m src.identity_manager warming --persona-id <id> --platform instagram
    python -m src.identity_manager burn --persona-id <id>
    python -m src.identity_manager clone --persona-id <id>
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
import json
import logging
import os
import random
import re
import string
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("identity_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "identities"
PERSONAS_FILE = DATA_DIR / "personas.json"
TEMPLATES_FILE = DATA_DIR / "persona_templates.json"
PLATFORM_PROFILES_FILE = DATA_DIR / "platform_profiles.json"
AVATAR_PROMPTS_FILE = DATA_DIR / "avatar_prompts.json"
GROUPS_FILE = DATA_DIR / "identity_groups.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

UTC = timezone.utc


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now_utc().isoformat()


# ---------------------------------------------------------------------------
# JSON persistence helpers (atomic writes)
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when the file is missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        if os.name == "nt":
            os.replace(str(tmp), str(path))
        else:
            tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Async/Sync dual interface helper
# ---------------------------------------------------------------------------

def _run_sync(coro):
    """Run an async coroutine from sync context, handling running loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ===================================================================
# ENUMS
# ===================================================================

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    UNSPECIFIED = "unspecified"


class AgeRange(str, Enum):
    TEEN = "teen"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    MIDDLE_AGED = "middle_aged"
    SENIOR = "senior"


AGE_RANGE_BOUNDS = {
    AgeRange.TEEN: (16, 19),
    AgeRange.YOUNG_ADULT: (20, 29),
    AgeRange.ADULT: (30, 44),
    AgeRange.MIDDLE_AGED: (45, 59),
    AgeRange.SENIOR: (60, 75),
}


class Platform(str, Enum):
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"
    PINTEREST = "pinterest"
    YOUTUBE = "youtube"
    REDDIT = "reddit"
    THREADS = "threads"
    SNAPCHAT = "snapchat"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    GMAIL = "gmail"
    OUTLOOK = "outlook"
    YAHOO = "yahoo"


class PersonaStatus(str, Enum):
    ACTIVE = "active"
    WARMING = "warming"
    SUSPENDED = "suspended"
    BURNED = "burned"
    RETIRED = "retired"
    TEMPLATE = "template"


class IdentityTier(str, Enum):
    DISPOSABLE = "disposable"
    STANDARD = "standard"
    PREMIUM = "premium"
    PRIMARY = "primary"


# ===================================================================
# PLATFORM BIO LIMITS
# ===================================================================

PLATFORM_BIO_LIMITS: Dict[Platform, Dict[str, int]] = {
    Platform.INSTAGRAM: {"bio": 150, "name": 30, "username": 30},
    Platform.TIKTOK: {"bio": 80, "name": 30, "username": 24},
    Platform.TWITTER: {"bio": 160, "name": 50, "username": 15},
    Platform.FACEBOOK: {"bio": 101, "name": 50, "username": 50},
    Platform.LINKEDIN: {"bio": 2600, "name": 50, "username": 100},
    Platform.PINTEREST: {"bio": 500, "name": 65, "username": 30},
    Platform.YOUTUBE: {"bio": 1000, "name": 100, "username": 30},
    Platform.REDDIT: {"bio": 200, "name": 30, "username": 20},
    Platform.THREADS: {"bio": 150, "name": 30, "username": 30},
}

# ===================================================================
# BUILT-IN NAME / LOCATION / INTEREST DATA
# ===================================================================

FIRST_NAMES_MALE = [
    "James", "John", "Robert", "Michael", "David", "William", "Richard", "Joseph",
    "Thomas", "Christopher", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Andrew", "Paul", "Joshua", "Kenneth", "Kevin", "Brian",
    "George", "Timothy", "Ronald", "Edward", "Jason", "Jeffrey", "Ryan",
    "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin",
    "Scott", "Brandon", "Benjamin", "Samuel", "Raymond", "Gregory", "Frank",
    "Alexander", "Patrick", "Jack", "Dennis", "Jerry", "Tyler", "Aaron", "Jose",
    "Nathan", "Henry", "Peter", "Douglas", "Zachary", "Kyle", "Noah", "Ethan",
    "Jeremy", "Walter", "Christian", "Keith", "Roger", "Terry", "Austin", "Sean",
    "Gerald", "Carl", "Harold", "Dylan", "Arthur", "Lawrence", "Jordan", "Jesse",
    "Bryan", "Billy", "Bruce", "Gabriel", "Joe", "Logan", "Albert", "Willie",
    "Alan", "Eugene", "Russell", "Vincent", "Philip", "Bobby", "Johnny", "Bradley",
    "Roy", "Ralph", "Eugene", "Randy", "Wayne", "Elijah", "Marcus", "Theodore",
]

FIRST_NAMES_FEMALE = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan",
    "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
    "Ashley", "Dorothy", "Kimberly", "Emily", "Donna", "Michelle", "Carol",
    "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura",
    "Cynthia", "Kathleen", "Amy", "Angela", "Shirley", "Brenda", "Emma", "Anna",
    "Pamela", "Nicole", "Helen", "Samantha", "Katherine", "Christine", "Debra",
    "Rachel", "Carolyn", "Janet", "Catherine", "Maria", "Heather", "Diane",
    "Ruth", "Julie", "Olivia", "Joyce", "Virginia", "Victoria", "Kelly", "Lauren",
    "Christina", "Joan", "Evelyn", "Judith", "Megan", "Andrea", "Cheryl", "Hannah",
    "Jacqueline", "Martha", "Gloria", "Teresa", "Ann", "Sara", "Madison", "Frances",
    "Kathryn", "Janice", "Jean", "Abigail", "Alice", "Judy", "Sophia", "Grace",
    "Denise", "Amber", "Doris", "Marilyn", "Danielle", "Beverly", "Isabella",
    "Theresa", "Diana", "Natalie", "Brittany", "Charlotte", "Marie", "Kayla",
    "Alexis", "Lori", "Alyssa", "Ella",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen",
    "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera",
    "Campbell", "Mitchell", "Carter", "Roberts", "Gomez", "Phillips", "Evans",
    "Turner", "Diaz", "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart",
    "Morris", "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz",
    "Morgan", "Cooper", "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos",
    "Kim", "Cox", "Ward", "Richardson", "Watson", "Brooks", "Chavez", "Wood",
    "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes", "Price", "Alvarez",
    "Castillo", "Sanders", "Patel", "Myers", "Long", "Ross", "Foster", "Jimenez",
]

US_CITIES: List[Dict[str, str]] = [
    {"city": "New York", "state": "New York", "timezone": "America/New_York"},
    {"city": "Los Angeles", "state": "California", "timezone": "America/Los_Angeles"},
    {"city": "Chicago", "state": "Illinois", "timezone": "America/Chicago"},
    {"city": "Houston", "state": "Texas", "timezone": "America/Chicago"},
    {"city": "Phoenix", "state": "Arizona", "timezone": "America/Phoenix"},
    {"city": "Philadelphia", "state": "Pennsylvania", "timezone": "America/New_York"},
    {"city": "San Antonio", "state": "Texas", "timezone": "America/Chicago"},
    {"city": "San Diego", "state": "California", "timezone": "America/Los_Angeles"},
    {"city": "Dallas", "state": "Texas", "timezone": "America/Chicago"},
    {"city": "San Jose", "state": "California", "timezone": "America/Los_Angeles"},
    {"city": "Austin", "state": "Texas", "timezone": "America/Chicago"},
    {"city": "Jacksonville", "state": "Florida", "timezone": "America/New_York"},
    {"city": "Fort Worth", "state": "Texas", "timezone": "America/Chicago"},
    {"city": "Columbus", "state": "Ohio", "timezone": "America/New_York"},
    {"city": "Charlotte", "state": "North Carolina", "timezone": "America/New_York"},
    {"city": "Indianapolis", "state": "Indiana", "timezone": "America/Indiana/Indianapolis"},
    {"city": "San Francisco", "state": "California", "timezone": "America/Los_Angeles"},
    {"city": "Seattle", "state": "Washington", "timezone": "America/Los_Angeles"},
    {"city": "Denver", "state": "Colorado", "timezone": "America/Denver"},
    {"city": "Nashville", "state": "Tennessee", "timezone": "America/Chicago"},
    {"city": "Oklahoma City", "state": "Oklahoma", "timezone": "America/Chicago"},
    {"city": "El Paso", "state": "Texas", "timezone": "America/Denver"},
    {"city": "Washington", "state": "District of Columbia", "timezone": "America/New_York"},
    {"city": "Boston", "state": "Massachusetts", "timezone": "America/New_York"},
    {"city": "Las Vegas", "state": "Nevada", "timezone": "America/Los_Angeles"},
    {"city": "Portland", "state": "Oregon", "timezone": "America/Los_Angeles"},
    {"city": "Memphis", "state": "Tennessee", "timezone": "America/Chicago"},
    {"city": "Louisville", "state": "Kentucky", "timezone": "America/New_York"},
    {"city": "Baltimore", "state": "Maryland", "timezone": "America/New_York"},
    {"city": "Milwaukee", "state": "Wisconsin", "timezone": "America/Chicago"},
    {"city": "Albuquerque", "state": "New Mexico", "timezone": "America/Denver"},
    {"city": "Tucson", "state": "Arizona", "timezone": "America/Phoenix"},
    {"city": "Fresno", "state": "California", "timezone": "America/Los_Angeles"},
    {"city": "Sacramento", "state": "California", "timezone": "America/Los_Angeles"},
    {"city": "Mesa", "state": "Arizona", "timezone": "America/Phoenix"},
    {"city": "Atlanta", "state": "Georgia", "timezone": "America/New_York"},
    {"city": "Kansas City", "state": "Missouri", "timezone": "America/Chicago"},
    {"city": "Omaha", "state": "Nebraska", "timezone": "America/Chicago"},
    {"city": "Colorado Springs", "state": "Colorado", "timezone": "America/Denver"},
    {"city": "Raleigh", "state": "North Carolina", "timezone": "America/New_York"},
    {"city": "Miami", "state": "Florida", "timezone": "America/New_York"},
    {"city": "Minneapolis", "state": "Minnesota", "timezone": "America/Chicago"},
    {"city": "Tampa", "state": "Florida", "timezone": "America/New_York"},
    {"city": "New Orleans", "state": "Louisiana", "timezone": "America/Chicago"},
    {"city": "Cleveland", "state": "Ohio", "timezone": "America/New_York"},
    {"city": "Pittsburgh", "state": "Pennsylvania", "timezone": "America/New_York"},
    {"city": "Cincinnati", "state": "Ohio", "timezone": "America/New_York"},
    {"city": "Orlando", "state": "Florida", "timezone": "America/New_York"},
    {"city": "St. Louis", "state": "Missouri", "timezone": "America/Chicago"},
    {"city": "Salt Lake City", "state": "Utah", "timezone": "America/Denver"},
]

OCCUPATIONS = [
    "Software Developer", "Graphic Designer", "Marketing Manager", "Teacher",
    "Nurse", "Freelance Writer", "Social Media Manager", "Photographer",
    "Data Analyst", "Project Manager", "Accountant", "Sales Representative",
    "Chef", "Interior Designer", "Physical Therapist", "Architect",
    "Veterinarian", "Real Estate Agent", "Financial Advisor", "Human Resources",
    "Content Creator", "UX Designer", "Yoga Instructor", "Personal Trainer",
    "Event Planner", "Librarian", "Journalist", "Musician", "Artist",
    "Therapist", "Nutritionist", "Web Developer", "Video Editor",
    "Copywriter", "Barista", "Florist", "Tattoo Artist", "Makeup Artist",
    "Electrician", "Plumber", "Carpenter", "Landscaper", "Baker",
    "Bartender", "Dental Hygienist", "Paralegal", "Travel Agent",
    "Fitness Coach", "Life Coach", "Podcast Host",
]

INTERESTS = [
    "hiking", "photography", "cooking", "reading", "yoga", "gardening",
    "painting", "travel", "music", "fitness", "meditation", "writing",
    "cycling", "surfing", "rock climbing", "camping", "baking", "pottery",
    "knitting", "woodworking", "chess", "board games", "video games",
    "anime", "manga", "cosplay", "astrology", "tarot", "crystals",
    "herbalism", "witchcraft", "moon rituals", "candle making",
    "essential oils", "journaling", "bullet journaling", "calligraphy",
    "watercolor", "digital art", "3D printing", "robotics", "coding",
    "AI tools", "smart home tech", "home automation", "DIY projects",
    "thrifting", "vintage clothing", "sustainable fashion", "veganism",
    "coffee", "tea ceremony", "wine tasting", "craft beer", "mixology",
    "running", "marathon training", "swimming", "kayaking", "skiing",
    "snowboarding", "skateboarding", "dancing", "salsa", "ballet",
    "martial arts", "boxing", "weightlifting", "pilates", "CrossFit",
    "bird watching", "stargazing", "astronomy", "philosophy", "psychology",
    "true crime", "podcasts", "film", "theater", "stand-up comedy",
    "volunteering", "animal rescue", "dog training", "cat fostering",
    "aquarium keeping", "plant care", "succulent collecting",
]

EDUCATION_LEVELS = [
    "High School Diploma",
    "Some College",
    "Associate's Degree",
    "Bachelor's Degree",
    "Master's Degree",
    "Doctorate",
]

RELATIONSHIP_STATUSES = [
    "Single", "In a relationship", "Engaged", "Married", "Divorced",
    "Widowed", "It's complicated",
]

# Niche-specific interests for targeted persona generation
NICHE_INTERESTS: Dict[str, List[str]] = {
    "witchcraft": [
        "witchcraft", "tarot", "crystals", "herbalism", "moon rituals",
        "candle making", "essential oils", "astrology", "meditation",
        "journaling", "nature walks", "gardening", "tea ceremony",
        "spell crafting", "divination", "pagan traditions",
    ],
    "smart home": [
        "smart home tech", "home automation", "AI tools", "coding",
        "robotics", "3D printing", "DIY projects", "gadgets",
        "voice assistants", "IoT devices", "home theater", "networking",
    ],
    "ai": [
        "AI tools", "coding", "robotics", "data science", "machine learning",
        "automation", "tech news", "startups", "productivity",
        "digital nomad", "side hustles", "online business",
    ],
    "parenting": [
        "cooking", "baking", "gardening", "reading", "crafts",
        "hiking", "camping", "board games", "volunteering",
        "photography", "journaling", "yoga", "meditation",
    ],
    "mythology": [
        "reading", "writing", "philosophy", "history", "archaeology",
        "fantasy", "world mythology", "folklore", "storytelling",
        "cosplay", "role-playing games", "theater", "art history",
    ],
    "bullet journal": [
        "bullet journaling", "calligraphy", "watercolor", "stickers",
        "washi tape", "planning", "productivity", "goal setting",
        "hand lettering", "art journaling", "organization",
    ],
    "fitness": [
        "running", "weightlifting", "yoga", "CrossFit", "martial arts",
        "swimming", "cycling", "hiking", "nutrition", "meal prep",
        "marathon training", "fitness tracking", "sports",
    ],
    "crafts": [
        "knitting", "crochet", "sewing", "pottery", "woodworking",
        "jewelry making", "candle making", "soap making", "macrame",
        "embroidery", "quilting", "resin art", "paper crafts",
    ],
}

# Communication style templates for fallback generation
COMM_STYLES = [
    "casual and friendly", "warm and encouraging", "witty and sarcastic",
    "enthusiastic and energetic", "thoughtful and reflective",
    "direct and no-nonsense", "playful and humorous", "calm and zen",
    "passionate and expressive", "professional but approachable",
]

POSTING_TONES = [
    "enthusiastic", "thoughtful", "inspiring", "humorous", "informative",
    "motivational", "chill", "edgy", "wholesome", "snarky",
    "poetic", "practical", "curious", "bold", "gentle",
]

# Backstory templates for fallback (when Haiku unavailable)
BACKSTORY_TEMPLATES = [
    "{first} is a {age}-year-old {occupation} from {city}, {state} who discovered a passion for {interest1} during {event}. When not working, {pronoun} loves {interest2} and sharing tips on {interest3}.",
    "Growing up in {city}, {first} always had a knack for {interest1}. Now a {occupation} by day, {pronoun} spends {possessive} free time diving deep into {interest2} and connecting with others who share a love for {interest3}.",
    "After years as a {occupation} in {city}, {first} found {possessive} true calling in {interest1}. {pronoun_cap} started sharing {possessive} journey online and quickly built a community around {interest2} and {interest3}.",
    "{first} moved to {city} at {age} to pursue a career in {occupation_field}. Along the way, {pronoun} fell in love with {interest1} and now balances work with {possessive} passion for {interest2}.",
    "A self-taught {interest1} enthusiast from {city}, {first} brings {possessive} {occupation} background to everything {pronoun} creates. {pronoun_cap}'s known among friends for {possessive} infectious love of {interest2} and {interest3}.",
]

BACKSTORY_EVENTS = [
    "college", "a cross-country road trip", "quarantine", "a breakup",
    "a career change", "a move to a new city", "a gap year",
    "a friend's recommendation", "a random YouTube video",
    "a local workshop", "a birthday gift",
]


# ===================================================================
# DATACLASSES
# ===================================================================

@dataclass
class PersonaDemographics:
    first_name: str = ""
    last_name: str = ""
    gender: Gender = Gender.UNSPECIFIED
    age: int = 25
    age_range: AgeRange = AgeRange.YOUNG_ADULT
    country: str = "United States"
    state: str = ""
    city: str = ""
    timezone: str = "America/New_York"
    language: str = "en"
    occupation: str = ""
    education: str = ""
    relationship_status: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gender"] = self.gender.value
        d["age_range"] = self.age_range.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PersonaDemographics:
        if not data:
            return cls()
        d = dict(data)
        if "gender" in d:
            d["gender"] = Gender(d["gender"]) if d["gender"] in [g.value for g in Gender] else Gender.UNSPECIFIED
        if "age_range" in d:
            d["age_range"] = AgeRange(d["age_range"]) if d["age_range"] in [a.value for a in AgeRange] else AgeRange.YOUNG_ADULT
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PersonaPersonality:
    interests: List[str] = field(default_factory=list)
    hobbies: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    communication_style: str = ""
    emoji_usage: str = "moderate"
    hashtag_style: str = "moderate"
    posting_tone: str = ""
    topics_to_avoid: List[str] = field(default_factory=list)
    backstory: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PersonaPersonality:
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PlatformProfile:
    platform: Platform
    username: str = ""
    display_name: str = ""
    bio: str = ""
    profile_url: str = ""
    avatar_prompt: str = ""
    header_prompt: str = ""
    account_id: str = ""
    status: PersonaStatus = PersonaStatus.TEMPLATE
    created_at: str = ""
    followers: int = 0
    following: int = 0
    posts: int = 0
    last_active: str = ""
    warming_schedule: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["platform"] = self.platform.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlatformProfile:
        if not data:
            return cls(platform=Platform.INSTAGRAM)
        d = dict(data)
        if "platform" in d:
            try:
                d["platform"] = Platform(d["platform"])
            except ValueError:
                d["platform"] = Platform.INSTAGRAM
        if "status" in d:
            try:
                d["status"] = PersonaStatus(d["status"])
            except ValueError:
                d["status"] = PersonaStatus.TEMPLATE
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EmailIdentity:
    provider: Platform
    address: str = ""
    password_ref: str = ""
    is_primary: bool = False
    is_recovery: bool = False
    phone_linked: str = ""
    created_at: str = ""
    last_checked: str = ""
    status: PersonaStatus = PersonaStatus.TEMPLATE

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provider"] = self.provider.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmailIdentity:
        if not data:
            return cls(provider=Platform.GMAIL)
        d = dict(data)
        if "provider" in d:
            try:
                d["provider"] = Platform(d["provider"])
            except ValueError:
                d["provider"] = Platform.GMAIL
        if "status" in d:
            try:
                d["status"] = PersonaStatus(d["status"])
            except ValueError:
                d["status"] = PersonaStatus.TEMPLATE
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Persona:
    id: str
    name: str
    demographics: PersonaDemographics = field(default_factory=PersonaDemographics)
    personality: PersonaPersonality = field(default_factory=PersonaPersonality)
    emails: List[EmailIdentity] = field(default_factory=list)
    platforms: List[PlatformProfile] = field(default_factory=list)
    phone_numbers: List[str] = field(default_factory=list)
    avatar_prompts: Dict[str, str] = field(default_factory=dict)
    tier: IdentityTier = IdentityTier.STANDARD
    status: PersonaStatus = PersonaStatus.TEMPLATE
    created_at: str = ""
    last_used: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    linked_personas: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "demographics": self.demographics.to_dict(),
            "personality": self.personality.to_dict(),
            "emails": [e.to_dict() for e in self.emails],
            "platforms": [p.to_dict() for p in self.platforms],
            "phone_numbers": list(self.phone_numbers),
            "avatar_prompts": dict(self.avatar_prompts),
            "tier": self.tier.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "notes": self.notes,
            "tags": list(self.tags),
            "linked_personas": list(self.linked_personas),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Persona:
        if not data:
            return cls(id=str(uuid.uuid4()), name="Unknown")
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Unknown"),
            demographics=PersonaDemographics.from_dict(data.get("demographics", {})),
            personality=PersonaPersonality.from_dict(data.get("personality", {})),
            emails=[EmailIdentity.from_dict(e) for e in data.get("emails", [])],
            platforms=[PlatformProfile.from_dict(p) for p in data.get("platforms", [])],
            phone_numbers=data.get("phone_numbers", []),
            avatar_prompts=data.get("avatar_prompts", {}),
            tier=IdentityTier(data["tier"]) if data.get("tier") in [t.value for t in IdentityTier] else IdentityTier.STANDARD,
            status=PersonaStatus(data["status"]) if data.get("status") in [s.value for s in PersonaStatus] else PersonaStatus.TEMPLATE,
            created_at=data.get("created_at", ""),
            last_used=data.get("last_used", ""),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
            linked_personas=data.get("linked_personas", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class IdentityGroup:
    id: str
    name: str
    description: str = ""
    persona_ids: List[str] = field(default_factory=list)
    purpose: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IdentityGroup:
        if not data:
            return cls(id=str(uuid.uuid4()), name="Unnamed")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ===================================================================
# IDENTITY MANAGER
# ===================================================================

class IdentityManager:
    """Generate and manage realistic digital identities (personas)."""

    def __init__(
        self,
        account_mgr: Any = None,
        data_dir: Path = None,
    ) -> None:
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._personas_file = self._data_dir / "personas.json"
        self._templates_file = self._data_dir / "persona_templates.json"
        self._profiles_file = self._data_dir / "platform_profiles.json"
        self._avatar_file = self._data_dir / "avatar_prompts.json"
        self._groups_file = self._data_dir / "identity_groups.json"

        self._lock = threading.Lock()
        self._account_mgr = account_mgr

        # Load data
        self._personas: Dict[str, Dict[str, Any]] = _load_json(self._personas_file, {})
        self._templates: Dict[str, Dict[str, Any]] = _load_json(self._templates_file, {})
        self._profiles: Dict[str, Dict[str, Any]] = _load_json(self._profiles_file, {})
        self._avatar_prompts: Dict[str, Dict[str, str]] = _load_json(self._avatar_file, {})
        self._groups: Dict[str, Dict[str, Any]] = _load_json(self._groups_file, {})

        logger.info(
            "IdentityManager loaded: %d personas, %d groups",
            len(self._personas), len(self._groups),
        )

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def _save_personas(self) -> None:
        _save_json(self._personas_file, self._personas)

    def _save_templates(self) -> None:
        _save_json(self._templates_file, self._templates)

    def _save_profiles(self) -> None:
        _save_json(self._profiles_file, self._profiles)

    def _save_avatar_prompts(self) -> None:
        _save_json(self._avatar_file, self._avatar_prompts)

    def _save_groups(self) -> None:
        _save_json(self._groups_file, self._groups)

    # -------------------------------------------------------------------
    # Haiku Integration
    # -------------------------------------------------------------------

    async def _call_haiku(self, prompt: str, max_tokens: int = 500) -> str:
        """Call Haiku for text generation, return empty string on failure."""
        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning("Haiku unavailable: %s", e)
            return ""

    # -------------------------------------------------------------------
    # Demographics Generation
    # -------------------------------------------------------------------

    def _generate_demographics(
        self,
        gender: Gender = None,
        age_range: AgeRange = None,
        country: str = None,
    ) -> PersonaDemographics:
        """Generate random realistic demographics."""
        if gender is None:
            gender = random.choice([Gender.MALE, Gender.FEMALE, Gender.NON_BINARY])

        if gender == Gender.MALE:
            first_name = random.choice(FIRST_NAMES_MALE)
        elif gender == Gender.FEMALE:
            first_name = random.choice(FIRST_NAMES_FEMALE)
        else:
            first_name = random.choice(FIRST_NAMES_MALE + FIRST_NAMES_FEMALE)

        last_name = random.choice(LAST_NAMES)

        if age_range is None:
            age_range = random.choice(list(AgeRange))
        low, high = AGE_RANGE_BOUNDS[age_range]
        age = random.randint(low, high)

        location = random.choice(US_CITIES)
        resolved_country = country or "United States"

        occupation = random.choice(OCCUPATIONS)
        education = random.choice(EDUCATION_LEVELS)
        relationship = random.choice(RELATIONSHIP_STATUSES)

        return PersonaDemographics(
            first_name=first_name,
            last_name=last_name,
            gender=gender,
            age=age,
            age_range=age_range,
            country=resolved_country,
            state=location["state"],
            city=location["city"],
            timezone=location["timezone"],
            language="en",
            occupation=occupation,
            education=education,
            relationship_status=relationship,
        )

    # -------------------------------------------------------------------
    # Personality Generation
    # -------------------------------------------------------------------

    def _generate_personality(
        self,
        demographics: PersonaDemographics,
        niche: str = None,
    ) -> PersonaPersonality:
        """Generate interests/style matching niche and demographics."""
        # Select interests based on niche
        if niche and niche.lower() in NICHE_INTERESTS:
            niche_pool = NICHE_INTERESTS[niche.lower()]
            core_interests = random.sample(niche_pool, min(4, len(niche_pool)))
            extra = random.sample(
                [i for i in INTERESTS if i not in core_interests],
                min(3, len(INTERESTS) - len(core_interests)),
            )
            interests = core_interests + extra
        else:
            interests = random.sample(INTERESTS, min(6, len(INTERESTS)))

        hobbies = random.sample(
            [i for i in INTERESTS if i not in interests],
            min(3, len(INTERESTS) - len(interests)),
        )

        values_pool = [
            "sustainability", "creativity", "community", "authenticity",
            "growth", "kindness", "adventure", "knowledge", "family",
            "independence", "spirituality", "health", "equality",
            "innovation", "simplicity", "humor", "courage", "empathy",
        ]
        values = random.sample(values_pool, min(3, len(values_pool)))

        comm_style = random.choice(COMM_STYLES)
        tone = random.choice(POSTING_TONES)

        # Adjust emoji/hashtag usage by age
        if demographics.age_range in (AgeRange.TEEN, AgeRange.YOUNG_ADULT):
            emoji_usage = random.choice(["moderate", "heavy"])
            hashtag_style = random.choice(["moderate", "many"])
        elif demographics.age_range == AgeRange.ADULT:
            emoji_usage = random.choice(["light", "moderate"])
            hashtag_style = random.choice(["few", "moderate"])
        else:
            emoji_usage = random.choice(["none", "light"])
            hashtag_style = random.choice(["none", "few"])

        topics_to_avoid = random.sample(
            ["politics", "religion", "violence", "controversy", "negativity", "gossip"],
            random.randint(1, 3),
        )

        return PersonaPersonality(
            interests=interests,
            hobbies=hobbies,
            values=values,
            communication_style=comm_style,
            emoji_usage=emoji_usage,
            hashtag_style=hashtag_style,
            posting_tone=tone,
            topics_to_avoid=topics_to_avoid,
            backstory="",
        )

    # -------------------------------------------------------------------
    # Backstory Generation
    # -------------------------------------------------------------------

    async def _generate_backstory(
        self,
        demographics: PersonaDemographics,
        personality: PersonaPersonality,
    ) -> str:
        """Generate a 2-3 sentence backstory. Uses Haiku with fallback."""
        prompt = (
            f"Write a natural, realistic 2-3 sentence backstory for a social media persona.\n"
            f"Name: {demographics.first_name} {demographics.last_name}\n"
            f"Age: {demographics.age}, Gender: {demographics.gender.value}\n"
            f"From: {demographics.city}, {demographics.state}\n"
            f"Occupation: {demographics.occupation}\n"
            f"Interests: {', '.join(personality.interests[:5])}\n"
            f"Communication style: {personality.communication_style}\n\n"
            f"Write ONLY the backstory, no labels or quotes. Make it feel authentic "
            f"and personal, like something someone would naturally say about themselves."
        )

        result = await self._call_haiku(prompt, max_tokens=200)
        if result:
            return result

        # Fallback: template-based backstory
        return self._template_backstory(demographics, personality)

    def _template_backstory(
        self,
        demographics: PersonaDemographics,
        personality: PersonaPersonality,
    ) -> str:
        """Deterministic backstory from templates."""
        template = random.choice(BACKSTORY_TEMPLATES)
        event = random.choice(BACKSTORY_EVENTS)

        # Pronoun mapping
        if demographics.gender == Gender.MALE:
            pronoun, pronoun_cap, possessive = "he", "He", "his"
        elif demographics.gender == Gender.FEMALE:
            pronoun, pronoun_cap, possessive = "she", "She", "her"
        else:
            pronoun, pronoun_cap, possessive = "they", "They", "their"

        interests = personality.interests or ["creative projects"]
        occupation_field = demographics.occupation.lower() if demographics.occupation else "their field"

        return template.format(
            first=demographics.first_name,
            age=demographics.age,
            occupation=demographics.occupation or "professional",
            occupation_field=occupation_field,
            city=demographics.city or "a small town",
            state=demographics.state or "the Midwest",
            interest1=interests[0] if len(interests) > 0 else "creativity",
            interest2=interests[1] if len(interests) > 1 else "learning new things",
            interest3=interests[2] if len(interests) > 2 else "connecting with others",
            event=event,
            pronoun=pronoun,
            pronoun_cap=pronoun_cap,
            possessive=possessive,
        )

    # -------------------------------------------------------------------
    # Username Generation
    # -------------------------------------------------------------------

    def _generate_username(
        self,
        demographics: PersonaDemographics,
        platform: Platform,
    ) -> str:
        """Generate a platform-appropriate username."""
        first = demographics.first_name.lower()
        last = demographics.last_name.lower()
        year_suffix = str(random.randint(90, 99)) if demographics.age > 25 else str(random.randint(0, 9))
        num2 = str(random.randint(1, 999))
        num3 = str(random.randint(10, 99))

        # Clean names for username use
        first_clean = re.sub(r'[^a-z]', '', first)
        last_clean = re.sub(r'[^a-z]', '', last)

        # Get limit for platform
        limits = PLATFORM_BIO_LIMITS.get(platform, {"username": 30})
        max_len = limits.get("username", 30)

        # Gather some interest words for creative usernames
        interest_words = []
        if demographics.occupation:
            occ_word = re.sub(r'[^a-z]', '', demographics.occupation.lower().split()[0])
            if occ_word:
                interest_words.append(occ_word)

        patterns: List[str] = []

        if platform == Platform.LINKEDIN:
            # LinkedIn: professional, first-last style
            patterns = [
                f"{first_clean}{last_clean}",
                f"{first_clean}.{last_clean}",
                f"{first_clean}-{last_clean}",
                f"{first_clean}{last_clean}{num3}",
            ]
        elif platform == Platform.REDDIT:
            # Reddit: creative, pseudo-anonymous
            adjectives = [
                "cosmic", "chill", "quiet", "happy", "wild", "lazy", "clever",
                "mighty", "sneaky", "brave", "noble", "swift", "gentle", "bold",
            ]
            nouns = [
                "panda", "fox", "wolf", "owl", "hawk", "bear", "cat",
                "raven", "otter", "lynx", "hare", "wren", "moth", "crow",
            ]
            adj = random.choice(adjectives)
            noun = random.choice(nouns)
            patterns = [
                f"{adj}_{noun}_{num2}",
                f"{adj}{noun}{num3}",
                f"the_{adj}_{noun}",
                f"{first_clean}_{adj}{num3}",
                f"{noun}_{first_clean}_{num3}",
            ]
        elif platform == Platform.TIKTOK:
            # TikTok: short, catchy, trendy
            patterns = [
                f"{first_clean}.{last_clean}",
                f"{first_clean}_{num2}",
                f"{first_clean}{year_suffix}",
                f"its{first_clean}",
                f"{first_clean}official",
                f"the.{first_clean}",
            ]
        elif platform == Platform.TWITTER:
            # Twitter: short constraint (15 char), punchy
            patterns = [
                f"{first_clean}{num3}",
                f"{first_clean}_{num3}",
                f"{first_clean}{last_clean[:3]}",
                f"the{first_clean}",
                f"{first_clean}x",
                f"real{first_clean}",
            ]
        elif platform == Platform.INSTAGRAM:
            # Instagram: aesthetic, period-separated or underscore
            patterns = [
                f"{first_clean}.{last_clean}",
                f"{first_clean}_{last_clean}",
                f"_{first_clean}.{last_clean}_",
                f"{first_clean}.{last_clean}{num3}",
                f"{first_clean}_{num2}",
                f"the.{first_clean}",
            ]
        elif platform == Platform.PINTEREST:
            # Pinterest: clean, brand-friendly
            patterns = [
                f"{first_clean}{last_clean}",
                f"{first_clean}.{last_clean}",
                f"{first_clean}{last_clean}{num3}",
                f"{first_clean}pins",
                f"{first_clean}boards",
            ]
        elif platform == Platform.YOUTUBE:
            # YouTube: channel-like names
            patterns = [
                f"{first_clean}{last_clean}",
                f"{first_clean}.{last_clean}",
                f"the{first_clean}channel",
                f"{first_clean}{num3}",
                f"{first_clean}official",
            ]
        elif platform == Platform.DISCORD:
            # Discord: creative display names
            patterns = [
                f"{first_clean}_{num2}",
                f"{first_clean}{num3}",
                f"{first_clean}.{last_clean}",
                f"x{first_clean}x",
                f"{first_clean}plays",
            ]
        elif platform == Platform.THREADS:
            # Threads: similar to Instagram
            patterns = [
                f"{first_clean}.{last_clean}",
                f"{first_clean}_{last_clean}",
                f"{first_clean}{num3}",
                f"the.{first_clean}",
            ]
        else:
            # Generic
            patterns = [
                f"{first_clean}{last_clean}{num3}",
                f"{first_clean}.{last_clean}",
                f"{first_clean}_{num2}",
                f"{first_clean}{year_suffix}",
            ]

        username = random.choice(patterns)

        # Enforce character limit
        if len(username) > max_len:
            username = username[:max_len]

        # Strip trailing dots/underscores for cleanliness
        username = username.strip("._ ")

        return username

    # -------------------------------------------------------------------
    # Avatar Prompt Generation
    # -------------------------------------------------------------------

    def _generate_avatar_prompt(
        self,
        demographics: PersonaDemographics,
        personality: PersonaPersonality,
        platform: Platform,
        style: str = "photo-realistic",
    ) -> str:
        """Generate a detailed prompt for AI avatar generation."""
        # Gender description
        if demographics.gender == Gender.MALE:
            gender_desc = "man"
        elif demographics.gender == Gender.FEMALE:
            gender_desc = "woman"
        else:
            gender_desc = "person"

        # Age descriptor
        if demographics.age < 20:
            age_desc = "young"
        elif demographics.age < 30:
            age_desc = "young adult"
        elif demographics.age < 45:
            age_desc = "adult"
        elif demographics.age < 60:
            age_desc = "middle-aged"
        else:
            age_desc = "mature"

        # Mood from posting tone
        tone = personality.posting_tone or "friendly"

        # Interest-based styling
        interests_str = ", ".join(personality.interests[:3]) if personality.interests else "general lifestyle"

        # Platform-specific framing
        if platform == Platform.LINKEDIN:
            setting = "professional headshot, neutral background, business attire"
        elif platform == Platform.INSTAGRAM:
            setting = "lifestyle photo, natural lighting, aesthetically pleasing background"
        elif platform == Platform.TIKTOK:
            setting = "casual selfie style, colorful background, energetic expression"
        elif platform == Platform.TWITTER:
            setting = "close-up headshot, simple background, engaging expression"
        elif platform == Platform.PINTEREST:
            setting = "bright, well-lit portrait, warm tones, inviting expression"
        elif platform == Platform.YOUTUBE:
            setting = "thumbnail-ready portrait, expressive face, vibrant background"
        elif platform == Platform.FACEBOOK:
            setting = "casual portrait, friendly expression, everyday setting"
        elif platform == Platform.REDDIT:
            setting = "casual avatar, can be illustrated or photo-realistic, neutral"
        else:
            setting = "clean portrait, simple background, approachable expression"

        prompt = (
            f"{style} portrait of a {age_desc} {gender_desc}, approximately {demographics.age} years old. "
            f"{setting}. Expression conveys a {tone} personality. "
            f"Person appears interested in {interests_str}. "
            f"High quality, natural-looking, suitable for a {platform.value} profile picture. "
            f"No text, no watermarks, no logos."
        )

        return prompt

    # -------------------------------------------------------------------
    # Bio Formatting (platform-specific)
    # -------------------------------------------------------------------

    def _format_bio_instagram(self, persona: Persona) -> str:
        """Instagram-style bio with line breaks and emojis."""
        d = persona.demographics
        p = persona.personality
        interests = p.interests[:3] if p.interests else ["lifestyle"]

        emoji_map = {
            "witchcraft": "moon", "tarot": "stars", "crystals": "gem",
            "cooking": "fork_and_knife", "photography": "camera",
            "hiking": "mountain", "yoga": "lotus", "music": "musical_note",
            "travel": "airplane", "reading": "book", "gardening": "seedling",
            "painting": "art", "fitness": "muscle", "writing": "pencil2",
            "coffee": "coffee", "coding": "computer",
        }

        lines = []
        # Name/occupation line
        if d.occupation:
            lines.append(d.occupation)

        # Interest line with emoji suggestions
        interest_line = " | ".join(interests[:3])
        lines.append(interest_line)

        # Location
        if d.city and d.state:
            lines.append(f"{d.city}, {d.state}")

        bio = "\n".join(lines)

        # Enforce 150 char limit
        if len(bio) > 150:
            bio = bio[:147] + "..."

        return bio

    def _format_bio_linkedin(self, persona: Persona) -> str:
        """Professional headline + summary for LinkedIn."""
        d = persona.demographics
        p = persona.personality
        interests = p.interests[:4] if p.interests else ["professional development"]

        headline = d.occupation or "Professional"
        if d.city and d.state:
            headline += f" | {d.city}, {d.state}"

        summary_parts = []
        if p.backstory:
            summary_parts.append(p.backstory)
        else:
            summary_parts.append(
                f"Experienced {d.occupation or 'professional'} passionate about "
                f"{', '.join(interests[:2])} and {interests[2] if len(interests) > 2 else 'growth'}."
            )

        summary_parts.append(
            f"Always looking to connect with others who share an interest in {interests[0]}."
        )

        bio = f"{headline}\n\n" + " ".join(summary_parts)

        if len(bio) > 2600:
            bio = bio[:2597] + "..."

        return bio

    def _format_bio_tiktok(self, persona: Persona) -> str:
        """Short, punchy, emoji-heavy TikTok bio."""
        d = persona.demographics
        p = persona.personality
        interests = p.interests[:2] if p.interests else ["vibes"]

        parts = []
        if d.occupation:
            parts.append(d.occupation.split()[0] if " " in d.occupation else d.occupation)

        parts.append(" + ".join(interests[:2]))

        if d.city:
            parts.append(d.city)

        bio = " | ".join(parts)

        if len(bio) > 80:
            bio = bio[:77] + "..."

        return bio

    def _format_bio_twitter(self, persona: Persona) -> str:
        """Witty, topic-focused Twitter bio."""
        d = persona.demographics
        p = persona.personality
        interests = p.interests[:3] if p.interests else ["stuff"]

        parts = []
        if d.occupation:
            parts.append(d.occupation)

        interest_str = " | ".join(interests)
        parts.append(interest_str)

        if p.posting_tone:
            parts.append(f"Mostly {p.posting_tone}")

        bio = ". ".join(parts)

        if len(bio) > 160:
            bio = bio[:157] + "..."

        return bio

    def _format_bio_pinterest(self, persona: Persona) -> str:
        """Pinterest bio - descriptive and inspirational."""
        d = persona.demographics
        p = persona.personality
        interests = p.interests[:4] if p.interests else ["inspiration"]

        lines = []
        if d.occupation:
            lines.append(f"{d.occupation} sharing what inspires me.")

        if interests:
            lines.append(f"Pinning: {', '.join(interests)}")

        if d.city and d.state:
            lines.append(f"Based in {d.city}, {d.state}")

        bio = " ".join(lines)

        if len(bio) > 500:
            bio = bio[:497] + "..."

        return bio

    def _format_bio_youtube(self, persona: Persona) -> str:
        """YouTube channel description."""
        d = persona.demographics
        p = persona.personality
        interests = p.interests[:4] if p.interests else ["content"]

        lines = []
        if p.backstory:
            lines.append(p.backstory)
        else:
            lines.append(
                f"Hey! I'm {d.first_name}, a {d.occupation or 'creator'} "
                f"from {d.city or 'the US'}."
            )

        lines.append(f"On this channel: {', '.join(interests)}")
        lines.append("Subscribe for new content!")

        bio = "\n".join(lines)

        if len(bio) > 1000:
            bio = bio[:997] + "..."

        return bio

    def _format_bio_reddit(self, persona: Persona) -> str:
        """Reddit bio - casual, potentially funny."""
        d = persona.demographics
        p = persona.personality
        interests = p.interests[:3] if p.interests else ["things"]

        parts = []
        parts.append(f"Into {', '.join(interests)}")
        if d.occupation:
            parts.append(f"{d.occupation} by day")

        bio = ". ".join(parts) + "."

        if len(bio) > 200:
            bio = bio[:197] + "..."

        return bio

    def _format_bio_facebook(self, persona: Persona) -> str:
        """Facebook intro bio."""
        d = persona.demographics
        p = persona.personality
        interests = p.interests[:3] if p.interests else ["life"]

        parts = []
        if d.occupation:
            parts.append(d.occupation)
        if d.city and d.state:
            parts.append(f"Lives in {d.city}, {d.state}")
        if interests:
            parts.append(f"Loves {', '.join(interests[:2])}")

        bio = " | ".join(parts)

        if len(bio) > 101:
            bio = bio[:98] + "..."

        return bio

    def _format_bio_threads(self, persona: Persona) -> str:
        """Threads bio - similar to Instagram."""
        return self._format_bio_instagram(persona)

    def _format_bio_generic(self, persona: Persona, platform: Platform) -> str:
        """Generic bio adaptation with character limit enforcement."""
        d = persona.demographics
        p = persona.personality
        interests = p.interests[:3] if p.interests else ["things"]

        parts = []
        if d.occupation:
            parts.append(d.occupation)
        parts.append(f"Into {', '.join(interests)}")
        if d.city:
            parts.append(d.city)

        bio = " | ".join(parts)

        limits = PLATFORM_BIO_LIMITS.get(platform, {"bio": 200})
        max_len = limits.get("bio", 200)
        if len(bio) > max_len:
            bio = bio[:max_len - 3] + "..."

        return bio

    # -------------------------------------------------------------------
    # Core: Persona Generation
    # -------------------------------------------------------------------

    async def generate_persona(
        self,
        niche: str = None,
        gender: Gender = None,
        age_range: AgeRange = None,
        country: str = None,
        tier: IdentityTier = IdentityTier.STANDARD,
    ) -> Persona:
        """Generate a full persona with demographics, personality, and backstory."""
        persona_id = str(uuid.uuid4())

        demographics = self._generate_demographics(gender, age_range, country)
        personality = self._generate_personality(demographics, niche)
        backstory = await self._generate_backstory(demographics, personality)
        personality.backstory = backstory

        name = f"{demographics.first_name} {demographics.last_name}"

        tags = []
        if niche:
            tags.append(niche)

        persona = Persona(
            id=persona_id,
            name=name,
            demographics=demographics,
            personality=personality,
            tier=tier,
            status=PersonaStatus.TEMPLATE,
            created_at=_now_iso(),
            tags=tags,
        )

        with self._lock:
            self._personas[persona_id] = persona.to_dict()
            self._save_personas()

        logger.info("Generated persona: %s (%s) [%s]", name, persona_id[:8], tier.value)
        return persona

    def generate_persona_sync(
        self,
        niche: str = None,
        gender: Gender = None,
        age_range: AgeRange = None,
        country: str = None,
        tier: IdentityTier = IdentityTier.STANDARD,
    ) -> Persona:
        """Synchronous wrapper for generate_persona."""
        return _run_sync(self.generate_persona(niche, gender, age_range, country, tier))

    async def generate_batch(
        self,
        count: int,
        niche: str = None,
        **kwargs,
    ) -> List[Persona]:
        """Batch generate multiple personas."""
        personas = []
        for i in range(count):
            persona = await self.generate_persona(niche=niche, **kwargs)
            personas.append(persona)
            logger.info("Batch progress: %d/%d", i + 1, count)
        return personas

    def generate_batch_sync(self, count: int, niche: str = None, **kwargs) -> List[Persona]:
        """Synchronous wrapper for generate_batch."""
        return _run_sync(self.generate_batch(count, niche, **kwargs))

    # -------------------------------------------------------------------
    # Platform Profile Adaptation
    # -------------------------------------------------------------------

    async def create_platform_profile(
        self,
        persona_id: str,
        platform: Platform,
    ) -> Optional[PlatformProfile]:
        """Generate a platform-specific profile from a persona."""
        persona = self.get_persona(persona_id)
        if persona is None:
            logger.warning("Persona %s not found", persona_id[:8] if persona_id else "None")
            return None

        # Check for existing profile on this platform
        for p in persona.platforms:
            if p.platform == platform:
                logger.info("Persona %s already has a %s profile", persona_id[:8], platform.value)
                return p

        username = self._generate_username(persona.demographics, platform)
        display_name = f"{persona.demographics.first_name} {persona.demographics.last_name}"

        # Enforce display name limits
        limits = PLATFORM_BIO_LIMITS.get(platform, {"name": 50})
        name_limit = limits.get("name", 50)
        if len(display_name) > name_limit:
            display_name = display_name[:name_limit]

        bio = await self.adapt_bio(persona_id, platform)
        avatar_prompt = self._generate_avatar_prompt(
            persona.demographics, persona.personality, platform
        )

        profile = PlatformProfile(
            platform=platform,
            username=username,
            display_name=display_name,
            bio=bio,
            avatar_prompt=avatar_prompt,
            status=PersonaStatus.TEMPLATE,
            created_at=_now_iso(),
        )

        # Add to persona
        with self._lock:
            persona_data = self._personas.get(persona_id)
            if persona_data:
                platforms_list = persona_data.get("platforms", [])
                platforms_list.append(profile.to_dict())
                persona_data["platforms"] = platforms_list
                self._personas[persona_id] = persona_data
                self._save_personas()

            # Also save to platform_profiles index
            profile_key = f"{persona_id}:{platform.value}"
            self._profiles[profile_key] = profile.to_dict()
            self._save_profiles()

            # Save avatar prompt
            if persona_id not in self._avatar_prompts:
                self._avatar_prompts[persona_id] = {}
            self._avatar_prompts[persona_id][platform.value] = avatar_prompt
            self._save_avatar_prompts()

        logger.info(
            "Created %s profile for %s: @%s",
            platform.value, persona.name, username,
        )
        return profile

    def create_platform_profile_sync(
        self,
        persona_id: str,
        platform: Platform,
    ) -> Optional[PlatformProfile]:
        """Synchronous wrapper for create_platform_profile."""
        return _run_sync(self.create_platform_profile(persona_id, platform))

    async def adapt_bio(self, persona_id: str, platform: Platform) -> str:
        """Create a bio within the platform's character limit, matching platform culture."""
        persona = self.get_persona(persona_id)
        if persona is None:
            return ""

        formatter_map = {
            Platform.INSTAGRAM: self._format_bio_instagram,
            Platform.TIKTOK: self._format_bio_tiktok,
            Platform.TWITTER: self._format_bio_twitter,
            Platform.LINKEDIN: self._format_bio_linkedin,
            Platform.PINTEREST: self._format_bio_pinterest,
            Platform.YOUTUBE: self._format_bio_youtube,
            Platform.REDDIT: self._format_bio_reddit,
            Platform.FACEBOOK: self._format_bio_facebook,
            Platform.THREADS: self._format_bio_threads,
        }

        formatter = formatter_map.get(platform)
        if formatter:
            return formatter(persona)
        return self._format_bio_generic(persona, platform)

    def adapt_bio_sync(self, persona_id: str, platform: Platform) -> str:
        """Synchronous wrapper for adapt_bio."""
        return _run_sync(self.adapt_bio(persona_id, platform))

    async def adapt_all_platforms(
        self,
        persona_id: str,
        platforms: List[Platform],
    ) -> List[PlatformProfile]:
        """Create profiles for multiple platforms."""
        profiles = []
        for platform in platforms:
            profile = await self.create_platform_profile(persona_id, platform)
            if profile:
                profiles.append(profile)
        return profiles

    def adapt_all_platforms_sync(
        self,
        persona_id: str,
        platforms: List[Platform],
    ) -> List[PlatformProfile]:
        """Synchronous wrapper for adapt_all_platforms."""
        return _run_sync(self.adapt_all_platforms(persona_id, platforms))

    # -------------------------------------------------------------------
    # Email Management
    # -------------------------------------------------------------------

    def assign_email(
        self,
        persona_id: str,
        provider: Platform,
        address: str,
        is_primary: bool = False,
    ) -> Optional[EmailIdentity]:
        """Link an email to a persona."""
        with self._lock:
            persona_data = self._personas.get(persona_id)
            if persona_data is None:
                logger.warning("Persona %s not found", persona_id[:8] if persona_id else "None")
                return None

            email = EmailIdentity(
                provider=provider,
                address=address,
                is_primary=is_primary,
                created_at=_now_iso(),
                status=PersonaStatus.ACTIVE,
            )

            # If setting as primary, unset any existing primary
            if is_primary:
                emails_list = persona_data.get("emails", [])
                for e in emails_list:
                    e["is_primary"] = False
                persona_data["emails"] = emails_list

            emails_list = persona_data.get("emails", [])
            emails_list.append(email.to_dict())
            persona_data["emails"] = emails_list
            self._personas[persona_id] = persona_data
            self._save_personas()

        logger.info("Assigned email %s to persona %s", address, persona_id[:8])
        return email

    def get_primary_email(self, persona_id: str) -> Optional[EmailIdentity]:
        """Get the primary email for a persona."""
        persona = self.get_persona(persona_id)
        if persona is None:
            return None
        for email in persona.emails:
            if email.is_primary:
                return email
        # Return first email if no primary set
        return persona.emails[0] if persona.emails else None

    def get_recovery_email(self, persona_id: str) -> Optional[EmailIdentity]:
        """Get the recovery email for a persona."""
        persona = self.get_persona(persona_id)
        if persona is None:
            return None
        for email in persona.emails:
            if email.is_recovery:
                return email
        return None

    def setup_email_chain(
        self,
        persona_id: str,
        primary_provider: Platform,
        recovery_provider: Platform,
    ) -> List[EmailIdentity]:
        """Create primary + recovery email references for a persona."""
        persona = self.get_persona(persona_id)
        if persona is None:
            return []

        first = persona.demographics.first_name.lower()
        last = persona.demographics.last_name.lower()
        num = str(random.randint(10, 99))

        provider_domains = {
            Platform.GMAIL: "gmail.com",
            Platform.OUTLOOK: "outlook.com",
            Platform.YAHOO: "yahoo.com",
        }

        primary_domain = provider_domains.get(primary_provider, "gmail.com")
        recovery_domain = provider_domains.get(recovery_provider, "outlook.com")

        primary_address = f"{first}.{last}{num}@{primary_domain}"
        recovery_address = f"{first}{last}{random.randint(100, 999)}@{recovery_domain}"

        emails = []

        primary_email = self.assign_email(persona_id, primary_provider, primary_address, is_primary=True)
        if primary_email:
            emails.append(primary_email)

        # Set recovery flag
        with self._lock:
            recovery_email = EmailIdentity(
                provider=recovery_provider,
                address=recovery_address,
                is_recovery=True,
                created_at=_now_iso(),
                status=PersonaStatus.ACTIVE,
            )
            persona_data = self._personas.get(persona_id)
            if persona_data:
                emails_list = persona_data.get("emails", [])
                emails_list.append(recovery_email.to_dict())
                persona_data["emails"] = emails_list
                self._personas[persona_id] = persona_data
                self._save_personas()
            emails.append(recovery_email)

        logger.info(
            "Set up email chain for %s: primary=%s, recovery=%s",
            persona_id[:8], primary_address, recovery_address,
        )
        return emails

    # -------------------------------------------------------------------
    # Phone Number Management
    # -------------------------------------------------------------------

    def assign_phone(self, persona_id: str, number: str) -> bool:
        """Assign a phone number to a persona."""
        with self._lock:
            persona_data = self._personas.get(persona_id)
            if persona_data is None:
                logger.warning("Persona %s not found", persona_id[:8] if persona_id else "None")
                return False
            phones = persona_data.get("phone_numbers", [])
            if number not in phones:
                phones.append(number)
                persona_data["phone_numbers"] = phones
                self._personas[persona_id] = persona_data
                self._save_personas()
            logger.info("Assigned phone %s to persona %s", number, persona_id[:8])
            return True

    def get_phone_numbers(self, persona_id: str) -> List[str]:
        """Get all phone numbers for a persona."""
        persona_data = self._personas.get(persona_id)
        if persona_data is None:
            return []
        return persona_data.get("phone_numbers", [])

    def remove_phone(self, persona_id: str, number: str) -> bool:
        """Remove a phone number from a persona."""
        with self._lock:
            persona_data = self._personas.get(persona_id)
            if persona_data is None:
                return False
            phones = persona_data.get("phone_numbers", [])
            if number in phones:
                phones.remove(number)
                persona_data["phone_numbers"] = phones
                self._personas[persona_id] = persona_data
                self._save_personas()
                logger.info("Removed phone %s from persona %s", number, persona_id[:8])
                return True
            return False

    # -------------------------------------------------------------------
    # Account Linking
    # -------------------------------------------------------------------

    def link_account(
        self,
        persona_id: str,
        platform: Platform,
        credential_ref: str,
    ) -> bool:
        """Link an account_manager credential to a persona's platform profile."""
        with self._lock:
            persona_data = self._personas.get(persona_id)
            if persona_data is None:
                logger.warning("Persona %s not found", persona_id[:8] if persona_id else "None")
                return False

            platforms_list = persona_data.get("platforms", [])
            found = False
            for p in platforms_list:
                if p.get("platform") == platform.value:
                    p["account_id"] = credential_ref
                    found = True
                    break

            if not found:
                # Create a minimal profile entry with the link
                new_profile = PlatformProfile(
                    platform=platform,
                    account_id=credential_ref,
                    created_at=_now_iso(),
                )
                platforms_list.append(new_profile.to_dict())

            persona_data["platforms"] = platforms_list
            self._personas[persona_id] = persona_data
            self._save_personas()

        logger.info(
            "Linked credential %s to persona %s on %s",
            credential_ref[:8] if credential_ref else "None",
            persona_id[:8],
            platform.value,
        )
        return True

    def get_linked_accounts(self, persona_id: str) -> Dict[str, str]:
        """Get all linked accounts: platform -> credential_ref."""
        persona = self.get_persona(persona_id)
        if persona is None:
            return {}
        result = {}
        for p in persona.platforms:
            if p.account_id:
                result[p.platform.value] = p.account_id
        return result

    def unlink_account(self, persona_id: str, platform: Platform) -> bool:
        """Unlink an account from a persona's platform profile."""
        with self._lock:
            persona_data = self._personas.get(persona_id)
            if persona_data is None:
                return False
            platforms_list = persona_data.get("platforms", [])
            for p in platforms_list:
                if p.get("platform") == platform.value:
                    p["account_id"] = ""
                    persona_data["platforms"] = platforms_list
                    self._personas[persona_id] = persona_data
                    self._save_personas()
                    logger.info("Unlinked %s from persona %s", platform.value, persona_id[:8])
                    return True
            return False

    # -------------------------------------------------------------------
    # Persona Management (CRUD)
    # -------------------------------------------------------------------

    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """Get a persona by ID."""
        data = self._personas.get(persona_id)
        if data is None:
            return None
        return Persona.from_dict(data)

    def list_personas(
        self,
        status: PersonaStatus = None,
        tier: IdentityTier = None,
        niche: str = None,
    ) -> List[Persona]:
        """List all personas, optionally filtered."""
        results = []
        for pid, data in self._personas.items():
            if status and data.get("status") != status.value:
                continue
            if tier and data.get("tier") != tier.value:
                continue
            if niche:
                tags = data.get("tags", [])
                if niche.lower() not in [t.lower() for t in tags]:
                    continue
            results.append(Persona.from_dict(data))
        return results

    def update_persona(self, persona_id: str, **fields) -> Optional[Persona]:
        """Update specific fields on a persona."""
        with self._lock:
            persona_data = self._personas.get(persona_id)
            if persona_data is None:
                logger.warning("Persona %s not found", persona_id[:8] if persona_id else "None")
                return None

            for key, value in fields.items():
                if key in ("id", "created_at"):
                    continue  # Immutable fields
                if isinstance(value, Enum):
                    persona_data[key] = value.value
                elif hasattr(value, "to_dict"):
                    persona_data[key] = value.to_dict()
                else:
                    persona_data[key] = value

            self._personas[persona_id] = persona_data
            self._save_personas()

        logger.info("Updated persona %s: %s", persona_id[:8], list(fields.keys()))
        return Persona.from_dict(persona_data)

    def delete_persona(self, persona_id: str) -> bool:
        """Permanently delete a persona."""
        with self._lock:
            if persona_id not in self._personas:
                return False
            name = self._personas[persona_id].get("name", "Unknown")
            del self._personas[persona_id]
            self._save_personas()

            # Clean up profiles index
            keys_to_remove = [k for k in self._profiles if k.startswith(f"{persona_id}:")]
            for k in keys_to_remove:
                del self._profiles[k]
            if keys_to_remove:
                self._save_profiles()

            # Clean up avatar prompts
            if persona_id in self._avatar_prompts:
                del self._avatar_prompts[persona_id]
                self._save_avatar_prompts()

            # Remove from groups
            for gid, gdata in self._groups.items():
                pids = gdata.get("persona_ids", [])
                if persona_id in pids:
                    pids.remove(persona_id)
                    gdata["persona_ids"] = pids
            self._save_groups()

        logger.info("Deleted persona %s (%s)", persona_id[:8], name)
        return True

    def archive_persona(self, persona_id: str) -> bool:
        """Set a persona to RETIRED status."""
        result = self.update_persona(persona_id, status=PersonaStatus.RETIRED)
        if result:
            logger.info("Archived persona %s", persona_id[:8])
            return True
        return False

    def burn_persona(self, persona_id: str) -> bool:
        """Mark a persona as BURNED (detected/banned)."""
        result = self.update_persona(
            persona_id,
            status=PersonaStatus.BURNED,
            last_used=_now_iso(),
            notes=f"Burned at {_now_iso()}",
        )
        if result:
            logger.info("Burned persona %s", persona_id[:8])
            return True
        return False

    def clone_persona(self, persona_id: str, new_name: str = None) -> Optional[Persona]:
        """Create a variation of an existing persona."""
        original = self.get_persona(persona_id)
        if original is None:
            logger.warning("Cannot clone: persona %s not found", persona_id[:8] if persona_id else "None")
            return None

        new_id = str(uuid.uuid4())
        clone_data = original.to_dict()
        clone_data["id"] = new_id
        clone_data["created_at"] = _now_iso()
        clone_data["last_used"] = ""
        clone_data["status"] = PersonaStatus.TEMPLATE.value
        clone_data["platforms"] = []  # Reset platforms for fresh generation
        clone_data["emails"] = []
        clone_data["phone_numbers"] = []
        clone_data["linked_personas"] = [persona_id]

        if new_name:
            clone_data["name"] = new_name
        else:
            # Vary the name slightly
            new_last = random.choice(LAST_NAMES)
            first = clone_data.get("demographics", {}).get("first_name", "Unknown")
            clone_data["name"] = f"{first} {new_last}"
            if "demographics" in clone_data:
                clone_data["demographics"]["last_name"] = new_last

        clone_data["notes"] = f"Cloned from {persona_id[:8]} at {_now_iso()}"

        with self._lock:
            self._personas[new_id] = clone_data
            self._save_personas()

            # Link the original to the clone
            orig_data = self._personas.get(persona_id, {})
            linked = orig_data.get("linked_personas", [])
            if new_id not in linked:
                linked.append(new_id)
                orig_data["linked_personas"] = linked
                self._personas[persona_id] = orig_data
                self._save_personas()

        clone = Persona.from_dict(clone_data)
        logger.info("Cloned persona %s -> %s (%s)", persona_id[:8], new_id[:8], clone.name)
        return clone

    # -------------------------------------------------------------------
    # Identity Groups
    # -------------------------------------------------------------------

    def create_group(
        self,
        name: str,
        purpose: str = "",
        persona_ids: List[str] = None,
    ) -> IdentityGroup:
        """Create a new identity group."""
        group_id = str(uuid.uuid4())
        group = IdentityGroup(
            id=group_id,
            name=name,
            description="",
            persona_ids=persona_ids or [],
            purpose=purpose,
            created_at=_now_iso(),
        )

        with self._lock:
            self._groups[group_id] = group.to_dict()
            self._save_groups()

        logger.info("Created group %s: %s (%d personas)", group_id[:8], name, len(group.persona_ids))
        return group

    def add_to_group(self, group_id: str, persona_id: str) -> bool:
        """Add a persona to a group."""
        with self._lock:
            gdata = self._groups.get(group_id)
            if gdata is None:
                logger.warning("Group %s not found", group_id[:8] if group_id else "None")
                return False
            if persona_id not in self._personas:
                logger.warning("Persona %s not found", persona_id[:8] if persona_id else "None")
                return False
            pids = gdata.get("persona_ids", [])
            if persona_id not in pids:
                pids.append(persona_id)
                gdata["persona_ids"] = pids
                self._groups[group_id] = gdata
                self._save_groups()
            logger.info("Added persona %s to group %s", persona_id[:8], group_id[:8])
            return True

    def remove_from_group(self, group_id: str, persona_id: str) -> bool:
        """Remove a persona from a group."""
        with self._lock:
            gdata = self._groups.get(group_id)
            if gdata is None:
                return False
            pids = gdata.get("persona_ids", [])
            if persona_id in pids:
                pids.remove(persona_id)
                gdata["persona_ids"] = pids
                self._groups[group_id] = gdata
                self._save_groups()
                logger.info("Removed persona %s from group %s", persona_id[:8], group_id[:8])
                return True
            return False

    def list_groups(self) -> List[IdentityGroup]:
        """List all identity groups."""
        return [IdentityGroup.from_dict(g) for g in self._groups.values()]

    def get_group(self, group_id: str) -> Optional[IdentityGroup]:
        """Get a group by ID."""
        gdata = self._groups.get(group_id)
        if gdata is None:
            return None
        return IdentityGroup.from_dict(gdata)

    def delete_group(self, group_id: str) -> bool:
        """Delete an identity group (does not delete the personas)."""
        with self._lock:
            if group_id not in self._groups:
                return False
            name = self._groups[group_id].get("name", "Unknown")
            del self._groups[group_id]
            self._save_groups()
        logger.info("Deleted group %s (%s)", group_id[:8], name)
        return True

    # -------------------------------------------------------------------
    # Warming Schedules
    # -------------------------------------------------------------------

    def generate_warming_schedule(
        self,
        persona_id: str,
        platform: Platform,
        days: int = 14,
    ) -> Dict[str, Any]:
        """Generate a gradual activity ramp-up plan for a persona on a platform."""
        persona = self.get_persona(persona_id)
        if persona is None:
            return {"error": "Persona not found"}

        schedule = self._default_warming_schedule(platform, days)

        # Attach to the platform profile
        with self._lock:
            persona_data = self._personas.get(persona_id)
            if persona_data:
                platforms_list = persona_data.get("platforms", [])
                found = False
                for p in platforms_list:
                    if p.get("platform") == platform.value:
                        p["warming_schedule"] = schedule
                        p["status"] = PersonaStatus.WARMING.value
                        found = True
                        break
                if not found:
                    new_profile = PlatformProfile(
                        platform=platform,
                        status=PersonaStatus.WARMING,
                        warming_schedule=schedule,
                        created_at=_now_iso(),
                    )
                    platforms_list.append(new_profile.to_dict())
                persona_data["platforms"] = platforms_list
                persona_data["status"] = PersonaStatus.WARMING.value
                self._personas[persona_id] = persona_data
                self._save_personas()

        logger.info(
            "Generated %d-day warming schedule for persona %s on %s",
            days, persona_id[:8], platform.value,
        )
        return schedule

    def get_warming_status(
        self,
        persona_id: str,
        platform: Platform,
    ) -> Dict[str, Any]:
        """Get current warming progress for a persona on a platform."""
        persona = self.get_persona(persona_id)
        if persona is None:
            return {"error": "Persona not found"}

        for p in persona.platforms:
            if p.platform == platform:
                schedule = p.warming_schedule
                if not schedule:
                    return {"status": "no_schedule", "platform": platform.value}

                total_days = len(schedule.get("days", []))
                now = _now_utc()
                start_str = schedule.get("start_date", "")
                if start_str:
                    try:
                        start_dt = datetime.fromisoformat(start_str)
                        if start_dt.tzinfo is None:
                            start_dt = start_dt.replace(tzinfo=UTC)
                        elapsed = (now - start_dt).days
                    except (ValueError, TypeError):
                        elapsed = 0
                else:
                    elapsed = 0

                current_day = min(elapsed + 1, total_days)
                progress_pct = round((current_day / total_days) * 100, 1) if total_days > 0 else 0

                return {
                    "persona_id": persona_id,
                    "platform": platform.value,
                    "total_days": total_days,
                    "current_day": current_day,
                    "progress_pct": progress_pct,
                    "status": p.status.value,
                    "schedule": schedule,
                }

        return {"status": "no_profile", "platform": platform.value}

    def _default_warming_schedule(
        self,
        platform: Platform,
        days: int = 14,
    ) -> Dict[str, Any]:
        """Sensible warming defaults per platform."""
        # Platform-specific daily action ramps
        if platform == Platform.INSTAGRAM:
            max_likes = 50
            max_follows = 20
            max_comments = 10
            max_posts = 2
        elif platform == Platform.TIKTOK:
            max_likes = 80
            max_follows = 30
            max_comments = 15
            max_posts = 3
        elif platform == Platform.TWITTER:
            max_likes = 60
            max_follows = 25
            max_comments = 20
            max_posts = 5
        elif platform == Platform.LINKEDIN:
            max_likes = 30
            max_follows = 15
            max_comments = 8
            max_posts = 1
        elif platform == Platform.PINTEREST:
            max_likes = 40
            max_follows = 20
            max_comments = 5
            max_posts = 10
        elif platform == Platform.YOUTUBE:
            max_likes = 30
            max_follows = 10
            max_comments = 10
            max_posts = 1
        elif platform == Platform.REDDIT:
            max_likes = 40
            max_follows = 10
            max_comments = 15
            max_posts = 3
        elif platform == Platform.FACEBOOK:
            max_likes = 40
            max_follows = 15
            max_comments = 10
            max_posts = 2
        else:
            max_likes = 30
            max_follows = 15
            max_comments = 10
            max_posts = 2

        day_plans = []
        for day_num in range(1, days + 1):
            progress = day_num / days
            # Gradual ramp: slow start, accelerate in middle, plateau near end
            ramp = min(1.0, (progress ** 0.7))

            day_plan = {
                "day": day_num,
                "likes": max(1, int(max_likes * ramp)),
                "follows": max(0, int(max_follows * ramp)),
                "comments": max(0, int(max_comments * ramp)),
                "posts": max(0, int(max_posts * ramp)),
                "browse_minutes": max(5, int(30 * ramp)),
                "sessions": max(1, int(3 * ramp)),
            }
            day_plans.append(day_plan)

        return {
            "platform": platform.value,
            "total_days": days,
            "start_date": _now_iso(),
            "days": day_plans,
            "notes": (
                f"Warming schedule for {platform.value}: {days} days. "
                f"Ramp gradually to max {max_likes} likes, {max_follows} follows, "
                f"{max_comments} comments, {max_posts} posts per day."
            ),
        }

    # -------------------------------------------------------------------
    # Avatar Generation
    # -------------------------------------------------------------------

    def generate_avatar_prompt(
        self,
        persona_id: str,
        platform: Platform = None,
        style: str = "photo-realistic",
    ) -> str:
        """Generate a detailed prompt for AI avatar image generation."""
        persona = self.get_persona(persona_id)
        if persona is None:
            return ""

        target_platform = platform or Platform.INSTAGRAM

        prompt = self._generate_avatar_prompt(
            persona.demographics,
            persona.personality,
            target_platform,
            style=style,
        )

        # Store it
        with self._lock:
            if persona_id not in self._avatar_prompts:
                self._avatar_prompts[persona_id] = {}
            self._avatar_prompts[persona_id][target_platform.value] = prompt
            self._save_avatar_prompts()

        return prompt

    def get_all_avatar_prompts(self, persona_id: str) -> Dict[str, str]:
        """Get all stored avatar prompts for a persona."""
        return dict(self._avatar_prompts.get(persona_id, {}))

    # -------------------------------------------------------------------
    # Search & Filter
    # -------------------------------------------------------------------

    def search_personas(self, query: str) -> List[Persona]:
        """Text search across all persona fields."""
        query_lower = query.lower()
        results = []
        for pid, data in self._personas.items():
            search_text = json.dumps(data, default=str).lower()
            if query_lower in search_text:
                results.append(Persona.from_dict(data))
        return results

    def find_by_platform(
        self,
        platform: Platform,
        username: str = None,
    ) -> List[Persona]:
        """Find personas that have a profile on a specific platform."""
        results = []
        for pid, data in self._personas.items():
            platforms_list = data.get("platforms", [])
            for p in platforms_list:
                if p.get("platform") == platform.value:
                    if username is None or p.get("username", "").lower() == username.lower():
                        results.append(Persona.from_dict(data))
                        break
        return results

    def find_by_niche(self, niche: str) -> List[Persona]:
        """Find personas tagged with a specific niche."""
        niche_lower = niche.lower()
        results = []
        for pid, data in self._personas.items():
            tags = [t.lower() for t in data.get("tags", [])]
            if niche_lower in tags:
                results.append(Persona.from_dict(data))
        return results

    def find_available(
        self,
        platform: Platform,
        tier: IdentityTier = None,
    ) -> List[Persona]:
        """Find personas that are active/template (not burned) for a platform."""
        valid_statuses = {PersonaStatus.ACTIVE.value, PersonaStatus.TEMPLATE.value, PersonaStatus.WARMING.value}
        results = []
        for pid, data in self._personas.items():
            if data.get("status") not in valid_statuses:
                continue
            if tier and data.get("tier") != tier.value:
                continue
            platforms_list = data.get("platforms", [])
            for p in platforms_list:
                if p.get("platform") == platform.value:
                    p_status = p.get("status", PersonaStatus.TEMPLATE.value)
                    if p_status not in (PersonaStatus.BURNED.value, PersonaStatus.SUSPENDED.value):
                        results.append(Persona.from_dict(data))
                        break
            else:
                # Also include personas that don't yet have a profile on this platform
                # (they can still be adapted)
                results.append(Persona.from_dict(data))
        return results

    # -------------------------------------------------------------------
    # Stats / Export / Import
    # -------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get identity statistics."""
        total = len(self._personas)

        by_status: Dict[str, int] = {}
        by_tier: Dict[str, int] = {}
        by_platform: Dict[str, int] = {}

        for pid, data in self._personas.items():
            status = data.get("status", "template")
            tier = data.get("tier", "standard")
            by_status[status] = by_status.get(status, 0) + 1
            by_tier[tier] = by_tier.get(tier, 0) + 1

            for p in data.get("platforms", []):
                plat = p.get("platform", "unknown")
                by_platform[plat] = by_platform.get(plat, 0) + 1

        return {
            "total_personas": total,
            "total_groups": len(self._groups),
            "by_status": by_status,
            "by_tier": by_tier,
            "by_platform": by_platform,
        }

    def export_personas(
        self,
        path: str,
        persona_ids: List[str] = None,
    ) -> int:
        """Export personas to a JSON file. Returns count exported."""
        if persona_ids:
            export_data = {pid: self._personas[pid] for pid in persona_ids if pid in self._personas}
        else:
            export_data = dict(self._personas)

        export_path = Path(path)
        _save_json(export_path, export_data)
        logger.info("Exported %d personas to %s", len(export_data), path)
        return len(export_data)

    def import_personas(self, path: str) -> int:
        """Import personas from a JSON file. Returns count imported."""
        import_path = Path(path)
        import_data = _load_json(import_path, {})

        count = 0
        with self._lock:
            for pid, pdata in import_data.items():
                if pid not in self._personas:
                    self._personas[pid] = pdata
                    count += 1
                else:
                    logger.info("Skipping duplicate persona %s", pid[:8])
            self._save_personas()

        logger.info("Imported %d personas from %s", count, path)
        return count


# ===================================================================
# SINGLETON
# ===================================================================

_identity_manager: Optional[IdentityManager] = None


def get_identity_manager(account_mgr: Any = None) -> IdentityManager:
    """Get or create the singleton IdentityManager instance."""
    global _identity_manager
    if _identity_manager is None:
        _identity_manager = IdentityManager(account_mgr=account_mgr)
    return _identity_manager


# ===================================================================
# TABLE FORMATTING HELPER
# ===================================================================

def _format_table(headers: List[str], rows: List[List[str]], max_col: int = 40) -> str:
    """Format a simple ASCII table."""
    if not rows:
        return "  (no data)\n"

    # Truncate cell values
    def trunc(val: str, limit: int) -> str:
        s = str(val)
        return s[:limit - 3] + "..." if len(s) > limit else s

    all_rows = [headers] + [[trunc(str(c), max_col) for c in r] for r in rows]
    col_widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(headers))]

    lines = []
    # Header
    header_line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("  " + "  ".join("-" * w for w in col_widths))

    for row in all_rows[1:]:
        line = "  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_widths))
        lines.append(line)

    return "\n".join(lines)


# ===================================================================
# CLI
# ===================================================================

def _cli_main() -> None:
    """CLI entry point for identity_manager."""
    parser = argparse.ArgumentParser(
        prog="identity_manager",
        description="Digital Identity Generator & Persona Management",
    )
    sub = parser.add_subparsers(dest="command")

    # -- generate --
    gen_p = sub.add_parser("generate", help="Generate new persona(s)")
    gen_p.add_argument("--niche", type=str, default=None, help="Niche/interest focus")
    gen_p.add_argument("--gender", type=str, default=None, choices=["male", "female", "non_binary"])
    gen_p.add_argument("--age-range", type=str, default=None, choices=["teen", "young_adult", "adult", "middle_aged", "senior"])
    gen_p.add_argument("--tier", type=str, default="standard", choices=["disposable", "standard", "premium", "primary"])
    gen_p.add_argument("--count", type=int, default=1, help="Number of personas to generate")

    # -- list --
    list_p = sub.add_parser("list", help="List personas")
    list_p.add_argument("--status", type=str, default=None, choices=["active", "warming", "suspended", "burned", "retired", "template"])
    list_p.add_argument("--tier", type=str, default=None, choices=["disposable", "standard", "premium", "primary"])
    list_p.add_argument("--niche", type=str, default=None)

    # -- show --
    show_p = sub.add_parser("show", help="Show persona details")
    show_p.add_argument("--id", type=str, required=True, help="Persona ID")

    # -- profile --
    prof_p = sub.add_parser("profile", help="Create platform profile")
    prof_p.add_argument("--persona-id", type=str, required=True)
    prof_p.add_argument("--platform", type=str, required=True, choices=[p.value for p in Platform])

    # -- search --
    srch_p = sub.add_parser("search", help="Search personas")
    srch_p.add_argument("--query", type=str, required=True)

    # -- group --
    grp_p = sub.add_parser("group", help="Manage identity groups")
    grp_sub = grp_p.add_subparsers(dest="group_command")

    grp_create = grp_sub.add_parser("create", help="Create group")
    grp_create.add_argument("--name", type=str, required=True)
    grp_create.add_argument("--purpose", type=str, default="")

    grp_list = grp_sub.add_parser("list", help="List groups")

    grp_add = grp_sub.add_parser("add", help="Add persona to group")
    grp_add.add_argument("--group-id", type=str, required=True)
    grp_add.add_argument("--persona-id", type=str, required=True)

    grp_remove = grp_sub.add_parser("remove", help="Remove persona from group")
    grp_remove.add_argument("--group-id", type=str, required=True)
    grp_remove.add_argument("--persona-id", type=str, required=True)

    grp_delete = grp_sub.add_parser("delete", help="Delete group")
    grp_delete.add_argument("--group-id", type=str, required=True)

    # -- export --
    exp_p = sub.add_parser("export", help="Export personas to file")
    exp_p.add_argument("--output", type=str, required=True)
    exp_p.add_argument("--ids", type=str, nargs="*", default=None, help="Specific persona IDs")

    # -- import --
    imp_p = sub.add_parser("import", help="Import personas from file")
    imp_p.add_argument("--input", type=str, required=True)

    # -- stats --
    sub.add_parser("stats", help="Show identity statistics")

    # -- warming --
    warm_p = sub.add_parser("warming", help="Show/generate warming schedule")
    warm_p.add_argument("--persona-id", type=str, required=True)
    warm_p.add_argument("--platform", type=str, required=True, choices=[p.value for p in Platform])
    warm_p.add_argument("--days", type=int, default=14)
    warm_p.add_argument("--show", action="store_true", help="Show existing schedule instead of generating")

    # -- burn --
    burn_p = sub.add_parser("burn", help="Mark persona as burned")
    burn_p.add_argument("--persona-id", type=str, required=True)

    # -- clone --
    clone_p = sub.add_parser("clone", help="Clone a persona")
    clone_p.add_argument("--persona-id", type=str, required=True)
    clone_p.add_argument("--name", type=str, default=None, help="New name for clone")

    args = parser.parse_args()
    mgr = get_identity_manager()

    if args.command == "generate":
        gender = Gender(args.gender) if args.gender else None
        age_range = AgeRange(args.age_range) if args.age_range else None
        tier = IdentityTier(args.tier)

        if args.count == 1:
            persona = mgr.generate_persona_sync(
                niche=args.niche, gender=gender, age_range=age_range, tier=tier,
            )
            _print_persona_summary(persona)
        else:
            personas = mgr.generate_batch_sync(
                count=args.count, niche=args.niche, gender=gender,
                age_range=age_range, tier=tier,
            )
            print(f"\n  Generated {len(personas)} personas:\n")
            headers = ["ID", "Name", "Age", "City", "Tier", "Status"]
            rows = []
            for p in personas:
                rows.append([
                    p.id[:12],
                    p.name,
                    str(p.demographics.age),
                    p.demographics.city,
                    p.tier.value,
                    p.status.value,
                ])
            print(_format_table(headers, rows))
            print()

    elif args.command == "list":
        status = PersonaStatus(args.status) if args.status else None
        tier = IdentityTier(args.tier) if args.tier else None
        personas = mgr.list_personas(status=status, tier=tier, niche=args.niche)

        print(f"\n  Personas  --  {len(personas)} found\n")
        if personas:
            headers = ["ID", "Name", "Age", "City", "Tier", "Status", "Platforms"]
            rows = []
            for p in personas:
                platform_count = len(p.platforms)
                rows.append([
                    p.id[:12],
                    p.name,
                    str(p.demographics.age),
                    p.demographics.city,
                    p.tier.value,
                    p.status.value,
                    str(platform_count),
                ])
            print(_format_table(headers, rows))
        print()

    elif args.command == "show":
        persona = mgr.get_persona(args.id)
        if persona is None:
            # Try partial match
            matches = [p for pid, p in mgr._personas.items() if pid.startswith(args.id)]
            if matches:
                persona = Persona.from_dict(matches[0])
            else:
                print(f"\n  Persona not found: {args.id}\n")
                return
        _print_persona_detail(persona)

    elif args.command == "profile":
        platform = Platform(args.platform)
        profile = mgr.create_platform_profile_sync(args.persona_id, platform)
        if profile:
            print(f"\n  Platform Profile Created")
            print(f"  {'=' * 40}")
            print(f"  Platform:     {profile.platform.value}")
            print(f"  Username:     @{profile.username}")
            print(f"  Display Name: {profile.display_name}")
            print(f"  Bio:          {profile.bio}")
            print(f"  Status:       {profile.status.value}")
            if profile.avatar_prompt:
                print(f"  Avatar Prompt: {profile.avatar_prompt[:80]}...")
            print()
        else:
            print(f"\n  Failed to create profile. Check persona ID.\n")

    elif args.command == "search":
        results = mgr.search_personas(args.query)
        print(f"\n  Search results for \"{args.query}\"  --  {len(results)} found\n")
        if results:
            headers = ["ID", "Name", "City", "Status", "Tags"]
            rows = []
            for p in results:
                rows.append([
                    p.id[:12],
                    p.name,
                    p.demographics.city,
                    p.status.value,
                    ", ".join(p.tags[:3]),
                ])
            print(_format_table(headers, rows))
        print()

    elif args.command == "group":
        if args.group_command == "create":
            group = mgr.create_group(name=args.name, purpose=args.purpose)
            print(f"\n  Created group: {group.name} ({group.id[:12]})\n")

        elif args.group_command == "list":
            groups = mgr.list_groups()
            print(f"\n  Identity Groups  --  {len(groups)}\n")
            if groups:
                headers = ["ID", "Name", "Purpose", "Personas"]
                rows = []
                for g in groups:
                    rows.append([
                        g.id[:12],
                        g.name,
                        g.purpose[:30] if g.purpose else "-",
                        str(len(g.persona_ids)),
                    ])
                print(_format_table(headers, rows))
            print()

        elif args.group_command == "add":
            ok = mgr.add_to_group(args.group_id, args.persona_id)
            if ok:
                print(f"\n  Added persona {args.persona_id[:12]} to group {args.group_id[:12]}\n")
            else:
                print(f"\n  Failed to add. Check IDs.\n")

        elif args.group_command == "remove":
            ok = mgr.remove_from_group(args.group_id, args.persona_id)
            if ok:
                print(f"\n  Removed persona {args.persona_id[:12]} from group {args.group_id[:12]}\n")
            else:
                print(f"\n  Failed to remove. Check IDs.\n")

        elif args.group_command == "delete":
            ok = mgr.delete_group(args.group_id)
            if ok:
                print(f"\n  Deleted group {args.group_id[:12]}\n")
            else:
                print(f"\n  Group not found: {args.group_id[:12]}\n")

        else:
            grp_p.print_help()

    elif args.command == "export":
        count = mgr.export_personas(args.output, persona_ids=args.ids)
        print(f"\n  Exported {count} personas to {args.output}\n")

    elif args.command == "import":
        count = mgr.import_personas(getattr(args, "input"))
        print(f"\n  Imported {count} personas from {getattr(args, 'input')}\n")

    elif args.command == "stats":
        stats = mgr.get_stats()
        print(f"\n  Identity Statistics")
        print(f"  {'=' * 40}")
        print(f"  Total Personas:  {stats['total_personas']}")
        print(f"  Total Groups:    {stats['total_groups']}")
        print()
        if stats["by_status"]:
            print(f"  By Status:")
            for k, v in sorted(stats["by_status"].items()):
                print(f"    {k:15s}  {v}")
        print()
        if stats["by_tier"]:
            print(f"  By Tier:")
            for k, v in sorted(stats["by_tier"].items()):
                print(f"    {k:15s}  {v}")
        print()
        if stats["by_platform"]:
            print(f"  By Platform:")
            for k, v in sorted(stats["by_platform"].items()):
                print(f"    {k:15s}  {v}")
        print()

    elif args.command == "warming":
        platform = Platform(args.platform)
        if args.show:
            status = mgr.get_warming_status(args.persona_id, platform)
            print(f"\n  Warming Status")
            print(f"  {'=' * 40}")
            for k, v in status.items():
                if k == "schedule":
                    print(f"  Schedule: ({len(v.get('days', []))} days)")
                    for day in v.get("days", [])[:5]:
                        print(f"    Day {day['day']}: {day['likes']} likes, {day['follows']} follows, {day['comments']} comments, {day['posts']} posts")
                    if len(v.get("days", [])) > 5:
                        print(f"    ... and {len(v['days']) - 5} more days")
                else:
                    print(f"  {k}: {v}")
            print()
        else:
            schedule = mgr.generate_warming_schedule(args.persona_id, platform, days=args.days)
            if "error" in schedule:
                print(f"\n  Error: {schedule['error']}\n")
            else:
                print(f"\n  Generated {schedule['total_days']}-day warming schedule for {platform.value}")
                print(f"  {'=' * 50}")
                for day in schedule.get("days", [])[:5]:
                    print(f"  Day {day['day']:2d}: {day['likes']:3d} likes, {day['follows']:2d} follows, {day['comments']:2d} comments, {day['posts']:1d} posts, {day['browse_minutes']:2d}min, {day['sessions']}x")
                remaining = len(schedule.get("days", [])) - 5
                if remaining > 0:
                    print(f"  ... and {remaining} more days")
                print()

    elif args.command == "burn":
        ok = mgr.burn_persona(args.persona_id)
        if ok:
            print(f"\n  Persona {args.persona_id[:12]} marked as BURNED\n")
        else:
            print(f"\n  Persona not found: {args.persona_id[:12]}\n")

    elif args.command == "clone":
        clone = mgr.clone_persona(args.persona_id, new_name=args.name)
        if clone:
            print(f"\n  Cloned persona:")
            _print_persona_summary(clone)
        else:
            print(f"\n  Failed to clone. Check persona ID.\n")

    else:
        parser.print_help()


# ===================================================================
# CLI Display Helpers
# ===================================================================

def _print_persona_summary(persona: Persona) -> None:
    """Print a concise persona summary."""
    d = persona.demographics
    p = persona.personality
    print(f"\n  Persona Generated")
    print(f"  {'=' * 40}")
    print(f"  ID:           {persona.id}")
    print(f"  Name:         {persona.name}")
    print(f"  Age:          {d.age} ({d.age_range.value})")
    print(f"  Gender:       {d.gender.value}")
    print(f"  Location:     {d.city}, {d.state}")
    print(f"  Occupation:   {d.occupation}")
    print(f"  Education:    {d.education}")
    print(f"  Tier:         {persona.tier.value}")
    print(f"  Status:       {persona.status.value}")
    print(f"  Interests:    {', '.join(p.interests[:5])}")
    print(f"  Style:        {p.communication_style}")
    print(f"  Tone:         {p.posting_tone}")
    if p.backstory:
        print(f"  Backstory:    {p.backstory[:120]}{'...' if len(p.backstory) > 120 else ''}")
    if persona.tags:
        print(f"  Tags:         {', '.join(persona.tags)}")
    print()


def _print_persona_detail(persona: Persona) -> None:
    """Print full persona details."""
    d = persona.demographics
    p = persona.personality
    print(f"\n  Persona Detail")
    print(f"  {'=' * 50}")
    print(f"  ID:               {persona.id}")
    print(f"  Name:             {persona.name}")
    print(f"  Created:          {persona.created_at}")
    print(f"  Last Used:        {persona.last_used or 'Never'}")
    print(f"  Tier:             {persona.tier.value}")
    print(f"  Status:           {persona.status.value}")
    print()
    print(f"  Demographics:")
    print(f"    Age:            {d.age} ({d.age_range.value})")
    print(f"    Gender:         {d.gender.value}")
    print(f"    Location:       {d.city}, {d.state}, {d.country}")
    print(f"    Timezone:       {d.timezone}")
    print(f"    Occupation:     {d.occupation}")
    print(f"    Education:      {d.education}")
    print(f"    Relationship:   {d.relationship_status}")
    print()
    print(f"  Personality:")
    print(f"    Interests:      {', '.join(p.interests)}")
    print(f"    Hobbies:        {', '.join(p.hobbies)}")
    print(f"    Values:         {', '.join(p.values)}")
    print(f"    Comm Style:     {p.communication_style}")
    print(f"    Emoji Usage:    {p.emoji_usage}")
    print(f"    Hashtag Style:  {p.hashtag_style}")
    print(f"    Posting Tone:   {p.posting_tone}")
    print(f"    Avoid Topics:   {', '.join(p.topics_to_avoid)}")
    if p.backstory:
        print(f"    Backstory:      {p.backstory}")
    print()

    if persona.emails:
        print(f"  Emails ({len(persona.emails)}):")
        for e in persona.emails:
            flags = []
            if e.is_primary:
                flags.append("PRIMARY")
            if e.is_recovery:
                flags.append("RECOVERY")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"    {e.address} ({e.provider.value}){flag_str}")
        print()

    if persona.platforms:
        print(f"  Platform Profiles ({len(persona.platforms)}):")
        for pp in persona.platforms:
            print(f"    {pp.platform.value}:")
            print(f"      Username:  @{pp.username}")
            print(f"      Name:      {pp.display_name}")
            print(f"      Bio:       {pp.bio[:80]}{'...' if len(pp.bio) > 80 else ''}")
            print(f"      Status:    {pp.status.value}")
            if pp.account_id:
                print(f"      Linked:    {pp.account_id[:12]}")
        print()

    if persona.phone_numbers:
        print(f"  Phone Numbers: {', '.join(persona.phone_numbers)}")
        print()

    if persona.tags:
        print(f"  Tags: {', '.join(persona.tags)}")

    if persona.linked_personas:
        print(f"  Linked Personas: {', '.join(pid[:12] for pid in persona.linked_personas)}")

    if persona.notes:
        print(f"  Notes: {persona.notes}")
    print()


if __name__ == "__main__":
    _cli_main()
