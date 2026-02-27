"""Voice profile loader — injects personality into system prompts."""

import os
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_voice_profiles() -> dict:
    """Load all voice profiles from config."""
    path = CONFIG_DIR / "voice_profiles.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_voice_prompt(profile_name: str = "maker_mentor") -> str:
    """Build a voice injection string for system prompts.

    Combines the profile's rules, vocabulary, and global anti-slop rules
    into a single string ready to inject into any system prompt.
    """
    data = load_voice_profiles()
    profiles = data.get("profiles", {})
    banned = data.get("banned_phrases", [])
    banned_patterns = data.get("banned_patterns", [])

    profile = profiles.get(profile_name)
    if not profile:
        available = ", ".join(profiles.keys())
        raise ValueError(f"Unknown voice profile '{profile_name}'. Available: {available}")

    parts = [
        f"Persona: {profile['name']} — {profile['description']}",
        f"Tone: {profile['tone']}",
        f"Perspective: {profile['perspective']}",
        "",
        "Writing rules:",
    ]
    for rule in profile.get("rules", []):
        parts.append(f"- {rule}")

    vocab = profile.get("vocabulary", {})
    if vocab.get("prefer"):
        parts.append("\nWord choices:")
        for pref in vocab["prefer"]:
            parts.append(f"- {pref}")
    if vocab.get("include"):
        parts.append("\nAlways include specifics like:")
        for inc in vocab["include"]:
            parts.append(f"- {inc}")

    parts.append("\nNEVER use these phrases (they sound like AI slop):")
    for phrase in banned:
        parts.append(f'- "{phrase}"')

    if banned_patterns:
        parts.append("\nNEVER use these patterns:")
        for pat in banned_patterns:
            parts.append(f'- "{pat}"')

    return "\n".join(parts)


def get_profile_for_content_type(content_type: str) -> str:
    """Return the best voice profile name for a content type."""
    mapping = {
        "article": "maker_mentor",
        "review": "gear_reviewer",
        "listing": "maker_mentor",
        "post": "community_voice",
    }
    return mapping.get(content_type, "maker_mentor")
