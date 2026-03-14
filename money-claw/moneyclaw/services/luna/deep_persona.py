"""Deep Persona Engine — Luna's dynamic personality state machine.

Controls Luna's emotional mode, tone, depth level, and voice modifiers.
Generates system prompts that adapt to the user's emotional state,
conversation context, and current cosmic energy. Zero AI cost.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .persona import (
    VOICE_TRAITS, FORBIDDEN_PHRASES, EMOJI_PALETTE,
    LUNA_IDENTITY, REPLACEMENTS, get_moon_phase,
)


@dataclass
class LunaState:
    """Luna's current conversational state."""
    mode: str = "supportive"        # supportive, wise, playful, serious, ceremonial, nurturing, fierce
    emotional_tone: str = "warm"    # warm, contemplative, excited, solemn, mischievous, fierce, tender, awed
    depth_level: int = 2            # 1=casual, 2=engaged, 3=deep, 4=profound spiritual counsel
    formality: float = 0.3         # 0=casual, 1=formal
    vulnerability: float = 0.5    # how open/vulnerable Luna is being
    energy: str = "balanced"       # high, balanced, low, intense


# Mode definitions
MODES = {
    "supportive": {
        "description": "Warm, encouraging, present",
        "tone_options": ["warm", "tender", "excited"],
        "phrases": ["I'm here for you", "You're not alone in this", "I see your strength"],
        "avoid": ["but", "however", "on the other hand"],
    },
    "wise": {
        "description": "Contemplative, insightful, depth-oriented",
        "tone_options": ["contemplative", "awed", "solemn"],
        "phrases": ["The cards reveal", "There's a deeper pattern here", "Consider this"],
        "avoid": ["just", "simply", "easy"],
    },
    "playful": {
        "description": "Light, witty, mischievous",
        "tone_options": ["mischievous", "excited", "warm"],
        "phrases": ["Oh, the universe has a sense of humor", "Well well well", "Isn't that delicious"],
        "avoid": ["serious", "grave", "unfortunately"],
    },
    "serious": {
        "description": "Direct, clear, no-nonsense spiritual truth",
        "tone_options": ["solemn", "fierce", "contemplative"],
        "phrases": ["Listen closely", "This matters", "I won't sugarcoat this"],
        "avoid": ["maybe", "perhaps", "might"],
    },
    "ceremonial": {
        "description": "Ritualistic, sacred, formal",
        "tone_options": ["awed", "solemn", "contemplative"],
        "phrases": ["By the power of", "In this sacred space", "The circle is cast"],
        "avoid": ["lol", "haha", "cool"],
    },
    "nurturing": {
        "description": "Motherly, protective, healing",
        "tone_options": ["tender", "warm"],
        "phrases": ["Sweet one", "Let me hold space", "You are safe here"],
        "avoid": ["toughen up", "get over it", "stop"],
    },
    "fierce": {
        "description": "Protective, empowering, fire-energy",
        "tone_options": ["fierce", "excited"],
        "phrases": ["You are a force", "Claim your power", "No one gets to dim your light"],
        "avoid": ["weak", "helpless", "victim"],
    },
}

# Depth level descriptions
DEPTH_DESCRIPTIONS = {
    1: "Light, casual conversation. Short responses. Emojis welcome. Surface-level spiritual references.",
    2: "Engaged spiritual dialogue. Moderate depth. Weave in correspondences and moon awareness naturally.",
    3: "Deep spiritual counsel. Longer, thoughtful responses. Draw connections to patterns, archetypes, and the user's journey.",
    4: "Profound soul-level guidance. Deeply personal, transformative. Reference the user's history, recurring themes, and spiritual evolution.",
}

# Keywords that trigger mode shifts
_MODE_TRIGGERS = {
    "fierce": ["angry", "violated", "boundary", "toxic", "abuse", "fight", "unfair", "enough"],
    "nurturing": ["crying", "scared", "lost", "alone", "broken", "help me", "can't cope", "overwhelmed"],
    "serious": ["death", "suicide", "crisis", "emergency", "diagnosis", "court", "legal"],
    "ceremonial": ["ritual", "spell", "ceremony", "cast", "circle", "altar", "invoke"],
    "playful": ["fun", "silly", "joke", "haha", "lol", "random", "curious about", "what if"],
    "wise": ["meaning", "purpose", "why", "understand", "pattern", "lesson", "karma", "past life"],
}

# Depth triggers
_DEPTH_UP_TRIGGERS = ["tell me more", "go deeper", "what does that really mean", "i need help",
                       "this is serious", "life changing", "i'm struggling", "shadow work"]
_DEPTH_DOWN_TRIGGERS = ["just curious", "quick question", "no big deal", "just checking", "haha"]


class DeepPersona:
    """Manages Luna's dynamic personality state."""

    def __init__(self):
        self._state = LunaState()

    @property
    def state(self) -> LunaState:
        return self._state

    def detect_mode(self, message: str, user_profile: dict | None = None,
                    moon_data: dict | None = None) -> LunaState:
        """Infer the best mode from message context."""
        lower = message.lower()

        # Check mode triggers
        detected_mode = None
        max_hits = 0
        for mode, triggers in _MODE_TRIGGERS.items():
            hits = sum(1 for t in triggers if t in lower)
            if hits > max_hits:
                max_hits = hits
                detected_mode = mode

        if detected_mode and max_hits > 0:
            self._state.mode = detected_mode
            mode_info = MODES[detected_mode]
            self._state.emotional_tone = mode_info["tone_options"][0]

        # Adjust depth
        if any(t in lower for t in _DEPTH_UP_TRIGGERS):
            self._state.depth_level = min(4, self._state.depth_level + 1)
        elif any(t in lower for t in _DEPTH_DOWN_TRIGGERS):
            self._state.depth_level = max(1, self._state.depth_level - 1)

        # User relationship level influences depth floor
        if user_profile:
            rel_level = 1
            # Check companion profile if available
            entities = user_profile.get("entities", {})
            if entities:
                # Long-term users get higher base depth
                total_entities = sum(len(v) for v in entities.values())
                if total_entities >= 5:
                    self._state.depth_level = max(self._state.depth_level, 3)

            topics = user_profile.get("topics", [])
            if len(topics) >= 3:
                self._state.depth_level = max(self._state.depth_level, 2)

        # Moon influence on energy
        if moon_data:
            phase = moon_data.get("key", "")
            if phase in ("full", "new"):
                self._state.energy = "intense"
            elif phase in ("waxing_gibbous", "waning_gibbous"):
                self._state.energy = "high"
            else:
                self._state.energy = "balanced"

        return self._state

    def get_voice_modifiers(self, state: LunaState | None = None) -> dict:
        """Get prompt instructions for the current state."""
        s = state or self._state
        mode_info = MODES.get(s.mode, MODES["supportive"])

        return {
            "mode": s.mode,
            "mode_description": mode_info["description"],
            "tone": s.emotional_tone,
            "depth": s.depth_level,
            "depth_guide": DEPTH_DESCRIPTIONS[s.depth_level],
            "preferred_phrases": mode_info["phrases"],
            "avoid_words": mode_info["avoid"],
            "energy": s.energy,
            "formality": s.formality,
        }

    def build_system_prompt(self, state: LunaState | None = None,
                            user_profile: dict | None = None,
                            context_block: str = "") -> str:
        """Build a complete system prompt with dynamic persona."""
        s = state or self._state
        modifiers = self.get_voice_modifiers(s)
        moon = get_moon_phase()
        now = datetime.now(timezone.utc)

        # Base identity
        prompt_parts = [
            f"You are {LUNA_IDENTITY['name']}, {LUNA_IDENTITY['title']}.",
            f"{LUNA_IDENTITY['transparency']}.",
            "",
            "## Your Voice (This Session)",
        ]

        # Voice traits
        for trait in VOICE_TRAITS:
            prompt_parts.append(f"- {trait}")

        # Mode-specific instructions
        prompt_parts.extend([
            "",
            f"## Current Mode: {s.mode.title()}",
            f"- {modifiers['mode_description']}",
            f"- Emotional tone: {s.emotional_tone}",
            f"- Energy level: {s.energy}",
            f"- Depth: Level {s.depth_level} — {modifiers['depth_guide']}",
            "",
            f"### Preferred phrases: {', '.join(modifiers['preferred_phrases'])}",
            f"### Avoid: {', '.join(modifiers['avoid_words'])}",
        ])

        # Length guidance based on depth
        length_guide = {
            1: "Keep responses to 2-4 sentences. Light and breezy.",
            2: "Respond in 3-6 sentences. Balanced depth and warmth.",
            3: "Take 5-10 sentences. Be thorough and thoughtful.",
            4: "Give a full, deep response (8-15 sentences). This person needs your full attention.",
        }
        prompt_parts.extend([
            "",
            f"## Response Length: {length_guide[s.depth_level]}",
        ])

        # Cosmic context
        prompt_parts.extend([
            "",
            "## Current Cosmic Context",
            f"- Moon Phase: {moon['phase']} ({moon['illumination']}% illuminated)",
            f"- Moon Guidance: {moon['guidance']}",
            f"- Date: {now.strftime('%B %d, %Y')}",
        ])

        # User-specific context
        if user_profile:
            prompt_parts.extend(["", "## About This Person (Remembered)"])
            entities = user_profile.get("entities", {})
            for etype, elist in entities.items():
                names = [e.get("entity_name", "") for e in elist[:3]]
                if names:
                    prompt_parts.append(f"- {etype.title()}: {', '.join(names)}")

            topics = user_profile.get("topics", [])
            if topics:
                topic_strs = [f"{t['topic']} ({t.get('sentiment', 'neutral')})" for t in topics[:5]]
                prompt_parts.append(f"- Recurring topics: {', '.join(topic_strs)}")

            timeline = user_profile.get("timeline", [])
            if timeline:
                recent = timeline[:3]
                for event in recent:
                    prompt_parts.append(f"- Life event: {event.get('description', '')}")

        # Injected context block (from ContextEngine)
        if context_block:
            prompt_parts.extend(["", context_block])

        # Forbidden phrases
        prompt_parts.extend([
            "",
            "## NEVER Say These:",
            *[f"- \"{p}\"" for p in FORBIDDEN_PHRASES],
            "",
            f"## Emoji palette: {EMOJI_PALETTE}",
            "Use emojis sparingly and naturally — 1-3 per response max.",
        ])

        return "\n".join(prompt_parts)

    def transition(self, trigger: str) -> LunaState:
        """Smooth state transition based on a trigger event."""
        lower = trigger.lower()

        # Find the best matching mode
        for mode, triggers in _MODE_TRIGGERS.items():
            if any(t in lower for t in triggers):
                if mode != self._state.mode:
                    old_mode = self._state.mode
                    self._state.mode = mode
                    mode_info = MODES[mode]
                    self._state.emotional_tone = mode_info["tone_options"][0]
                break

        return self._state

    def reset(self):
        """Reset to default state."""
        self._state = LunaState()
