"""Luna Engine — master orchestrator for the Luna Intelligence System.

Routes incoming messages through the full intelligence pipeline:
1. Extract entities from user message (ConversationalMemory)
2. Build context bundle (ContextEngine)
3. Detect mode and build system prompt (DeepPersona)
4. Route to handler (reading, chat, spell/ritual, knowledge)
5. Score response quality (QualityGate)
6. Record follow-up promises
7. Update companion XP/streak
8. Return response with metadata

Replaces direct calls to readings/companion for chat interactions.
"""

import logging
import re
import time
from dataclasses import dataclass, field

from ...config import get_config
from ...memory import Memory
from .conv_memory import ConversationalMemory
from .context import ContextEngine, ContextBundle
from .deep_persona import DeepPersona, LunaState
from .quality_gate import ResponseQualityGate
from .grimoire_service import GrimoireService
from .persona import get_moon_phase, READING_SIGNOFFS

logger = logging.getLogger("moneyclaw.luna_engine")

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class LunaResponse:
    """Response from the Luna Engine."""
    text: str
    mode: str = "supportive"
    quality_score: int = 0
    cost_cents: int = 0
    model: str = ""
    handler: str = "chat"
    retried: bool = False
    context_summary: dict = field(default_factory=dict)


# Keywords that indicate reading requests
_READING_TRIGGERS = [
    "reading", "tarot", "card pull", "draw a card", "pull a card",
    "celtic cross", "past present future", "what do the cards say",
    "yes or no", "spread", "daily card",
]

# Keywords that indicate spell/ritual requests
_SPELL_TRIGGERS = [
    "spell", "ritual", "ceremony", "cast", "circle",
    "protection spell", "love spell", "money spell",
    "how to", "craft a", "create a ritual",
]

# Keywords that indicate knowledge queries
_KNOWLEDGE_TRIGGERS = [
    "what is", "tell me about", "meaning of", "properties of",
    "what does", "how does", "explain", "what are",
    "herb for", "crystal for", "stone for",
]

# Follow-up promise patterns
_FOLLOWUP_PATTERNS = [
    re.compile(r"(?:i'll|let me|i will)\s+check\s+(?:back|in)\s+(?:on|about|with)\s+(.{5,50})", re.IGNORECASE),
    re.compile(r"(?:we'll|we can)\s+(?:explore|revisit|discuss)\s+(.{5,50}?)(?:\s+(?:next|later|soon))", re.IGNORECASE),
    re.compile(r"(?:come back|return)\s+(?:and|to)\s+(?:tell|share|let)\s+me\s+(?:about|how)\s+(.{5,40})", re.IGNORECASE),
]


class LunaEngine:
    """Master orchestrator for all Luna interactions."""

    def __init__(self, memory: Memory | None = None):
        self.memory = memory or Memory()
        self.config = get_config()
        self.conv_memory = ConversationalMemory(self.memory)
        self.context_engine = ContextEngine(self.memory)
        self.persona = DeepPersona()
        self.quality_gate = ResponseQualityGate()
        self.grimoire = GrimoireService()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if anthropic is None:
                raise RuntimeError("anthropic package not installed")
            self._client = anthropic.Anthropic(
                api_key=self.config.anthropic.api_key
            )
        return self._client

    def respond(self, user_id: str, message: str,
                channel: str = "web") -> LunaResponse:
        """Generate a Luna response — the main entry point."""
        t0 = time.monotonic()

        # 1. Extract entities from user message
        self.conv_memory.extract_and_store(user_id, message)

        # 2. Build context bundle
        context = self.context_engine.build_context(user_id, message, channel)

        # 3. Detect mode and build system prompt
        state = self.persona.detect_mode(
            message,
            user_profile=context.user_profile,
            moon_data=context.moon_data,
        )

        # 4. Route to handler
        handler, response_text, model, cost_cents = self._route(
            user_id, message, context, state
        )

        # 5. Quality gate
        quality_context = {
            "depth_level": state.depth_level,
            "user_profile": context.user_profile,
        }
        score, deductions = self.quality_gate.score(
            response_text, message, quality_context
        )

        retried = False
        if score < self.quality_gate.threshold and handler == "chat":
            # Retry with improvement hints
            suggestions = self.quality_gate.suggest_improvements(
                response_text, message, quality_context
            )
            hint = "\n".join(f"- {s}" for s in suggestions)
            enhanced_message = f"{message}\n\n[Quality improvements needed: {hint}]"

            _, response_text, model, retry_cost = self._route(
                user_id, enhanced_message, context, state
            )
            cost_cents += retry_cost
            retried = True

            score, _ = self.quality_gate.score(
                response_text, message, quality_context
            )

        # 6. Detect follow-up promises in response
        self._detect_followups(user_id, response_text)

        # 7. Update companion (XP, interaction tracking)
        self._update_companion(user_id, channel, handler)

        # 8. Log interaction
        duration_ms = int((time.monotonic() - t0) * 1000)
        self.memory.log_interaction(
            customer_id=user_id,
            channel=channel,
            interaction_type=handler,
            service="luna_engine",
            question=message[:200],
            response=response_text[:500],
            cost_cents=cost_cents,
            duration_ms=duration_ms,
        )

        return LunaResponse(
            text=response_text,
            mode=state.mode,
            quality_score=score,
            cost_cents=cost_cents,
            model=model,
            handler=handler,
            retried=retried,
            context_summary={
                "moon": context.moon_data.get("phase", ""),
                "topics": context.detected_topics,
                "followups": len(context.pending_followups),
                "depth": state.depth_level,
            },
        )

    def _route(self, user_id: str, message: str,
               context: ContextBundle, state: LunaState
               ) -> tuple[str, str, str, int]:
        """Route message to the appropriate handler.

        Returns (handler_name, response_text, model_used, cost_cents).
        """
        lower = message.lower()

        # Reading request
        if any(t in lower for t in _READING_TRIGGERS):
            return self._handle_reading_request(user_id, message, context, state)

        # Spell/ritual request
        if any(t in lower for t in _SPELL_TRIGGERS):
            return self._handle_spell_request(user_id, message, context, state)

        # Knowledge query
        if any(t in lower for t in _KNOWLEDGE_TRIGGERS):
            return self._handle_knowledge_query(user_id, message, context, state)

        # Default: conversational chat
        return self._handle_chat(user_id, message, context, state)

    def _handle_chat(self, user_id: str, message: str,
                     context: ContextBundle, state: LunaState
                     ) -> tuple[str, str, str, int]:
        """Handle conversational chat — the most common case."""
        context_block = self.context_engine.format_for_prompt(context)
        system_prompt = self.persona.build_system_prompt(
            state, context.user_profile, context_block
        )

        # Casual chat uses Haiku for cost efficiency
        model = self.config.anthropic.haiku_model
        max_tokens = {1: 200, 2: 400, 3: 800, 4: 1200}.get(state.depth_level, 400)

        text, cost = self._call_claude(system_prompt, message, model, max_tokens)
        return "chat", text, model, cost

    def _handle_reading_request(self, user_id: str, message: str,
                                context: ContextBundle, state: LunaState
                                ) -> tuple[str, str, str, int]:
        """Handle reading requests — delegates to ReadingEngine for full readings."""
        # For quick readings (daily, yes/no), handle inline
        lower = message.lower()

        if "yes" in lower and ("no" in lower or "or" in lower):
            # Yes/no reading
            from .readings import ReadingEngine
            engine = ReadingEngine(self.memory)
            result = engine.quick_yes_no(message, customer_id=user_id)
            return "reading", result["reading"], self.config.anthropic.haiku_model, 0

        # For other readings, suggest using the full reading system
        context_block = self.context_engine.format_for_prompt(context)
        system_prompt = self.persona.build_system_prompt(
            state, context.user_profile, context_block
        )
        system_prompt += "\n\n## Task: The user wants a reading. Guide them to choose a spread type and help them formulate their question. Be warm and inviting."

        model = self.config.anthropic.haiku_model
        text, cost = self._call_claude(system_prompt, message, model, 400)
        return "reading", text, model, cost

    def _handle_spell_request(self, user_id: str, message: str,
                              context: ContextBundle, state: LunaState
                              ) -> tuple[str, str, str, int]:
        """Handle spell/ritual requests using GrimoireService."""
        # Extract intention from message
        intention = self._extract_intention(message)

        if "ritual" in message.lower():
            data = self.grimoire.craft_ritual(intention)
        else:
            data = self.grimoire.craft_spell(intention)

        # Generate Luna's presentation of the spell/ritual
        context_block = self.context_engine.format_for_prompt(context)
        system_prompt = self.persona.build_system_prompt(
            state, context.user_profile, context_block
        )

        spell_info = f"""
## Spell/Ritual Data (Present this warmly as Luna)
Title: {data['title']}
Ingredients: {', '.join(data['ingredients'].get('herbs', []))} + {', '.join(data['ingredients'].get('crystals', []))}
Candle: {data['ingredients'].get('candle_color', 'white')}
Best timing: {data.get('best_timing', {}).get('moon_phase', 'Any')} on {data.get('best_timing', {}).get('day_of_week', 'any day')}
Steps: {chr(10).join(data.get('steps', []))}
"""
        system_prompt += spell_info

        model = self.config.anthropic.default_model  # Sonnet for quality
        max_tokens = 1200
        text, cost = self._call_claude(system_prompt, message, model, max_tokens)
        return "spell", text, model, cost

    def _handle_knowledge_query(self, user_id: str, message: str,
                                context: ContextBundle, state: LunaState
                                ) -> tuple[str, str, str, int]:
        """Handle knowledge questions using grimoire data + Claude synthesis."""
        lower = message.lower()

        # Try to find specific herb/crystal/card
        knowledge_block = ""

        # Check herbs
        for herb_name in ["lavender", "rosemary", "sage", "chamomile", "mugwort",
                          "basil", "thyme", "cinnamon"]:
            if herb_name in lower:
                data = self.grimoire.lookup_herb(herb_name)
                if data:
                    knowledge_block += f"\n## Herb Data: {data.get('name', herb_name)}\n"
                    knowledge_block += f"Element: {data.get('element', 'Unknown')}\n"
                    knowledge_block += f"Uses: {', '.join(data.get('uses', []))}\n"

        # Check crystals
        for crystal_name in ["amethyst", "quartz", "tourmaline", "citrine", "selenite",
                              "obsidian", "labradorite", "moonstone"]:
            if crystal_name in lower:
                data = self.grimoire.lookup_crystal(crystal_name)
                if data:
                    knowledge_block += f"\n## Crystal Data: {data.get('name', crystal_name)}\n"
                    knowledge_block += f"Element: {data.get('element', 'Unknown')}\n"
                    knowledge_block += f"Uses: {', '.join(data.get('uses', []))}\n"

        context_block = self.context_engine.format_for_prompt(context)
        system_prompt = self.persona.build_system_prompt(
            state, context.user_profile, context_block
        )
        if knowledge_block:
            system_prompt += f"\n\n## Knowledge to Share:\n{knowledge_block}"
        system_prompt += "\n\nShare this knowledge warmly as Luna, adding practical advice for working with these materials."

        model = self.config.anthropic.haiku_model
        text, cost = self._call_claude(system_prompt, message, model, 600)
        return "knowledge", text, model, cost

    def _call_claude(self, system_prompt: str, user_message: str,
                     model: str, max_tokens: int) -> tuple[str, int]:
        """Call Claude and return (response_text, cost_cents)."""
        try:
            msg = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_message}],
            )

            text = msg.content[0].text
            in_tokens = msg.usage.input_tokens
            out_tokens = msg.usage.output_tokens

            # Estimate cost
            if "haiku" in model:
                cost_cents = int(
                    (in_tokens * 0.80 / 1_000_000 + out_tokens * 4.00 / 1_000_000) * 100
                )
            else:
                cost_cents = int(
                    (in_tokens * 3.00 / 1_000_000 + out_tokens * 15.00 / 1_000_000) * 100
                )

            return text, cost_cents

        except Exception as e:
            logger.error("Claude API call failed: %s", e)
            return self._fallback_response(), 0

    def _fallback_response(self) -> str:
        """Emergency fallback when API fails."""
        moon = get_moon_phase()
        return (
            f"The {moon['phase']} holds its own wisdom tonight, darling. "
            f"While I gather my thoughts, remember: {moon['guidance']}. "
            f"Try reaching out again in a moment — the ether was briefly disturbed. 🌙"
        )

    def _detect_followups(self, user_id: str, response: str):
        """Detect follow-up promises in Luna's response."""
        for pattern in _FOLLOWUP_PATTERNS:
            match = pattern.search(response)
            if match:
                promise = match.group(1).strip()
                if len(promise) >= 5:
                    self.conv_memory.record_followup(user_id, promise, due_days=7)
                    break

    def _update_companion(self, user_id: str, channel: str, handler: str):
        """Update companion XP for the interaction."""
        try:
            from .companion import CompanionEngine
            comp = CompanionEngine(self.memory)
            if handler == "reading":
                comp.award_xp(user_id, "reading_free")
            else:
                comp.award_xp(user_id, "chat_message")
        except Exception:
            pass

    def _extract_intention(self, message: str) -> str:
        """Extract the intention/goal from a spell/ritual request."""
        lower = message.lower()

        # Priority: keyword-based detection (most precise)
        if "protect" in lower:
            return "protection"
        if "love" in lower or "heart" in lower:
            return "love"
        if "money" in lower or "prosper" in lower:
            return "money"
        if "heal" in lower or "health" in lower:
            return "healing"
        if "peace" in lower or "calm" in lower:
            return "peace"

        # Fallback: regex extraction for less common intentions
        patterns = [
            re.compile(r"(?:spell|ritual)\s+(?:for|to)\s+(.{3,30}?)(?:\.|!|\?|$)", re.IGNORECASE),
            re.compile(r"(?:help\s+me\s+with|help\s+with|need)\s+(.{3,30}?)(?:\.|!|\?|$)", re.IGNORECASE),
        ]
        for p in patterns:
            m = p.search(message)
            if m:
                return m.group(1).strip()

        return "guidance"
