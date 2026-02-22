"""GrimoireEngine — Master orchestrator connecting all modules."""

import time
import datetime
from pathlib import Path

from grimoire.models import (
    RitualPlan, GeneratedSpell, GeneratedRitual, GeneratedMeditation,
    AmplifyResult, EnhancedQuery, PracticeEntry, TarotReading,
    DailyPractice, WeeklyForecast, ConsultResult, GrimoireResult,
    JourneyInsight, QueryType, IntentionCategory,
)
from grimoire.forge.spell_scout import SpellScout
from grimoire.forge.ritual_sentinel import RitualSentinel
from grimoire.forge.moon_oracle import MoonOracle
from grimoire.knowledge.planetary_hours import get_day_ruler
from grimoire.forge.spell_smith import SpellSmith
from grimoire.forge.practice_codex import PracticeCodex
from grimoire.amplify.amplify_pipeline import AmplifyPipeline
from grimoire.enhancer.mystic_enhancer import MysticEnhancer
from grimoire.voice import get_opening, get_closing, get_encouragement


class GrimoireEngine:
    """Master orchestrator for the Grimoire Intelligence System.

    Wires together FORGE (5 intelligence modules), AMPLIFY (6-stage pipeline),
    and the Mystic Prompt Enhancer into a unified interface.
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(Path(__file__).parent / "data" / "grimoire.db")
        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # FORGE modules
        self.scout = SpellScout()
        self.sentinel = RitualSentinel()
        self.oracle = MoonOracle()
        self.smith = SpellSmith()
        self.codex = PracticeCodex(db_path)

        # AMPLIFY pipeline
        self.amplify = AmplifyPipeline()

        # Prompt enhancer
        self.enhancer = MysticEnhancer()

    # ── Primary methods ────────────────────────────────────────────────

    def consult(self, query: str) -> ConsultResult:
        """Ask anything — auto-routes to appropriate modules.

        This is the main entry point. Takes any natural language query about
        witchcraft practice and returns a comprehensive, helpful response.
        """
        start = time.time()

        # Enhance the query
        enhanced = self.enhancer.enhance(query)

        # Get current energy context
        energy = self.oracle.get_current_energy()

        # Detect what kind of help they need
        query_type = enhanced.query_type

        # Route to appropriate handler
        response_parts = []
        correspondences = {}
        practice_suggestions = []
        related_topics = []

        if query_type == QueryType.SPELL_REQUEST:
            result = self._handle_spell_request(query, enhanced)
            response_parts = result["response_parts"]
            correspondences = result["correspondences"]
            practice_suggestions = result["suggestions"]
            related_topics = ["timing", "correspondences", "spell types"]

        elif query_type == QueryType.HERB_CRYSTAL_QUERY:
            result = self._handle_herb_crystal_query(query, enhanced)
            response_parts = result["response_parts"]
            correspondences = result["correspondences"]
            practice_suggestions = result["suggestions"]
            related_topics = ["magical properties", "safety", "pairings"]

        elif query_type == QueryType.MOON_QUERY:
            result = self._handle_moon_query(query, enhanced)
            response_parts = result["response_parts"]
            practice_suggestions = result["suggestions"]
            related_topics = ["lunar magick", "timing", "moon rituals"]

        elif query_type == QueryType.SABBAT_PLANNING:
            result = self._handle_sabbat_query(query, enhanced)
            response_parts = result["response_parts"]
            correspondences = result["correspondences"]
            practice_suggestions = result["suggestions"]
            related_topics = ["wheel of year", "seasonal magick"]

        elif query_type == QueryType.DIVINATION_QUESTION:
            result = self._handle_divination_query(query, enhanced)
            response_parts = result["response_parts"]
            practice_suggestions = result["suggestions"]
            related_topics = ["tarot", "scrying", "pendulum"]

        elif query_type == QueryType.MEDITATION_REQUEST:
            result = self._handle_meditation_query(query, enhanced)
            response_parts = result["response_parts"]
            practice_suggestions = result["suggestions"]
            related_topics = ["grounding", "visualization", "chakras"]

        elif query_type == QueryType.SHADOW_WORK:
            result = self._handle_shadow_work_query(query, enhanced)
            response_parts = result["response_parts"]
            practice_suggestions = result["suggestions"]
            related_topics = ["inner work", "journaling", "self-compassion"]

        elif query_type == QueryType.TAROT_QUERY:
            result = self._handle_tarot_query(query, enhanced)
            response_parts = result["response_parts"]
            practice_suggestions = result["suggestions"]
            related_topics = ["spreads", "card meanings", "divination"]

        else:
            result = self._handle_general_query(query, enhanced)
            response_parts = result["response_parts"]
            practice_suggestions = result["suggestions"]
            related_topics = ["practice tips", "getting started"]

        # Build moon context
        moon_context = (
            f"{energy.phase_name} — {energy.magical_energy}"
        )

        # Build timing advice
        timing_advice = ""
        intention = self.enhancer._extract_intention(query)
        if intention:
            dates = self.oracle.suggest_best_dates(intention, days_ahead=14)
            if dates:
                best = dates[0]
                timing_advice = f"Best upcoming date: {best['date']} ({best['reason']})"

        response = "\n\n".join(response_parts)

        elapsed = (time.time() - start) * 1000

        return ConsultResult(
            query=query,
            query_type=query_type,
            response=response,
            correspondences=correspondences,
            moon_context=moon_context,
            timing_advice=timing_advice,
            practice_suggestions=practice_suggestions,
            related_topics=related_topics,
            sources=["Grimoire Knowledge Base", "Moon Oracle", "Spell Scout"],
        )

    def current_energy(self) -> dict:
        """What's the magical energy right now?"""
        energy = self.oracle.get_current_energy()
        guidance = self.oracle.get_daily_guidance()

        return {
            "moon_phase": energy.phase_name,
            "illumination": energy.illumination,
            "zodiac_sign": energy.zodiac_sign,
            "magical_energy": energy.magical_energy,
            "best_for": energy.best_for,
            "avoid": energy.avoid,
            "daily_guidance": guidance,
            "element": energy.element,
            "keywords": energy.keywords,
        }

    def craft_spell(
        self, intention: str, difficulty: str = "beginner", spell_type: str = "candle",
        amplify_result: bool = True
    ) -> GrimoireResult:
        """Generate + optionally amplify a complete spell."""
        start = time.time()
        summary = []

        # Scout the intention
        scout_result = self.scout.analyze(intention)
        summary.append(f"Intention: {scout_result.category.value} (score: {scout_result.completeness_score})")

        # Generate the spell
        spell = self.smith.craft_spell(intention, spell_type, difficulty)
        summary.append(f"Generated: {spell.title}")

        # Convert to RitualPlan for scoring and amplification
        plan = self.smith.to_ritual_plan(spell)

        # Score it
        score_result = self.sentinel.score(plan)
        summary.append(f"Score: {score_result.total_score} ({score_result.grade})")

        # Amplify if requested
        amp_result = None
        if amplify_result:
            amp_result = self.amplify.amplify(plan)
            summary.append(f"Amplified: quality {amp_result.quality_score}")

        elapsed = (time.time() - start) * 1000

        return GrimoireResult(
            action="craft_spell",
            data=spell,
            forge_intel={
                "scout": {
                    "category": scout_result.category.value,
                    "completeness": scout_result.completeness_score,
                    "quick_start": scout_result.quick_start,
                },
                "sentinel": {
                    "score": score_result.total_score,
                    "grade": score_result.grade,
                    "suggestions": score_result.suggestions,
                },
            },
            amplify_result=amp_result,
            processing_time_ms=elapsed,
            summary=summary,
        )

    def craft_ritual(
        self, occasion: str, intention: str, difficulty: str = "beginner",
        amplify_result: bool = True
    ) -> GrimoireResult:
        """Generate + optionally amplify a complete ritual."""
        start = time.time()
        summary = []

        # Generate the ritual
        ritual = self.smith.craft_ritual(occasion, intention, difficulty)
        summary.append(f"Generated: {ritual.title}")

        # Create a RitualPlan for scoring
        plan = RitualPlan(
            title=ritual.title,
            intention=ritual.intention,
            category=occasion,
            difficulty=ritual.difficulty,
            materials=[],
            steps=ritual.body,
            timing=ritual.timing_notes,
            correspondences_used=ritual.correspondences,
            safety_notes=ritual.safety_notes,
            preparation=ritual.preparation,
            aftercare=ritual.aftercare,
        )

        # Score
        score_result = self.sentinel.score(plan)
        summary.append(f"Score: {score_result.total_score} ({score_result.grade})")

        # Amplify
        amp_result = None
        if amplify_result:
            amp_result = self.amplify.amplify(plan)
            summary.append(f"Amplified: quality {amp_result.quality_score}")

        elapsed = (time.time() - start) * 1000

        return GrimoireResult(
            action="craft_ritual",
            data=ritual,
            forge_intel={
                "sentinel": {
                    "score": score_result.total_score,
                    "grade": score_result.grade,
                    "suggestions": score_result.suggestions,
                },
            },
            amplify_result=amp_result,
            processing_time_ms=elapsed,
            summary=summary,
        )

    def craft_meditation(self, intention: str, difficulty: str = "beginner") -> GrimoireResult:
        """Generate a guided meditation."""
        start = time.time()
        meditation = self.smith.craft_meditation(intention, difficulty)
        elapsed = (time.time() - start) * 1000

        return GrimoireResult(
            action="craft_meditation",
            data=meditation,
            processing_time_ms=elapsed,
            summary=[f"Generated: {meditation.title}"],
        )

    def daily_practice(self, dt: datetime.date | None = None) -> DailyPractice:
        """Get a personalized daily practice suggestion."""
        energy = self.oracle.get_current_energy(
            datetime.datetime.combine(dt, datetime.time()) if dt else None
        )
        practice = self.smith.generate_daily_practice(
            dt=dt,
            moon_phase=energy.phase_name,
            day_ruler=self._get_day_ruler_name(dt or datetime.date.today()),
        )
        return practice

    def weekly_forecast(self, start_date: datetime.date | None = None) -> WeeklyForecast:
        """Get a 7-day magical calendar."""
        return self.oracle.get_weekly_forecast(start_date)

    def log_practice(self, entry: PracticeEntry) -> dict:
        """Log a practice session and check for milestones."""
        practice_id = self.codex.log_practice(entry)
        milestones = self.codex._check_milestones(entry.practice_type)
        return {
            "practice_id": practice_id,
            "milestones": milestones,
            "message": "Practice logged! " + (
                f"Milestone achieved: {milestones[0]}" if milestones
                else get_encouragement()
            ),
        }

    def log_tarot(self, reading: TarotReading) -> dict:
        """Log a tarot reading."""
        reading_id = self.codex.log_tarot_reading(reading)
        return {"reading_id": reading_id, "message": "Tarot reading logged."}

    def my_journey(self) -> JourneyInsight:
        """Get practice insights and growth summary."""
        return self.codex.get_growth_summary()

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_day_ruler_name(self, dt: datetime.date) -> str:
        """Get the ruling planet name for a given date."""
        weekday = dt.weekday()  # 0=Monday
        return get_day_ruler(weekday)

    # ── Query handlers ─────────────────────────────────────────────────

    def _handle_spell_request(self, query: str, enhanced: EnhancedQuery) -> dict:
        intention = self.enhancer._extract_intention(query)
        scout_result = self.scout.analyze(intention or query)
        correspondences = scout_result.correspondences

        parts = [
            f"{get_opening(self.oracle.get_current_energy().phase_name)} here's what I recommend for your {scout_result.category.value} working:",
            f"**Correspondences:**",
        ]
        for cat, items in correspondences.items():
            if items:
                parts.append(f"- **{cat.title()}:** {', '.join(items[:5])}")

        parts.append(f"\n**Quick Start:** {scout_result.quick_start}")

        if scout_result.suggestions:
            parts.append("**Suggestions:**")
            for s in scout_result.suggestions[:3]:
                parts.append(f"- {s}")

        parts.append(get_closing())

        return {
            "response_parts": parts,
            "correspondences": correspondences,
            "suggestions": scout_result.suggestions,
        }

    def _handle_herb_crystal_query(self, query: str, enhanced: EnhancedQuery) -> dict:
        from grimoire.knowledge.correspondences import get_herb, get_crystal

        subjects = self.enhancer._extract_subjects(query)
        parts = []
        correspondences = {"herbs": [], "crystals": []}

        for subject in subjects[:3]:
            herb = get_herb(subject)
            crystal = get_crystal(subject)
            if herb:
                correspondences["herbs"].append(herb["name"])
                parts.append(f"**{herb['name']}** — {', '.join(herb['magical_properties'][:5])}")
                parts.append(f"  Element: {herb['element']} | Planet: {herb['planet']}")
                if herb.get("safety_notes"):
                    parts.append(f"  Safety: {herb['safety_notes'][0]}")
                if herb.get("beginner_tip"):
                    parts.append(f"  Tip: {herb['beginner_tip']}")
            elif crystal:
                correspondences["crystals"].append(crystal["name"])
                parts.append(f"**{crystal['name']}** — {', '.join(crystal['magical_properties'][:5])}")
                parts.append(f"  Element: {crystal['element']} | Chakra: {crystal.get('chakra', 'varies')}")
                if crystal.get("beginner_tip"):
                    parts.append(f"  Tip: {crystal['beginner_tip']}")

        if not parts:
            intention = self.enhancer._extract_intention(query)
            from grimoire.knowledge.correspondences import get_correspondences_for_intention
            corr = get_correspondences_for_intention(intention or query)
            parts.append(f"For **{intention or query}**, I recommend:")
            for cat, items in corr.items():
                if items:
                    parts.append(f"- **{cat.title()}:** {', '.join(items[:5])}")
            correspondences = corr

        parts.append(f"\n{get_closing()}")
        return {
            "response_parts": parts,
            "correspondences": correspondences,
            "suggestions": ["Try combining complementary herbs and crystals", "Always research safety before using herbs topically"],
        }

    def _handle_moon_query(self, query: str, enhanced: EnhancedQuery) -> dict:
        energy = self.oracle.get_current_energy()
        guidance = self.oracle.get_daily_guidance()

        parts = [
            f"**Current Moon:** {energy.phase_name}",
            f"**Energy:** {energy.magical_energy}",
            f"**Best For:** {', '.join(energy.best_for[:5])}",
            f"**Avoid:** {', '.join(energy.avoid[:3])}",
            "",
            guidance,
        ]

        return {
            "response_parts": parts,
            "suggestions": [
                f"This {energy.phase_name} is ideal for {energy.best_for[0] if energy.best_for else 'reflection'}",
                "Track your moon observations in a journal",
            ],
        }

    def _handle_sabbat_query(self, query: str, enhanced: EnhancedQuery) -> dict:
        from grimoire.knowledge.wheel_of_year import get_next_sabbat, get_sabbat, SABBATS

        # Check if specific sabbat mentioned
        query_lower = query.lower()
        target_sabbat = None
        for name in SABBATS:
            if name in query_lower:
                target_sabbat = get_sabbat(name)
                break

        if not target_sabbat:
            sab_name, target_sabbat, days = get_next_sabbat(
                datetime.date.today().month, datetime.date.today().day
            )
            parts = [f"**Upcoming Sabbat:** {target_sabbat['name']} (in {days} days)"]
        else:
            parts = [f"**{target_sabbat['name']}** ({target_sabbat.get('pronunciation', '')})"]

        parts.append(f"**Meaning:** {target_sabbat['meaning']}")
        parts.append(f"**Themes:** {', '.join(target_sabbat['themes'][:5])}")

        corr = target_sabbat.get("correspondences", {})
        if corr:
            parts.append("**Correspondences:**")
            for cat in ["colors", "herbs", "crystals", "foods"]:
                if cat in corr:
                    parts.append(f"  - {cat.title()}: {', '.join(corr[cat][:5])}")

        if target_sabbat.get("rituals"):
            parts.append("**Suggested Rituals:**")
            for r in target_sabbat["rituals"][:3]:
                parts.append(f"  - {r}")

        parts.append(f"\n{get_closing()}")

        return {
            "response_parts": parts,
            "correspondences": corr,
            "suggestions": target_sabbat.get("journal_prompts", [])[:3],
        }

    def _handle_divination_query(self, query: str, enhanced: EnhancedQuery) -> dict:
        energy = self.oracle.get_current_energy()
        parts = [
            f"**Divination Guidance:**",
            f"The current {energy.phase_name} {'supports' if 'divination' in energy.best_for else 'is neutral for'} divinatory work.",
            "",
            "**Suggested Approaches:**",
            "- **Tarot:** Draw a single card for daily guidance, or use a three-card spread",
            "- **Scrying:** Use a dark mirror or bowl of water during the waning/dark moon",
            "- **Pendulum:** Best for yes/no questions; cleanse before each session",
            "- **Bibliomancy:** Open a meaningful book to a random page for guidance",
        ]

        if "tarot" in query.lower():
            from grimoire.knowledge.tarot import draw_cards, SPREADS
            cards = draw_cards(1)
            card = cards[0]
            parts.append(f"\n**Your Card of the Moment:** {card['name']} ({card['orientation']})")
            meaning_key = f"{card['orientation']}_meaning" if f"{card['orientation']}_meaning" in card else "keywords_upright"
            if card["orientation"] == "upright" and "keywords_upright" in card:
                parts.append(f"Keywords: {', '.join(card['keywords_upright'][:4])}")
            elif card["orientation"] == "reversed" and "keywords_reversed" in card:
                parts.append(f"Keywords: {', '.join(card['keywords_reversed'][:4])}")

        parts.append(f"\n{get_closing()}")
        return {
            "response_parts": parts,
            "suggestions": ["Keep a divination journal", "Cleanse your tools regularly", "Trust your first impression"],
        }

    def _handle_meditation_query(self, query: str, enhanced: EnhancedQuery) -> dict:
        intention = self.enhancer._extract_intention(query) or "grounding"
        meditation = self.smith.craft_meditation(intention)

        parts = [
            f"**{meditation.title}**",
            f"Duration: {meditation.duration_minutes} minutes | Difficulty: {meditation.difficulty}",
            "",
            "**Preparation:**",
        ]
        for p in meditation.preparation:
            parts.append(f"- {p}")

        parts.append(f"\n**Grounding:** {meditation.grounding}")
        parts.append("\n**Journey:**")
        for step in meditation.body[:5]:
            parts.append(f"- {step}")

        parts.append(f"\n**Peak:** {meditation.peak_experience}")
        parts.append(f"\n{get_closing()}")

        return {
            "response_parts": parts,
            "suggestions": meditation.journal_prompts[:3],
        }

    def _handle_shadow_work_query(self, query: str, enhanced: EnhancedQuery) -> dict:
        from grimoire.knowledge.journal_prompts import SHADOW_WORK_PROMPTS
        import random

        prompts = random.sample(SHADOW_WORK_PROMPTS, min(3, len(SHADOW_WORK_PROMPTS)))

        parts = [
            "**Shadow Work Guidance**",
            "",
            "Shadow work is the practice of exploring the parts of yourself that you've hidden, "
            "repressed, or denied. It's deep, transformative work that requires self-compassion.",
            "",
            "**Important:** Go at your own pace. Shadow work is not a race. If emotions become "
            "overwhelming, pause and ground yourself. This practice does not replace professional "
            "mental health support.",
            "",
            "**Journal Prompts for Today:**",
        ]
        for p in prompts:
            parts.append(f"- {p}")

        parts.append("\n**Supportive Correspondences:**")
        parts.append("- Crystals: obsidian, black tourmaline, smoky quartz, apache tear")
        parts.append("- Herbs: mugwort, wormwood (for dreaming/insight, not ingestion)")
        parts.append("- Colors: black (protection), deep purple (transformation)")

        parts.append(f"\n{get_encouragement()}")

        return {
            "response_parts": parts,
            "suggestions": prompts,
        }

    def _handle_tarot_query(self, query: str, enhanced: EnhancedQuery) -> dict:
        from grimoire.knowledge.tarot import draw_cards, SPREADS, get_card

        parts = ["**Tarot Guidance**", ""]

        # Check if asking about a specific card
        card_data = None
        query_lower = query.lower()
        for card_name_check in ["fool", "magician", "priestess", "empress", "emperor",
                                 "hierophant", "lovers", "chariot", "strength", "hermit",
                                 "wheel", "justice", "hanged", "death", "temperance",
                                 "devil", "tower", "star", "moon", "sun", "judgement", "world"]:
            if card_name_check in query_lower:
                card_data = get_card(card_name_check)
                break

        if card_data:
            parts.append(f"**{card_data['name']}**")
            if "keywords_upright" in card_data:
                parts.append(f"Upright: {', '.join(card_data['keywords_upright'][:5])}")
            if "keywords_reversed" in card_data:
                parts.append(f"Reversed: {', '.join(card_data['keywords_reversed'][:5])}")
            if "upright_meaning" in card_data:
                parts.append(f"\n{card_data['upright_meaning']}")
            if "advice" in card_data:
                parts.append(f"\n**Advice:** {card_data['advice']}")
        else:
            # Draw a card
            cards = draw_cards(3)
            parts.append("**Three-Card Draw:**")
            positions = ["Past", "Present", "Future"]
            for i, card in enumerate(cards):
                kw_key = f"keywords_{card['orientation']}"
                keywords = card.get(kw_key, card.get("keywords_upright", []))
                parts.append(f"- **{positions[i]}:** {card['name']} ({card['orientation']}) — {', '.join(keywords[:3])}")

        parts.append(f"\n{get_closing()}")
        return {
            "response_parts": parts,
            "suggestions": ["Record this reading in your tarot journal", "Revisit this reading in a week"],
        }

    def _handle_general_query(self, query: str, enhanced: EnhancedQuery) -> dict:
        energy = self.oracle.get_current_energy()
        intention = self.enhancer._extract_intention(query)

        parts = [
            f"{get_opening(energy.phase_name)}",
            "",
        ]

        if intention:
            scout_result = self.scout.analyze(intention)
            parts.append(f"For your interest in **{scout_result.category.value}**, here's a starting point:")
            parts.append(f"\n{scout_result.quick_start}")
            if scout_result.suggestions:
                parts.append("\n**Next Steps:**")
                for s in scout_result.suggestions[:3]:
                    parts.append(f"- {s}")
        else:
            parts.append("I'm here to help with your witchcraft practice. You can ask me about:")
            parts.append("- **Spells & Rituals** — crafting, timing, and correspondences")
            parts.append("- **Moon Magick** — current phase, best timing, lunar calendar")
            parts.append("- **Herbs & Crystals** — magical properties, safety, pairings")
            parts.append("- **Tarot** — card meanings, spreads, interpretation")
            parts.append("- **Sabbats** — celebrations, correspondences, rituals")
            parts.append("- **Meditation** — guided journeys, grounding, chakra work")
            parts.append("- **Shadow Work** — journal prompts, guidance")
            parts.append("- **Daily Practice** — personalized suggestions")

        parts.append(f"\n{get_closing()}")
        return {
            "response_parts": parts,
            "suggestions": [
                "Try asking 'What's the energy today?'",
                "Ask me to craft a spell for any intention",
                "Request a weekly forecast",
            ],
        }
