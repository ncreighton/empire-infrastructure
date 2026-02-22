"""
RitualSentinel — Scoring and Enhancement Module for Ritual Plans.

Part of the Grimoire Intelligence System's FORGE layer. Follows the Sentinel
pattern: score a ritual/spell plan across 6 quality criteria (total 100 points),
provide actionable feedback, and auto-enhance if the score falls below threshold.

Scoring Criteria (100 points total):
    intention_clarity          (0-20)  How clear and specific is the intention?
    correspondence_alignment   (0-20)  Do materials match the intention?
    timing_awareness           (0-15)  Is lunar/planetary timing considered?
    structural_completeness    (0-15)  Opening, body, closing, aftercare present?
    safety_ethics              (0-15)  Safety warnings and ethical notes included?
    personalization            (0-15)  Room for personal adaptation?
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import field
from typing import Any

from grimoire.models import RitualPlan, RitualScore
from grimoire.knowledge.correspondences import (
    HERBS,
    CRYSTALS,
    COLORS,
    INTENTION_MAP,
    get_correspondences_for_intention,
)

logger = logging.getLogger("grimoire.forge.ritual_sentinel")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORING_CRITERIA: dict[str, dict[str, Any]] = {
    "intention_clarity": {
        "weight": 20,
        "description": "How clear and specific is the intention?",
    },
    "correspondence_alignment": {
        "weight": 20,
        "description": "Do materials/correspondences match the intention?",
    },
    "timing_awareness": {
        "weight": 15,
        "description": "Is lunar/planetary timing considered?",
    },
    "structural_completeness": {
        "weight": 15,
        "description": "Does it have opening, body, closing, aftercare?",
    },
    "safety_ethics": {
        "weight": 15,
        "description": "Are safety warnings and ethical considerations included?",
    },
    "personalization": {
        "weight": 15,
        "description": "Is there room for personal adaptation?",
    },
}

# Words that signal vague, non-specific intentions
_VAGUE_WORDS = {
    "stuff", "things", "something", "whatever", "anything", "some",
    "maybe", "kind of", "sort of", "idk", "i guess", "general",
}

# Intention categories we can detect from plan text
_INTENTION_KEYWORDS: dict[str, list[str]] = {
    "protection": ["protect", "shield", "ward", "guard", "defense", "defend", "safe"],
    "love": ["love", "romance", "attract", "relationship", "heart", "partner", "soulmate"],
    "prosperity": ["prosper", "wealth", "money", "abundance", "rich", "financial", "fortune"],
    "healing": ["heal", "health", "recovery", "cure", "wellness", "restore", "mend"],
    "divination": ["divin", "scry", "psychic", "vision", "prophetic", "foresight", "tarot"],
    "banishing": ["banish", "remove", "rid", "expel", "exile", "release", "let go"],
    "cleansing": ["cleans", "purif", "clear", "smudge", "wash", "renew", "refresh"],
    "creativity": ["creativ", "inspir", "muse", "art", "imaginat", "innovat"],
    "wisdom": ["wisdom", "knowledge", "learn", "understand", "enlighten", "insight"],
    "confidence": ["confiden", "self-esteem", "courage", "bold", "empower", "asserti"],
    "communication": ["communicat", "speak", "voice", "express", "eloquen", "truth-tell"],
    "grounding": ["ground", "center", "root", "stabiliz", "earth", "anchor"],
    "transformation": ["transform", "change", "metamorphos", "evolv", "transmut", "rebirth"],
    "peace": ["peace", "calm", "serenity", "tranquil", "harmoni", "sooth", "stillness"],
    "courage": ["courag", "brave", "fearless", "bold", "valor", "warrior", "strength"],
}

# Moon phases and their magical alignment
_MOON_ALIGNMENT: dict[str, list[str]] = {
    "new_moon": ["new beginnings", "intention setting", "planting seeds", "divination"],
    "new moon": ["new beginnings", "intention setting", "planting seeds", "divination"],
    "waxing_crescent": ["growth", "attraction", "building", "courage"],
    "waxing crescent": ["growth", "attraction", "building", "courage"],
    "first_quarter": ["action", "determination", "decision", "challenges"],
    "first quarter": ["action", "determination", "decision", "challenges"],
    "waxing_gibbous": ["refinement", "patience", "adjustment", "nurturing"],
    "waxing gibbous": ["refinement", "patience", "adjustment", "nurturing"],
    "waxing moon": ["growth", "attraction", "building", "prosperity", "love"],
    "waxing": ["growth", "attraction", "building", "prosperity", "love"],
    "full_moon": ["power", "completion", "divination", "healing", "love", "prosperity"],
    "full moon": ["power", "completion", "divination", "healing", "love", "prosperity"],
    "waning_gibbous": ["gratitude", "sharing", "teaching", "introspection"],
    "waning gibbous": ["gratitude", "sharing", "teaching", "introspection"],
    "last_quarter": ["release", "forgiveness", "letting go", "banishing"],
    "last quarter": ["release", "forgiveness", "letting go", "banishing"],
    "waning_crescent": ["rest", "surrender", "healing", "reflection"],
    "waning crescent": ["rest", "surrender", "healing", "reflection"],
    "waning moon": ["banishing", "releasing", "cleansing", "protection"],
    "waning": ["banishing", "releasing", "cleansing", "protection"],
    "dark_moon": ["shadow work", "banishing", "deep divination", "rest"],
    "dark moon": ["shadow work", "banishing", "deep divination", "rest"],
}

# Materials that involve fire or burning
_FIRE_MATERIALS = {
    "candle", "candles", "incense", "charcoal", "fire", "flame", "burn",
    "cauldron", "smudge", "smoke", "match", "lighter", "fireproof",
}

# Materials that involve potentially harmful herbs
_SENSITIVE_HERBS = {
    "mugwort", "wormwood", "belladonna", "hemlock", "henbane", "datura",
    "mandrake", "comfrey", "st. john's wort", "valerian",
}


# ---------------------------------------------------------------------------
# RitualSentinel
# ---------------------------------------------------------------------------


class RitualSentinel:
    """Scores ritual/spell plans on 6 criteria (100 points) and auto-enhances.

    Usage::

        sentinel = RitualSentinel()
        score = sentinel.score(plan)
        print(f"{score.grade}: {score.total_score}/100")

        # Or score-and-enhance in one call:
        score, enhanced_plan = sentinel.score_and_enhance(plan, threshold=85.0)
    """

    # ── Public API ────────────────────────────────────────────────────────

    def score(self, plan: RitualPlan) -> RitualScore:
        """Score a RitualPlan across all 6 criteria.

        Args:
            plan: The ritual plan to evaluate.

        Returns:
            A RitualScore with total, grade, per-criterion scores,
            suggestions, and enhancements.
        """
        all_suggestions: list[str] = []
        all_enhancements: list[str] = []

        # Score each criterion
        ic_score, ic_suggestions = self._score_intention_clarity(plan)
        ca_score, ca_suggestions = self._score_correspondence_alignment(plan)
        ta_score, ta_suggestions = self._score_timing_awareness(plan)
        sc_score, sc_suggestions = self._score_structural_completeness(plan)
        se_score, se_suggestions = self._score_safety_ethics(plan)
        pe_score, pe_suggestions = self._score_personalization(plan)

        all_suggestions.extend(ic_suggestions)
        all_suggestions.extend(ca_suggestions)
        all_suggestions.extend(ta_suggestions)
        all_suggestions.extend(sc_suggestions)
        all_suggestions.extend(se_suggestions)
        all_suggestions.extend(pe_suggestions)

        total = ic_score + ca_score + ta_score + sc_score + se_score + pe_score
        total = round(min(total, 100.0), 1)

        # Build enhancement notes for weak areas
        if ic_score < SCORING_CRITERIA["intention_clarity"]["weight"] * 0.7:
            all_enhancements.append("Rewrite intention with more specificity")
        if ca_score < SCORING_CRITERIA["correspondence_alignment"]["weight"] * 0.7:
            all_enhancements.append("Add matching correspondences for the intention")
        if ta_score < SCORING_CRITERIA["timing_awareness"]["weight"] * 0.7:
            all_enhancements.append("Add lunar/planetary timing guidance")
        if sc_score < SCORING_CRITERIA["structural_completeness"]["weight"] * 0.7:
            all_enhancements.append("Add missing structural sections")
        if se_score < SCORING_CRITERIA["safety_ethics"]["weight"] * 0.7:
            all_enhancements.append("Inject safety warnings and ethical notes")
        if pe_score < SCORING_CRITERIA["personalization"]["weight"] * 0.7:
            all_enhancements.append("Add personalization options and alternatives")

        grade = self.grade(total)

        result = RitualScore(
            total_score=total,
            grade=grade,
            intention_clarity=round(ic_score, 1),
            correspondence_alignment=round(ca_score, 1),
            timing_awareness=round(ta_score, 1),
            structural_completeness=round(sc_score, 1),
            safety_ethics=round(se_score, 1),
            personalization=round(pe_score, 1),
            suggestions=all_suggestions,
            enhancements=all_enhancements,
        )

        logger.info(
            "RitualSentinel scored '%s': %s (%s) — %d suggestions, %d enhancements",
            plan.title, total, grade, len(all_suggestions), len(all_enhancements),
        )
        return result

    def grade(self, score: float) -> str:
        """Map a numeric score (0-100) to a letter grade.

        Args:
            score: The total score from 0 to 100.

        Returns:
            Grade string: S, A+, A, B+, B, C+, C, or Needs Work.
        """
        if score >= 95:
            return "S"
        elif score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "C+"
        elif score >= 65:
            return "C"
        else:
            return "Needs Work"

    def auto_enhance(self, plan: RitualPlan, score_result: RitualScore) -> RitualPlan:
        """Automatically enhance weak areas of a scored ritual plan.

        Takes a plan and its score, then patches every criterion that scored
        below 70% of its maximum weight. Returns a new RitualPlan with
        enhancements applied (the original is not mutated).

        Args:
            plan: The original ritual plan.
            score_result: The scoring result from :meth:`score`.

        Returns:
            A new RitualPlan with enhancements applied.
        """
        enhanced = copy.deepcopy(plan)
        detected_category = self._detect_category(enhanced)

        # -- Intention Clarity --
        max_ic = SCORING_CRITERIA["intention_clarity"]["weight"]
        if score_result.intention_clarity < max_ic * 0.7:
            enhanced = self._enhance_intention_clarity(enhanced, detected_category)

        # -- Correspondence Alignment --
        max_ca = SCORING_CRITERIA["correspondence_alignment"]["weight"]
        if score_result.correspondence_alignment < max_ca * 0.7:
            enhanced = self._enhance_correspondence_alignment(enhanced, detected_category)

        # -- Timing Awareness --
        max_ta = SCORING_CRITERIA["timing_awareness"]["weight"]
        if score_result.timing_awareness < max_ta * 0.7:
            enhanced = self._enhance_timing_awareness(enhanced, detected_category)

        # -- Structural Completeness --
        max_sc = SCORING_CRITERIA["structural_completeness"]["weight"]
        if score_result.structural_completeness < max_sc * 0.7:
            enhanced = self._enhance_structural_completeness(enhanced)

        # -- Safety & Ethics --
        max_se = SCORING_CRITERIA["safety_ethics"]["weight"]
        if score_result.safety_ethics < max_se * 0.7:
            enhanced = self._enhance_safety_ethics(enhanced)

        # -- Personalization --
        max_pe = SCORING_CRITERIA["personalization"]["weight"]
        if score_result.personalization < max_pe * 0.7:
            enhanced = self._enhance_personalization(enhanced)

        logger.info("RitualSentinel auto-enhanced '%s'", plan.title)
        return enhanced

    def score_and_enhance(
        self,
        plan: RitualPlan,
        threshold: float = 85.0,
    ) -> tuple[RitualScore, RitualPlan]:
        """Score a plan and, if below threshold, auto-enhance and re-score.

        This is the main convenience entry point. It scores the plan once;
        if the total falls below *threshold*, it enhances the plan and
        returns the new score alongside the enhanced plan. If the plan
        already meets the threshold, the original plan is returned.

        Args:
            plan: The ritual plan to evaluate.
            threshold: Minimum acceptable score (default 85.0).

        Returns:
            A tuple of (final_score, final_plan). If no enhancement was
            needed, final_plan is the original plan.
        """
        initial_score = self.score(plan)

        if initial_score.total_score >= threshold:
            logger.info(
                "Plan '%s' scored %.1f (>= %.1f threshold) — no enhancement needed",
                plan.title, initial_score.total_score, threshold,
            )
            return initial_score, plan

        logger.info(
            "Plan '%s' scored %.1f (< %.1f threshold) — auto-enhancing",
            plan.title, initial_score.total_score, threshold,
        )
        enhanced_plan = self.auto_enhance(plan, initial_score)
        final_score = self.score(enhanced_plan)

        return final_score, enhanced_plan

    # ── Private Scoring Methods ───────────────────────────────────────────

    def _score_intention_clarity(self, plan: RitualPlan) -> tuple[float, list[str]]:
        """Score intention clarity (0-20).

        Full points for: specific intention (>10 chars), a clear goal
        stated in present tense, and a detectable category. Deductions
        for vague language, missing category, or overly broad framing.

        Returns:
            (score, suggestions) tuple.
        """
        max_score = 20.0
        score = 0.0
        suggestions: list[str] = []
        intention = plan.intention.strip()

        # --- Length / Specificity (0-7) ---
        if len(intention) > 50:
            score += 7.0
        elif len(intention) > 30:
            score += 5.0
        elif len(intention) > 10:
            score += 3.0
        else:
            suggestions.append(
                "Intention is very short. Expand it to describe exactly what "
                "you want to manifest, e.g. 'I draw protective energy around "
                "my home, shielding my family from negativity.'"
            )

        # --- Vagueness check (deduction up to -4) ---
        intention_lower = intention.lower()
        vague_count = sum(1 for w in _VAGUE_WORDS if w in intention_lower)
        if vague_count > 0:
            deduction = min(vague_count * 2.0, 4.0)
            score = max(score - deduction, 0.0)
            suggestions.append(
                "Remove vague words like 'stuff', 'things', or 'something'. "
                "Replace with concrete, specific language."
            )

        # --- Clear goal keyword (0-5) ---
        goal_patterns = [
            r"\b(i (?:am|have|draw|attract|release|banish|create|invoke|invoke|protect))",
            r"\b(my (?:home|family|heart|mind|spirit|energy|path))",
            r"\b(bring|manifest|cultivate|nurture|strengthen|restore)\b",
        ]
        goal_found = any(re.search(p, intention_lower) for p in goal_patterns)
        if goal_found:
            score += 5.0
        else:
            suggestions.append(
                "Use active, present-tense phrasing for your intention, e.g. "
                "'I am protected' or 'I draw abundance into my life.'"
            )

        # --- Category detected (0-4) ---
        detected = self._detect_category(plan)
        if detected:
            score += 4.0
        elif plan.category:
            score += 4.0
        else:
            suggestions.append(
                "Add a category to your plan (e.g. protection, love, prosperity, "
                "healing) so correspondences can be properly aligned."
            )

        # --- Present tense bonus (0-4) ---
        present_patterns = [
            r"\b(i am|i have|i draw|i attract|i release|i create)\b",
            r"\b(is|are|flows|grows|shines|blooms|strengthens)\b",
        ]
        present_found = any(re.search(p, intention_lower) for p in present_patterns)
        if present_found:
            score += 4.0
        elif len(intention) > 10:
            # Partial credit if intention exists but isn't present tense
            score += 1.5
            suggestions.append(
                "Consider rephrasing in present tense to state the intention "
                "as already manifesting, e.g. 'My home is shielded' rather "
                "than 'I want my home to be shielded.'"
            )

        return min(score, max_score), suggestions

    def _score_correspondence_alignment(self, plan: RitualPlan) -> tuple[float, list[str]]:
        """Score correspondence alignment (0-20).

        Full points when materials, herbs, crystals, and colors match the
        detected intention category. Deductions for mismatched or missing
        correspondences.

        Returns:
            (score, suggestions) tuple.
        """
        max_score = 20.0
        score = 0.0
        suggestions: list[str] = []

        detected = self._detect_category(plan)
        if not detected:
            # Cannot assess alignment without a category
            score += 5.0  # Baseline: we can't deduct for what we can't check
            suggestions.append(
                "No clear intention category detected. Add a category to "
                "enable correspondence alignment checking."
            )
            return min(score, max_score), suggestions

        # Get the ideal correspondences for this category
        ideal = INTENTION_MAP.get(detected, {})
        ideal_herbs = set(h.lower() for h in ideal.get("herbs", []))
        ideal_crystals = set(c.lower() for c in ideal.get("crystals", []))
        ideal_colors = set(c.lower() for c in ideal.get("colors", []))

        # Collect what the plan actually uses
        plan_herbs: set[str] = set()
        plan_crystals: set[str] = set()
        plan_colors: set[str] = set()

        # From correspondences_used dict
        corr = plan.correspondences_used or {}
        for h in corr.get("herbs", []):
            plan_herbs.add(h.lower().strip())
        for c in corr.get("crystals", []):
            plan_crystals.add(c.lower().strip())
        for c in corr.get("colors", []):
            plan_colors.add(c.lower().strip())

        # Also scan materials list for known herbs, crystals, and colors
        all_known_herbs = set(HERBS.keys())
        all_known_crystals = set(CRYSTALS.keys())
        all_known_colors = set(COLORS.keys())

        for mat in plan.materials:
            mat_lower = mat.lower().strip()
            for herb_name in all_known_herbs:
                if herb_name in mat_lower:
                    plan_herbs.add(herb_name)
            for crystal_name in all_known_crystals:
                if crystal_name in mat_lower:
                    plan_crystals.add(crystal_name)
            for color_name in all_known_colors:
                if color_name in mat_lower:
                    plan_colors.add(color_name)

        # --- Herb alignment (0-7) ---
        if plan_herbs:
            matching = plan_herbs & ideal_herbs
            if matching:
                ratio = len(matching) / max(len(plan_herbs), 1)
                score += 7.0 * ratio
            else:
                score += 1.0  # Has herbs, but none match
                mismatched = plan_herbs - ideal_herbs
                if mismatched:
                    suggestions.append(
                        f"Herbs {', '.join(sorted(mismatched))} don't align with "
                        f"'{detected}' intention. Consider: {', '.join(sorted(list(ideal_herbs)[:4]))}."
                    )
        else:
            suggestions.append(
                f"No herbs detected. For '{detected}' work, consider: "
                f"{', '.join(sorted(list(ideal_herbs)[:4]))}."
            )

        # --- Crystal alignment (0-7) ---
        if plan_crystals:
            matching = plan_crystals & ideal_crystals
            if matching:
                ratio = len(matching) / max(len(plan_crystals), 1)
                score += 7.0 * ratio
            else:
                score += 1.0
                mismatched = plan_crystals - ideal_crystals
                if mismatched:
                    suggestions.append(
                        f"Crystals {', '.join(sorted(mismatched))} don't align with "
                        f"'{detected}' intention. Consider: {', '.join(sorted(list(ideal_crystals)[:4]))}."
                    )
        else:
            suggestions.append(
                f"No crystals detected. For '{detected}' work, consider: "
                f"{', '.join(sorted(list(ideal_crystals)[:4]))}."
            )

        # --- Color alignment (0-6) ---
        if plan_colors:
            matching = plan_colors & ideal_colors
            if matching:
                ratio = len(matching) / max(len(plan_colors), 1)
                score += 6.0 * ratio
            else:
                score += 1.0
                suggestions.append(
                    f"Colors used don't align with '{detected}'. "
                    f"Consider: {', '.join(sorted(ideal_colors))}."
                )
        else:
            suggestions.append(
                f"No color correspondences detected. For '{detected}' work, "
                f"use: {', '.join(sorted(ideal_colors))}."
            )

        return min(score, max_score), suggestions

    def _score_timing_awareness(self, plan: RitualPlan) -> tuple[float, list[str]]:
        """Score timing awareness (0-15).

        Full points when moon_phase is specified and appropriate for the
        intention, and when planetary day/hour is mentioned. Partial credit
        for mentioning timing without checking alignment.

        Returns:
            (score, suggestions) tuple.
        """
        max_score = 15.0
        score = 0.0
        suggestions: list[str] = []
        detected = self._detect_category(plan)

        moon = (plan.moon_phase or "").strip().lower()
        timing_text = (plan.timing or "").strip().lower()
        combined = f"{moon} {timing_text}"

        # --- Moon phase specified (0-6) ---
        if moon:
            score += 3.0  # Credit for specifying at all

            # Check alignment with intention
            if detected:
                ideal = INTENTION_MAP.get(detected, {})
                ideal_moon = (ideal.get("moon_phase", "") or "").lower()

                # Check if the plan's moon phase aligns
                if ideal_moon and ideal_moon in moon:
                    score += 3.0
                elif ideal_moon:
                    # Not aligned, but still specified
                    suggestions.append(
                        f"Moon phase '{plan.moon_phase}' may not be optimal for "
                        f"'{detected}' work. Consider: {ideal_moon}."
                    )
                    score += 1.0
                else:
                    score += 2.0  # Category has no strong moon preference
        else:
            # Check if moon phase is mentioned in timing text
            moon_mentioned = any(
                phrase in combined
                for phrase in ["moon", "lunar", "new moon", "full moon", "waxing", "waning"]
            )
            if moon_mentioned:
                score += 2.0
                suggestions.append(
                    "Moon phase is mentioned in timing text but not set as a "
                    "field. Set moon_phase explicitly for better alignment scoring."
                )
            else:
                suggestions.append(
                    "No moon phase specified. Consider the lunar cycle: waxing "
                    "for growth/attraction, full for power/divination, waning "
                    "for release/banishing."
                )

        # --- Planetary day mentioned (0-5) ---
        day_keywords = [
            "sunday", "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday",
        ]
        day_found = any(d in combined for d in day_keywords)
        planetary_keywords = ["planetary", "hour", "ruler", "day of"]
        planetary_found = any(k in combined for k in planetary_keywords)

        if day_found and planetary_found:
            score += 5.0
        elif day_found:
            score += 3.0
            suggestions.append(
                "Good: a day is specified. Add planetary hour awareness for "
                "even stronger timing alignment."
            )
        elif planetary_found:
            score += 2.0
        else:
            if detected:
                ideal = INTENTION_MAP.get(detected, {})
                ideal_day = ideal.get("day", "")
                if ideal_day:
                    suggestions.append(
                        f"No day of the week specified. For '{detected}' work, "
                        f"{ideal_day.title()} (ruled by {ideal.get('planet', 'its planet')}) "
                        f"is ideal."
                    )
            else:
                suggestions.append(
                    "No planetary day specified. Each day of the week is ruled "
                    "by a planet with specific magical affinities."
                )

        # --- General timing notes (0-4) ---
        timing_signals = [
            "dawn", "dusk", "midnight", "noon", "sunrise", "sunset",
            "equinox", "solstice", "sabbat", "esbat", "seasonal",
        ]
        timing_found = sum(1 for t in timing_signals if t in combined)
        if timing_found >= 2:
            score += 4.0
        elif timing_found == 1:
            score += 2.0
        elif timing_text:
            score += 1.0  # Has some timing text, just not specific

        return min(score, max_score), suggestions

    def _score_structural_completeness(self, plan: RitualPlan) -> tuple[float, list[str]]:
        """Score structural completeness (0-15).

        Checks for: title (2), preparation steps (3), main steps with 3+
        entries (4), closing (3), aftercare (3). Deductions for missing
        sections.

        Returns:
            (score, suggestions) tuple.
        """
        max_score = 15.0
        score = 0.0
        suggestions: list[str] = []

        # --- Title (0-2) ---
        if plan.title and len(plan.title.strip()) > 3:
            score += 2.0
        elif plan.title:
            score += 1.0
            suggestions.append(
                "Title is very short. A descriptive title helps set the "
                "ritual's tone, e.g. 'Full Moon Abundance Ritual' rather "
                "than just 'Abundance'."
            )
        else:
            suggestions.append("Add a descriptive title to the ritual plan.")

        # --- Preparation steps (0-3) ---
        if plan.preparation and len(plan.preparation) >= 2:
            score += 3.0
        elif plan.preparation and len(plan.preparation) == 1:
            score += 1.5
            suggestions.append(
                "Only one preparation step. Add more: gather materials, "
                "cleanse the space, ground yourself, set up the altar."
            )
        else:
            suggestions.append(
                "No preparation steps found. Add steps such as: gather "
                "materials, cleanse the space, cast a circle (if desired), "
                "and ground/center yourself."
            )

        # --- Main steps with 3+ entries (0-4) ---
        if plan.steps and len(plan.steps) >= 5:
            score += 4.0
        elif plan.steps and len(plan.steps) >= 3:
            score += 3.0
        elif plan.steps and len(plan.steps) >= 1:
            score += 1.5
            suggestions.append(
                f"Only {len(plan.steps)} main step(s). A well-structured "
                f"ritual typically has at least 3-5 steps: invocation, "
                f"main working, energy raising, directing energy, and grounding."
            )
        else:
            suggestions.append(
                "No main steps found. Add the core ritual actions: invocation, "
                "the main magical working, energy direction, and grounding."
            )

        # --- Closing present (0-3) ---
        # Check steps, enrichments, or expansions for closing language
        has_closing = False
        closing_keywords = [
            "close", "closing", "release", "dismiss", "thank", "farewell",
            "so mote it be", "blessed be", "open the circle", "ground",
        ]
        all_text = " ".join(plan.steps + plan.aftercare).lower()
        if any(k in all_text for k in closing_keywords):
            has_closing = True
        # Also check if steps list ends with something that sounds like closing
        if plan.steps and len(plan.steps) >= 2:
            last_step = plan.steps[-1].lower()
            if any(k in last_step for k in closing_keywords):
                has_closing = True

        if has_closing:
            score += 3.0
        else:
            suggestions.append(
                "No closing detected. Always close your ritual: thank any "
                "entities invoked, ground excess energy, open the circle, "
                "and mark the end with words like 'So mote it be.'"
            )

        # --- Aftercare (0-3) ---
        if plan.aftercare and len(plan.aftercare) >= 2:
            score += 3.0
        elif plan.aftercare and len(plan.aftercare) == 1:
            score += 1.5
            suggestions.append(
                "Only one aftercare note. Add more: ground with food/drink, "
                "journal your experience, rest, note any signs or dreams."
            )
        else:
            suggestions.append(
                "No aftercare section. Add post-ritual care: eat grounding "
                "food, drink water, journal your experience, note signs and "
                "dreams over the following days."
            )

        return min(score, max_score), suggestions

    def _score_safety_ethics(self, plan: RitualPlan) -> tuple[float, list[str]]:
        """Score safety and ethics (0-15).

        Full points when safety_notes are present, fire/herb warnings
        included where relevant, ethical disclaimers present, and
        substitutions offered.

        Returns:
            (score, suggestions) tuple.
        """
        max_score = 15.0
        score = 0.0
        suggestions: list[str] = []

        safety_text = " ".join(plan.safety_notes).lower() if plan.safety_notes else ""
        all_materials = " ".join(plan.materials).lower()
        all_steps = " ".join(plan.steps).lower()
        combined_text = f"{all_materials} {all_steps} {safety_text}"

        # --- Safety notes present (0-4) ---
        if plan.safety_notes and len(plan.safety_notes) >= 2:
            score += 4.0
        elif plan.safety_notes and len(plan.safety_notes) == 1:
            score += 2.0
            suggestions.append(
                "Only one safety note. Consider adding warnings for fire "
                "safety, herb interactions, ventilation, and allergy awareness."
            )
        else:
            suggestions.append(
                "No safety notes found. Always include safety information, "
                "especially for fire, herbs, incense, and crystal handling."
            )

        # --- Fire safety where relevant (0-3) ---
        uses_fire = any(f in combined_text for f in _FIRE_MATERIALS)
        if uses_fire:
            fire_warnings = [
                "fire safe", "fireproof", "fire-safe", "never leave",
                "unattended", "extinguish", "fire safety", "heat-proof",
                "burn safely", "ventilat",
            ]
            has_fire_warning = any(w in safety_text for w in fire_warnings)
            if has_fire_warning:
                score += 3.0
            else:
                suggestions.append(
                    "This ritual uses fire/candles/incense but has no fire "
                    "safety warnings. Add: 'Never leave flames unattended. "
                    "Use a fire-safe surface. Ensure proper ventilation.'"
                )
        else:
            score += 3.0  # Not applicable, full marks

        # --- Herb safety where relevant (0-3) ---
        uses_sensitive_herb = any(h in combined_text for h in _SENSITIVE_HERBS)
        if uses_sensitive_herb:
            herb_warnings = [
                "toxic", "poison", "ingest", "pregnancy", "allergic",
                "allergy", "medication", "interact", "external only",
                "do not eat", "do not consume", "consult",
            ]
            has_herb_warning = any(w in safety_text for w in herb_warnings)
            if has_herb_warning:
                score += 3.0
            else:
                suggestions.append(
                    "Sensitive herbs are used but no herb safety warnings are "
                    "present. Add warnings about ingestion, pregnancy, "
                    "allergies, and medication interactions."
                )
        else:
            score += 3.0  # Not applicable, full marks

        # --- Ethical disclaimers (0-3) ---
        ethical_phrases = [
            "harm none", "free will", "consent", "ethical", "for the highest good",
            "with harm to none", "respect", "do no harm", "rede",
            "an it harm none", "and it harm none",
        ]
        has_ethics = any(e in combined_text for e in ethical_phrases)
        if has_ethics:
            score += 3.0
        else:
            suggestions.append(
                "No ethical disclaimer detected. Consider adding: 'With harm "
                "to none, for the highest good of all' or similar ethical "
                "framing to your intention or closing."
            )

        # --- Substitutions offered (0-2) ---
        sub_phrases = [
            "substitut", "alternative", "instead of", "if you don't have",
            "replace with", "or use", "you can also use", "swap",
        ]
        has_subs = any(s in combined_text for s in sub_phrases)
        if has_subs:
            score += 2.0
        else:
            suggestions.append(
                "No substitutions offered. Include alternatives for materials "
                "that may be hard to find or expensive."
            )

        return min(score, max_score), suggestions

    def _score_personalization(self, plan: RitualPlan) -> tuple[float, list[str]]:
        """Score personalization (0-15).

        Full points when difficulty is specified, alternatives offered,
        journal prompts included, and 'make it your own' language present.

        Returns:
            (score, suggestions) tuple.
        """
        max_score = 15.0
        score = 0.0
        suggestions: list[str] = []

        all_text_parts: list[str] = [plan.intention, plan.timing]
        all_text_parts.extend(plan.steps)
        all_text_parts.extend(plan.preparation)
        all_text_parts.extend(plan.aftercare)
        all_text_parts.extend(plan.safety_notes)
        combined = " ".join(all_text_parts).lower()

        # --- Difficulty specified (0-4) ---
        if plan.difficulty and plan.difficulty.strip():
            difficulty = plan.difficulty.strip().lower()
            valid_difficulties = {"beginner", "intermediate", "advanced"}
            if difficulty in valid_difficulties:
                score += 4.0
            else:
                score += 2.0
                suggestions.append(
                    f"Difficulty '{plan.difficulty}' is non-standard. "
                    f"Use 'beginner', 'intermediate', or 'advanced'."
                )
        else:
            suggestions.append(
                "No difficulty level specified. Set difficulty to help "
                "practitioners choose rituals appropriate for their experience."
            )

        # --- Alternatives / variations offered (0-4) ---
        variation_phrases = [
            "alternative", "variation", "adapt", "modify", "adjust",
            "simpler version", "advanced version", "you may also",
            "feel free to", "another option", "you could also",
            "if you prefer", "optional", "for a simpler",
        ]
        variation_count = sum(1 for v in variation_phrases if v in combined)
        if variation_count >= 3:
            score += 4.0
        elif variation_count >= 1:
            score += 2.0
        else:
            suggestions.append(
                "No alternatives or variations offered. Add notes like "
                "'For a simpler version...' or 'You may adapt this by...' "
                "to make the ritual accessible at multiple skill levels."
            )

        # --- Journal prompts or reflection (0-4) ---
        journal_phrases = [
            "journal", "reflect", "write down", "record",
            "ask yourself", "contemplate", "meditate on",
            "what did you feel", "what came up", "notice",
        ]
        journal_count = sum(1 for j in journal_phrases if j in combined)
        if journal_count >= 2:
            score += 4.0
        elif journal_count == 1:
            score += 2.0
        else:
            suggestions.append(
                "No journal prompts or reflection cues. Add prompts like: "
                "'After the ritual, journal about what you felt, saw, or "
                "sensed. What messages came through?'"
            )

        # --- "Make it your own" language (0-3) ---
        personal_phrases = [
            "make it your own", "personal", "your own words",
            "in your own way", "trust your intuition", "follow your instinct",
            "what feels right", "customize", "your unique",
            "there is no wrong way", "your practice",
        ]
        personal_count = sum(1 for p in personal_phrases if p in combined)
        if personal_count >= 2:
            score += 3.0
        elif personal_count == 1:
            score += 1.5
        else:
            suggestions.append(
                "No personalization language found. Encourage practitioners "
                "with phrases like 'Make it your own', 'Trust your intuition', "
                "or 'There is no wrong way to do this.'"
            )

        return min(score, max_score), suggestions

    # ── Private Enhancement Methods ───────────────────────────────────────

    def _enhance_intention_clarity(
        self, plan: RitualPlan, category: str
    ) -> RitualPlan:
        """Rewrite the intention to be more specific and present-tense."""
        intention = plan.intention.strip()

        if not category:
            category = "general magical"

        # Rebuild the intention with present-tense, specific language
        if len(intention) < 10:
            plan.intention = (
                f"I call upon the energy of {category} to create powerful, "
                f"positive change in my life. My will is clear, my intention "
                f"is focused, and I am aligned with my purpose."
            )
        else:
            # Enhance existing intention
            if not any(
                p in intention.lower()
                for p in ["i am", "i draw", "i have", "i create", "i attract"]
            ):
                plan.intention = (
                    f"I am aligned with the energy of {category}. {intention} "
                    f"My intention is clear, and I manifest this with focused will."
                )

        if not plan.category:
            plan.category = category

        return plan

    def _enhance_correspondence_alignment(
        self, plan: RitualPlan, category: str
    ) -> RitualPlan:
        """Inject matching correspondences from the knowledge base."""
        if not category:
            return plan

        ideal = INTENTION_MAP.get(category, {})
        if not ideal:
            return plan

        corr = plan.correspondences_used
        if not corr:
            corr = {}

        # Add herbs if missing
        if not corr.get("herbs"):
            corr["herbs"] = ideal.get("herbs", [])[:4]
        else:
            # Supplement with aligned herbs
            current = set(h.lower() for h in corr["herbs"])
            for herb in ideal.get("herbs", [])[:3]:
                if herb.lower() not in current:
                    corr["herbs"].append(herb)
                    break

        # Add crystals if missing
        if not corr.get("crystals"):
            corr["crystals"] = ideal.get("crystals", [])[:3]
        else:
            current = set(c.lower() for c in corr["crystals"])
            for crystal in ideal.get("crystals", [])[:3]:
                if crystal.lower() not in current:
                    corr["crystals"].append(crystal)
                    break

        # Add colors if missing
        if not corr.get("colors"):
            corr["colors"] = ideal.get("colors", [])[:3]

        plan.correspondences_used = corr

        # Also update materials list to include key correspondences
        existing_materials = set(m.lower() for m in plan.materials)
        for herb in corr.get("herbs", [])[:2]:
            if herb.lower() not in existing_materials:
                plan.materials.append(herb.title())
        for crystal in corr.get("crystals", [])[:2]:
            if crystal.lower() not in existing_materials:
                plan.materials.append(crystal.title())

        return plan

    def _enhance_timing_awareness(
        self, plan: RitualPlan, category: str
    ) -> RitualPlan:
        """Add timing notes including moon phase and planetary day."""
        ideal = INTENTION_MAP.get(category, {}) if category else {}

        # Set moon phase if missing
        if not plan.moon_phase and ideal.get("moon_phase"):
            plan.moon_phase = ideal["moon_phase"]

        # Build timing notes
        timing_parts: list[str] = []
        if plan.timing:
            timing_parts.append(plan.timing.strip())

        if ideal.get("day"):
            timing_parts.append(
                f"Best performed on {ideal['day'].title()}, "
                f"ruled by {ideal.get('planet', 'its planetary ruler').title()}."
            )

        if ideal.get("moon_phase"):
            timing_parts.append(
                f"Ideal moon phase: {ideal['moon_phase'].replace('_', ' ').title()}."
            )

        if not any("hour" in t.lower() for t in timing_parts):
            if ideal.get("planet"):
                timing_parts.append(
                    f"For maximum potency, work during the planetary hour of "
                    f"{ideal['planet'].title()}."
                )

        plan.timing = " ".join(timing_parts)
        return plan

    def _enhance_structural_completeness(self, plan: RitualPlan) -> RitualPlan:
        """Add missing structural sections (preparation, closing, aftercare)."""
        # Add preparation if missing
        if not plan.preparation:
            plan.preparation = [
                "Gather all materials and arrange them on your altar or workspace.",
                "Cleanse your space with smoke, sound, or visualization.",
                "Ground and center yourself with three deep breaths.",
            ]

        # Add closing step if steps don't have one
        if plan.steps:
            closing_keywords = [
                "close", "release", "dismiss", "thank", "farewell",
                "so mote it be", "blessed be",
            ]
            last_step_lower = plan.steps[-1].lower()
            has_closing = any(k in last_step_lower for k in closing_keywords)
            if not has_closing:
                plan.steps.append(
                    "Thank any energies, deities, or spirits you invoked. "
                    "Ground any excess energy by pressing your palms to the "
                    "earth or floor. Say: 'This circle is open but never "
                    "broken. So mote it be.'"
                )
        else:
            plan.steps = [
                "Set your intention clearly by speaking it aloud.",
                "Perform the main working with focused will and visualization.",
                "Raise energy through chanting, drumming, or breathwork, "
                "then direct it toward your intention.",
                "Ground any excess energy. Thank any entities or energies invoked.",
                "Close the ritual with the words: 'So mote it be. Blessed be.'",
            ]

        # Add aftercare if missing
        if not plan.aftercare:
            plan.aftercare = [
                "Eat something grounding (bread, chocolate, root vegetables) "
                "and drink water.",
                "Journal your experience: what you felt, saw, or sensed.",
                "Over the next few days, stay alert for signs, synchronicities, "
                "and relevant dreams.",
            ]

        # Add title if missing
        if not plan.title or len(plan.title.strip()) <= 3:
            category = self._detect_category(plan) or "magical"
            plan.title = f"{category.title()} Working Ritual"

        return plan

    def _enhance_safety_ethics(self, plan: RitualPlan) -> RitualPlan:
        """Inject safety notes and ethical reminders."""
        if not plan.safety_notes:
            plan.safety_notes = []

        existing_text = " ".join(plan.safety_notes).lower()
        all_materials = " ".join(plan.materials).lower()
        all_steps = " ".join(plan.steps).lower()
        combined = f"{all_materials} {all_steps}"

        # Fire safety
        uses_fire = any(f in combined for f in _FIRE_MATERIALS)
        if uses_fire and "fire" not in existing_text and "flame" not in existing_text:
            plan.safety_notes.append(
                "Fire safety: Never leave candles or incense unattended. "
                "Use a fire-safe dish or holder. Ensure proper ventilation "
                "when burning incense or herbs."
            )

        # Herb safety
        for herb in _SENSITIVE_HERBS:
            if herb in combined and herb not in existing_text:
                herb_data = HERBS.get(herb, {})
                herb_warnings = herb_data.get("safety_notes", [])
                if herb_warnings:
                    plan.safety_notes.append(
                        f"Herb safety ({herb.title()}): {' '.join(herb_warnings)}"
                    )
                break  # Add one herb warning to avoid wall of text

        # Ethical framing
        ethical_phrases = [
            "harm none", "free will", "consent", "ethical", "highest good",
        ]
        if not any(e in existing_text for e in ethical_phrases):
            plan.safety_notes.append(
                "Ethics: Perform this working with harm to none and for the "
                "highest good of all involved. Respect the free will of others. "
                "Magic is a complement to, not a replacement for, professional "
                "medical, legal, or psychological advice."
            )

        # Substitutions note
        if not any("substitut" in n.lower() for n in plan.safety_notes):
            plan.safety_notes.append(
                "Substitutions: If you don't have a specific material, "
                "clear quartz can substitute for any crystal, white candles "
                "for any color, and rosemary for most herbs."
            )

        return plan

    def _enhance_personalization(self, plan: RitualPlan) -> RitualPlan:
        """Add difficulty variants and personalization notes."""
        # Set difficulty if missing
        if not plan.difficulty or plan.difficulty.strip().lower() not in {
            "beginner", "intermediate", "advanced",
        }:
            plan.difficulty = "beginner"

        # Add personalization to aftercare
        personal_notes = [
            "Make it your own: feel free to adapt any words, gestures, or "
            "materials to what resonates with your personal practice.",
            "Journal prompt: After the ritual, reflect on what you felt. "
            "What images or sensations arose? What messages came through?",
        ]

        existing_aftercare = " ".join(plan.aftercare).lower() if plan.aftercare else ""
        for note in personal_notes:
            if note[:20].lower() not in existing_aftercare:
                plan.aftercare.append(note)

        # Add variation note to steps
        variation_note = (
            "Variation: For a simpler version, you may skip the circle casting "
            "and work with just a single candle and your intention. Trust your "
            "intuition — there is no wrong way to connect with your magic."
        )
        existing_steps = " ".join(plan.steps).lower() if plan.steps else ""
        if "variation" not in existing_steps and "simpler version" not in existing_steps:
            plan.steps.append(variation_note)

        return plan

    # ── Private Helpers ───────────────────────────────────────────────────

    def _detect_category(self, plan: RitualPlan) -> str:
        """Detect the magical intention category from plan fields.

        Checks plan.category first, then scans intention and title text
        for known intention keywords.

        Returns:
            The detected category string (lowercase), or empty string if
            no category can be determined.
        """
        # Direct category field
        if plan.category:
            cat = plan.category.strip().lower()
            if cat in INTENTION_MAP:
                return cat

        # Scan intention + title for keywords
        search_text = f"{plan.intention} {plan.title}".lower()
        for category, keywords in _INTENTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in search_text:
                    return category

        return ""
