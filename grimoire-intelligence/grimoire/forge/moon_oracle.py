"""MoonOracle -- timing intelligence for magical practice.

Provides current energy readings, optimal timing suggestions, weekly
forecasts, and daily guidance by combining lunar phase data, planetary
hour correspondences, zodiac sign influences, and Wheel-of-the-Year
seasonal context.

Part of the Grimoire Intelligence System's FORGE layer.
"""

import datetime
import math

from grimoire.models import MoonInfo, MoonPhase, TimingRecommendation, WeeklyForecast
from grimoire.knowledge.moon_phases import (
    MOON_PHASES,
    MOON_IN_SIGNS,
    calculate_moon_phase,
    calculate_moon_phase_precise,
    get_phase_data,
    get_sign_data,
    get_combined_energy,
)
from grimoire.knowledge.planetary_hours import (
    get_day_ruler,
    get_current_planetary_hour,
    get_best_planetary_hour,
    PLANET_CORRESPONDENCES,
    INTENTION_TO_PLANET,
)
from grimoire.knowledge.wheel_of_year import get_next_sabbat, get_seasonal_context
from grimoire.knowledge.journal_prompts import get_moon_prompts


# ---------------------------------------------------------------------------
#  Zodiac helpers -- approximate moon-sign lookup
# ---------------------------------------------------------------------------

_ZODIAC_SIGNS: list[str] = [
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces",
]

# The moon spends roughly 2.33 days in each sign (29.53 / 12 ~ 2.46).
_DAYS_PER_SIGN: float = 29.53058867 / 12.0

# ---------------------------------------------------------------------------
#  Intention-keyword mappings used by scoring helpers
# ---------------------------------------------------------------------------

# Map broad intention keywords to the moon phase families that best serve them.
_INTENTION_PHASE_MAP: dict[str, list[str]] = {
    # Growth / attraction
    "love":          ["waxing_crescent", "first_quarter", "waxing_gibbous", "full_moon"],
    "prosperity":    ["waxing_crescent", "first_quarter", "waxing_gibbous", "full_moon"],
    "abundance":     ["waxing_gibbous", "full_moon"],
    "creativity":    ["waxing_crescent", "first_quarter", "waxing_gibbous", "full_moon"],
    "healing":       ["waxing_gibbous", "full_moon", "waning_gibbous"],
    "confidence":    ["first_quarter", "waxing_gibbous", "full_moon"],
    "courage":       ["first_quarter", "full_moon"],
    "communication": ["waxing_crescent", "first_quarter"],
    "wisdom":        ["full_moon", "waning_gibbous"],
    "peace":         ["waxing_gibbous", "full_moon", "waning_gibbous"],

    # Banishing / release
    "banishing":       ["waning_gibbous", "last_quarter", "waning_crescent"],
    "cleansing":       ["last_quarter", "waning_crescent", "new_moon"],
    "protection":      ["full_moon", "waning_gibbous", "last_quarter"],
    "transformation":  ["waning_gibbous", "last_quarter", "waning_crescent", "new_moon"],
    "grounding":       ["last_quarter", "waning_crescent", "new_moon"],

    # New beginnings
    "new_beginnings":  ["new_moon", "waxing_crescent"],
    "intention":       ["new_moon", "waxing_crescent"],

    # Divination / psychic
    "divination":  ["full_moon", "new_moon", "waning_crescent"],
    "psychic":     ["full_moon", "new_moon", "waning_crescent"],
}

# Map intention keywords to zodiac sign elements/qualities that support them.
_INTENTION_SIGN_MAP: dict[str, list[str]] = {
    "love":          ["libra", "taurus", "cancer", "leo", "pisces"],
    "prosperity":    ["taurus", "capricorn", "leo", "sagittarius"],
    "abundance":     ["taurus", "sagittarius", "leo"],
    "creativity":    ["leo", "pisces", "gemini", "libra"],
    "healing":       ["virgo", "pisces", "cancer"],
    "confidence":    ["aries", "leo", "sagittarius"],
    "courage":       ["aries", "leo", "scorpio"],
    "communication": ["gemini", "libra", "aquarius"],
    "wisdom":        ["sagittarius", "aquarius", "scorpio"],
    "peace":         ["libra", "pisces", "cancer", "taurus"],
    "banishing":     ["scorpio", "capricorn", "aries"],
    "cleansing":     ["virgo", "aquarius", "pisces"],
    "protection":    ["scorpio", "capricorn", "aries", "cancer"],
    "transformation":["scorpio", "pisces", "capricorn"],
    "grounding":     ["taurus", "capricorn", "virgo"],
    "divination":    ["pisces", "scorpio", "cancer", "aquarius"],
    "psychic":       ["pisces", "scorpio", "cancer"],
    "new_beginnings":["aries", "gemini", "sagittarius"],
    "intention":     ["aries", "capricorn"],
}

# Seasonal month groupings for scoring seasonal alignment.
_INTENTION_SEASON_MAP: dict[str, list[int]] = {
    "love":          [4, 5, 6, 7],         # spring-summer passion
    "prosperity":    [7, 8, 9, 10],        # harvest months
    "abundance":     [6, 7, 8, 9],
    "creativity":    [2, 3, 4, 5],         # spring awakening
    "healing":       [2, 3, 6, 9],         # equinox / Imbolc
    "confidence":    [5, 6, 7],
    "courage":       [3, 4, 5, 6],
    "communication": [3, 5, 6],
    "wisdom":        [9, 10, 11, 12],      # darkening year, introspection
    "peace":         [9, 12, 1, 2],
    "banishing":     [10, 11, 12, 1],      # waning year
    "cleansing":     [2, 3, 9],            # Imbolc, equinoxes
    "protection":    [10, 11, 12],
    "transformation":[10, 11, 1],          # Samhain corridor
    "grounding":     [8, 9, 10, 12],
    "divination":    [10, 11, 12],         # thin-veil season
    "psychic":       [10, 11, 12, 1],
    "new_beginnings":[1, 2, 3],
    "intention":     [1, 2, 10],           # New Year & Witch's New Year
}

# Day-of-week names (isoweekday() - 1 index).
_DAY_NAMES: list[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


# =========================================================================== #
#  MoonOracle                                                                  #
# =========================================================================== #


class MoonOracle:
    """Timing intelligence engine for magical practice.

    Combines lunar, planetary, zodiacal, and seasonal data to answer:
    * What is the energy right now?
    * When is the best time for a specific intention?
    * What does the coming week look like?
    * Which dates optimise a particular working?
    """

    def __init__(self, lat: float | None = None, lon: float | None = None):
        self.lat = lat
        self.lon = lon

    # ------------------------------------------------------------------ #
    #  Public methods                                                      #
    # ------------------------------------------------------------------ #

    def get_current_energy(
        self, dt: datetime.datetime | None = None
    ) -> MoonInfo:
        """Return a complete magical-energy profile for *dt* (default: now).

        Combines moon phase, precise zodiac sign (via ephem when available),
        day ruler, planetary hour, and seasonal context into a single
        :class:`MoonInfo` object.
        """
        if dt is None:
            dt = datetime.datetime.now()

        # 1. Moon phase, illumination, and zodiac sign (precise when ephem available)
        phase_key, illumination, precise_zodiac = calculate_moon_phase_precise(
            dt.year, dt.month, dt.day, dt.hour
        )
        phase_data = get_phase_data(phase_key) or MOON_PHASES["new_moon"]

        # 2. Map phase key to MoonPhase enum
        try:
            phase_enum = MoonPhase(phase_key)
        except ValueError:
            phase_enum = MoonPhase.NEW_MOON

        # 3. Zodiac sign — prefer precise from ephem, fallback to approximate
        zodiac = precise_zodiac if precise_zodiac else self.get_moon_sign_for_date(dt.date())
        sign_data = get_sign_data(zodiac)

        # 4. Day ruler & planetary hour (with real sunrise/sunset when available)
        weekday = dt.weekday()  # Monday=0
        day_ruler = get_day_ruler(weekday)
        current_hour = get_current_planetary_hour(
            weekday, dt.hour, lat=self.lat, lon=self.lon
        )

        # 5. Seasonal context
        seasonal = get_seasonal_context(dt.month)

        # 6. Build magical-energy description
        sign_label = sign_data["sign"] if sign_data else zodiac.title()
        sign_energy = sign_data["energy"] if sign_data else ""
        energy_desc = (
            f"The {phase_data['name']} in {sign_label} brings "
            f"{phase_data['magical_energy']} energy"
        )
        if sign_energy:
            energy_desc += f", coloured by {sign_label}'s {sign_energy} influence"
        energy_desc += (
            f". Today is ruled by {day_ruler} "
            f"({PLANET_CORRESPONDENCES.get(day_ruler, {}).get('day', _DAY_NAMES[weekday])})."
        )

        # 7. Best-for / avoid lists -- merge phase + sign
        best_for = list(phase_data.get("best_for", []))
        avoid = list(phase_data.get("avoid", []))
        if sign_data:
            for item in sign_data.get("best_for", []):
                if item not in best_for:
                    best_for.append(item)
            for item in sign_data.get("avoid", []):
                if item not in avoid:
                    avoid.append(item)

        # 8. Daily guidance -- combine phase guidance with seasonal colour
        guidance = phase_data.get("daily_guidance", "")
        guidance += f" {seasonal[:120]}..."  # trim seasonal snippet

        # 9. Element and keywords
        element = phase_data.get("element", "")
        keywords = list(phase_data.get("keywords", []))

        return MoonInfo(
            phase=phase_enum,
            phase_name=phase_data["name"],
            illumination=illumination,
            zodiac_sign=sign_label,
            magical_energy=energy_desc,
            best_for=best_for,
            avoid=avoid,
            daily_guidance=guidance,
            element=element,
            keywords=keywords,
        )

    # ------------------------------------------------------------------ #

    def get_optimal_timing(
        self, intention: str, days_ahead: int = 30
    ) -> list[TimingRecommendation]:
        """Find the best upcoming dates for *intention* within *days_ahead*.

        Each candidate day is scored (0-100) across four dimensions:
        - Moon phase alignment (40 pts)
        - Day ruler alignment  (25 pts)
        - Zodiac sign alignment (20 pts)
        - Seasonal alignment   (15 pts)

        Returns the top 5 dates sorted by ``alignment_score`` descending.
        """
        today = datetime.date.today()
        norm_intention = self._normalise_intention(intention)
        candidates: list[TimingRecommendation] = []

        for offset in range(days_ahead):
            d = today + datetime.timedelta(days=offset)
            phase_key, illumination, precise_zodiac = calculate_moon_phase_precise(
                d.year, d.month, d.day
            )
            phase_data = get_phase_data(phase_key) or {}
            weekday = d.weekday()
            day_ruler = get_day_ruler(weekday)
            zodiac = precise_zodiac if precise_zodiac else self.get_moon_sign_for_date(d)

            # Scoring
            s_phase = self._score_moon_phase_for_intention(phase_key, norm_intention)
            s_ruler = self._score_day_ruler_for_intention(day_ruler, norm_intention)
            s_zodiac = self._score_zodiac_for_intention(zodiac, norm_intention)
            s_season = self._score_seasonal_alignment(d.month, norm_intention)
            total = s_phase + s_ruler + s_zodiac + s_season

            # Build reasons and cautions
            reasons: list[str] = []
            cautions: list[str] = []

            if s_phase >= 30:
                reasons.append(
                    f"{phase_data.get('name', phase_key)} is ideal for {intention}"
                )
            elif s_phase >= 15:
                reasons.append(
                    f"{phase_data.get('name', phase_key)} supports {intention}"
                )

            if s_ruler >= 18:
                reasons.append(
                    f"{_DAY_NAMES[weekday]} ({day_ruler} day) strongly aligns with {intention}"
                )
            elif s_ruler >= 10:
                reasons.append(
                    f"{_DAY_NAMES[weekday]} ({day_ruler} day) offers secondary support"
                )

            if s_zodiac >= 15:
                reasons.append(
                    f"Moon in {zodiac.title()} amplifies this working"
                )

            if s_season >= 10:
                reasons.append("Seasonal energy is supportive")

            # Cautions
            if s_phase < 10:
                cautions.append(
                    f"The {phase_data.get('name', phase_key)} is not optimal for {intention}; "
                    "consider adapting your approach"
                )
            for item in phase_data.get("avoid", []):
                if norm_intention in item.lower():
                    cautions.append(f"Phase caution: {item}")

            candidates.append(
                TimingRecommendation(
                    date=d.isoformat(),
                    moon_phase=phase_data.get("name", phase_key),
                    zodiac_sign=zodiac.title(),
                    day_ruler=day_ruler,
                    planetary_hour="",  # best hour varies by user schedule
                    alignment_score=round(total, 1),
                    reasons=reasons,
                    cautions=cautions,
                )
            )

        # Sort by score desc and return top 5
        candidates.sort(key=lambda r: r.alignment_score, reverse=True)
        return candidates[:5]

    # ------------------------------------------------------------------ #

    def get_weekly_forecast(
        self, start_date: datetime.date | None = None
    ) -> WeeklyForecast:
        """Return a 7-day magical calendar beginning on *start_date*.

        Each day entry includes moon phase, day ruler, approximate zodiac
        sign, energy description, best-for and avoid lists, and a tip.
        The forecast also provides highlights, upcoming sabbat info, and
        a weekly theme derived from the dominant energies.
        """
        if start_date is None:
            start_date = datetime.date.today()

        days: list[dict] = []
        phase_counts: dict[str, int] = {}
        highlights: list[str] = []

        # Upcoming sabbat (computed once)
        sabbat_name, sabbat_data, days_until = get_next_sabbat(
            start_date.month, start_date.day
        )

        for offset in range(7):
            d = start_date + datetime.timedelta(days=offset)
            phase_key, illumination, precise_zodiac = calculate_moon_phase_precise(
                d.year, d.month, d.day
            )
            phase_data = get_phase_data(phase_key) or MOON_PHASES["new_moon"]
            weekday = d.weekday()
            day_ruler = get_day_ruler(weekday)
            zodiac = precise_zodiac if precise_zodiac else self.get_moon_sign_for_date(d)
            sign_data = get_sign_data(zodiac) or {}

            # Track phase counts for weekly theme
            family = self._phase_family(phase_key)
            phase_counts[family] = phase_counts.get(family, 0) + 1

            # Energy description
            energy = (
                f"{phase_data['magical_energy'].capitalize()} energy "
                f"under {zodiac.title()}'s {sign_data.get('energy', 'influence')}"
            )

            # Tip
            tip = phase_data.get("daily_guidance", "Follow your intuition today.")

            day_entry = {
                "date": d.isoformat(),
                "day_name": _DAY_NAMES[weekday],
                "moon_phase": phase_data["name"],
                "illumination": illumination,
                "zodiac_sign": zodiac.title(),
                "day_ruler": day_ruler,
                "energy": energy,
                "best_for": list(phase_data.get("best_for", [])),
                "avoid": list(phase_data.get("avoid", [])),
                "tip": tip,
            }
            days.append(day_entry)

            # Highlight notable events
            if phase_key == "full_moon":
                highlights.append(f"Full Moon on {_DAY_NAMES[weekday]} {d.isoformat()}")
            elif phase_key == "new_moon":
                highlights.append(f"New Moon on {_DAY_NAMES[weekday]} {d.isoformat()}")

            # Check sabbat proximity
            sabbat_delta = days_until - offset
            if sabbat_delta == 0:
                highlights.append(f"{sabbat_name} falls on {_DAY_NAMES[weekday]} {d.isoformat()}")
            elif 0 < sabbat_delta <= 3:
                highlights.append(
                    f"{sabbat_name} is {sabbat_delta} day{'s' if sabbat_delta != 1 else ''} away "
                    f"({_DAY_NAMES[weekday]} {d.isoformat()})"
                )

        # Weekly theme
        weekly_theme = self._derive_weekly_theme(phase_counts)

        # Upcoming sabbat string
        upcoming = f"{sabbat_name} in {days_until} day{'s' if days_until != 1 else ''}"

        return WeeklyForecast(
            start_date=start_date.isoformat(),
            days=days,
            highlights=highlights if highlights else ["A steady week with no major celestial events."],
            upcoming_sabbat=upcoming,
            weekly_theme=weekly_theme,
        )

    # ------------------------------------------------------------------ #

    def suggest_best_dates(
        self, intention: str, days_ahead: int = 30
    ) -> list[dict]:
        """Simplified date suggestions -- top 3 dates as plain dicts.

        Each dict contains ``date``, ``reason``, and ``score`` keys.
        """
        recs = self.get_optimal_timing(intention, days_ahead)
        results: list[dict] = []
        for rec in recs[:3]:
            reason = "; ".join(rec.reasons) if rec.reasons else "General alignment"
            results.append({
                "date": rec.date,
                "reason": reason,
                "score": rec.alignment_score,
            })
        return results

    # ------------------------------------------------------------------ #

    def get_daily_guidance(
        self, dt: datetime.datetime | None = None
    ) -> str:
        """Return a paragraph of natural-sounding daily magical guidance.

        Blends moon phase, day ruler, zodiac sign, and seasonal context
        into an encouraging, readable narrative.
        """
        if dt is None:
            dt = datetime.datetime.now()

        phase_key, illumination, precise_zodiac = calculate_moon_phase_precise(
            dt.year, dt.month, dt.day, dt.hour
        )
        phase_data = get_phase_data(phase_key) or MOON_PHASES["new_moon"]
        weekday = dt.weekday()
        day_ruler = get_day_ruler(weekday)
        zodiac = precise_zodiac if precise_zodiac else self.get_moon_sign_for_date(dt.date())
        sign_data = get_sign_data(zodiac) or {}
        seasonal = get_seasonal_context(dt.month)

        # Planetary domains for day ruler
        ruler_data = PLANET_CORRESPONDENCES.get(day_ruler, {})
        domains = ruler_data.get("magical_domains", [])
        domains_str = ", ".join(domains[:3]) if domains else "general practice"

        # Journal prompts
        prompts = get_moon_prompts(phase_key)
        prompt = prompts[0] if prompts else "What does your intuition whisper today?"

        # Build guidance paragraph
        sign_label = sign_data.get("sign", zodiac.title())
        sign_quality = sign_data.get("energy", "balanced")

        guidance = (
            f"Today the {phase_data['name']} rests in {sign_label}, weaving "
            f"{phase_data['magical_energy']} lunar energy with {sign_label}'s "
            f"{sign_quality} nature. It is {_DAY_NAMES[weekday]}, ruled by "
            f"{day_ruler}, whose gifts of {domains_str} are close at hand. "
            f"{phase_data.get('daily_guidance', '')} "
            f"The season whispers: {seasonal[:150]} "
            f"A question to carry with you: {prompt}"
        )
        return guidance

    # ------------------------------------------------------------------ #

    def get_moon_sign_for_date(self, dt: datetime.date) -> str:
        """Return an approximate zodiac sign for the moon on *dt*.

        This is a simplified model: it uses the moon's age in the current
        synodic cycle to step through the twelve signs. The starting sign
        for the cycle is derived from a known reference lunation. The
        result is approximate (within about 1 sign) and suitable for a
        practice-oriented tool -- not an ephemeris replacement.
        """
        # Julian Day Number (simplified Gregorian)
        a = (14 - dt.month) // 12
        y = dt.year + 4800 - a
        m = dt.month + 12 * a - 3
        jdn = (
            dt.day
            + (153 * m + 2) // 5
            + 365 * y
            + y // 4
            - y // 100
            + y // 400
            - 32045
        )

        # Known new-moon epoch and synodic period (mirrors moon_phases module)
        known_new_moon_jdn = 2451550.1 + 0.76  # ~Jan 6 2000
        synodic_month = 29.53058867

        days_since = jdn - known_new_moon_jdn
        moon_age = days_since % synodic_month  # 0 = new moon

        # The moon enters a new sign roughly every 2.46 days.
        # At the reference new moon (Jan 6 2000) the moon was approximately
        # in Capricorn. We use sign index 9 (capricorn) as the base and
        # advance by moon_age / _DAYS_PER_SIGN.
        base_sign_index = 9  # capricorn for reference lunation start
        # Also account for how many full cycles have passed -- each full
        # synodic month advances the base sign by ~0.9 signs.
        full_cycles = int(days_since / synodic_month)
        # Each successive new moon starts about 1 sign later in the zodiac
        cycle_shift = full_cycles % 12
        current_index = (base_sign_index + cycle_shift + int(moon_age / _DAYS_PER_SIGN)) % 12

        return _ZODIAC_SIGNS[current_index]

    # ------------------------------------------------------------------ #
    #  Internal scoring helpers                                            #
    # ------------------------------------------------------------------ #

    def _score_moon_phase_for_intention(
        self, phase: str, intention: str
    ) -> float:
        """Score how well *phase* serves *intention* (0-40).

        Growth/attraction intentions score high during waxing/full phases.
        Banishing/release intentions score high during waning phases.
        New-beginning intentions peak at new moon.
        Divination peaks at full and dark moons.
        """
        ideal_phases = _INTENTION_PHASE_MAP.get(intention, [])
        if not ideal_phases:
            # Unknown intention: give a neutral score
            return 15.0

        if phase in ideal_phases:
            rank = ideal_phases.index(phase)
            # First listed phase is the best match
            if rank == 0:
                return 40.0
            elif rank == 1:
                return 32.0
            elif rank == 2:
                return 25.0
            else:
                return 20.0

        # Phase not in ideal list -- check if it's at least in the same
        # family (waxing vs waning vs new/full).
        phase_family = self._phase_family(phase)
        ideal_families = {self._phase_family(p) for p in ideal_phases}
        if phase_family in ideal_families:
            return 12.0

        return 5.0

    def _score_day_ruler_for_intention(
        self, day_ruler: str, intention: str
    ) -> float:
        """Score how well *day_ruler* supports *intention* (0-25).

        Uses INTENTION_TO_PLANET from the planetary_hours knowledge base.
        """
        ideal_planets = INTENTION_TO_PLANET.get(intention, [])
        if not ideal_planets:
            # Try matching day ruler's domains directly
            ruler_data = PLANET_CORRESPONDENCES.get(day_ruler, {})
            domains = ruler_data.get("magical_domains", [])
            if intention in domains:
                return 22.0
            return 8.0

        if day_ruler in ideal_planets:
            rank = ideal_planets.index(day_ruler)
            if rank == 0:
                return 25.0
            else:
                return 18.0

        # Secondary: check if ruler's domains overlap with intention
        ruler_data = PLANET_CORRESPONDENCES.get(day_ruler, {})
        domains = ruler_data.get("magical_domains", [])
        if intention in domains:
            return 14.0

        return 5.0

    def _score_zodiac_for_intention(
        self, sign: str, intention: str
    ) -> float:
        """Score how well *sign* supports *intention* (0-20).

        Uses the _INTENTION_SIGN_MAP lookup.
        """
        ideal_signs = _INTENTION_SIGN_MAP.get(intention, [])
        if not ideal_signs:
            return 8.0

        sign_lower = sign.lower()
        if sign_lower in ideal_signs:
            rank = ideal_signs.index(sign_lower)
            if rank == 0:
                return 20.0
            elif rank == 1:
                return 16.0
            else:
                return 12.0

        # Element affinity: if the sign's element matches a top-sign element
        sign_data = get_sign_data(sign_lower)
        if sign_data:
            sign_element = sign_data.get("element", "")
            for ideal_sign in ideal_signs[:2]:
                ideal_data = get_sign_data(ideal_sign)
                if ideal_data and ideal_data.get("element") == sign_element:
                    return 8.0

        return 3.0

    def _score_seasonal_alignment(
        self, month: int, intention: str
    ) -> float:
        """Score how well the current *month* aligns with *intention* (0-15).

        Harvest-time months boost prosperity work; dark months boost
        banishing and divination; spring boosts new beginnings, etc.
        """
        ideal_months = _INTENTION_SEASON_MAP.get(intention, [])
        if not ideal_months:
            return 7.0  # neutral

        if month in ideal_months:
            return 15.0

        # Adjacent month gets partial credit
        for m in ideal_months:
            if abs(month - m) == 1 or abs(month - m) == 11:
                return 10.0

        return 4.0

    # ------------------------------------------------------------------ #
    #  Private utilities                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _phase_family(phase_key: str) -> str:
        """Classify a moon phase into a broad family for theme analysis."""
        if phase_key in ("new_moon",):
            return "new"
        if phase_key in ("waxing_crescent", "first_quarter", "waxing_gibbous"):
            return "waxing"
        if phase_key in ("full_moon",):
            return "full"
        if phase_key in ("waning_gibbous", "last_quarter", "waning_crescent"):
            return "waning"
        return "mixed"

    @staticmethod
    def _derive_weekly_theme(phase_counts: dict[str, int]) -> str:
        """Derive a prose weekly theme from the dominant phase family."""
        if not phase_counts:
            return "A week of mixed and balanced energies."

        dominant = max(phase_counts, key=phase_counts.get)  # type: ignore[arg-type]

        themes = {
            "new": (
                "This week is anchored in New Moon energy. Focus on setting "
                "intentions, beginning new projects, and spending time in quiet "
                "reflection. The slate is clean -- dream boldly."
            ),
            "waxing": (
                "Waxing energy dominates the week ahead. This is a time for "
                "building, attracting, and taking action on your intentions. "
                "Momentum is on your side -- lean into growth."
            ),
            "full": (
                "Full Moon energy illuminates the week. Expect heightened "
                "intuition, emotional intensity, and powerful manifestation "
                "potential. Celebrate what you have created and charge your tools."
            ),
            "waning": (
                "Waning energy shapes the week. This is a powerful period for "
                "releasing what no longer serves you, banishing, cleansing, and "
                "completing loose ends. Let go with grace."
            ),
        }
        return themes.get(dominant, "A week of shifting energies -- stay present and adaptable.")

    @staticmethod
    def _normalise_intention(intention: str) -> str:
        """Normalise an intention string into a lookup key.

        Strips whitespace, lowercases, and attempts to find the closest
        matching key in _INTENTION_PHASE_MAP.
        """
        raw = intention.strip().lower().replace("-", "_").replace(" ", "_")

        # Direct match
        if raw in _INTENTION_PHASE_MAP:
            return raw

        # Partial / fuzzy match -- check if the raw string contains a known key
        for key in _INTENTION_PHASE_MAP:
            if key in raw or raw in key:
                return key

        # Check INTENTION_TO_PLANET as a secondary source
        for key in INTENTION_TO_PLANET:
            if key in raw or raw in key:
                return key

        # Fallback: return as-is; scoring helpers will give neutral values
        return raw
