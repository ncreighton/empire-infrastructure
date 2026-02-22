"""VariationEngine — weighted pool selection with anti-repetition tracking.

Ensures every output is fresh and unique by:
  - Maintaining a SQLite recency log of all selections
  - Weighting recently-used items lower (anti-repetition)
  - Biasing toward never-used items (discovery)
  - Providing pick() for single selection and pick_n() for N unique items

Reuses the existing grimoire.db database.
"""

import random
import sqlite3
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Recency table schema
# ---------------------------------------------------------------------------

_RECENCY_SCHEMA = """
CREATE TABLE IF NOT EXISTS recency_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    pool_name TEXT NOT NULL,
    selected_value TEXT NOT NULL
);
"""


# ===========================================================================
# VariationEngine
# ===========================================================================

class VariationEngine:
    """Weighted pool selection with SQLite-backed anti-repetition tracking."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(
                Path(__file__).resolve().parent.parent / "data" / "grimoire.db"
            )
        self.db_path = db_path
        self._is_memory = db_path == ":memory:"
        self._shared_conn: sqlite3.Connection | None = None

        if not self._is_memory:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_table()

    def _init_table(self) -> None:
        with self._connect() as conn:
            conn.executescript(_RECENCY_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        if self._is_memory:
            if self._shared_conn is None:
                self._shared_conn = sqlite3.connect(":memory:")
                self._shared_conn.row_factory = sqlite3.Row
            return self._shared_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pick(self, pool_name: str, variants: list[str]) -> str:
        """Select one variant with anti-repetition weighting.

        Weights:
          - Used in last 3 days: 0.1
          - Used 4-14 days ago: 0.5
          - Used 15-30 days ago: 1.0
          - Never used or >30 days: 2.0 (discovery bias)

        Args:
            pool_name: Identifier for this pool (e.g. "affirmation_protection").
            variants: List of candidate strings.

        Returns:
            A single selected string.
        """
        if not variants:
            return ""
        if len(variants) == 1:
            self._log_selection(pool_name, variants[0])
            return variants[0]

        weights = self._get_recency_weights(pool_name, variants)
        selected = random.choices(variants, weights=weights, k=1)[0]
        self._log_selection(pool_name, selected)
        return selected

    def pick_n(self, pool_name: str, variants: list[str], n: int) -> list[str]:
        """Select N unique variants with anti-repetition weighting.

        Args:
            pool_name: Identifier for this pool.
            variants: List of candidate strings.
            n: Number of items to select.

        Returns:
            A list of N unique selected strings (or fewer if pool is small).
        """
        if not variants:
            return []
        n = min(n, len(variants))

        weights = self._get_recency_weights(pool_name, variants)
        selected: list[str] = []
        remaining = list(variants)
        remaining_weights = list(weights)

        for _ in range(n):
            if not remaining:
                break
            choice = random.choices(remaining, weights=remaining_weights, k=1)[0]
            selected.append(choice)
            idx = remaining.index(choice)
            remaining.pop(idx)
            remaining_weights.pop(idx)

        for item in selected:
            self._log_selection(pool_name, item)

        return selected

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_selection(self, pool_name: str, selected: str) -> None:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO recency_log (timestamp, pool_name, selected_value) VALUES (?, ?, ?)",
                (ts, pool_name, selected),
            )
            conn.commit()

    def _get_recency_weights(self, pool_name: str, variants: list[str]) -> list[float]:
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff_30 = (now - datetime.timedelta(days=30)).isoformat()

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT selected_value, MAX(timestamp) AS last_used
                FROM recency_log
                WHERE pool_name = ? AND timestamp > ?
                GROUP BY selected_value
                """,
                (pool_name, cutoff_30),
            ).fetchall()

        recent_map: dict[str, datetime.datetime] = {}
        for row in rows:
            try:
                recent_map[row["selected_value"]] = datetime.datetime.fromisoformat(
                    row["last_used"]
                )
            except (ValueError, TypeError):
                pass

        weights: list[float] = []
        for variant in variants:
            last_used = recent_map.get(variant)
            if last_used is None:
                weights.append(2.0)  # discovery bias
            else:
                days_ago = (now - last_used).total_seconds() / 86400
                if days_ago < 3:
                    weights.append(0.1)
                elif days_ago < 14:
                    weights.append(0.5)
                elif days_ago < 30:
                    weights.append(1.0)
                else:
                    weights.append(2.0)

        return weights


# ===========================================================================
# TEXT POOLS — All variant texts organized by category
# ===========================================================================

# ---------------------------------------------------------------------------
# Affirmation pools: 5-6 variants per intention
# ---------------------------------------------------------------------------

AFFIRMATION_POOLS: dict[str, list[str]] = {
    "protection": [
        "I am surrounded by an unbreakable shield of light. Nothing that is not for my highest good may enter.",
        "I stand in a fortress of my own making. My boundaries are sacred and strong.",
        "Like ancient stone walls, my protection holds firm against all that would cause harm.",
        "I am wrapped in the armor of my ancestors. Their strength flows through my veins.",
        "My energy field is sealed and sovereign. Only love and light may pass through.",
        "I am the guardian of my own sacred space. My will is an impenetrable ward.",
    ],
    "love": [
        "I am worthy of deep, abundant love. My heart is open and magnetic.",
        "Love flows to me as naturally as rivers flow to the sea. I am ready to receive.",
        "My heart radiates warmth that draws kindred souls near. I am a beacon of love.",
        "I deserve love that nourishes, uplifts, and honors my wholeness.",
        "Every cell of my being vibrates with the frequency of love. I attract what I am.",
        "I open my heart without fear, knowing that love is my birthright and my strength.",
    ],
    "prosperity": [
        "Abundance flows to me from expected and unexpected sources. I am a magnet for prosperity.",
        "I am aligned with the energy of wealth. Opportunities multiply around me.",
        "The universe conspires in my favor. Prosperity is my natural state.",
        "I release all scarcity thinking. I am worthy of overflowing abundance.",
        "Golden energy surrounds me, drawing wealth, opportunity, and financial freedom.",
        "My cup overflows. I have more than enough and I share generously.",
    ],
    "healing": [
        "My body, mind, and spirit align in radiant health. Healing energy flows through me with every breath.",
        "I am being restored to wholeness. Every cell remembers its perfect blueprint.",
        "Healing light pours through me, washing away pain and replacing it with vitality.",
        "I am patient with my healing journey. Every day brings me closer to balance.",
        "My body is a temple of resilience. I trust its ancient wisdom to heal.",
    ],
    "divination": [
        "My inner sight is clear and true. I trust the messages the universe reveals to me.",
        "The veil parts easily for me. I see what needs to be seen with clarity and courage.",
        "I am a clear channel for divine wisdom. Messages flow to me effortlessly.",
        "My intuition is a compass that never fails. I trust its direction completely.",
        "The symbols speak and I listen. Every sign carries meaning meant for me.",
    ],
    "banishing": [
        "I release all that no longer serves me. Negativity dissolves in the light of my will.",
        "What does not belong in my life now falls away. I am free of unwanted attachments.",
        "I sever all cords that drain my energy. My power returns to me tenfold.",
        "Darkness flees before my light. I banish all that stands between me and my peace.",
        "I command all harmful influences to depart. My word is law in my own domain.",
    ],
    "cleansing": [
        "I am purified in body, mind, and spirit. Only what is clean and true remains within my space.",
        "Like rain washing the earth, I am renewed. All impurities dissolve and drain away.",
        "I stand in a waterfall of cleansing light. Every shadow is washed from my being.",
        "My space is sacred and pure. I reclaim it with every conscious breath.",
        "I release all stagnant energy. Fresh, clean vitality fills every corner of my life.",
    ],
    "creativity": [
        "Creative energy flows through me freely and joyfully. I am an open channel for inspiration.",
        "Ideas come to me like sparks from a sacred fire. I fan them into brilliant flame.",
        "My imagination is boundless. I create with confidence and wild abandon.",
        "I am a vessel of creative magick. Art flows through me as naturally as breath.",
        "Every moment holds the seed of a new creation. I am fertile ground for inspiration.",
    ],
    "wisdom": [
        "Ancient wisdom lives within me. I access deeper knowing with every still moment I create.",
        "The library of the universe is open to me. I read its pages with reverence and clarity.",
        "Wisdom flows to me from root and star alike. I am a student and a sage.",
        "I trust the deep knowing that lives beneath thought. My wisdom is ancestral and eternal.",
        "Each experience is a teacher. I extract the gold from every lesson life offers.",
    ],
    "confidence": [
        "I stand in my power, radiant and unshakable. I trust myself completely and act with bold certainty.",
        "I am the author of my own story. My voice carries weight and my presence commands respect.",
        "Solar fire blazes within me. I shine without apology and inspire others by my example.",
        "I walk through the world with the certainty of mountains. Nothing can diminish my worth.",
        "My confidence is rooted in self-knowledge. I know who I am and I am enough.",
    ],
    "communication": [
        "My words carry truth and clarity. I speak with confidence and listen with compassion.",
        "I express myself with eloquence and authenticity. My voice is heard and valued.",
        "Words flow from me like mercury — quick, bright, and precisely aimed.",
        "I communicate with grace and power. My message lands exactly where it needs to.",
        "The gift of clear speech is mine. I speak truth wrapped in kindness.",
    ],
    "grounding": [
        "I am rooted deep in the earth, stable and strong. No storm can shake my foundation.",
        "Like an ancient oak, I draw strength from deep roots and stand tall through every season.",
        "I am anchored to the bedrock of my being. Chaos swirls but cannot move me.",
        "The earth holds me. I am supported, stable, and profoundly present.",
        "My roots extend deep into the heart of the earth. I am unmovable and at peace.",
    ],
    "transformation": [
        "I welcome change as a sacred teacher. I release the old self and step into who I am becoming.",
        "I am the phoenix. What burns away was never truly mine. What remains is indestructible.",
        "Transformation is my birthright. I shed what no longer fits and emerge renewed.",
        "I trust the darkness of the cocoon. What dissolves within me is becoming wings.",
        "I am not breaking — I am breaking open. Each ending seeds a more powerful beginning.",
    ],
    "peace": [
        "Serenity fills me from crown to root. I am calm, centered, and at peace with all that is.",
        "Stillness dwells at my core. No external noise can reach the quiet place within me.",
        "I am the eye of every storm. Peace is not something I seek — it is what I am.",
        "I exhale tension and inhale tranquility. Every breath deepens my calm.",
        "Like still water reflecting the moon, I am undisturbed, clear, and luminous.",
    ],
    "courage": [
        "I face every challenge with the heart of a warrior and the grace of a healer. Fear does not rule me.",
        "Courage flows through me like liquid fire. I act boldly even when my hands tremble.",
        "I was born for this moment. My bravery is not the absence of fear but the mastery of it.",
        "I draw courage from every ancestor who stood firm before me. Their strength is mine.",
        "Fear is fuel for my flame. I transmute doubt into decisive, powerful action.",
    ],
}

# ---------------------------------------------------------------------------
# Element imagery pools: 3-4 variants per element
# ---------------------------------------------------------------------------

ELEMENT_IMAGERY_POOLS: dict[str, list[str]] = {
    "fire": [
        "See a brilliant flame kindling before you, its warmth washing over your skin. The fire dances with purpose, each flicker shaping your intention into golden light. Feel the heat gather at your solar plexus, igniting your willpower and burning away all that stands between you and your goal.",
        "A circle of candles blazes around you, each flame a sentinel of your will. The fire roars upward, carrying your intention skyward on columns of heat and light. Your spirit kindles — you are the flame, fierce and unstoppable.",
        "Deep within you, embers glow red-hot, waiting for your breath. As you exhale intention, they ignite into a bonfire of purpose. The flames leap and dance, painting the darkness with the colors of your desire. You are forged in sacred fire.",
        "A phoenix of golden flame rises from the ashes before you. Its wings spread wide, each feather a prayer made visible. The heat is fierce but does not burn — it purifies, transforms, and empowers everything it touches.",
    ],
    "water": [
        "Imagine a still, moonlit pool at your feet. As you breathe, gentle ripples spread outward, carrying your intention across the water. Feel the cool, cleansing energy wash through you — purifying, healing, flowing into every corner of your being like a sacred tide returning home.",
        "You stand at the edge of a midnight ocean. Each wave that touches your feet carries away what you release, and each retreating tide brings a gift from the deep. You are both the shore and the sea — patient, powerful, and endlessly renewing.",
        "A sacred spring wells up from the earth beneath you. Crystal-clear water rises, pooling around your hands, singing with the memory of every healing it has ever performed. You drink deeply and feel restoration flow through every vein.",
        "Rain falls gently around you, each drop a blessing from the sky. The water traces silver paths down your skin, washing away weariness and worry, leaving you clean, clear, and shimmering with renewed intention.",
    ],
    "earth": [
        "Visualize rich, dark soil beneath your hands. Feel its weight and warmth, teeming with life just beneath the surface. Press your intention into the earth like a seed. Feel roots extend from your body downward, anchoring you, while green shoots of manifestation reach toward the light.",
        "You stand in an ancient forest where trees older than memory surround you. Their roots intertwine beneath your feet, connecting you to a web of strength that spans continents. You are part of this network now — grounded, supported, unshakable.",
        "A mountain rises within you, stone by stone, solid and eternal. You feel the weight of granite in your bones, the patience of geological time in your breath. You are immovable, unhurried, and profoundly real.",
    ],
    "air": [
        "A gentle breeze rises around you, carrying the scent of herbs and distant rain. Speak your intention into the wind and watch the words become silver threads spiraling upward. Feel clarity flood your mind as the air sharpens your focus and lifts your spirit like wings unfolding.",
        "You stand on a high peak where the wind sings with the voices of all who have sought wisdom. The air is thin and electric, crackling with possibility. Each breath fills you with crystal clarity and the courage to speak your truth.",
        "A whirlwind of golden light spirals around you, lifting thoughts and worries from your shoulders like autumn leaves. In the eye of this gentle storm, your mind is perfectly still, perfectly clear, perfectly free.",
    ],
    "spirit": [
        "A column of luminous white light descends from above, passing through your crown and filling your entire being. You stand at the crossroads of all elements, all directions, all possibilities. Your intention resonates outward in every direction at once, connecting you to the web of all magick.",
        "The boundaries between worlds dissolve. You float in a space of infinite starlight where every possibility exists simultaneously. Your intention is a seed of pure light, and the universe conspires to help it grow.",
        "You are everywhere and nowhere — a point of awareness in an ocean of consciousness. All knowledge is available to you. All power flows through you. You need only choose your intention and the cosmos aligns.",
    ],
}

# ---------------------------------------------------------------------------
# Aftercare pools: 6 variant sets
# ---------------------------------------------------------------------------

AFTERCARE_POOLS: list[list[str]] = [
    [
        "Ground yourself by eating something, drinking water, or touching the earth.",
        "Record your experience in your journal while the details are fresh.",
        "Rest gently and let the magick settle into your bones.",
    ],
    [
        "Place your palms flat on the ground and breathe out any excess energy.",
        "Drink a warm cup of tea and reflect on what you felt.",
        "Give thanks silently for the energies that assisted you.",
    ],
    [
        "Wash your hands slowly and deliberately, symbolically releasing the working.",
        "Sketch or doodle your impressions — words are not always necessary.",
        "Step outside for a few breaths of fresh air to transition gently.",
    ],
    [
        "Eat something earthy — bread, root vegetables, dark chocolate — to ground.",
        "Write three words that capture the essence of your experience.",
        "Sit quietly for a minute, feeling gratitude radiate from your heart.",
    ],
    [
        "Stretch your body gently — neck, shoulders, hands — releasing held tension.",
        "Drink a full glass of water to replenish and ground your energy.",
        "Take three slow, deep breaths and whisper 'it is done' on the last exhale.",
    ],
    [
        "Touch something natural — a stone, a leaf, bark — to reconnect with the earth.",
        "Journal one sentence about what surprised you during the working.",
        "Allow yourself to rest without guilt. Energy work can be tiring.",
    ],
]

# ---------------------------------------------------------------------------
# Preparation pools: 6 variant sets
# ---------------------------------------------------------------------------

PREPARATION_POOLS: list[list[str]] = [
    [
        "Cleanse your space with smoke, sound, or salt water.",
        "Gather all materials before beginning.",
        "Center yourself with three deep breaths.",
    ],
    [
        "Open a window to invite fresh energy into your space.",
        "Arrange your tools with intention — placement matters.",
        "Stand in silence for a moment and feel the energy of the room.",
    ],
    [
        "Sweep the floor (physically or energetically) to clear stagnant energy.",
        "Light a cleansing candle or diffuse purifying essential oils.",
        "Set a clear intention in your mind before touching any tools.",
    ],
    [
        "Wash your hands with intention, imagining mundane energy rinsing away.",
        "Place all materials within easy reach so you won't break focus mid-working.",
        "Take a moment to thank the materials for their service.",
    ],
    [
        "Ring a bell, clap, or use a singing bowl to break up stale energy.",
        "Lay out a clean cloth as your working surface.",
        "Close your eyes, ground your feet, and invite sacred space to form around you.",
    ],
    [
        "Sprinkle salt water at the four corners of your space.",
        "Silence all notifications and commit fully to the present moment.",
        "Breathe in through your nose for four counts, hold for four, exhale for four. Repeat three times.",
    ],
]

# ---------------------------------------------------------------------------
# Timing advice pools: 3 variants per intention
# ---------------------------------------------------------------------------

TIMING_ADVICE_POOLS: dict[str, list[str]] = {
    "protection": [
        "Best performed during the waning moon to banish negativity, or on a Tuesday for Mars energy. The hour of Mars or Saturn strengthens protective workings.",
        "Cast protection magick at dusk, when the boundary between day and night mirrors the boundaries you are building. Saturday's Saturn energy fortifies wards.",
        "The dark moon is the most potent time for invisible, impenetrable shields. Tuesday and Saturday are your strongest days.",
    ],
    "love": [
        "Ideal on a Friday, the day of Venus, during the waxing moon to draw love toward you. The full moon amplifies all love workings.",
        "Perform love magick at dawn on a Friday, when Venus rules and the world awakens to new possibilities. The waxing gibbous moon intensifies attraction.",
        "The hour of Venus on any day opens the heart's channel. Pair with a waxing or full moon for maximum drawing power.",
    ],
    "prosperity": [
        "Thursday (Jupiter) during the waxing moon is the classic timing for abundance work. Sunrise amplifies prosperity spells.",
        "Perform prosperity workings when the moon is waxing and Jupiter rules the hour. Thursday at noon is the golden window.",
        "The waxing crescent moon is ideal for planting financial seeds. Pair with Thursday's Jupiter energy or Sunday's solar abundance.",
    ],
    "healing": [
        "Monday (the Moon) during the full moon enhances healing magick. Perform at dawn for renewal energy.",
        "Healing flows strongest during the waxing moon as energy builds toward fullness. Monday's lunar energy amplifies restorative workings.",
        "The full moon at midnight bathes all in healing light. Wednesday also supports healing through Mercury's influence on the mind-body connection.",
    ],
    "divination": [
        "Monday or Wednesday during the full moon. Midnight is traditionally most potent for scrying and prophetic work.",
        "The dark moon opens the deepest channels of psychic sight. Practice at the liminal hours — dawn, dusk, or midnight.",
        "Mercury's day (Wednesday) during any moon phase sharpens the inner eye. The full moon amplifies but the new moon deepens.",
    ],
    "banishing": [
        "Saturday (Saturn) during the waning or dark moon. Perform at midnight or during the hour of Saturn.",
        "The last quarter moon is the knife that cuts cords. Work on a Saturday during Saturn's hour for maximum severing power.",
        "Banishing workings gain strength as the moon diminishes. Perform between midnight and dawn on a Tuesday or Saturday.",
    ],
    "cleansing": [
        "Monday during the waning moon for releasing impurities. Dawn sweeps away the old with new light.",
        "Cleansing rituals performed at dawn on a Monday harness the moon's purifying energy. The waning crescent is ideal.",
        "Any waning moon phase supports cleansing. Pair with the hour of the Moon or perform during a rain shower for amplified purification.",
    ],
    "creativity": [
        "Sunday (the Sun) during the waxing moon to build creative energy. Noon amplifies solar creative power.",
        "The waxing crescent moon sparks new ideas. Work on a Sunday or Friday to channel solar fire or Venusian artistry.",
        "Mid-morning on a Sunday, when solar energy is ascending, is the creative sweet spot. The waxing gibbous moon refines raw inspiration.",
    ],
    "wisdom": [
        "Thursday (Jupiter) during the full moon for illumination. Twilight hours open the doors of perception.",
        "Wednesday's Mercury sharpens the mind while Thursday's Jupiter expands understanding. The full moon reveals hidden knowledge.",
        "Seek wisdom during the waning gibbous moon, when the teacher energy is strongest. Thursday evenings under starlight are especially potent.",
    ],
    "confidence": [
        "Sunday (the Sun) during the waxing moon. Perform at noon when solar energy peaks.",
        "The waxing gibbous moon builds inner fire toward its fullest expression. Work at high noon on a Sunday for maximum solar confidence.",
        "Tuesday's Mars energy fuels warrior confidence. Pair with the first quarter moon for decisive, action-oriented self-assurance.",
    ],
    "communication": [
        "Wednesday (Mercury) during the waxing moon for clear expression. Dawn brings fresh clarity to the voice.",
        "Mercury's hour on any day sharpens the tongue and clarifies the mind. The first quarter moon empowers decisive communication.",
        "Perform communication workings at mid-morning on a Wednesday, when Mercury's influence is sharpest and the mind is keen.",
    ],
    "grounding": [
        "Saturday (Saturn) during the waning moon. Midnight is the earth's stillest hour for maximum effect.",
        "Work barefoot on the earth during Saturn's day for the deepest grounding. The last quarter moon supports release and settling.",
        "Dawn and dusk on a Saturday anchor you between light and dark. The dark moon grounds you into the deepest stillness.",
    ],
    "transformation": [
        "Saturday (Saturn) during the dark moon for the deepest transformations. The hour between midnight and dawn is liminal and potent.",
        "Samhain season amplifies transformation work. The waning crescent to new moon transition mirrors the death-rebirth cycle.",
        "Work during the last quarter moon when the old must be cut away. Saturday at midnight is the crossroads of change.",
    ],
    "peace": [
        "Monday (the Moon) during the full moon for illuminated serenity. Perform at dusk as the world softens.",
        "Friday's Venus brings harmony while Monday's Moon brings calm. The waning gibbous moon gently releases tension.",
        "Twilight on any day carries peace energy. Pair with the full moon's serene radiance for deepest tranquility.",
    ],
    "courage": [
        "Tuesday (Mars) during the waxing moon to build strength. Sunrise is the time of brave first steps.",
        "The first quarter moon challenges and strengthens. Work during Mars hour on a Tuesday for warrior energy.",
        "Dawn on a Tuesday channels the fierce courage of Mars. The waxing gibbous moon builds power toward confident action.",
    ],
}

# ---------------------------------------------------------------------------
# Daily suggestion pools: 3-4 variants per planet ruler
# ---------------------------------------------------------------------------

DAILY_SUGGESTION_POOLS: dict[str, list[str]] = {
    "moon": [
        "Connect with your intuition tonight. Hold a moonstone or selenite and sit in quiet reflection for 10 minutes.",
        "Fill a bowl with water and gaze at its surface by candlelight. Let images and impressions rise without judgment.",
        "Anoint your third eye with a drop of water and whisper a question to the moon. Listen for the answer in your dreams tonight.",
        "Brew a cup of chamomile or jasmine tea and drink it slowly, sending gratitude to the lunar energies that guide your intuition.",
    ],
    "mars": [
        "Light a red candle and speak your intentions with the fierce energy of Mars. Today favors courage and bold action.",
        "Write your most challenging goal on a piece of paper. Hold it in both hands and declare it aloud three times with increasing power.",
        "Carry a piece of red jasper or carnelian today. When you feel hesitation, grip it and channel the warrior within.",
        "Do something brave today — even something small. Mars rewards action, not perfection.",
    ],
    "mercury": [
        "Write a letter to your future self, or pull a single tarot card and journal about its message.",
        "Speak an affirmation into a glass of water, then drink it. Mercury carries your words into every cell.",
        "Organize your altar or workspace today. Mercury thrives on clarity and order — so does your magick.",
        "Learn one new thing about your craft today. Read a page, listen to a podcast, or research a new herb.",
    ],
    "jupiter": [
        "Light a green candle for Thursday's Jupiter energy and speak abundance affirmations. Today magnifies prosperity workings.",
        "Write down three things you are grateful for. Jupiter expands whatever you focus on — choose abundance.",
        "Share your knowledge with someone today. Teaching is Jupiter's gift, and it returns to you multiplied.",
        "Visualize your life one year from now, filled with every abundance you desire. Hold that vision for five minutes.",
    ],
    "venus": [
        "Create a small love or beauty ritual — add rose petals to your bath, or carry rose quartz today.",
        "Wear something that makes you feel beautiful. Venus energy amplifies self-love and attraction.",
        "Prepare a meal with love and intention. Kitchen witchery is Venus magick at its finest.",
        "Write a love note to yourself. List three qualities you admire about who you are becoming.",
    ],
    "saturn": [
        "Perform a grounding meditation. Sit with black tourmaline and release one thing that no longer serves you.",
        "Set one firm boundary today. Saturn rewards discipline and the courage to say no.",
        "Organize or cleanse a cluttered space. Saturn's energy transforms chaos into structure.",
        "Reflect on one lesson that hardship has taught you. Saturn is the stern teacher whose gifts endure.",
    ],
    "sun": [
        "Stand in sunlight for five minutes and absorb solar vitality. Carry citrine or tiger's eye for confidence.",
        "Light a gold or yellow candle and affirm your personal power. Today, you shine without apology.",
        "Do something creative today — paint, sing, write, dance. The Sun fuels self-expression and joy.",
        "Wear gold or warm colors and notice how it shifts your energy. You are radiant today.",
    ],
}

# ---------------------------------------------------------------------------
# Quick practice pools: 3-4 variants per planet ruler
# ---------------------------------------------------------------------------

QUICK_PRACTICE_POOLS: dict[str, list[str]] = {
    "moon": [
        "Hold your hands over a bowl of water and whisper one wish into it.",
        "Close your eyes and imagine moonlight filling your body from crown to root.",
        "Touch your heart and say: 'I trust my intuition completely.'",
    ],
    "mars": [
        "Light a match, state one bold intention, and let the flame carry it.",
        "Stomp your feet three times and declare: 'I am powerful and unstoppable.'",
        "Hold a red stone (or imagine one) and breathe fire into your solar plexus.",
    ],
    "mercury": [
        "Write three words that describe your intention and carry the paper with you.",
        "Tap your temples gently three times and say: 'My mind is sharp and clear.'",
        "Read one line from a meaningful book and let it guide your day.",
    ],
    "jupiter": [
        "Blow cinnamon toward your front door for abundance.",
        "Hold a coin in your palm and affirm: 'Abundance flows to me from all directions.'",
        "Write one thing you want to expand in your life and tuck it in your pocket.",
    ],
    "venus": [
        "Hold rose quartz to your heart for one minute and breathe self-love.",
        "Trace a heart on your palm with your finger and send love to someone you care about.",
        "Smell something beautiful — a flower, a perfume, fresh herbs — and let it fill you with joy.",
    ],
    "saturn": [
        "Place both palms flat on the ground and breathe out tension five times.",
        "Hold a dark stone and name one thing you are releasing today.",
        "Stand still for 30 seconds and feel the weight of your own bones. You are solid.",
    ],
    "sun": [
        "Face the sun (or a bright light), close your eyes, and affirm your power.",
        "Place your hand on your solar plexus and say: 'I am radiant and alive.'",
        "Stretch your arms wide and take three deep, golden breaths.",
    ],
}

# ---------------------------------------------------------------------------
# Challenge pools (for AMPLIFY anticipate stage)
# ---------------------------------------------------------------------------

CHALLENGE_POOLS: list[dict[str, str]] = [
    {"challenge": "Candle won't stay lit", "solution": "Try a draft-free location. Cup your hands around the flame gently. An LED candle is an equally valid substitute — the intention matters, not the flame."},
    {"challenge": "Mind keeps wandering during meditation or visualization", "solution": "This is completely normal. Gently return your focus without judgment each time. Wandering is not failure; noticing it is awareness."},
    {"challenge": "Don't feel anything during the working", "solution": "Magick does not require dramatic sensations. Many experienced practitioners feel nothing in the moment. Trust the process."},
    {"challenge": "Interrupted mid-ritual", "solution": "Pause, handle the interruption calmly, then return. You can resume where you left off or close gracefully. The energy waits for you."},
    {"challenge": "Materials unavailable", "solution": "Intention is more powerful than any tool. Use substitutions or practice with visualization alone. Focus outshines any material."},
    {"challenge": "Feeling anxious or emotional during the working", "solution": "This can happen, especially with transformation or healing work. Pause and ground yourself. You can always stop and return another day."},
    {"challenge": "Unsure if the words are right", "solution": "There is no single correct incantation. Speak from your heart in your own words. Authenticity is more powerful than perfection."},
    {"challenge": "Spell didn't seem to work", "solution": "Magick works on its own timeline. Look for subtle shifts in the days ahead. Sometimes the universe delivers what you need, not what you asked for."},
    {"challenge": "Feeling drained after the working", "solution": "Eat something grounding, drink water, and rest. Energy work takes real energy. This is normal and temporary."},
    {"challenge": "Not sure if I'm doing it right", "solution": "If you set an intention and performed the working with sincerity, you did it right. There is no certification needed to practice magick."},
    {"challenge": "Smoke from incense is too strong", "solution": "Open a window, use a fan to disperse, or switch to essential oil diffusers, simmer pots, or sound cleansing instead."},
    {"challenge": "Cat/pet keeps interrupting the ritual", "solution": "Animals are sensitive to energy shifts. Some practitioners welcome them as guardians of the circle. Otherwise, gently close the door."},
    {"challenge": "Can't visualize clearly", "solution": "Not everyone is visual. Try using other senses: feel the energy, hear the intention, or speak it aloud. Aphantasia doesn't diminish magick."},
    {"challenge": "Roommates or family don't understand", "solution": "Your practice is personal. A closed door, headphones, and a small, discreet altar are all you need. Not everyone needs to witness your magick."},
    {"challenge": "Weather prevents outdoor ritual", "solution": "Open a window and listen to the rain or wind. Play nature sounds. The elements are always accessible through visualization and intention."},
]

# ---------------------------------------------------------------------------
# Preparation checklist pools (for AMPLIFY anticipate stage)
# ---------------------------------------------------------------------------

CHECKLIST_POOLS: list[str] = [
    "Choose your date and time (consult the timing guidance in optimizations).",
    "Gather all materials and place them in your working area.",
    "Cleanse your space: open a window, burn cleansing herbs, or sprinkle salt water.",
    "Cleanse yourself: wash your hands, take a ritual bath, or simply take three conscious breaths.",
    "Silence your phone and minimize potential interruptions.",
    "Set up your altar or working surface with materials arranged intentionally.",
    "Review the steps of your working so you are not reading mid-ritual.",
    "Ground and center: stand or sit quietly, feel your connection to the earth.",
    "State your intention clearly to yourself before beginning.",
    "Check that you have water and a grounding snack nearby for after the working.",
    "Set the mood: light candles, play ambient music, or sit in natural light.",
    "Ensure pets are settled and doors are closed if privacy is needed.",
    "Take a moment of gratitude for the practice itself before you begin.",
    "Wear comfortable clothing or ritual garments that make you feel empowered.",
    "If working with fire, confirm your fireproof surfaces and safety measures.",
]

# ---------------------------------------------------------------------------
# Aftercare AMPLIFY pools (for AMPLIFY anticipate stage)
# ---------------------------------------------------------------------------

AFTERCARE_AMPLIFY_POOLS: list[str] = [
    "Ground yourself: eat something, drink water, place your hands on the earth.",
    "Record your experience in a journal or grimoire while it is fresh.",
    "Rest if you feel drained — energy work can be tiring.",
    "Avoid immediately scrolling social media or engaging in stressful tasks.",
    "Take a warm shower or bath to wash away any residual energy.",
    "Spend a few minutes in gratitude for the practice itself.",
    "Be gentle with yourself for the rest of the day.",
    "Step outside for fresh air and feel the transition back to ordinary awareness.",
    "Drink a full glass of water — hydration helps ground post-ritual energy.",
    "Stretch gently, paying special attention to your neck and shoulders.",
    "Light a grounding incense (sandalwood, cedar, or patchouli) to transition.",
    "Call a friend or connect with someone to gently re-enter the social world.",
]

# ---------------------------------------------------------------------------
# Ethical note pools (for AMPLIFY fortify stage)
# ---------------------------------------------------------------------------

ETHICAL_NOTE_POOLS: list[str] = [
    "'An it harm none, do what ye will.' Consider the ripple effects of every working.",
    "Love magick must never override another person's free will or consent. Focus on self-love, attracting compatible energy, or strengthening existing mutual bonds.",
    "Source herbs and crystals ethically. White sage is over-harvested from Indigenous lands; garden sage, rosemary, or cedar are excellent alternatives.",
    "Cultural respect: research the origins of practices you adopt. Credit traditions and avoid claiming sacred practices from closed communities as your own.",
    "Never use magick to manipulate, control, or harm another being.",
    "Respect the land and environment when gathering materials outdoors. Take only what you need and leave an offering of gratitude.",
    "Consent extends to spirits and deities. Approach with respect, not demands. Offerings and reciprocity strengthen spiritual relationships.",
    "Consider the environmental impact of your practice. Biodegradable offerings, reusable tools, and ethical sourcing honor the earth.",
    "Your practice is your own, but be mindful when sharing or teaching. Not everyone is ready for every topic, and some practices require initiation.",
    "Banishing and binding have ethical nuances. Banish behaviors and energies, not people. Binding should protect, not punish.",
]
