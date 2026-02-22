"""
Complete 78-card Rider-Waite-Smith tarot deck with spread templates.

Pure Python — no external dependencies.
Data: 22 Major Arcana, 56 Minor Arcana (4 suits x 14 cards), 10 spreads.
"""

import random
from typing import Optional

# ---------------------------------------------------------------------------
# MAJOR ARCANA  (0 – XXI)
# ---------------------------------------------------------------------------

MAJOR_ARCANA: list[dict] = [
    {
        "number": 0,
        "name": "The Fool",
        "element": "air",
        "planet": "",
        "zodiac": "uranus",
        "keywords_upright": ["new beginnings", "innocence", "spontaneity", "free spirit"],
        "keywords_reversed": ["recklessness", "fear of unknown", "holding back"],
        "upright_meaning": "A leap of faith into the unknown, carrying only trust and wonder. The universe supports your fresh start.",
        "reversed_meaning": "Hesitation or reckless naivety blocks your path. Ground yourself before jumping.",
        "yes_or_no": "yes",
        "advice": "Take the leap — overthinking will not serve you here.",
        "journal_prompt": "Where in my life am I being called to begin again with a beginner's mind?",
        "correspondences": {"herbs": ["peppermint", "ginseng"], "crystals": ["clear quartz", "turquoise"], "colors": ["yellow", "sky blue"]},
    },
    {
        "number": 1,
        "name": "The Magician",
        "element": "air",
        "planet": "mercury",
        "zodiac": "",
        "keywords_upright": ["manifestation", "willpower", "resourcefulness", "skill"],
        "keywords_reversed": ["manipulation", "untapped potential", "trickery"],
        "upright_meaning": "You possess every tool you need. Channel your will and focus to manifest your vision into reality.",
        "reversed_meaning": "Talents lie dormant or are being misused. Realign intention with action.",
        "yes_or_no": "yes",
        "advice": "Stop waiting for permission — you already have what you need.",
        "journal_prompt": "What latent skill or resource am I neglecting that could change everything?",
        "correspondences": {"herbs": ["lavender", "marjoram"], "crystals": ["citrine", "tiger's eye"], "colors": ["yellow", "white"]},
    },
    {
        "number": 2,
        "name": "The High Priestess",
        "element": "water",
        "planet": "moon",
        "zodiac": "",
        "keywords_upright": ["intuition", "mystery", "subconscious", "inner wisdom"],
        "keywords_reversed": ["secrets", "disconnection from intuition", "withdrawal"],
        "upright_meaning": "Trust the quiet voice within. Hidden knowledge is surfacing — be still and receive it.",
        "reversed_meaning": "You are ignoring your intuition or hiding truths from yourself. Go inward.",
        "yes_or_no": "maybe",
        "advice": "Sit in silence for ten minutes today and listen to what arises.",
        "journal_prompt": "What truth have I been avoiding that my intuition already knows?",
        "correspondences": {"herbs": ["mugwort", "jasmine"], "crystals": ["moonstone", "lapis lazuli"], "colors": ["blue", "silver"]},
    },
    {
        "number": 3,
        "name": "The Empress",
        "element": "earth",
        "planet": "venus",
        "zodiac": "",
        "keywords_upright": ["abundance", "nurturing", "fertility", "sensuality"],
        "keywords_reversed": ["creative block", "neglect", "dependence"],
        "upright_meaning": "A season of growth and creative abundance. Nurture yourself and your projects with love.",
        "reversed_meaning": "Self-care is lacking, or you are smothering instead of nurturing. Restore balance.",
        "yes_or_no": "yes",
        "advice": "Invest in beauty and comfort today — it will replenish your creative well.",
        "journal_prompt": "How can I bring more nurturing energy into my daily life?",
        "correspondences": {"herbs": ["rose", "vervain"], "crystals": ["rose quartz", "emerald"], "colors": ["green", "pink"]},
    },
    {
        "number": 4,
        "name": "The Emperor",
        "element": "fire",
        "planet": "",
        "zodiac": "aries",
        "keywords_upright": ["authority", "structure", "stability", "leadership"],
        "keywords_reversed": ["rigidity", "tyranny", "lack of discipline"],
        "upright_meaning": "Build solid foundations through discipline and clear boundaries. Your leadership is needed.",
        "reversed_meaning": "Control has become excessive or absent. Examine where power is misused or surrendered.",
        "yes_or_no": "yes",
        "advice": "Create a concrete plan with deadlines — structure will set you free.",
        "journal_prompt": "Where do I need to establish better boundaries or stronger discipline?",
        "correspondences": {"herbs": ["cedar", "frankincense"], "crystals": ["carnelian", "red jasper"], "colors": ["red", "gold"]},
    },
    {
        "number": 5,
        "name": "The Hierophant",
        "element": "earth",
        "planet": "",
        "zodiac": "taurus",
        "keywords_upright": ["tradition", "spiritual wisdom", "mentorship", "conformity"],
        "keywords_reversed": ["rebellion", "unorthodoxy", "questioning beliefs"],
        "upright_meaning": "Seek guidance from trusted traditions or a mentor. Shared wisdom holds the key.",
        "reversed_meaning": "Blind conformity stifles your growth. Question inherited beliefs and find your own path.",
        "yes_or_no": "maybe",
        "advice": "Consult someone whose experience exceeds your own before deciding.",
        "journal_prompt": "Which inherited belief no longer serves me, and what would I replace it with?",
        "correspondences": {"herbs": ["sage", "sandalwood"], "crystals": ["lapis lazuli", "sapphire"], "colors": ["red", "grey"]},
    },
    {
        "number": 6,
        "name": "The Lovers",
        "element": "air",
        "planet": "",
        "zodiac": "gemini",
        "keywords_upright": ["love", "harmony", "partnership", "alignment"],
        "keywords_reversed": ["disharmony", "imbalance", "misalignment of values"],
        "upright_meaning": "A profound union or choice rooted in authentic values. Follow your heart with open eyes.",
        "reversed_meaning": "Conflict between head and heart, or values are out of alignment. Reassess before committing.",
        "yes_or_no": "yes",
        "advice": "Choose the option that aligns with your deepest values, not just comfort.",
        "journal_prompt": "What does true partnership — with others or with myself — look like for me?",
        "correspondences": {"herbs": ["rose", "damiana"], "crystals": ["rose quartz", "rhodonite"], "colors": ["pink", "orange"]},
    },
    {
        "number": 7,
        "name": "The Chariot",
        "element": "water",
        "planet": "",
        "zodiac": "cancer",
        "keywords_upright": ["determination", "willpower", "victory", "direction"],
        "keywords_reversed": ["lack of direction", "aggression", "loss of control"],
        "upright_meaning": "Harness opposing forces through sheer determination. Victory comes to those who steer with focus.",
        "reversed_meaning": "Willpower has become aggression, or you have lost your sense of direction entirely.",
        "yes_or_no": "yes",
        "advice": "Set one clear goal and pursue it relentlessly this week.",
        "journal_prompt": "What opposing inner forces do I need to unite to move forward?",
        "correspondences": {"herbs": ["bay laurel", "wormwood"], "crystals": ["black onyx", "chalcedony"], "colors": ["silver", "blue"]},
    },
    {
        "number": 8,
        "name": "Strength",
        "element": "fire",
        "planet": "",
        "zodiac": "leo",
        "keywords_upright": ["courage", "inner strength", "patience", "compassion"],
        "keywords_reversed": ["self-doubt", "weakness", "raw emotion"],
        "upright_meaning": "True power is gentle. Tame your fears with compassion, not force.",
        "reversed_meaning": "Self-doubt erodes your confidence, or emotions run unchecked. Reclaim your inner calm.",
        "yes_or_no": "yes",
        "advice": "Approach your hardest challenge today with gentleness instead of force.",
        "journal_prompt": "When has quiet courage served me better than loud bravado?",
        "correspondences": {"herbs": ["chamomile", "sunflower"], "crystals": ["tiger's eye", "garnet"], "colors": ["gold", "red"]},
    },
    {
        "number": 9,
        "name": "The Hermit",
        "element": "earth",
        "planet": "",
        "zodiac": "virgo",
        "keywords_upright": ["solitude", "introspection", "inner guidance", "wisdom"],
        "keywords_reversed": ["isolation", "loneliness", "withdrawal"],
        "upright_meaning": "Withdraw from noise to find your own light. Solitude is not loneliness — it is a pilgrimage inward.",
        "reversed_meaning": "Isolation has become unhealthy, or you resist necessary self-reflection.",
        "yes_or_no": "maybe",
        "advice": "Spend time alone in nature this week to reconnect with your inner voice.",
        "journal_prompt": "What wisdom is waiting for me in the quiet spaces I have been avoiding?",
        "correspondences": {"herbs": ["sage", "myrrh"], "crystals": ["amethyst", "smoky quartz"], "colors": ["grey", "dark blue"]},
    },
    {
        "number": 10,
        "name": "Wheel of Fortune",
        "element": "fire",
        "planet": "jupiter",
        "zodiac": "",
        "keywords_upright": ["cycles", "fate", "luck", "turning point"],
        "keywords_reversed": ["bad luck", "resistance to change", "stagnation"],
        "upright_meaning": "The wheel turns in your favor. Embrace the cycle and ride the momentum of change.",
        "reversed_meaning": "Resisting inevitable change creates suffering. Release control and trust the cycle.",
        "yes_or_no": "yes",
        "advice": "Accept that change is the only constant and position yourself to benefit from it.",
        "journal_prompt": "What cycle in my life is completing, and what is beginning?",
        "correspondences": {"herbs": ["clove", "star anise"], "crystals": ["labradorite", "aventurine"], "colors": ["purple", "blue"]},
    },
    {
        "number": 11,
        "name": "Justice",
        "element": "air",
        "planet": "",
        "zodiac": "libra",
        "keywords_upright": ["fairness", "truth", "accountability", "law"],
        "keywords_reversed": ["dishonesty", "injustice", "avoidance of accountability"],
        "upright_meaning": "Truth and fairness prevail. Act with integrity and accept the consequences of past choices.",
        "reversed_meaning": "Dishonesty — toward self or others — creates imbalance. Own your part.",
        "yes_or_no": "maybe",
        "advice": "Make the fair decision, even if it is not the easy one.",
        "journal_prompt": "Where have I been avoiding accountability, and what would honesty look like?",
        "correspondences": {"herbs": ["marigold", "pennyroyal"], "crystals": ["bloodstone", "jade"], "colors": ["green", "blue"]},
    },
    {
        "number": 12,
        "name": "The Hanged Man",
        "element": "water",
        "planet": "neptune",
        "zodiac": "",
        "keywords_upright": ["surrender", "new perspective", "letting go", "pause"],
        "keywords_reversed": ["stalling", "martyrdom", "resistance"],
        "upright_meaning": "Suspend action and see from a new angle. What feels like delay is actually deep transformation.",
        "reversed_meaning": "You are stalling out of fear or playing the martyr. Stop resisting the pause.",
        "yes_or_no": "maybe",
        "advice": "Stop pushing — let the answer come to you instead.",
        "journal_prompt": "What would I see differently if I stopped trying to control the outcome?",
        "correspondences": {"herbs": ["lotus", "willow bark"], "crystals": ["aquamarine", "beryl"], "colors": ["blue", "grey"]},
    },
    {
        "number": 13,
        "name": "Death",
        "element": "water",
        "planet": "",
        "zodiac": "scorpio",
        "keywords_upright": ["transformation", "endings", "rebirth", "release"],
        "keywords_reversed": ["resistance to change", "stagnation", "fear of endings"],
        "upright_meaning": "A chapter closes to make way for profound renewal. Let what is dying fall away gracefully.",
        "reversed_meaning": "Clinging to what has expired creates decay. Release the old to welcome the new.",
        "yes_or_no": "no",
        "advice": "Identify one thing you have outgrown and consciously release it today.",
        "journal_prompt": "What must I allow to die so that something new can be born?",
        "correspondences": {"herbs": ["cypress", "wormwood"], "crystals": ["obsidian", "black tourmaline"], "colors": ["black", "white"]},
    },
    {
        "number": 14,
        "name": "Temperance",
        "element": "fire",
        "planet": "",
        "zodiac": "sagittarius",
        "keywords_upright": ["balance", "moderation", "patience", "alchemy"],
        "keywords_reversed": ["excess", "imbalance", "impatience"],
        "upright_meaning": "Blend opposing forces with patience and care. Moderation and steady alchemy create lasting results.",
        "reversed_meaning": "Excess in one area starves another. Rebalance before burnout arrives.",
        "yes_or_no": "yes",
        "advice": "Find the middle path in a situation where you have been leaning to extremes.",
        "journal_prompt": "Where is my life out of balance, and what small adjustment could restore harmony?",
        "correspondences": {"herbs": ["echinacea", "elderflower"], "crystals": ["amethyst", "kunzite"], "colors": ["blue", "purple"]},
    },
    {
        "number": 15,
        "name": "The Devil",
        "element": "earth",
        "planet": "",
        "zodiac": "capricorn",
        "keywords_upright": ["bondage", "shadow self", "attachment", "materialism"],
        "keywords_reversed": ["release", "breaking free", "reclaiming power"],
        "upright_meaning": "Unhealthy attachments or illusions of powerlessness bind you. The chains are looser than you think.",
        "reversed_meaning": "You are beginning to break free from a destructive pattern. Keep going.",
        "yes_or_no": "no",
        "advice": "Name the habit or attachment that controls you — awareness is the first key to freedom.",
        "journal_prompt": "What am I pretending I cannot change, and what would freedom look like?",
        "correspondences": {"herbs": ["patchouli", "black pepper"], "crystals": ["black obsidian", "smoky quartz"], "colors": ["black", "dark red"]},
    },
    {
        "number": 16,
        "name": "The Tower",
        "element": "fire",
        "planet": "mars",
        "zodiac": "",
        "keywords_upright": ["upheaval", "sudden change", "revelation", "liberation"],
        "keywords_reversed": ["fear of change", "avoidance", "delayed disaster"],
        "upright_meaning": "Structures built on false foundations crumble. The destruction clears ground for something true.",
        "reversed_meaning": "You sense the cracks but avoid confronting them. Delayed collapse only magnifies the fall.",
        "yes_or_no": "no",
        "advice": "Let the false thing fall — rebuilding on truth is always worth the rubble.",
        "journal_prompt": "What in my life is built on shaky ground, and what truth am I avoiding?",
        "correspondences": {"herbs": ["dragon's blood", "nettle"], "crystals": ["ruby", "garnet"], "colors": ["red", "grey"]},
    },
    {
        "number": 17,
        "name": "The Star",
        "element": "air",
        "planet": "",
        "zodiac": "aquarius",
        "keywords_upright": ["hope", "inspiration", "renewal", "serenity"],
        "keywords_reversed": ["despair", "disconnection", "lack of faith"],
        "upright_meaning": "After the storm, peace. Pour your healing gifts freely — hope and inspiration return in abundance.",
        "reversed_meaning": "Faith has dimmed and you feel disconnected from purpose. Seek even the smallest light.",
        "yes_or_no": "yes",
        "advice": "Do one thing today purely for the joy of it, with no outcome attached.",
        "journal_prompt": "What gives me hope even in dark times, and how can I nurture that light?",
        "correspondences": {"herbs": ["lavender", "chamomile"], "crystals": ["aquamarine", "celestite"], "colors": ["light blue", "silver"]},
    },
    {
        "number": 18,
        "name": "The Moon",
        "element": "water",
        "planet": "",
        "zodiac": "pisces",
        "keywords_upright": ["illusion", "fear", "subconscious", "intuition"],
        "keywords_reversed": ["clarity", "release of fear", "truth emerging"],
        "upright_meaning": "Things are not as they seem. Navigate by intuition through the fog — fear distorts perception.",
        "reversed_meaning": "Illusions dissolve and suppressed truths surface. Clarity returns gradually.",
        "yes_or_no": "no",
        "advice": "Do not make major decisions until the fog lifts — trust the process.",
        "journal_prompt": "What fear is distorting my perception, and what might I see without it?",
        "correspondences": {"herbs": ["mugwort", "poppy"], "crystals": ["moonstone", "selenite"], "colors": ["violet", "silver"]},
    },
    {
        "number": 19,
        "name": "The Sun",
        "element": "fire",
        "planet": "sun",
        "zodiac": "",
        "keywords_upright": ["joy", "success", "vitality", "clarity"],
        "keywords_reversed": ["temporary setback", "dimmed enthusiasm", "delayed success"],
        "upright_meaning": "Radiant success and joy illuminate your path. Bask in the warmth — you have earned this light.",
        "reversed_meaning": "The sun is momentarily behind clouds. Success is not lost, only briefly delayed.",
        "yes_or_no": "yes",
        "advice": "Celebrate your wins today, no matter how small they seem.",
        "journal_prompt": "What brings me the purest, most uncomplicated joy?",
        "correspondences": {"herbs": ["sunflower", "St. John's wort"], "crystals": ["sunstone", "amber"], "colors": ["yellow", "orange"]},
    },
    {
        "number": 20,
        "name": "Judgement",
        "element": "fire",
        "planet": "pluto",
        "zodiac": "",
        "keywords_upright": ["rebirth", "inner calling", "absolution", "reckoning"],
        "keywords_reversed": ["self-doubt", "refusal of calling", "harsh self-judgment"],
        "upright_meaning": "A higher calling summons you to rise. Forgive the past and answer the call of your truest self.",
        "reversed_meaning": "You doubt your worthiness or refuse the transformation being asked of you.",
        "yes_or_no": "yes",
        "advice": "Forgive yourself for one past mistake and commit to moving forward today.",
        "journal_prompt": "What is my soul calling me toward, and what must I forgive to get there?",
        "correspondences": {"herbs": ["frankincense", "lotus"], "crystals": ["labradorite", "moldavite"], "colors": ["gold", "white"]},
    },
    {
        "number": 21,
        "name": "The World",
        "element": "earth",
        "planet": "saturn",
        "zodiac": "",
        "keywords_upright": ["completion", "integration", "fulfillment", "wholeness"],
        "keywords_reversed": ["incompletion", "shortcuts", "lack of closure"],
        "upright_meaning": "A grand cycle reaches fulfillment. You are whole, accomplished, and ready for the next spiral.",
        "reversed_meaning": "Something remains unfinished, blocking your sense of completion. Tie up loose ends.",
        "yes_or_no": "yes",
        "advice": "Finish what you started before beginning anything new.",
        "journal_prompt": "What cycle is completing in my life, and how do I honor that achievement?",
        "correspondences": {"herbs": ["bay laurel", "comfrey"], "crystals": ["lapis lazuli", "opal"], "colors": ["blue", "violet"]},
    },
]

# ---------------------------------------------------------------------------
# MINOR ARCANA  (56 cards — 4 suits x 14 cards each)
# ---------------------------------------------------------------------------

MINOR_ARCANA: dict = {
    "wands": {
        "element": "fire",
        "themes": ["passion", "creativity", "action", "will"],
        "cards": [
            {"name": "Ace of Wands", "number": 1, "keywords_upright": ["inspiration", "new venture", "creative spark"], "keywords_reversed": ["delays", "lack of motivation", "missed opportunity"], "upright_meaning": "A powerful surge of creative energy ignites a new venture.", "reversed_meaning": "A promising idea stalls due to hesitation or poor timing.", "yes_or_no": "yes", "advice": "Act on the spark of inspiration before it fades."},
            {"name": "Two of Wands", "number": 2, "keywords_upright": ["planning", "future vision", "discovery"], "keywords_reversed": ["fear of unknown", "poor planning", "indecision"], "upright_meaning": "You stand at the threshold of expansion — plan your bold next move.", "reversed_meaning": "Fear of the unknown keeps you stuck in a safe but stagnant position.", "yes_or_no": "maybe", "advice": "Map out your vision before taking the next step."},
            {"name": "Three of Wands", "number": 3, "keywords_upright": ["expansion", "foresight", "momentum"], "keywords_reversed": ["setbacks", "frustration", "limited vision"], "upright_meaning": "Your ships are coming in — expansion and progress are underway.", "reversed_meaning": "Delays in expected progress test your patience.", "yes_or_no": "yes", "advice": "Keep your eyes on the horizon; momentum is building."},
            {"name": "Four of Wands", "number": 4, "keywords_upright": ["celebration", "harmony", "homecoming"], "keywords_reversed": ["instability", "lack of support", "tension at home"], "upright_meaning": "A joyous milestone deserves celebration with those you love.", "reversed_meaning": "Domestic tension or lack of stability undermines your happiness.", "yes_or_no": "yes", "advice": "Pause to celebrate how far you have come."},
            {"name": "Five of Wands", "number": 5, "keywords_upright": ["conflict", "competition", "disagreement"], "keywords_reversed": ["avoidance", "compromise", "end of conflict"], "upright_meaning": "Healthy competition or clashing egos demand you stand your ground.", "reversed_meaning": "The conflict is resolving, or you are avoiding necessary confrontation.", "yes_or_no": "no", "advice": "Engage the conflict directly rather than letting it fester."},
            {"name": "Six of Wands", "number": 6, "keywords_upright": ["victory", "recognition", "public acclaim"], "keywords_reversed": ["fall from grace", "ego", "lack of recognition"], "upright_meaning": "Public recognition and triumph reward your efforts.", "reversed_meaning": "Success breeds ego, or your achievements go unnoticed.", "yes_or_no": "yes", "advice": "Accept praise graciously and share credit generously."},
            {"name": "Seven of Wands", "number": 7, "keywords_upright": ["perseverance", "defense", "standing your ground"], "keywords_reversed": ["overwhelm", "giving up", "being overrun"], "upright_meaning": "Hold your position — challenges test your resolve but you have the high ground.", "reversed_meaning": "The pressure feels insurmountable and you consider surrendering.", "yes_or_no": "yes", "advice": "Defend what matters to you, even when outnumbered."},
            {"name": "Eight of Wands", "number": 8, "keywords_upright": ["swift action", "momentum", "rapid progress"], "keywords_reversed": ["delays", "scattered energy", "miscommunication"], "upright_meaning": "Events accelerate rapidly — act decisively while momentum carries you.", "reversed_meaning": "Messages cross, plans stall, and energy scatters in too many directions.", "yes_or_no": "yes", "advice": "Strike while the iron is hot; speed is your ally now."},
            {"name": "Nine of Wands", "number": 9, "keywords_upright": ["resilience", "persistence", "last stand"], "keywords_reversed": ["exhaustion", "stubbornness", "paranoia"], "upright_meaning": "You are weary but not defeated — one final push brings you through.", "reversed_meaning": "Stubbornness masquerades as resilience; know when to rest.", "yes_or_no": "yes", "advice": "Rest if you must, but do not quit — the finish line is near."},
            {"name": "Ten of Wands", "number": 10, "keywords_upright": ["burden", "overcommitment", "hard work"], "keywords_reversed": ["delegation", "release", "burnout"], "upright_meaning": "You carry too much alone — the load is unsustainable.", "reversed_meaning": "You are finally learning to delegate or drop what does not serve you.", "yes_or_no": "no", "advice": "Put down at least one burden that is not truly yours to carry."},
            {"name": "Page of Wands", "number": 11, "keywords_upright": ["enthusiasm", "exploration", "new ideas"], "keywords_reversed": ["impatience", "lack of direction", "scattered energy"], "upright_meaning": "A youthful burst of enthusiasm fuels exploration and discovery.", "reversed_meaning": "Excitement without focus leads to half-finished projects.", "yes_or_no": "yes", "advice": "Follow your curiosity but commit to one idea at a time."},
            {"name": "Knight of Wands", "number": 12, "keywords_upright": ["adventure", "boldness", "passion"], "keywords_reversed": ["impulsiveness", "recklessness", "hot temper"], "upright_meaning": "Charge forward with passion and daring — adventure awaits the bold.", "reversed_meaning": "Impulsive action without forethought leads to avoidable consequences.", "yes_or_no": "yes", "advice": "Channel your fire into purposeful action, not impulsive reaction."},
            {"name": "Queen of Wands", "number": 13, "keywords_upright": ["confidence", "warmth", "determination"], "keywords_reversed": ["jealousy", "insecurity", "selfishness"], "upright_meaning": "Radiate confidence and inspire others with your warmth and creative vision.", "reversed_meaning": "Insecurity turns warmth into jealousy or controlling behavior.", "yes_or_no": "yes", "advice": "Lead with warmth and watch others rise to match your energy."},
            {"name": "King of Wands", "number": 14, "keywords_upright": ["leadership", "vision", "entrepreneurship"], "keywords_reversed": ["tyranny", "vagueness", "ruthlessness"], "upright_meaning": "Bold visionary leadership inspires action and transforms ideas into empires.", "reversed_meaning": "Visionary energy devolves into arrogance or empty promises.", "yes_or_no": "yes", "advice": "Lead by example and let your actions speak louder than your plans."},
        ],
    },
    "cups": {
        "element": "water",
        "themes": ["emotions", "relationships", "intuition", "creativity"],
        "cards": [
            {"name": "Ace of Cups", "number": 1, "keywords_upright": ["new love", "emotional awakening", "compassion"], "keywords_reversed": ["emotional loss", "blocked feelings", "emptiness"], "upright_meaning": "A new wave of love, joy, or creative inspiration overflows from within.", "reversed_meaning": "Emotions are blocked or a promising connection fails to materialize.", "yes_or_no": "yes", "advice": "Open your heart to receive the love and joy being offered."},
            {"name": "Two of Cups", "number": 2, "keywords_upright": ["partnership", "mutual attraction", "unity"], "keywords_reversed": ["imbalance", "breakup", "disharmony"], "upright_meaning": "A beautiful partnership forms through mutual respect and genuine connection.", "reversed_meaning": "An imbalance in giving and receiving strains a relationship.", "yes_or_no": "yes", "advice": "Invest equally in the partnership that matters most to you."},
            {"name": "Three of Cups", "number": 3, "keywords_upright": ["celebration", "friendship", "community"], "keywords_reversed": ["overindulgence", "gossip", "isolation"], "upright_meaning": "Joyful celebration with friends and community lifts your spirits.", "reversed_meaning": "Social excess or exclusion dampens the joy of connection.", "yes_or_no": "yes", "advice": "Gather your people and celebrate together."},
            {"name": "Four of Cups", "number": 4, "keywords_upright": ["contemplation", "apathy", "reevaluation"], "keywords_reversed": ["renewed interest", "new awareness", "acceptance"], "upright_meaning": "Boredom or dissatisfaction hides an unnoticed opportunity right in front of you.", "reversed_meaning": "You finally notice what you have been overlooking and re-engage.", "yes_or_no": "no", "advice": "Look at what is being offered before dismissing it."},
            {"name": "Five of Cups", "number": 5, "keywords_upright": ["grief", "loss", "regret"], "keywords_reversed": ["acceptance", "moving on", "finding peace"], "upright_meaning": "Grief over what was lost blinds you to what still remains.", "reversed_meaning": "You begin to accept loss and turn toward what can still be saved.", "yes_or_no": "no", "advice": "Mourn what is gone, then turn around and see what remains."},
            {"name": "Six of Cups", "number": 6, "keywords_upright": ["nostalgia", "innocence", "reunion"], "keywords_reversed": ["stuck in past", "unrealistic memories", "moving forward"], "upright_meaning": "Sweet memories and past connections offer comfort and healing.", "reversed_meaning": "Clinging to an idealized past prevents present growth.", "yes_or_no": "yes", "advice": "Honor your memories without letting them anchor you to the past."},
            {"name": "Seven of Cups", "number": 7, "keywords_upright": ["fantasy", "choices", "wishful thinking"], "keywords_reversed": ["clarity", "decisiveness", "reality check"], "upright_meaning": "Many alluring options shimmer before you, but not all are real — choose wisely.", "reversed_meaning": "The fog of fantasy clears, revealing which options are truly viable.", "yes_or_no": "no", "advice": "Test your dreams against reality before committing resources."},
            {"name": "Eight of Cups", "number": 8, "keywords_upright": ["walking away", "seeking deeper meaning", "disillusionment"], "keywords_reversed": ["fear of change", "stagnation", "clinging"], "upright_meaning": "What once fulfilled you no longer does — it is time to walk away and seek deeper meaning.", "reversed_meaning": "You know you should leave but fear of the unknown keeps you.", "yes_or_no": "no", "advice": "Have the courage to leave what no longer fulfills you."},
            {"name": "Nine of Cups", "number": 9, "keywords_upright": ["wish fulfillment", "contentment", "satisfaction"], "keywords_reversed": ["dissatisfaction", "greed", "materialism"], "upright_meaning": "Your deepest wish is granted — savor this moment of genuine contentment.", "reversed_meaning": "Getting what you wanted brings less joy than expected.", "yes_or_no": "yes", "advice": "Enjoy what you have achieved without immediately wanting more."},
            {"name": "Ten of Cups", "number": 10, "keywords_upright": ["emotional fulfillment", "family harmony", "lasting happiness"], "keywords_reversed": ["broken family", "disharmony", "misaligned values"], "upright_meaning": "Deep, lasting happiness rooted in loving relationships and emotional wholeness.", "reversed_meaning": "Family conflict or broken bonds disrupt your sense of belonging.", "yes_or_no": "yes", "advice": "Nurture the relationships that form the foundation of your happiness."},
            {"name": "Page of Cups", "number": 11, "keywords_upright": ["creative message", "intuitive nudge", "curiosity"], "keywords_reversed": ["emotional immaturity", "creative block", "escapism"], "upright_meaning": "A gentle intuitive message or creative inspiration surfaces unexpectedly.", "reversed_meaning": "Emotional immaturity or escapism clouds your creative instincts.", "yes_or_no": "yes", "advice": "Pay attention to the quiet creative nudges appearing in your life."},
            {"name": "Knight of Cups", "number": 12, "keywords_upright": ["romance", "charm", "following the heart"], "keywords_reversed": ["moodiness", "unrealistic expectations", "jealousy"], "upright_meaning": "A romantic or creative pursuit calls you to follow your heart with grace.", "reversed_meaning": "Idealism turns to moodiness when reality fails to match the fantasy.", "yes_or_no": "yes", "advice": "Follow your heart but keep at least one foot on the ground."},
            {"name": "Queen of Cups", "number": 13, "keywords_upright": ["compassion", "emotional security", "intuition"], "keywords_reversed": ["codependency", "emotional manipulation", "insecurity"], "upright_meaning": "Deep emotional wisdom and compassion flow freely to those around you.", "reversed_meaning": "Caretaking becomes codependency, or emotions are used to manipulate.", "yes_or_no": "yes", "advice": "Hold space for others without absorbing their pain as your own."},
            {"name": "King of Cups", "number": 14, "keywords_upright": ["emotional balance", "diplomacy", "calm authority"], "keywords_reversed": ["emotional volatility", "manipulation", "coldness"], "upright_meaning": "Mastery of emotion allows you to lead with calm, compassionate authority.", "reversed_meaning": "Suppressed emotions erupt unexpectedly, or detachment becomes coldness.", "yes_or_no": "yes", "advice": "Lead with emotional intelligence — feel deeply but respond thoughtfully."},
        ],
    },
    "swords": {
        "element": "air",
        "themes": ["intellect", "truth", "conflict", "communication"],
        "cards": [
            {"name": "Ace of Swords", "number": 1, "keywords_upright": ["clarity", "breakthrough", "truth"], "keywords_reversed": ["confusion", "misinformation", "clouded judgment"], "upright_meaning": "A powerful moment of mental clarity cuts through confusion to reveal truth.", "reversed_meaning": "Clouded thinking or misinformation leads you astray.", "yes_or_no": "yes", "advice": "Speak the truth clearly, even if your voice shakes."},
            {"name": "Two of Swords", "number": 2, "keywords_upright": ["indecision", "stalemate", "difficult choice"], "keywords_reversed": ["information overload", "lesser of two evils", "avoidance"], "upright_meaning": "A difficult choice demands you remove the blindfold and face facts.", "reversed_meaning": "Avoiding a decision only prolongs the painful stalemate.", "yes_or_no": "maybe", "advice": "Gather the facts you need and make the call — indecision is a choice too."},
            {"name": "Three of Swords", "number": 3, "keywords_upright": ["heartbreak", "sorrow", "painful truth"], "keywords_reversed": ["recovery", "forgiveness", "releasing pain"], "upright_meaning": "Painful truth pierces the heart, but honest grief is the path to healing.", "reversed_meaning": "The worst of the heartbreak is passing and recovery begins.", "yes_or_no": "no", "advice": "Allow yourself to grieve fully so healing can begin."},
            {"name": "Four of Swords", "number": 4, "keywords_upright": ["rest", "recovery", "contemplation"], "keywords_reversed": ["restlessness", "burnout", "forced activity"], "upright_meaning": "Retreat and rest are essential now — your mind and body demand recovery.", "reversed_meaning": "You resist necessary rest, pushing yourself toward collapse.", "yes_or_no": "maybe", "advice": "Rest is not laziness — it is preparation for what comes next."},
            {"name": "Five of Swords", "number": 5, "keywords_upright": ["conflict", "defeat", "hollow victory"], "keywords_reversed": ["reconciliation", "moving on", "cutting losses"], "upright_meaning": "Winning this battle costs more than losing it — consider the true price of victory.", "reversed_meaning": "The fight is over; it is time to make peace and move on.", "yes_or_no": "no", "advice": "Ask yourself whether winning this fight is worth what it costs."},
            {"name": "Six of Swords", "number": 6, "keywords_upright": ["transition", "moving on", "calmer waters"], "keywords_reversed": ["resistance to change", "unfinished business", "baggage"], "upright_meaning": "You are leaving troubled waters behind and heading toward peace.", "reversed_meaning": "Emotional baggage weighs down your journey to calmer shores.", "yes_or_no": "yes", "advice": "Keep moving forward — better shores are ahead."},
            {"name": "Seven of Swords", "number": 7, "keywords_upright": ["deception", "strategy", "stealth"], "keywords_reversed": ["exposure", "confession", "coming clean"], "upright_meaning": "Someone is not being forthright — use discernment and protect yourself.", "reversed_meaning": "Hidden truths are exposed, or conscience demands confession.", "yes_or_no": "no", "advice": "Examine who might not be telling you the full story, including yourself."},
            {"name": "Eight of Swords", "number": 8, "keywords_upright": ["restriction", "helplessness", "self-imposed limitation"], "keywords_reversed": ["freedom", "new perspective", "empowerment"], "upright_meaning": "You feel trapped, but the bonds are largely self-imposed — open your eyes.", "reversed_meaning": "You realize the prison was an illusion and begin to free yourself.", "yes_or_no": "no", "advice": "Challenge the belief that you are stuck — most of the barriers are mental."},
            {"name": "Nine of Swords", "number": 9, "keywords_upright": ["anxiety", "nightmares", "despair"], "keywords_reversed": ["hope", "recovery", "reaching out"], "upright_meaning": "Anxiety and worst-case thinking torment you — the fear is worse than reality.", "reversed_meaning": "The darkest night is ending; reaching out for help brings relief.", "yes_or_no": "no", "advice": "Talk to someone you trust about what keeps you up at night."},
            {"name": "Ten of Swords", "number": 10, "keywords_upright": ["painful ending", "rock bottom", "betrayal"], "keywords_reversed": ["recovery", "regeneration", "worst is over"], "upright_meaning": "A painful ending or betrayal marks rock bottom — the only way now is up.", "reversed_meaning": "The worst is behind you and the slow process of recovery begins.", "yes_or_no": "no", "advice": "Accept that this chapter is over and begin to heal."},
            {"name": "Page of Swords", "number": 11, "keywords_upright": ["curiosity", "new ideas", "communication"], "keywords_reversed": ["gossip", "scattered thoughts", "cynicism"], "upright_meaning": "A keen, curious mind thirsts for knowledge and new perspectives.", "reversed_meaning": "Mental restlessness turns into gossip or unconstructive criticism.", "yes_or_no": "maybe", "advice": "Channel your sharp mind into learning something genuinely new."},
            {"name": "Knight of Swords", "number": 12, "keywords_upright": ["ambition", "action", "fast thinking"], "keywords_reversed": ["impulsiveness", "aggression", "no direction"], "upright_meaning": "Swift, decisive action fueled by sharp intellect charges toward the goal.", "reversed_meaning": "Haste without thought creates collateral damage.", "yes_or_no": "yes", "advice": "Move quickly but think first — speed without direction is just chaos."},
            {"name": "Queen of Swords", "number": 13, "keywords_upright": ["independence", "clear boundaries", "direct communication"], "keywords_reversed": ["coldness", "bitterness", "overly critical"], "upright_meaning": "Clear thinking and honest communication cut through pretense with compassion.", "reversed_meaning": "Emotional walls and sharp words push people away unnecessarily.", "yes_or_no": "yes", "advice": "Speak your truth clearly and without apology, but with kindness."},
            {"name": "King of Swords", "number": 14, "keywords_upright": ["intellectual authority", "truth", "ethical leadership"], "keywords_reversed": ["manipulation", "cruelty", "abuse of power"], "upright_meaning": "Authority wielded through clear thinking, fairness, and unwavering ethics.", "reversed_meaning": "Intellect divorced from empathy becomes cold manipulation.", "yes_or_no": "maybe", "advice": "Make decisions based on principles, not emotions or politics."},
        ],
    },
    "pentacles": {
        "element": "earth",
        "themes": ["material world", "finances", "health", "manifestation"],
        "cards": [
            {"name": "Ace of Pentacles", "number": 1, "keywords_upright": ["new opportunity", "prosperity", "manifestation"], "keywords_reversed": ["missed chance", "poor planning", "scarcity mindset"], "upright_meaning": "A tangible new opportunity for wealth, health, or stability appears.", "reversed_meaning": "A promising material opportunity slips away due to poor planning.", "yes_or_no": "yes", "advice": "Seize the practical opportunity in front of you and build on it."},
            {"name": "Two of Pentacles", "number": 2, "keywords_upright": ["balance", "adaptability", "juggling priorities"], "keywords_reversed": ["overwhelm", "disorganization", "financial stress"], "upright_meaning": "You skillfully juggle multiple demands — stay flexible and keep your rhythm.", "reversed_meaning": "Too many balls in the air leads to dropped responsibilities.", "yes_or_no": "maybe", "advice": "Prioritize ruthlessly and let the non-essentials wait."},
            {"name": "Three of Pentacles", "number": 3, "keywords_upright": ["teamwork", "craftsmanship", "collaboration"], "keywords_reversed": ["poor teamwork", "mediocrity", "lack of effort"], "upright_meaning": "Skilled collaboration produces work of genuine quality and lasting value.", "reversed_meaning": "Poor communication or lack of effort undermines the team's potential.", "yes_or_no": "yes", "advice": "Seek expert input and collaborate — the result will exceed solo effort."},
            {"name": "Four of Pentacles", "number": 4, "keywords_upright": ["security", "conservation", "control"], "keywords_reversed": ["greed", "hoarding", "fear of loss"], "upright_meaning": "Protecting resources wisely, but guard against gripping too tightly.", "reversed_meaning": "Fear of loss leads to hoarding that blocks the flow of abundance.", "yes_or_no": "maybe", "advice": "Save wisely, but do not let fear of scarcity control your generosity."},
            {"name": "Five of Pentacles", "number": 5, "keywords_upright": ["hardship", "loss", "isolation"], "keywords_reversed": ["recovery", "help arriving", "end of hard times"], "upright_meaning": "Material hardship or feeling left out in the cold — help is closer than you think.", "reversed_meaning": "The worst of the hardship passes and support becomes available.", "yes_or_no": "no", "advice": "Swallow your pride and ask for help — it is available."},
            {"name": "Six of Pentacles", "number": 6, "keywords_upright": ["generosity", "giving and receiving", "charity"], "keywords_reversed": ["strings attached", "power imbalance", "one-sided generosity"], "upright_meaning": "Generosity flows in both directions — give freely and receive graciously.", "reversed_meaning": "Giving comes with hidden strings, or the balance of exchange is unfair.", "yes_or_no": "yes", "advice": "Give without expectation and receive without guilt."},
            {"name": "Seven of Pentacles", "number": 7, "keywords_upright": ["patience", "long-term investment", "assessment"], "keywords_reversed": ["impatience", "wasted effort", "poor returns"], "upright_meaning": "Seeds you planted are growing — be patient and assess your progress.", "reversed_meaning": "Impatience or poor strategy threatens to waste your investment.", "yes_or_no": "maybe", "advice": "Review your long-term investments and adjust course if needed."},
            {"name": "Eight of Pentacles", "number": 8, "keywords_upright": ["diligence", "mastery", "skill development"], "keywords_reversed": ["perfectionism", "shortcuts", "lack of ambition"], "upright_meaning": "Dedicated practice and attention to craft build mastery over time.", "reversed_meaning": "Shortcuts undermine quality, or perfectionism stalls progress.", "yes_or_no": "yes", "advice": "Commit to the daily practice that builds real expertise."},
            {"name": "Nine of Pentacles", "number": 9, "keywords_upright": ["abundance", "independence", "self-sufficiency"], "keywords_reversed": ["financial dependence", "overwork", "superficiality"], "upright_meaning": "You enjoy the fruits of your labor in elegant self-sufficiency.", "reversed_meaning": "Luxury masks loneliness, or independence was bought at too high a cost.", "yes_or_no": "yes", "advice": "Enjoy what you have built and let yourself rest in abundance."},
            {"name": "Ten of Pentacles", "number": 10, "keywords_upright": ["legacy", "inheritance", "long-term success"], "keywords_reversed": ["family disputes", "financial failure", "loss of legacy"], "upright_meaning": "Lasting wealth and legacy provide security for generations to come.", "reversed_meaning": "Family disputes or poor planning threaten the legacy you have built.", "yes_or_no": "yes", "advice": "Think about what you are building that will outlast you."},
            {"name": "Page of Pentacles", "number": 11, "keywords_upright": ["ambition", "studiousness", "new financial opportunity"], "keywords_reversed": ["laziness", "procrastination", "missed opportunity"], "upright_meaning": "An eager student of life, ready to turn knowledge into tangible results.", "reversed_meaning": "Procrastination or lack of follow-through wastes a promising start.", "yes_or_no": "yes", "advice": "Start learning the practical skill you have been putting off."},
            {"name": "Knight of Pentacles", "number": 12, "keywords_upright": ["reliability", "hard work", "steady progress"], "keywords_reversed": ["stubbornness", "stagnation", "boredom"], "upright_meaning": "Slow, steady, and utterly reliable effort builds unshakable results.", "reversed_meaning": "Routine becomes rut, and reliability hardens into stubbornness.", "yes_or_no": "yes", "advice": "Keep showing up consistently — reliability is your superpower."},
            {"name": "Queen of Pentacles", "number": 13, "keywords_upright": ["nurturing abundance", "practicality", "comfort"], "keywords_reversed": ["smothering", "work-life imbalance", "neglect of self"], "upright_meaning": "Abundant nurturing creates a warm, secure environment for all to thrive.", "reversed_meaning": "Giving too much to others depletes your own reserves.", "yes_or_no": "yes", "advice": "Create a comfortable sanctuary and nourish yourself as well as others."},
            {"name": "King of Pentacles", "number": 14, "keywords_upright": ["wealth", "business acumen", "stability"], "keywords_reversed": ["greed", "materialism", "financial mismanagement"], "upright_meaning": "Masterful stewardship of resources builds enduring wealth and security.", "reversed_meaning": "The pursuit of wealth overshadows what truly matters.", "yes_or_no": "yes", "advice": "Build wealth that serves your values, not the other way around."},
        ],
    },
}

# ---------------------------------------------------------------------------
# SPREADS  (10 templates)
# ---------------------------------------------------------------------------

SPREADS: dict = {
    "single_card": {
        "name": "Single Card",
        "card_count": 1,
        "positions": ["Guidance"],
        "best_for": "quick daily insight",
        "description": "Draw one card for immediate clarity on a question or as a daily meditation focus.",
    },
    "three_card": {
        "name": "Past, Present, Future",
        "card_count": 3,
        "positions": ["Past", "Present", "Future"],
        "best_for": "understanding a situation's trajectory",
        "description": "A classic three-card layout revealing the energies of what was, what is, and what is coming.",
    },
    "celtic_cross": {
        "name": "Celtic Cross",
        "card_count": 10,
        "positions": [
            "Present situation",
            "Challenge or crossing energy",
            "Subconscious foundation",
            "Recent past",
            "Best possible outcome",
            "Near future",
            "Your attitude",
            "External influences",
            "Hopes and fears",
            "Final outcome",
        ],
        "best_for": "in-depth analysis of a complex situation",
        "description": "The most comprehensive traditional spread, revealing layers of influence from subconscious to outcome.",
    },
    "relationship": {
        "name": "Relationship Spread",
        "card_count": 7,
        "positions": [
            "You in the relationship",
            "The other person",
            "The connection between you",
            "Strengths of the relationship",
            "Challenges to address",
            "What you both need",
            "Where this is heading",
        ],
        "best_for": "understanding any relationship dynamic",
        "description": "Explore the dynamics, strengths, challenges, and trajectory of a relationship.",
    },
    "decision": {
        "name": "Two Paths Spread",
        "card_count": 5,
        "positions": [
            "The heart of the decision",
            "Path A — what unfolds",
            "Path A — outcome",
            "Path B — what unfolds",
            "Path B — outcome",
        ],
        "best_for": "choosing between two options",
        "description": "Lay out two potential paths side by side to illuminate the likely unfolding and outcome of each.",
    },
    "shadow_work": {
        "name": "Shadow Work Spread",
        "card_count": 5,
        "positions": [
            "The shadow aspect surfacing",
            "How it originated",
            "How it manifests in daily life",
            "The gift hidden within it",
            "How to integrate it",
        ],
        "best_for": "deep self-exploration and inner healing",
        "description": "Illuminate a hidden aspect of the psyche, trace its origins, and discover the gift within the shadow.",
    },
    "new_moon": {
        "name": "New Moon Spread",
        "card_count": 6,
        "positions": [
            "Seed energy of this cycle",
            "What to release from last cycle",
            "Intention to plant",
            "Action to nurture growth",
            "Obstacle to be aware of",
            "Harvest to expect",
        ],
        "best_for": "new moon intention setting",
        "description": "Align with the new moon cycle by clarifying intentions, necessary releases, and the growth ahead.",
    },
    "full_moon": {
        "name": "Full Moon Spread",
        "card_count": 5,
        "positions": [
            "What is illuminated",
            "What has reached fullness",
            "What must be released",
            "Gratitude focus",
            "Guidance for the waning phase",
        ],
        "best_for": "full moon reflection and release",
        "description": "Harness the full moon's illumination to see clearly, celebrate completion, and release what no longer serves.",
    },
    "sabbat": {
        "name": "Sabbat / Seasonal Spread",
        "card_count": 4,
        "positions": [
            "Theme of this season",
            "Lesson to learn",
            "Energy to embody",
            "Gift this season brings",
        ],
        "best_for": "Wheel of the Year sabbat celebrations",
        "description": "Attune to the energy of the current sabbat or seasonal turning point on the Wheel of the Year.",
    },
    "year_ahead": {
        "name": "Year Ahead Spread",
        "card_count": 13,
        "positions": [
            "Overall theme of the year",
            "January", "February", "March",
            "April", "May", "June",
            "July", "August", "September",
            "October", "November", "December",
        ],
        "best_for": "yearly overview and planning",
        "description": "A card for each month plus an overarching theme, mapping the energies of the entire year ahead.",
    },
}

# ---------------------------------------------------------------------------
# INTENTION -> KEYWORD MAPPING  (used by get_cards_for_intention)
# ---------------------------------------------------------------------------

_INTENTION_KEYWORDS: dict[str, list[str]] = {
    "love": ["love", "partnership", "romance", "mutual attraction", "compassion", "harmony", "emotional fulfillment"],
    "career": ["ambition", "leadership", "entrepreneurship", "hard work", "mastery", "new venture", "skill development"],
    "money": ["prosperity", "wealth", "abundance", "manifestation", "financial", "inheritance", "new opportunity"],
    "healing": ["recovery", "forgiveness", "release", "compassion", "renewal", "inner strength", "acceptance"],
    "creativity": ["creative spark", "inspiration", "new ideas", "enthusiasm", "creative", "vision"],
    "guidance": ["wisdom", "intuition", "inner guidance", "clarity", "truth", "contemplation"],
    "protection": ["defense", "boundaries", "security", "standing your ground", "resilience", "strength"],
    "transformation": ["transformation", "rebirth", "endings", "change", "turning point", "liberation"],
    "abundance": ["abundance", "prosperity", "wealth", "generosity", "self-sufficiency", "long-term success"],
    "confidence": ["confidence", "willpower", "courage", "determination", "victory", "leadership"],
    "peace": ["serenity", "balance", "harmony", "rest", "calm", "moderation", "contentment"],
    "spirituality": ["spiritual wisdom", "inner calling", "subconscious", "mystery", "intuition", "introspection"],
    "new beginnings": ["new beginnings", "new venture", "fresh start", "seed energy", "new opportunity", "exploration"],
    "letting go": ["release", "walking away", "surrender", "moving on", "endings", "breaking free"],
    "success": ["victory", "success", "fulfillment", "completion", "recognition", "mastery", "triumph"],
    "family": ["family harmony", "homecoming", "nurturing", "legacy", "community", "celebration"],
    "truth": ["truth", "clarity", "breakthrough", "direct communication", "honesty", "accountability"],
    "courage": ["courage", "boldness", "adventure", "perseverance", "determination", "standing your ground"],
}

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def _all_cards() -> list[dict]:
    """Return a flat list of all 78 cards with a 'source' field added."""
    cards = []
    for card in MAJOR_ARCANA:
        entry = dict(card)
        entry["source"] = "major"
        cards.append(entry)
    for suit_name, suit_data in MINOR_ARCANA.items():
        for card in suit_data["cards"]:
            entry = dict(card)
            entry["source"] = suit_name
            entry["element"] = suit_data["element"]
            cards.append(entry)
    return cards


def get_card(name: str) -> Optional[dict]:
    """Search both Major and Minor Arcana by exact or case-insensitive name.

    Returns the card dict or None if not found.
    """
    name_lower = name.strip().lower()
    for card in MAJOR_ARCANA:
        if card["name"].lower() == name_lower:
            return dict(card)
    for suit_data in MINOR_ARCANA.values():
        for card in suit_data["cards"]:
            if card["name"].lower() == name_lower:
                return dict(card)
    return None


def get_major(number: int) -> Optional[dict]:
    """Return a Major Arcana card by its number (0-21), or None."""
    for card in MAJOR_ARCANA:
        if card["number"] == number:
            return dict(card)
    return None


def get_spread(name: str) -> Optional[dict]:
    """Return a spread template by key name (e.g. 'celtic_cross'), or None."""
    spread = SPREADS.get(name)
    return dict(spread) if spread else None


def get_suit(suit_name: str) -> Optional[dict]:
    """Return full suit data by name (wands, cups, swords, pentacles), or None."""
    suit = MINOR_ARCANA.get(suit_name.strip().lower())
    return dict(suit) if suit else None


def draw_cards(count: int, allow_reversed: bool = True) -> list[dict]:
    """Draw *count* random cards from the full 78-card deck.

    Each drawn card gets an ``orientation`` field: ``"upright"`` or
    ``"reversed"`` (if *allow_reversed* is True; roughly 50/50 chance).
    Cards are drawn without replacement.

    Returns a list of card dicts.
    """
    all_cards = _all_cards()
    if count > len(all_cards):
        count = len(all_cards)
    drawn = random.sample(all_cards, count)
    for card in drawn:
        if allow_reversed:
            card["orientation"] = random.choice(["upright", "reversed"])
        else:
            card["orientation"] = "upright"
    return drawn


def get_cards_for_intention(intention: str) -> list[dict]:
    """Return cards thematically related to an intention string.

    Matches intention text against a curated keyword map, then scans every
    card's upright keywords. Returns a deduplicated list sorted by
    relevance (number of keyword hits), most relevant first.
    """
    intention_lower = intention.strip().lower()

    # Collect relevant target keywords from the intention map
    target_keywords: set[str] = set()
    for key, kw_list in _INTENTION_KEYWORDS.items():
        if key in intention_lower:
            target_keywords.update(kw_list)

    # If no mapped keywords found, use the raw intention words as fallback
    if not target_keywords:
        target_keywords = {w for w in intention_lower.split() if len(w) > 2}

    all_cards = _all_cards()
    scored: list[tuple[int, dict]] = []

    for card in all_cards:
        card_keywords = " ".join(card.get("keywords_upright", [])).lower()
        card_meaning = card.get("upright_meaning", "").lower()
        combined = card_keywords + " " + card_meaning

        hits = sum(1 for kw in target_keywords if kw in combined)
        if hits > 0:
            scored.append((hits, card))

    # Sort by descending relevance
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [card for _, card in scored]
