"""
Wheel of the Year — the 8 sabbats of the Wiccan/Pagan calendar.

Pure Python reference data with helper functions for date lookup,
seasonal context, and sabbat retrieval. No external dependencies.
"""

from datetime import date, timedelta

# ── Sabbat Data ────────────────────────────────────────────────────────────────

SABBATS: dict[str, dict] = {
    "samhain": {
        "name": "Samhain",
        "pronunciation": "SOW-in",
        "dates": {
            "northern": "October 31 - November 1",
            "southern": "April 30 - May 1",
        },
        "alternate_names": [
            "Halloween",
            "Witch's New Year",
            "Festival of the Dead",
            "Third Harvest",
            "Ancestor Night",
        ],
        "meaning": (
            "Samhain marks the end of the harvest season and the beginning of "
            "the dark half of the year. The veil between the living and the dead "
            "is at its thinnest, making it the most potent night for communion "
            "with ancestors and spirits. It is the Witch's New Year, a time of "
            "endings that seed new beginnings."
        ),
        "themes": [
            "ancestors",
            "death and rebirth",
            "divination",
            "shadow work",
            "honoring the dead",
            "transformation",
            "the thinning veil",
        ],
        "energy": (
            "Deeply introspective and liminal. The boundary between worlds "
            "dissolves. Power builds in stillness and darkness. This is the "
            "energy of the crone, the final exhale before renewal."
        ),
        "correspondences": {
            "colors": ["black", "orange", "dark red", "purple", "gold"],
            "herbs": ["mugwort", "wormwood", "rosemary", "sage", "mandrake", "nightshade"],
            "crystals": ["obsidian", "onyx", "smoky quartz", "jet", "bloodstone"],
            "foods": ["apples", "pumpkin", "pomegranate", "root vegetables", "soul cakes", "colcannon"],
            "animals": ["bat", "cat", "owl", "raven", "spider"],
            "deities": ["The Crone", "Hecate", "Cerridwen", "The Morrigan", "Anubis", "Hel", "Persephone"],
            "incense": ["myrrh", "patchouli", "mugwort", "wormwood", "copal"],
            "symbols": ["jack-o-lantern", "cauldron", "skull", "besom", "black mirror"],
        },
        "rituals": [
            "Set a Dumb Supper — a silent meal with a place set for the beloved dead",
            "Scry with a black mirror or bowl of dark water by candlelight",
            "Build an ancestor altar with photos, heirlooms, and offerings",
            "Write letters to the departed and burn them to send the messages through the veil",
            "Perform a year-end release ritual, burning what no longer serves you",
        ],
        "journal_prompts": [
            "What has died or ended this year that I need to honor and release?",
            "If my ancestors could speak to me tonight, what would they say?",
            "What seeds am I planting in the dark soil of winter for future growth?",
            "What shadow aspects of myself am I ready to face and integrate?",
            "How has the cycle of death and rebirth shown up in my life this year?",
        ],
        "altar_setup": (
            "Cover the altar in black or dark purple cloth. Place photos or "
            "mementos of ancestors at the center. Add a cauldron, black candles, "
            "apples, pomegranates, and a bowl of water for scrying. Include "
            "mugwort for visionary work and rosemary for remembrance."
        ),
        "meditation_theme": (
            "Journey through the thinning veil to meet an ancestor or guide. "
            "Sit in the darkness between the worlds and listen for the whispers "
            "of those who came before."
        ),
        "element": "Water",
        "direction": "West",
        "approx_month": 10,
        "approx_day": 31,
    },
    "yule": {
        "name": "Yule",
        "pronunciation": "YOOL",
        "dates": {
            "northern": "December 20 - 23 (Winter Solstice)",
            "southern": "June 20 - 23",
        },
        "alternate_names": [
            "Winter Solstice",
            "Midwinter",
            "Alban Arthan",
            "Saturnalia",
            "Festival of Light",
        ],
        "meaning": (
            "Yule celebrates the rebirth of the Sun God at the Winter Solstice, "
            "the longest night of the year. From this darkest point, the light "
            "begins its return. It is a festival of hope, renewal, and the "
            "triumph of light over darkness."
        ),
        "themes": [
            "rebirth of the sun",
            "hope in darkness",
            "renewal",
            "rest and reflection",
            "gratitude",
            "inner light",
            "family and hearth",
        ],
        "energy": (
            "Quiet, warm, and gently powerful. The longest night holds the "
            "promise of returning light. Energy turns inward toward hearth and "
            "home. There is deep magic in stillness and patience."
        ),
        "correspondences": {
            "colors": ["red", "green", "gold", "white", "silver"],
            "herbs": ["holly", "ivy", "mistletoe", "pine", "cedar", "frankincense", "bayberry"],
            "crystals": ["garnet", "ruby", "clear quartz", "bloodstone", "green aventurine"],
            "foods": ["wassail", "gingerbread", "roasted meats", "yule log cake", "spiced cider", "oranges"],
            "animals": ["stag", "robin", "wren", "bear"],
            "deities": ["The Oak King", "The Holly King", "Brigid", "Saturn", "Freya", "Odin", "Sol Invictus"],
            "incense": ["frankincense", "myrrh", "pine", "cedar", "cinnamon", "bayberry"],
            "symbols": ["yule log", "evergreen wreath", "sun wheel", "holly", "mistletoe", "candles"],
        },
        "rituals": [
            "Burn a Yule log to honor the returning sun and carry blessings into the new year",
            "Hold an all-night vigil from sunset to sunrise, welcoming back the light",
            "Decorate an evergreen tree with symbols of wishes for the coming year",
            "Light candles at dawn to celebrate the Sun God's rebirth",
            "Exchange handmade gifts charged with intention and blessing",
        ],
        "journal_prompts": [
            "What light have I kept alive within me during my darkest times?",
            "What am I gestating in the darkness that will be born with the returning sun?",
            "What traditions and roots anchor me and give me strength?",
            "How can I bring more warmth and generosity into the world?",
            "What am I most grateful for as this year comes to a close?",
        ],
        "altar_setup": (
            "Drape the altar in red and green. Place a Yule log at the center, "
            "studded with holly, pine, and three candles (red, green, white). "
            "Surround with evergreen boughs, pinecones, oranges studded with "
            "cloves, and gold or silver ornaments. Light candles to honor the "
            "returning sun."
        ),
        "meditation_theme": (
            "Sit in the darkness of the longest night. Visualize a single spark "
            "of golden light in your heart, growing brighter with each breath "
            "until it fills you entirely — the sun reborn within."
        ),
        "element": "Earth",
        "direction": "North",
        "approx_month": 12,
        "approx_day": 21,
    },
    "imbolc": {
        "name": "Imbolc",
        "pronunciation": "IM-bolk",
        "dates": {
            "northern": "February 1 - 2",
            "southern": "August 1 - 2",
        },
        "alternate_names": [
            "Candlemas",
            "Brigid's Day",
            "Festival of Lights",
            "Feast of Torches",
            "Oimelc",
        ],
        "meaning": (
            "Imbolc honors the first stirrings of spring beneath the frozen "
            "ground. Sacred to the goddess Brigid, it celebrates the growing "
            "light, purification, and the quickening of new life. The ewes "
            "begin to lactate — the first sign that winter's grip is loosening."
        ),
        "themes": [
            "purification",
            "new beginnings",
            "inspiration",
            "healing",
            "the returning light",
            "creativity",
            "dedication",
        ],
        "energy": (
            "Tender, hopeful, and cleansing. The first whispers of spring stir "
            "beneath the soil. Creative and poetic energy rises. This is the "
            "maiden's first breath after winter's sleep."
        ),
        "correspondences": {
            "colors": ["white", "pale yellow", "light green", "lavender", "pink"],
            "herbs": ["chamomile", "angelica", "basil", "bay laurel", "heather", "snowdrops"],
            "crystals": ["amethyst", "moonstone", "clear quartz", "turquoise", "sunstone"],
            "foods": ["dairy", "seeds", "herbed breads", "honey cakes", "blackberry pie", "spiced wine"],
            "animals": ["ewe", "lamb", "swan", "groundhog", "deer"],
            "deities": ["Brigid", "Vesta", "Athena", "Gaia", "The Maiden"],
            "incense": ["vanilla", "chamomile", "myrrh", "basil", "cinnamon"],
            "symbols": ["Brigid's cross", "candle wheel", "snowdrops", "white flowers", "flame"],
        },
        "rituals": [
            "Weave a Brigid's cross from rushes or straw and hang it for protection",
            "Light every candle in the house at sundown to welcome the returning light",
            "Perform a house cleansing with salt water and a besom to sweep away winter stagnation",
            "Leave a cloth outside overnight for Brigid to bless (Brat Bhride)",
            "Write creative intentions and dedicate them to Brigid's sacred flame",
        ],
        "journal_prompts": [
            "What is stirring beneath the surface of my life, waiting to emerge?",
            "How can I purify my space, body, and mind to welcome new growth?",
            "What creative projects or dreams want my attention right now?",
            "Where in my life do I need healing, and how can I tend to it?",
            "What does the returning light illuminate that I could not see in the dark?",
        ],
        "altar_setup": (
            "Cover the altar in white or pale yellow cloth. Place a Brigid's "
            "cross at the center alongside white candles. Add snowdrops or other "
            "early flowers, a bowl of milk and honey, seeds, and a small cauldron "
            "with a lit candle inside to represent Brigid's sacred flame."
        ),
        "meditation_theme": (
            "Visualize yourself as a seed buried in dark earth. Feel warmth "
            "reaching you from above as the light returns. Sense your shell "
            "softening, a green shoot pushing upward toward the sun."
        ),
        "element": "Fire",
        "direction": "South",
        "approx_month": 2,
        "approx_day": 1,
    },
    "ostara": {
        "name": "Ostara",
        "pronunciation": "oh-STAR-ah",
        "dates": {
            "northern": "March 19 - 22 (Spring Equinox)",
            "southern": "September 19 - 22",
        },
        "alternate_names": [
            "Spring Equinox",
            "Vernal Equinox",
            "Alban Eilir",
            "Eostre's Day",
            "Lady Day",
        ],
        "meaning": (
            "Ostara celebrates the Spring Equinox, when day and night stand in "
            "perfect balance before the light overtakes the dark. Named for the "
            "Germanic goddess Eostre, it is a festival of fertility, balance, "
            "and explosive new growth."
        ),
        "themes": [
            "balance",
            "fertility",
            "new growth",
            "equality of light and dark",
            "planting seeds",
            "joy",
            "resurrection",
        ],
        "energy": (
            "Vibrant, expansive, and joyful. Life bursts forth everywhere. "
            "The world is waking up, and the energy is one of action, planting, "
            "and forward momentum. Balance is the key — equal parts doing and being."
        ),
        "correspondences": {
            "colors": ["pastel green", "yellow", "pink", "lavender", "robin's egg blue"],
            "herbs": ["lemon balm", "jasmine", "violet", "daffodil", "honeysuckle", "tansy"],
            "crystals": ["aquamarine", "rose quartz", "moonstone", "jasper", "amazonite"],
            "foods": ["eggs", "honey", "sprouts", "leafy greens", "hot cross buns", "spring salads"],
            "animals": ["hare", "rabbit", "chick", "butterfly", "robin"],
            "deities": ["Eostre", "Ostara", "Persephone", "The Green Man", "Flora", "Freya"],
            "incense": ["jasmine", "rose", "violet", "sage", "strawberry"],
            "symbols": ["eggs", "hare", "seeds", "spring flowers", "butterflies", "equal-armed cross"],
        },
        "rituals": [
            "Plant seeds (literal or symbolic) imbued with intentions for the growing season",
            "Dye or decorate eggs with sigils and symbols of your desires",
            "Take a dawn walk to greet the equinox sunrise and welcome the season",
            "Perform a balance ritual, honoring both the light and dark within yourself",
            "Create a spring altar garden with soil, seeds, and fresh flowers",
        ],
        "journal_prompts": [
            "Where do I need more balance between action and rest in my life?",
            "What seeds of intention am I planting this spring, and how will I tend them?",
            "What has been dormant in me that is ready to awaken and grow?",
            "How can I welcome more joy and playfulness into my daily life?",
            "What does fertility mean to me beyond the physical — creative, spiritual, emotional?",
        ],
        "altar_setup": (
            "Drape the altar in pastel colors — green, yellow, pink. Place "
            "decorated eggs, a vase of daffodils or tulips, seeds, a potted "
            "seedling, and images of hares or rabbits. Add an equal-armed cross "
            "to symbolize balance. Include a bowl of soil for planting intentions."
        ),
        "meditation_theme": (
            "Stand at the threshold between equal darkness and light. Feel "
            "perfect balance within your body. Then step forward into the growing "
            "light, carrying balance with you as the world blooms around you."
        ),
        "element": "Air",
        "direction": "East",
        "approx_month": 3,
        "approx_day": 20,
    },
    "beltane": {
        "name": "Beltane",
        "pronunciation": "BEL-tayn",
        "dates": {
            "northern": "May 1",
            "southern": "November 1",
        },
        "alternate_names": [
            "May Day",
            "Walpurgis Night",
            "Cetsamhain",
            "Festival of Fires",
            "Roodmas",
        ],
        "meaning": (
            "Beltane celebrates the height of spring and the sacred union of "
            "the God and Goddess. It is a fire festival of passion, fertility, "
            "and the full flowering of life. The veil thins again as it did at "
            "Samhain, but here the spirits are those of the fae and the wild."
        ),
        "themes": [
            "passion",
            "fertility",
            "sacred union",
            "fire and vitality",
            "the fae",
            "sensuality",
            "abundance",
        ],
        "energy": (
            "Ecstatic, passionate, and wild. Life force energy is at its peak. "
            "The earth is lush, desire is heightened, and creativity flows "
            "without restraint. This is the energy of the lover, the dancer, "
            "the bonfire leaper."
        ),
        "correspondences": {
            "colors": ["red", "white", "green", "dark pink", "yellow"],
            "herbs": ["hawthorn", "roses", "woodruff", "meadowsweet", "lilac", "foxglove"],
            "crystals": ["emerald", "malachite", "rose quartz", "carnelian", "rhodonite"],
            "foods": ["oat cakes", "honey", "strawberries", "maypole cake", "mead", "fresh salads"],
            "animals": ["bee", "cow", "horse", "swallow", "white stag"],
            "deities": ["The May Queen", "The Green Man", "Aphrodite", "Cernunnos", "Flora", "Pan"],
            "incense": ["rose", "frankincense", "lilac", "vanilla", "jasmine"],
            "symbols": ["maypole", "bonfire", "flower crown", "ribbons", "hawthorn blossom"],
        },
        "rituals": [
            "Dance the maypole to weave intentions of abundance and community",
            "Jump the Beltane bonfire for purification, luck, and fertility",
            "Weave flower crowns and wear them to embody the spirit of the May Queen or King",
            "Collect morning dew at dawn to wash your face for beauty and youth blessings",
            "Create a faerie offering of milk, honey, and bread left at a hawthorn tree",
        ],
        "journal_prompts": [
            "What am I passionate about, and how can I pursue it more fully?",
            "Where in my life is energy wanting to move, grow, and create?",
            "How do I honor my body, desires, and sensual nature?",
            "What sacred unions or partnerships (creative, romantic, spiritual) am I cultivating?",
            "What would my life look like if I fully embraced joy and wild abandon?",
        ],
        "altar_setup": (
            "Adorn the altar with fresh flowers — roses, lilacs, hawthorn "
            "blossoms. Weave ribbons of red, white, and green around candles. "
            "Place a small maypole or phallic symbol at center alongside a "
            "chalice. Add honey, strawberries, and a bowl of morning dew. "
            "Light red and white candles for the sacred union."
        ),
        "meditation_theme": (
            "Walk through a flowering meadow as dawn breaks on May morning. "
            "Feel the life force of the earth rising through your feet, filling "
            "you with vitality, passion, and the wild joy of being alive."
        ),
        "element": "Fire",
        "direction": "South",
        "approx_month": 5,
        "approx_day": 1,
    },
    "litha": {
        "name": "Litha",
        "pronunciation": "LITH-ah",
        "dates": {
            "northern": "June 20 - 23 (Summer Solstice)",
            "southern": "December 20 - 23",
        },
        "alternate_names": [
            "Midsummer",
            "Summer Solstice",
            "Alban Hefin",
            "St. John's Eve",
            "Gathering Day",
        ],
        "meaning": (
            "Litha celebrates the Summer Solstice, the longest day and shortest "
            "night. The Sun God is at the height of his power. It is a time of "
            "maximum light, abundance, and outward energy — but also holds the "
            "bittersweet knowing that from this peak, the light begins to wane."
        ),
        "themes": [
            "peak power",
            "abundance",
            "the sun at its zenith",
            "faerie magic",
            "empowerment",
            "gratitude",
            "light and shadow",
        ],
        "energy": (
            "Radiant, powerful, and expansive. Everything is in full bloom. "
            "Energy is at its highest outward expression. There is tremendous "
            "power available for manifestation, but wisdom lies in also "
            "acknowledging the coming turn toward darkness."
        ),
        "correspondences": {
            "colors": ["gold", "yellow", "orange", "green", "blue"],
            "herbs": [
                "St. John's wort", "lavender", "chamomile", "sunflower",
                "mugwort", "vervain", "fern",
            ],
            "crystals": ["citrine", "sunstone", "tiger's eye", "amber", "lapis lazuli"],
            "foods": ["fresh fruits", "honey", "mead", "grilled foods", "sun-shaped bread", "lemonade"],
            "animals": ["bee", "butterfly", "firefly", "horse", "wren"],
            "deities": [
                "The Oak King", "The Holly King", "Lugh", "Ra",
                "Aine", "Sulis", "Apollo",
            ],
            "incense": ["lavender", "frankincense", "lemon", "sandalwood", "copal"],
            "symbols": ["sun wheel", "bonfire", "sunflower", "golden crown", "faerie ring"],
        },
        "rituals": [
            "Stay up to watch the sunset and sunrise, honoring the shortest night",
            "Light a Midsummer bonfire and make offerings of herbs to the flames",
            "Gather St. John's wort and other herbs at noon when their power is strongest",
            "Create a sun wheel from flowers and roll it downhill at dawn",
            "Leave offerings of honey and cream for the fae in a garden or wild place",
        ],
        "journal_prompts": [
            "Where in my life am I at the peak of my power, and how can I honor that?",
            "What abundance am I enjoying right now that I want to pause and appreciate?",
            "How do I balance outward energy and action with inner stillness?",
            "What might I need to release or let wane as the light begins to turn?",
            "If this moment were the zenith of a story, what is the story about?",
        ],
        "altar_setup": (
            "Cover the altar in gold or bright yellow cloth. Place sunflowers, "
            "a sun wheel or solar symbol at center, golden candles, citrine and "
            "amber stones. Add a bowl of fresh fruit, a jar of honey, and "
            "bundles of dried herbs. Include images or figures representing "
            "the Oak King and Holly King."
        ),
        "meditation_theme": (
            "Stand at high noon on the longest day. Feel the sun's full power "
            "pouring into you from above. Hold this golden light in your solar "
            "plexus and know your own radiance. Then gently release, knowing "
            "that fullness carries the seed of rest."
        ),
        "element": "Fire",
        "direction": "South",
        "approx_month": 6,
        "approx_day": 21,
    },
    "lughnasadh": {
        "name": "Lughnasadh",
        "pronunciation": "LOO-nah-sah",
        "dates": {
            "northern": "August 1 - 2",
            "southern": "February 1 - 2",
        },
        "alternate_names": [
            "Lammas",
            "First Harvest",
            "Loaf Mass",
            "Festival of First Fruits",
            "Hlafmas",
        ],
        "meaning": (
            "Lughnasadh is the first of three harvest festivals, celebrating "
            "the grain harvest and honoring the Celtic god Lugh. The God "
            "willingly sacrifices himself as the grain is cut so that the people "
            "may live. It is a time of gratitude, skill, and reaping what was sown."
        ),
        "themes": [
            "first harvest",
            "gratitude",
            "sacrifice",
            "skill and craft",
            "abundance",
            "community",
            "reaping what you sow",
        ],
        "energy": (
            "Warm, generous, and productive. The first fruits of labor are "
            "coming in. There is satisfaction in seeing results and sharing "
            "bounty with others. Energy shifts from peak outward expression "
            "toward gathering and assessing."
        ),
        "correspondences": {
            "colors": ["gold", "orange", "bronze", "deep yellow", "green", "brown"],
            "herbs": ["wheat", "corn", "barley", "meadowsweet", "mint", "heather", "blackberry"],
            "crystals": ["peridot", "citrine", "aventurine", "tiger's eye", "carnelian"],
            "foods": [
                "fresh bread", "corn", "berries", "grains", "beer",
                "cider", "pie", "first fruits of the garden",
            ],
            "animals": ["rooster", "crow", "crane", "salmon", "pig"],
            "deities": ["Lugh", "Demeter", "Ceres", "John Barleycorn", "Tailtiu", "Danu"],
            "incense": ["sandalwood", "wheat", "frankincense", "heather", "corn silk"],
            "symbols": ["corn dolly", "bread loaf", "scythe", "wheat sheaf", "sun"],
        },
        "rituals": [
            "Bake bread from scratch and share it with loved ones as a communion of harvest",
            "Make a corn dolly from the last sheaf and keep it until Imbolc for blessings",
            "Hold games or contests of skill in honor of Lugh's funeral games for Tailtiu",
            "Give thanks for the first fruits by leaving offerings at a crossroads or field",
            "Assess your personal harvest — what goals have come to fruition this year?",
        ],
        "journal_prompts": [
            "What have I worked hard for that is now bearing fruit?",
            "What sacrifices have I made this year, and what did they yield?",
            "How can I share my abundance and skills with my community?",
            "What am I most proud of creating or accomplishing so far this year?",
            "What still needs tending before the final harvest at Mabon?",
        ],
        "altar_setup": (
            "Drape the altar in gold, orange, or brown cloth. Place a loaf of "
            "fresh bread at the center alongside a sheaf of wheat or corn. Add "
            "seasonal fruits and vegetables, a corn dolly, golden candles, and "
            "a small sickle or blade. Include a cup of ale or cider as a "
            "libation offering."
        ),
        "meditation_theme": (
            "Walk through a golden field of ripe grain at sunset. Feel the "
            "warmth of the harvest sun on your skin. Reach down and take a "
            "handful of grain — this is the fruit of your labor, your "
            "dedication made manifest."
        ),
        "element": "Earth",
        "direction": "North",
        "approx_month": 8,
        "approx_day": 1,
    },
    "mabon": {
        "name": "Mabon",
        "pronunciation": "MAY-bon",
        "dates": {
            "northern": "September 21 - 24 (Autumn Equinox)",
            "southern": "March 19 - 22",
        },
        "alternate_names": [
            "Autumn Equinox",
            "Fall Equinox",
            "Second Harvest",
            "Harvest Home",
            "Alban Elfed",
        ],
        "meaning": (
            "Mabon is the Autumn Equinox, when day and night are again in "
            "perfect balance before the dark overtakes the light. It is the "
            "second harvest and a time of deep gratitude, reflection, and "
            "preparation. The God descends into the underworld, and the earth "
            "begins its slow exhale into winter."
        ),
        "themes": [
            "balance",
            "gratitude",
            "second harvest",
            "reflection",
            "preparation",
            "letting go",
            "the descent",
        ],
        "energy": (
            "Contemplative, grateful, and grounding. The frenzy of summer "
            "settles into a mature calm. Energy turns inward, assessing what "
            "has been gained and what must be released. There is beauty in "
            "the dying light."
        ),
        "correspondences": {
            "colors": ["deep red", "orange", "brown", "maroon", "gold", "dark green"],
            "herbs": ["sage", "rosemary", "marigold", "thistle", "apple blossom", "hops", "yarrow"],
            "crystals": ["amber", "sapphire", "lapis lazuli", "smoky quartz", "tiger's eye"],
            "foods": [
                "apples", "wine", "grapes", "nuts", "squash",
                "root vegetables", "corn", "autumn stews",
            ],
            "animals": ["goose", "salmon", "stag", "blackbird", "wolf"],
            "deities": [
                "Mabon ap Modron", "Persephone", "Demeter", "Dionysus",
                "The Green Man", "Pomona", "Thor",
            ],
            "incense": ["sage", "myrrh", "benzoin", "apple", "cinnamon", "clove"],
            "symbols": [
                "cornucopia", "scales", "apple", "wine",
                "grapevine", "autumn leaves",
            ],
        },
        "rituals": [
            "Hold a gratitude feast and share the bounty of the harvest with community",
            "Create a gratitude list or altar with items representing each blessing of the year",
            "Perform a balance meditation, honoring the equal dark and light within you",
            "Make wine or cider and bless it as a libation for the coming dark months",
            "Walk in nature and collect fallen leaves, acorns, and seeds as altar offerings",
        ],
        "journal_prompts": [
            "What am I most grateful for at this moment in my life?",
            "What do I need to release before winter, and how can I do so with grace?",
            "Where is the balance between giving and receiving in my life?",
            "What wisdom have I harvested from this year's experiences?",
            "How can I prepare — physically, emotionally, spiritually — for the dark half of the year?",
        ],
        "altar_setup": (
            "Cover the altar in autumn colors — deep red, orange, brown. Place "
            "a cornucopia overflowing with apples, gourds, grapes, and nuts at "
            "the center. Add autumn leaves, a pair of balanced scales, wine or "
            "cider in a chalice, and amber or brown candles. Include a small "
            "mirror to reflect on the past season."
        ),
        "meditation_theme": (
            "Stand in an orchard at twilight as the equinox sun sets. Feel the "
            "perfect balance of light and dark within and around you. Pick an "
            "apple from the tree — it holds the wisdom of the year. Take a "
            "bite and let that knowing fill you."
        ),
        "element": "Water",
        "direction": "West",
        "approx_month": 9,
        "approx_day": 22,
    },
}

# ── Ordered list for date calculations ─────────────────────────────────────────

_SABBAT_ORDER = [
    "imbolc",       # ~Feb 1
    "ostara",       # ~Mar 20
    "beltane",      # ~May 1
    "litha",        # ~Jun 21
    "lughnasadh",   # ~Aug 1
    "mabon",        # ~Sep 22
    "samhain",      # ~Oct 31
    "yule",         # ~Dec 21
]

_SEASON_MAP = {
    "spring": ["imbolc", "ostara"],
    "summer": ["beltane", "litha"],
    "autumn": ["lughnasadh", "mabon"],
    "fall":   ["lughnasadh", "mabon"],
    "winter": ["samhain", "yule"],
}


def _approx_date(sabbat_key: str, year: int) -> date:
    """Return the approximate date for a sabbat in a given year."""
    s = SABBATS[sabbat_key]
    return date(year, s["approx_month"], s["approx_day"])


# ── Helper Functions ───────────────────────────────────────────────────────────


def get_sabbat(name: str) -> dict | None:
    """
    Look up a sabbat by name (case-insensitive). Accepts primary names,
    alternate names, and common variants.

    Returns the sabbat dict or None if not found.
    """
    key = name.strip().lower().replace("'", "'").replace("\u2019", "'")

    # Direct key match
    if key in SABBATS:
        return SABBATS[key]

    # Search by display name and alternate names
    for sabbat_key, data in SABBATS.items():
        if key == data["name"].lower():
            return data
        for alt in data["alternate_names"]:
            if key == alt.lower():
                return data

    return None


def get_current_sabbat(month: int, day: int) -> dict:
    """
    Return the sabbat whose season we are currently within.

    The Wheel is divided into eight arcs. This function returns the most
    recently passed sabbat (the one whose energy currently holds sway).
    """
    today = date(2000, month, day)  # year is arbitrary for comparison

    # Build a list of (approx_date, key) sorted chronologically
    dated = []
    for key in _SABBAT_ORDER:
        s = SABBATS[key]
        dated.append((date(2000, s["approx_month"], s["approx_day"]), key))

    # Find the most recent sabbat on or before today
    current_key = _SABBAT_ORDER[-1]  # default to Yule (wraps around)
    for d, key in dated:
        if d <= today:
            current_key = key
        else:
            break

    return SABBATS[current_key]


def get_next_sabbat(month: int, day: int) -> tuple[str, dict, int]:
    """
    Return (name, sabbat_dict, days_until) for the next upcoming sabbat.

    Uses the current year (or next year for wrap-around) to compute an
    accurate day count.
    """
    today = date.today().replace(month=month, day=day)
    year = today.year

    best_name = None
    best_data = None
    best_days = 999

    for key in _SABBAT_ORDER:
        s_date = _approx_date(key, year)
        delta = (s_date - today).days
        if delta < 0:
            # Try next year
            s_date = _approx_date(key, year + 1)
            delta = (s_date - today).days
        if 0 < delta < best_days:
            best_days = delta
            best_name = SABBATS[key]["name"]
            best_data = SABBATS[key]

    # Edge case: if today IS a sabbat, look ahead to the following one
    if best_name is None:
        # Fallback: return the first sabbat of the next cycle
        key = _SABBAT_ORDER[0]
        s_date = _approx_date(key, year + 1)
        best_days = (s_date - today).days
        best_name = SABBATS[key]["name"]
        best_data = SABBATS[key]

    return best_name, best_data, best_days


def get_sabbat_by_season(season: str) -> list[dict]:
    """
    Return sabbats associated with a season.

    Accepted seasons: spring, summer, autumn (or fall), winter.
    Returns a list of sabbat dicts (typically 2 per season).
    """
    key = season.strip().lower()
    if key not in _SEASON_MAP:
        return []
    return [SABBATS[k] for k in _SEASON_MAP[key]]


def get_seasonal_context(month: int) -> str:
    """
    Return a prose description of the current seasonal energy based on where
    we sit on the Wheel of the Year (Northern Hemisphere perspective).
    """
    contexts = {
        1: (
            "Deep winter holds the land. Yule's returning light is still a "
            "whisper. The earth sleeps, and so should you — rest, dream, and "
            "gestate your visions in the dark. Imbolc's first stirrings are "
            "near, a promise of warmth beneath the frost."
        ),
        2: (
            "Imbolc energy stirs the frozen ground. The light grows stronger "
            "each day. Brigid's sacred flame kindles inspiration and creativity. "
            "This is a time for purification, setting intentions, and nursing "
            "the first tender shoots of new projects and dreams."
        ),
        3: (
            "The Spring Equinox approaches, bringing Ostara's gift of perfect "
            "balance. Day and night stand equal. Seeds planted now — literal and "
            "metaphorical — carry tremendous potential. The world is waking up, "
            "and so is your creative power."
        ),
        4: (
            "Spring is in full bloom between Ostara and Beltane. Growth is "
            "accelerating. Flowers open, plans take shape, and the energy of "
            "renewal is everywhere. Channel this expansive energy into your "
            "most important creative and spiritual work."
        ),
        5: (
            "Beltane fire blazes through the land. Passion, fertility, and "
            "life force energy are at fever pitch. The sacred marriage of earth "
            "and sky plays out in every blossom and birdsong. Honor your body, "
            "your desires, and the wild joy of being alive."
        ),
        6: (
            "Midsummer approaches — Litha, the Summer Solstice, the longest "
            "day. The Sun God stands at his zenith. This is a time of maximum "
            "power, radiance, and outward expression. Celebrate what you have "
            "built, but remember: from the peak, the wheel turns toward harvest."
        ),
        7: (
            "The sun's power begins its slow descent after Litha. Lughnasadh "
            "draws near with the promise of the first grain harvest. The long "
            "golden days invite gratitude and gathering. Begin to assess what "
            "your efforts have yielded and prepare for the reaping."
        ),
        8: (
            "Lughnasadh has arrived — the First Harvest. The grain is cut, "
            "bread is baked, and the God willingly sacrifices for the people's "
            "sustenance. Share your skills and your bounty. Give thanks for "
            "what has grown, and tend what still ripens on the vine."
        ),
        9: (
            "Mabon, the Autumn Equinox, brings another moment of perfect "
            "balance before the dark half claims the year. The Second Harvest "
            "is a cornucopia of gratitude. Reflect on what you have gathered — "
            "materially, emotionally, spiritually — and release what is done."
        ),
        10: (
            "The veil thins as Samhain approaches. Autumn deepens, leaves fall, "
            "and the world prepares for the descent into darkness. Ancestor "
            "energy builds. Turn inward, honor what has passed, and prepare "
            "for the Witch's New Year at month's end."
        ),
        11: (
            "Samhain's echo lingers in the darkening days. The Witch's New Year "
            "has begun. The veil is still thin, and communion with the spirit "
            "world remains potent. Use this introspective time for divination, "
            "shadow work, and deep inner knowing. Yule's light is weeks away."
        ),
        12: (
            "The longest night approaches. Yule, the Winter Solstice, is the "
            "great turning point — from this deepest darkness, the light is "
            "reborn. Celebrate with warmth, generosity, and the quiet faith "
            "that the sun will return. Rest. Reflect. Be still and know."
        ),
    }

    return contexts.get(month, "The Wheel turns ever onward.")
