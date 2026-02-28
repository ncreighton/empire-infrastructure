"""Per-niche domain expertise — real products, facts, tips, and visual subjects.

Injected into AI script prompts so the LLM generates specific, actionable content
instead of generic filler. Also provides niche-appropriate visual descriptions
and image style suffixes.
"""

DOMAIN_EXPERTISE = {
    # ── Tech (5 niches) ───────────────────────────────────────────────────

    "smarthomewizards": {
        "key_products": [
            "Amazon Echo Dot", "Google Nest Hub", "Philips Hue",
            "Ring Video Doorbell", "ecobee SmartThermostat", "TP-Link Kasa",
            "Lutron Caseta", "Aqara sensors", "HomePod Mini", "SmartThings Hub",
            "Nanoleaf panels", "Wyze Cam", "Yale Assure Lock", "Roomba j7",
        ],
        "expert_tips": [
            "Start with smart lighting — cheapest entry point with the biggest visual impact",
            "Always check Matter and Thread compatibility before buying in 2026",
            "Use routines and automations, not just voice commands — that's where the real time savings are",
            "Put your smart hub on ethernet, not WiFi — reliability goes up dramatically",
            "Group devices by room in your app before setting up automations",
            "Use smart plugs to make any dumb device smart for under ten dollars",
            "Set up a guest WiFi network for all IoT devices to improve security",
        ],
        "talking_points": {
            "smart lighting": "Philips Hue has the best ecosystem and integration, Govee is budget-friendly with great effects, LIFX needs no hub but costs more per bulb",
            "voice assistants": "Alexa leads in smart home integrations with over 100,000 skills, Google has better natural language understanding, Siri works best in Apple-only homes",
            "security": "Ring is budget-friendly at $99, Arlo has the best video quality at 2K, Ubiquiti is the prosumer pick — all need subscriptions for cloud storage",
            "automation": "Home Assistant is free and runs locally for power users, SmartThings is the easiest hub-based platform, Apple Home wins on privacy",
            "thermostats": "ecobee includes a room sensor in the box, Nest learns your schedule automatically, Honeywell T9 has the best multi-room support",
        },
        "visual_subjects": {
            "default": "smart home device on a modern countertop, clean minimalist interior, soft ambient LED lighting",
            "lighting": "smart LED strip lights glowing in a modern living room, ambient purple and blue colors, cozy evening atmosphere",
            "security": "smart doorbell camera on a modern front door, ring of light glowing, sleek design, daytime",
            "voice": "smart speaker on a nightstand, blue LED ring glowing, modern minimalist bedroom",
            "automation": "smartphone showing smart home dashboard app, modern kitchen background, touch screen interface",
        },
        "style_suffix": ", clean product photography, soft ambient lighting, modern minimalist interior, shallow depth of field, editorial style, 4K",
    },

    "smarthomegearreviews": {
        "key_products": [
            "Google Nest Wifi Pro", "eero Pro 6E", "Ring Alarm Pro",
            "Aqara Hub M3", "Sonos Era 100", "Roborock S8 MaxV Ultra",
            "SwitchBot Hub 2", "Meross smart garage opener", "Govee RGBIC strips",
            "Eufy Video Lock", "Reolink Argus 4 Pro", "Shelly Plus relays",
        ],
        "expert_tips": [
            "Always test device range before permanent installation — WiFi dead zones kill smart home reliability",
            "Check if the device works locally or only via cloud — cloud-only devices stop working when the internet goes down",
            "Read the return policy before buying — some smart home gear is non-returnable once activated",
            "Mesh WiFi is almost mandatory for homes with more than 15 smart devices",
            "Thread-enabled devices form a mesh network that gets more reliable as you add more devices",
        ],
        "talking_points": {
            "mesh wifi": "Google Nest Wifi Pro supports WiFi 6E and doubles as a smart speaker, eero Pro 6E has the fastest speeds, TP-Link Deco is the budget pick",
            "robot vacuums": "Roborock S8 MaxV Ultra has the best obstacle avoidance, Roomba j7 handles pet hair best, Ecovacs X2 Omni has the strongest suction",
            "smart locks": "Yale Assure Lock 2 has the best design and Matter support, August WiFi Smart Lock is easiest to install, Schlage Encode Plus works with Apple Home Key",
            "sensors": "Aqara sensors are the cheapest Zigbee option at $15 each, Hue motion sensors work great for lighting automation, Eve sensors keep all data local",
        },
        "visual_subjects": {
            "default": "smart home device close-up on a clean white surface, product photography, soft studio lighting",
            "wifi": "mesh WiFi router system on a modern desk, multiple nodes visible, clean setup",
            "vacuum": "robot vacuum on hardwood floor in a modern living room, clean path visible",
            "locks": "smart lock installed on a modern front door, touchscreen keypad glowing, close-up",
        },
        "style_suffix": ", clean product photography, soft ambient lighting, modern minimalist interior, shallow depth of field, editorial style, 4K",
    },

    "wearablegearreviews": {
        "key_products": [
            "Apple Watch Ultra 2", "Samsung Galaxy Watch 6", "Garmin Venu 3",
            "Fitbit Charge 6", "Oura Ring Gen 3", "Whoop 4.0",
            "AirPods Pro 2", "Sony WF-1000XM5", "Pixel Buds Pro 2",
            "Meta Ray-Ban smart glasses", "Bose Ultra Open Earbuds",
        ],
        "expert_tips": [
            "Battery life matters more than features — a dead watch tracks nothing",
            "For accurate heart rate, wear the watch two finger widths above your wrist bone",
            "GPS accuracy varies hugely between watches — Garmin consistently leads in outdoor tracking",
            "Subscription costs add up — Whoop is $30 a month, Fitbit Premium is $10, Apple Watch has no subscription",
            "AMOLED screens are easier to read outdoors than LCD, but drain battery faster with always-on display",
        ],
        "talking_points": {
            "smartwatches": "Apple Watch Ultra 2 has the best ecosystem and health features, Garmin Venu 3 wins on battery life with 14 days, Samsung Galaxy Watch 6 is the best Android option",
            "fitness trackers": "Fitbit Charge 6 is the best value at $99, Oura Ring is the most discreet option, Whoop has the deepest recovery analytics but requires a subscription",
            "earbuds": "AirPods Pro 2 have the best noise canceling for Apple users, Sony XM5 has the best sound quality overall, Pixel Buds Pro 2 have the best Google Assistant integration",
            "sleep tracking": "Oura Ring is the gold standard for sleep tracking accuracy, Apple Watch Series 9 added sleep apnea detection, Garmin tracks sleep stages reliably",
        },
        "visual_subjects": {
            "default": "smartwatch on a wrist, modern lifestyle setting, clean background, product showcase",
            "earbuds": "wireless earbuds in charging case, close-up product photography, clean white background",
            "fitness": "fitness tracker on wrist during workout, gym setting, motion blur, dynamic energy",
            "sleep": "sleep tracker on nightstand next to bed, soft warm lighting, cozy bedroom",
        },
        "style_suffix": ", clean product photography, soft ambient lighting, modern minimalist interior, shallow depth of field, editorial style, 4K",
    },

    "aiinactionhub": {
        "key_products": [
            "ChatGPT", "Claude", "Midjourney", "Cursor AI", "GitHub Copilot",
            "Perplexity AI", "Notion AI", "Zapier", "Make.com",
            "Runway ML", "ElevenLabs", "Descript", "Gamma AI",
        ],
        "expert_tips": [
            "Chain multiple AI tools together for 10x output — use ChatGPT for ideation, Claude for writing, Midjourney for visuals",
            "Always review and edit AI output — use it as a first draft, not a final product",
            "Custom GPTs and Claude Projects save hours by pre-loading your context and instructions",
            "Use Zapier or Make.com to automate repetitive AI workflows without coding",
            "Prompt engineering tip: give the AI a role, context, and specific output format for dramatically better results",
        ],
        "talking_points": {
            "writing": "Claude excels at long-form writing and analysis, ChatGPT is better for creative brainstorming, Perplexity is best for research with sources",
            "coding": "Cursor AI is the best AI code editor with inline completions, GitHub Copilot integrates with VS Code, Claude handles complex codebases well",
            "images": "Midjourney produces the most artistic images, DALL-E 3 is easiest to prompt, Flux Pro is best for photorealism",
            "automation": "Zapier connects 6,000+ apps with AI steps, Make.com offers more complex workflows for less money, n8n is the free self-hosted option",
        },
        "visual_subjects": {
            "default": "laptop screen showing AI interface, modern desk setup, clean workspace, soft monitor glow",
            "coding": "code editor with AI completions visible, dark theme, modern developer workspace",
            "writing": "AI writing interface on screen, split view with document, clean workspace",
            "automation": "workflow automation diagram on screen, connected nodes and arrows, modern dashboard",
        },
        "style_suffix": ", clean product photography, soft ambient lighting, modern minimalist interior, shallow depth of field, editorial style, 4K",
    },

    "pulsegearreviews": {
        "key_products": [
            "Garmin Forerunner 965", "Apple Watch Ultra 2", "Whoop 4.0",
            "Theragun Pro", "Hyperice Hypervolt 2 Pro", "NordicTrack treadmill",
            "Rogue Echo Bike", "TRX Suspension Trainer", "Bose Sport Earbuds",
            "Jabra Elite 8 Active", "NOBULL trainers", "Hoka Clifton 9",
        ],
        "expert_tips": [
            "Heart rate zones matter more than total calories — zone 2 training builds your aerobic base",
            "A good recovery tool pays for itself by reducing injury risk — the Theragun is worth every penny for serious athletes",
            "GPS watches are 2-5% off on distance — calibrate with a known route before racing",
            "Wrist-based heart rate is unreliable during strength training — use a chest strap for lifting",
            "Invest in shoes first, gadgets second — the wrong shoes cause more damage than the right watch prevents",
        ],
        "talking_points": {
            "running watches": "Garmin Forerunner 965 has the best GPS accuracy and training metrics, Apple Watch Ultra 2 has the best display, COROS PACE 3 is the best value at $229",
            "recovery": "Theragun Pro hits deepest at 60 lbs of force, Hyperice Hypervolt 2 Pro is quietest, Theragun Mini is best for travel at $199",
            "home gym": "Rogue Echo Bike is the most durable cardio machine, TRX trains full body with no space needed, Peloton Bike+ has the best classes ecosystem",
            "earbuds": "Jabra Elite 8 Active are the toughest workout earbuds with IP68, Bose Sport Earbuds have the best fit, Beats Fit Pro have the best bass for motivation",
        },
        "visual_subjects": {
            "default": "fitness gear product close-up, gym setting background, dramatic lighting, high energy",
            "running": "running watch on wrist mid-stride, outdoor trail, motion blur, dynamic action",
            "gym": "home gym setup with equipment, clean garage gym, motivational atmosphere",
            "recovery": "recovery massage gun in use on muscle, athletic person, studio lighting",
        },
        "style_suffix": ", dynamic action photography, high contrast, gym or outdoor setting, motivational energy, sharp focus, sports photography, 4K",
    },

    # ── AI News (3 niches) ────────────────────────────────────────────────

    "aidiscoverydigest": {
        "key_products": [
            "ChatGPT", "Claude", "Gemini", "Llama 3", "Mistral",
            "Midjourney", "Sora", "Perplexity", "OpenAI", "Anthropic",
            "Google DeepMind", "Meta AI", "Hugging Face", "Stability AI",
        ],
        "expert_tips": [
            "The AI landscape changes weekly — follow Anthropic, OpenAI, and Google DeepMind announcements directly",
            "Open-source models like Llama 3 are closing the gap with proprietary ones — they're now viable for production use",
            "Multimodal AI is the biggest trend in 2026 — models that understand text, images, audio, and video together",
            "AI agents that can browse the web, write code, and use tools autonomously are the next major shift",
            "Context windows keep growing — Claude offers 200K tokens, meaning it can process entire codebases at once",
        ],
        "talking_points": {
            "language models": "Claude leads in safety and long-form analysis, GPT-4 is the most versatile, Gemini has the best multimodal integration with Google services",
            "image generation": "Midjourney v6 produces the most artistic results, DALL-E 3 understands text prompts best, Stable Diffusion 3 is the best open-source option",
            "open source": "Llama 3 from Meta is the most capable open model, Mistral is the best European AI company, Hugging Face is the GitHub of AI models",
            "ai safety": "Anthropic leads on constitutional AI and safety research, OpenAI has the largest safety team, EU AI Act is shaping global regulation",
        },
        "visual_subjects": {
            "default": "futuristic AI visualization, neural network nodes glowing, digital data streams, blue and purple tones",
            "language models": "AI chatbot interface on screen, conversation visible, sleek dark UI, modern tech aesthetic",
            "robots": "humanoid robot in modern setting, soft lighting, futuristic but approachable design",
            "data": "holographic data visualization, floating charts and graphs, dark background with neon accents",
        },
        "style_suffix": ", futuristic digital environment, holographic displays, neon accents, clean tech aesthetic, sharp focus, 4K, professional photography",
    },

    "clearainews": {
        "key_products": [
            "GPT-5", "Claude Opus", "Gemini Ultra", "Sora",
            "OpenAI", "Anthropic", "Google DeepMind", "Meta AI",
            "NVIDIA", "Microsoft Copilot", "Apple Intelligence",
            "EU AI Act", "US AI Executive Order",
        ],
        "expert_tips": [
            "Big Tech is spending over $100 billion on AI infrastructure in 2026 alone",
            "NVIDIA controls 80% of the AI chip market — their earnings predict the entire AI industry's health",
            "AI regulation is accelerating — the EU AI Act took effect in 2025, other countries are following",
            "Enterprise AI adoption doubled in 2025 — the real money is in B2B AI tools, not consumer chatbots",
            "AI job displacement is real but slower than headlines suggest — most roles are being augmented, not replaced",
        ],
        "talking_points": {
            "industry": "Microsoft invested $13 billion in OpenAI, Google is betting on Gemini, Anthropic raised $7.3 billion — the AI arms race shows no signs of slowing",
            "regulation": "EU AI Act bans social scoring and real-time facial recognition, requires transparency for high-risk AI, penalties up to 7% of global revenue",
            "chips": "NVIDIA H100 GPUs cost $25,000 each and have year-long waitlists, AMD MI300X is the main competitor, custom chips from Google TPU and Amazon Trainium are gaining ground",
            "jobs": "Goldman Sachs estimates AI could affect 300 million jobs globally, but McKinsey says it will create as many new roles as it displaces",
        },
        "visual_subjects": {
            "default": "breaking news style AI graphic, bold headline text, tech news studio aesthetic, red and blue accents",
            "chips": "close-up of AI processor chip, circuit board detail, blue LED lighting, tech macro photography",
            "regulation": "government building with digital AI overlay, scales of justice, formal authoritative aesthetic",
            "companies": "modern tech company headquarters exterior, glass building, Silicon Valley aesthetic",
        },
        "style_suffix": ", futuristic digital environment, holographic displays, neon accents, clean tech aesthetic, sharp focus, 4K, professional photography",
    },

    "wealthfromai": {
        "key_products": [
            "ChatGPT", "Claude", "Jasper AI", "Copy.ai", "Canva AI",
            "Shopify AI", "Zapier", "Gumroad", "Teachable", "Carrd",
            "Midjourney", "ElevenLabs", "Descript", "Synthesia",
        ],
        "expert_tips": [
            "The fastest AI income path is selling services, not building products — offer AI-powered writing, design, or consulting",
            "AI content creation tools can cut production time by 80% — that's your competitive edge as a solopreneur",
            "Build once, sell forever — use AI to create digital products like templates, courses, and ebooks",
            "Most AI side hustles fail because people build what's cool, not what people pay for — start with demand",
            "Automate client onboarding with AI chatbots and Zapier — it's the highest-ROI automation for freelancers",
        ],
        "talking_points": {
            "side hustles": "AI content writing services earn $3,000-10,000 per month, AI-generated art prints sell well on Etsy, AI tutoring is an emerging market",
            "tools": "Jasper AI is best for marketing copy at $49/mo, Copy.ai has a generous free tier, Claude is the most capable for complex writing tasks",
            "passive income": "Sell AI prompt packs on Gumroad starting at $0, create AI courses on Teachable, build niche AI chatbots as a subscription service",
            "automation": "Zapier can automate 90% of admin work for free, Make.com handles complex multi-step workflows, AI email assistants save 5+ hours per week",
        },
        "visual_subjects": {
            "default": "laptop showing revenue dashboard, modern workspace, green accent lighting, entrepreneurial energy",
            "money": "growing revenue chart on screen, upward trend line, green and gold colors, success visualization",
            "tools": "multiple AI tool interfaces on screen, split view dashboard, modern tech workspace",
            "automation": "automated workflow visualization, connected nodes, data flowing between apps, futuristic dashboard",
        },
        "style_suffix": ", professional corporate aesthetic, clean modern workspace, data visualization, confident atmosphere, editorial photography, 4K",
    },

    # ── Witchcraft (3 niches) ─────────────────────────────────────────────

    "witchcraftforbeginners": {
        "key_products": [
            "white sage bundle", "clear quartz crystal", "amethyst",
            "black tourmaline", "beeswax candles", "cast iron cauldron",
            "Book of Shadows journal", "tarot deck (Rider-Waite-Smith)",
            "dried lavender", "rose quartz", "moonstone", "selenite wand",
            "mortar and pestle", "altar cloth",
        ],
        "expert_tips": [
            "Start with just one crystal and one herb — you don't need a full apothecary to begin practicing",
            "Clear quartz is called the master healer because it amplifies the energy of any other crystal near it",
            "Always cleanse new crystals before use — moonlight, sound, or smoke all work",
            "The best time for manifestation spells is during the waxing moon, especially the three days before full moon",
            "Keep a spell journal — tracking your practice is how you develop your intuition over time",
            "Lavender is the most versatile herb in witchcraft — it works for protection, love, sleep, and purification",
            "Your intention matters more than expensive tools — a birthday candle works just as well as a hand-dipped ritual candle",
        ],
        "talking_points": {
            "crystals": "Clear quartz amplifies everything, amethyst enhances intuition, black tourmaline is the best protection stone, rose quartz opens the heart chakra",
            "herbs": "Lavender for peace and protection, rosemary for memory and cleansing, cinnamon for prosperity spells, mugwort for divination and dreams",
            "moon phases": "New moon for new beginnings, waxing moon for growth and attraction, full moon for maximum power and charging crystals, waning moon for releasing and banishing",
            "candle magic": "White candles are universal and work for any spell, green for money, pink for love, black for protection and banishing — always dress candles with oil from bottom to top for attraction",
            "protection": "Black tourmaline by your front door blocks negative energy, salt circles create sacred space, rosemary hung above doorways protects the home",
        },
        "visual_subjects": {
            "default": "crystal collection on dark altar cloth, candlelight, dried herbs, mystical atmosphere",
            "crystals": "amethyst and clear quartz crystals on dark surface, candlelight reflection, mystical purple glow",
            "herbs": "dried herbs in glass jars on wooden shelf, mortar and pestle, apothecary aesthetic, warm lighting",
            "moon": "full moon in night sky, silhouette of tree branches, mystical blue glow, ethereal atmosphere",
            "candles": "ritual candles burning on altar, melted wax dripping, warm golden light, sacred space",
            "tarot": "tarot cards spread on dark velvet cloth, candlelight, mystical symbols, divination setup",
        },
        "style_suffix": ", mystical atmosphere, candlelight, soft ethereal glow, dark moody background, sacred space aesthetic, shallow depth of field, fine art photography",
    },

    "moonrituallibrary": {
        "key_products": [
            "moonstone", "selenite", "labradorite", "silver candles",
            "moon water jar", "lunar calendar", "white sage",
            "clear quartz", "obsidian", "moon phase wall art",
            "ritual bath salts", "intention journal",
        ],
        "expert_tips": [
            "Make moon water by leaving a sealed jar of water under the full moon overnight — use it in spells, baths, and cleansing",
            "New moon is for setting intentions and planting seeds, full moon is for releasing what no longer serves you",
            "Track your energy across moon phases for three months — you'll notice clear patterns in your productivity and emotions",
            "Moonstone is the most powerful crystal for lunar work — wear it during rituals to strengthen your moon connection",
            "The void-of-course moon is not ideal for starting new projects — use this time for reflection and rest instead",
        ],
        "talking_points": {
            "full moon": "Full moon is peak energy for charging crystals, releasing rituals, and gratitude ceremonies — the three days around full moon are all powerful",
            "new moon": "New moon is the dark reset — best time for intention setting, starting new projects, and planting metaphorical seeds",
            "eclipses": "Lunar eclipses amplify full moon energy tenfold — avoid starting new spells during eclipses, instead observe and receive messages",
            "moon water": "Collect water under the full moon for cleansing sprays, add to baths for emotional healing, use in spell jars for amplified intentions",
        },
        "visual_subjects": {
            "default": "full moon reflecting on still water, night scene, mystical blue light, serene landscape",
            "ritual": "moon ritual setup with candles and crystals, moonlight streaming in, sacred ceremony space",
            "crystals": "moonstone and selenite under moonlight, glowing ethereal quality, dark velvet background",
            "water": "glass jar of moon water catching moonlight, crystals surrounding it, mystical atmosphere",
        },
        "style_suffix": ", mystical atmosphere, candlelight, soft ethereal glow, dark moody background, sacred space aesthetic, shallow depth of field, fine art photography",
    },

    "manifestandalign": {
        "key_products": [
            "citrine crystal", "green aventurine", "pyrite", "rose quartz",
            "manifestation journal", "vision board supplies", "affirmation cards",
            "meditation cushion", "essential oil diffuser", "gratitude journal",
            "369 method worksheet", "scripting journal",
        ],
        "expert_tips": [
            "The 369 method works: write your manifestation 3 times in the morning, 6 times in the afternoon, 9 times before bed",
            "Gratitude is the fastest frequency raiser — spend 5 minutes each morning listing 10 things you're grateful for",
            "Scripting means writing about your desired reality as if it's already happened — present tense, specific details, and emotions",
            "Citrine is the abundance crystal — keep one in your wallet or on your desk to attract prosperity",
            "The biggest manifestation mistake is trying to control the HOW — set the intention, then release attachment to the path",
        ],
        "talking_points": {
            "methods": "369 method for focused repetition, scripting for detailed visualization, two-cup method for quantum jumping, pillow method for effortless sleep manifestation",
            "abundance": "Citrine attracts wealth and success, green aventurine is the luckiest crystal, pyrite is called fool's gold but attracts real prosperity",
            "mindset": "Your thoughts create your reality — neuroscience confirms that visualization activates the same brain regions as real experience",
            "daily practice": "Morning gratitude journal, midday affirmations, evening scripting — consistency matters more than duration",
        },
        "visual_subjects": {
            "default": "golden light streaming through window, crystals and journal on table, warm abundant atmosphere",
            "crystals": "citrine and green aventurine crystals in golden light, abundance energy, warm tones",
            "journal": "beautiful manifestation journal with pen, golden hour lighting, inspiring workspace",
            "meditation": "person meditating in golden light, peaceful setting, cosmic energy visualization",
        },
        "style_suffix": ", mystical atmosphere, candlelight, soft ethereal glow, dark moody background, sacred space aesthetic, shallow depth of field, fine art photography",
    },

    # ── Mythology (1 niche) ───────────────────────────────────────────────

    "mythicalarchives": {
        "key_products": [
            "Zeus", "Odin", "Thor", "Athena", "Anubis", "Ra",
            "Medusa", "Hercules", "Loki", "Aphrodite",
            "Mjolnir", "Excalibur", "Pandora's Box", "Trident of Poseidon",
        ],
        "expert_tips": [
            "Greek mythology alone has over 300 named gods and titans — every culture has equally rich stories",
            "Most mythological monsters represent real human fears — Medusa represents the danger of unchecked anger",
            "Norse mythology predicted the end of the world in Ragnarok — but it also promised rebirth after destruction",
            "Egyptian mythology is the oldest recorded — the Pyramid Texts date back to 2400 BCE",
            "Many myths share the same archetypes across cultures — the flood myth appears in Greek, Mesopotamian, Hindu, and Native American traditions",
        ],
        "talking_points": {
            "greek": "Zeus wielded lightning and ruled Olympus, Athena was born from his head fully armored, Poseidon controlled all oceans, Hades ruled the underworld and was not evil",
            "norse": "Odin sacrificed his eye for wisdom at Mimir's Well, Thor's hammer Mjolnir could level mountains, Loki was a shapeshifter who caused Ragnarok",
            "egyptian": "Ra sailed his sun boat across the sky each day, Anubis weighed hearts against a feather to judge the dead, Isis reassembled Osiris after Set murdered him",
            "monsters": "Medusa's gaze turned men to stone, the Minotaur was trapped in the Labyrinth of Crete, Fenrir the wolf was so powerful the gods chained him with magic",
            "artifacts": "Excalibur could only be drawn by the true king, Pandora's Box released all evils but kept hope inside, the Holy Grail promised eternal life",
        },
        "visual_subjects": {
            "default": "ancient Greek temple with white marble columns, bright blue sky with white clouds, sunlit Mediterranean landscape",
            "gods": "powerful deity figure in white marble hall, bright natural daylight streaming in, vivid blue and white palette, epic scale",
            "monsters": "mythological creature in vivid fantasy landscape, bright colorful scenery, epic digital art, clear blue sky",
            "battles": "epic mythological battle scene, bright vivid sky, bold saturated colors, clear sharp details",
            "artifacts": "legendary artifact with bright magical glow, white marble pedestal, clean bright setting, vivid blue and gold details",
        },
        "style_suffix": ", vivid bright digital painting, clean natural daylight, saturated bold colors, sharp detailed textures, bright highlights, no shadows, no dark tones, 4K illustration, masterwork quality",
    },

    # ── Lifestyle (4 niches) ──────────────────────────────────────────────

    "bulletjournals": {
        "key_products": [
            "Leuchtturm1917 A5 dotted", "Archer & Olive notebook",
            "Tombow Dual Brush Pens", "Micron pens", "Mildliner highlighters",
            "washi tape", "Staedtler triplus fineliners", "ruler and stencils",
            "Pilot Juice gel pens", "correction tape",
        ],
        "expert_tips": [
            "Start with a simple index, future log, monthly spread, and daily log — that's all you need",
            "The Leuchtturm1917 is the gold standard bullet journal — it has numbered pages, an index, and lay-flat binding",
            "Don't compare your journal to Instagram — function over beauty every single time",
            "Use migration at the end of each month — move incomplete tasks forward and drop what doesn't matter anymore",
            "Habit trackers work best with 3-5 habits max — tracking too many leads to burnout and blank pages",
        ],
        "talking_points": {
            "setup": "Start with the official Ryder Carroll method: index, future log, monthly log, daily log — add collections only when you need them",
            "supplies": "Leuchtturm1917 for the best paper quality with no ghosting, Tombow markers for headers, Micron pens for everyday writing, Mildliners for subtle color coding",
            "spreads": "Mood trackers reveal emotional patterns over months, habit trackers build consistency, finance trackers replace budgeting apps, sleep logs improve rest",
            "minimalist": "Minimalist journals use only black ink and simple symbols — they're faster to maintain and just as effective as decorated spreads",
        },
        "visual_subjects": {
            "default": "open bullet journal with colorful spread, pens and washi tape nearby, cozy desk setup, overhead view",
            "supplies": "bullet journal supplies flat lay, pens markers washi tape, organized and colorful, clean white background",
            "spreads": "detailed bullet journal monthly spread, hand-lettered headers, colorful habit tracker, artistic layout",
            "minimal": "minimalist bullet journal page, clean black ink, simple layout, dotted grid visible, zen aesthetic",
        },
        "style_suffix": ", bright natural lighting, cozy warm interior, lifestyle photography, inviting atmosphere, editorial style, shallow depth of field, 4K",
    },

    "theconnectedhaven": {
        "key_products": [
            "Google Family Link", "Apple Screen Time", "Circle Home Plus",
            "Amazon Kids+", "Kindle Paperwhite Kids", "Tile tracker",
            "shared family calendar apps", "Cozi Family Organizer",
            "meal planning apps", "chore chart templates",
        ],
        "expert_tips": [
            "Set up a family command center — one place for calendars, to-do lists, and meal plans saves hours of mental load each week",
            "Screen time limits work better as routines than restrictions — 'screens off at 7pm' is easier to enforce than '2 hours per day'",
            "A Sunday 30-minute family meeting prevents most weekday chaos — review the calendar, plan meals, assign chores",
            "Use shared digital lists for groceries, tasks, and goals — Cozi and Apple Reminders both work great for families",
            "Batch cooking on Sundays saves 5+ hours of weeknight cooking — prep proteins, grains, and vegetables separately",
        ],
        "talking_points": {
            "organization": "A family command center with shared calendar, meal plan, and chore chart eliminates 90% of daily decision fatigue",
            "screen time": "Circle Home Plus manages every device on your WiFi for $99, Google Family Link is free for Android, Apple Screen Time is built into every iPhone",
            "meal planning": "Batch cooking saves 5+ hours per week, theme nights like Taco Tuesday reduce decision fatigue, involving kids in cooking builds life skills",
            "routines": "Morning routines should be no more than 5 steps for kids, visual routine charts work better than verbal reminders, consistency beats perfection",
        },
        "visual_subjects": {
            "default": "organized family kitchen with calendar on wall, warm lighting, cozy and functional home",
            "tech": "family using shared tablet app together, living room setting, warm comfortable atmosphere",
            "cooking": "family meal prep in bright kitchen, fresh ingredients on counter, warm welcoming atmosphere",
            "organization": "organized home office or command center, labels and calendars visible, clean productive space",
        },
        "style_suffix": ", bright natural lighting, cozy warm interior, lifestyle photography, inviting atmosphere, editorial style, shallow depth of field, 4K",
    },

    "familyflourish": {
        "key_products": [
            "STEM toys", "art supplies kits", "outdoor play equipment",
            "board games (Ticket to Ride, Catan Junior)", "cooking kits for kids",
            "educational apps (Khan Academy Kids, Duolingo)", "nature journals",
            "family budget planners", "chore reward systems",
        ],
        "expert_tips": [
            "Screen-free activities after school reduce meltdowns by creating a decompression window — try 30 minutes of outdoor play first",
            "Family game nights build problem-solving skills and emotional regulation — aim for once a week",
            "The 20-minute rule: if an activity keeps kids engaged for 20 minutes without a screen, it's a winner — stock up on those",
            "Involving kids in meal prep from age 3 builds independence and reduces picky eating",
            "Budget-friendly family fun: library visits are free, nature walks cost nothing, and cooking together beats eating out",
        ],
        "talking_points": {
            "activities": "STEM kits from KiwiCo teach science through play, nature scavenger hunts are free and educational, cooking together builds math skills and independence",
            "education": "Khan Academy Kids is the best free learning app for ages 2-8, Duolingo makes language learning a game, Reading Eggs teaches phonics effectively",
            "budgeting": "The envelope method works great for family budgets, meal planning saves $200+ per month on groceries, library programs replace expensive activities",
            "routines": "Consistent bedtime routines improve behavior the next day, morning charts reduce nagging by 80%, after-school routines prevent homework battles",
        },
        "visual_subjects": {
            "default": "happy family doing activity together, bright warm room, joyful atmosphere, natural lighting",
            "activities": "kids doing arts and crafts at table, colorful supplies, bright playroom, creative energy",
            "outdoor": "family playing outdoors in park, sunshine, green nature, active and happy",
            "cooking": "parent and child cooking together in kitchen, flour on hands, warm joyful moment",
        },
        "style_suffix": ", bright natural lighting, cozy warm interior, lifestyle photography, inviting atmosphere, editorial style, shallow depth of field, 4K",
    },

    "celebrationseason": {
        "key_products": [
            "balloon garland kits", "LED fairy lights", "tablescape supplies",
            "Cricut cutting machine", "printable party templates",
            "themed tableware sets", "photo booth props",
            "cake decorating supplies", "DIY centerpiece materials",
        ],
        "expert_tips": [
            "Balloon garlands are the highest-impact, lowest-cost decoration — a $15 kit creates a $200 look",
            "Always plan decorations around one focal point — a dessert table, photo wall, or entrance arch",
            "LED fairy lights in glass vases cost $5 and create stunning centerpieces for any event",
            "Shop dollar stores and Amazon basics for tableware — guests care about the experience, not the plate brand",
            "A Cricut machine pays for itself after 3 parties — custom banners, cake toppers, and invitations",
        ],
        "talking_points": {
            "decorations": "Balloon garlands are the number one party trend, a single color palette makes any space look cohesive, real greenery mixed with faux flowers looks expensive on a budget",
            "diy": "Cricut machines create custom party decor for pennies, printable templates save $50+ per party, mason jar centerpieces with fairy lights never go out of style",
            "themes": "Trending themes for 2026: cottagecore, disco ball, tropical, minimalist chic, retro 70s — pick one and commit to it",
            "food": "Grazing boards replace expensive catering and look stunning in photos, themed cookies from molds are impressive and easy, a signature cocktail elevates any party",
        },
        "visual_subjects": {
            "default": "beautifully decorated party table, colorful balloons, warm festive lighting, celebration atmosphere",
            "balloons": "stunning balloon garland arch, pastel colors, party decoration, elegant and festive",
            "table": "gorgeous tablescape with centerpieces, candles, place settings, elegant dinner party",
            "diy": "DIY party crafts in progress, supplies spread out, creative workspace, colorful materials",
        },
        "style_suffix": ", bright natural lighting, cozy warm interior, lifestyle photography, inviting atmosphere, editorial style, shallow depth of field, 4K",
    },
}

# Category-level style suffix fallbacks
_CATEGORY_STYLE_SUFFIXES = {
    "tech": ", clean product photography, soft ambient lighting, modern minimalist interior, shallow depth of field, editorial style, sharp focus, 4K, professional",
    "ai_news": ", futuristic digital environment, holographic displays, neon accents, clean tech aesthetic, sharp focus, 4K, professional photography",
    "witchcraft": ", mystical atmosphere, candlelight, soft ethereal glow, dark moody background, sacred space aesthetic, shallow depth of field, fine art photography",
    "mythology": ", vivid bright digital painting, clean natural daylight, saturated bold colors, sharp detailed textures, bright highlights, no shadows, no dark tones, 4K illustration, masterwork quality",
    "lifestyle": ", bright natural lighting, cozy warm interior, lifestyle photography, inviting atmosphere, editorial style, shallow depth of field, 4K",
    "fitness": ", dynamic action photography, high contrast, gym or outdoor setting, motivational energy, sharp focus, sports photography, 4K",
    "business": ", professional corporate aesthetic, clean modern workspace, data visualization, confident atmosphere, editorial photography, 4K",
}


def get_domain_expertise(niche: str, topic: str = None) -> dict:
    """Get domain expertise for a niche, optionally with topic-matched content surfaced.

    Args:
        niche: Niche ID (e.g. 'smarthomewizards')
        topic: Optional topic string to match against talking_points and visual_subjects

    Returns:
        Full expertise dict with 'matched_talking_point' and 'matched_visual' keys
        added when a topic match is found.
    """
    expertise = DOMAIN_EXPERTISE.get(niche, {})
    if not expertise:
        return {}

    result = dict(expertise)

    if topic:
        topic_lower = topic.lower()

        # Find best matching talking point
        best_match = None
        best_score = 0
        for key, value in expertise.get("talking_points", {}).items():
            # Check if any word from the talking point key appears in the topic
            key_words = key.lower().split()
            score = sum(1 for w in key_words if w in topic_lower)
            if score > best_score:
                best_score = score
                best_match = (key, value)

        if best_match:
            result["matched_talking_point"] = {
                "topic": best_match[0],
                "content": best_match[1],
            }

        # Find best matching visual subject
        best_visual = None
        best_visual_score = 0
        for key, value in expertise.get("visual_subjects", {}).items():
            if key == "default":
                continue
            key_words = key.lower().split()
            score = sum(1 for w in key_words if w in topic_lower)
            if score > best_visual_score:
                best_visual_score = score
                best_visual = (key, value)

        if best_visual:
            result["matched_visual"] = {
                "topic": best_visual[0],
                "description": best_visual[1],
            }

    return result


def get_style_suffix(niche: str) -> str:
    """Get the image style suffix for a niche.

    Returns niche-specific suffix if available, falls back to category suffix.
    """
    expertise = DOMAIN_EXPERTISE.get(niche, {})
    if expertise and "style_suffix" in expertise:
        return expertise["style_suffix"]

    # Fallback to category suffix
    from .niche_profiles import get_niche_profile
    profile = get_niche_profile(niche)
    category = profile.get("category", "lifestyle")
    return _CATEGORY_STYLE_SUFFIXES.get(category, _CATEGORY_STYLE_SUFFIXES["lifestyle"])
