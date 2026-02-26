#!/usr/bin/env python3
"""
ForgeFiles Caption Engine
===========================
Production-grade caption, hashtag, and metadata generation system.
20+ variants per platform, A/B testing variants, voiceover scripts,
UTM tracking, engagement hooks, and community-native tone per platform.
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime


# ============================================================================
# HASHTAG BANKS (researched, high-performing tags by category)
# ============================================================================

HASHTAG_BANKS = {
    "core_3dprint": [
        "#3dprinting", "#3dprint", "#3dprinted", "#3dprinter",
        "#stlfiles", "#stlfile", "#3dmodel", "#3ddesign",
        "#printinplace", "#fdm", "#resinprint", "#sla",
    ],
    "maker_community": [
        "#maker", "#makerspace", "#makersofinstagram", "#makersmovement",
        "#diy", "#diycrafts", "#handmade", "#crafts",
        "#thingiverse", "#printables", "#myminifactory",
    ],
    "engagement": [
        "#satisfying", "#oddlysatisfying", "#asmr", "#timelapse",
        "#cool", "#amazing", "#mindblowing", "#viral",
    ],
    "product_showcase": [
        "#productdesign", "#industrialdesign", "#prototype",
        "#designinspiration", "#productlaunch", "#newdesign",
    ],
    "hobby_printing": [
        "#3dprintingcommunity", "#3dprintinglife", "#ender3",
        "#prusa", "#bambulab", "#3dprintingprojects",
    ],
    "platform_tiktok": [
        "#fyp", "#foryou", "#foryoupage", "#trending", "#viral",
        "#techtok", "#makertok", "#3dprintingtiktok",
    ],
    "platform_reels": [
        "#reels", "#reelsinstagram", "#reelsviral",
        "#instareels", "#explorepage",
    ],
    "platform_pinterest": [
        "#pinterestinspired", "#pinterestfinds",
        "#homedecor", "#giftideas", "#unique",
    ],
}


def _pick_hashtags(categories, count=15):
    """Pick a diverse set of hashtags from specified categories."""
    pool = []
    for cat in categories:
        pool.extend(HASHTAG_BANKS.get(cat, []))
    pool = list(set(pool))
    random.shuffle(pool)
    return pool[:count]


# ============================================================================
# ENGAGEMENT HOOKS (first line / first 3 seconds)
# ============================================================================

HOOKS = {
    "curiosity": [
        "You need to see this.",
        "Wait for the reveal...",
        "This changes everything.",
        "I can't believe how this turned out.",
        "You've never seen anything like this.",
        "Stop scrolling — look at this.",
    ],
    "question": [
        "What would you print this in?",
        "Would you print this?",
        "Can your printer handle this?",
        "What color would you choose?",
        "Who needs one of these?",
        "Could this be your next print?",
    ],
    "pov": [
        "POV: You just found your next print project",
        "POV: Your printer queue just got longer",
        "POV: You discover the perfect STL file",
    ],
    "flex": [
        "Just dropped this.",
        "New design just dropped.",
        "Fresh off the build plate.",
        "Print-ready and waiting for you.",
    ],
    "challenge": [
        "Bet you can't print this in under 4 hours.",
        "Show me your print of this.",
        "Tag someone who needs to print this.",
        "Print this and tag me.",
    ],
}


def _pick_hook(style=None):
    """Pick a random engagement hook."""
    if style and style in HOOKS:
        return random.choice(HOOKS[style])
    all_hooks = []
    for hooks in HOOKS.values():
        all_hooks.extend(hooks)
    return random.choice(all_hooks)


# ============================================================================
# CALL-TO-ACTION VARIANTS
# ============================================================================

CTAS = {
    "tiktok": [
        "Link in bio for the STL",
        "STL link in bio",
        "Grab the file — link in bio",
        "Download link in bio",
        "Bio link for the file",
    ],
    "reels": [
        "Link in bio to download the STL file.",
        "Grab this design — link in bio.",
        "STL file available now. Link in bio.",
        "Download the print-ready file. Link in bio.",
    ],
    "youtube": [
        "Download the STL file — link in the description below.",
        "Grab this design from the link in the description.",
        "STL download link in the description. Happy printing!",
    ],
    "pinterest": [
        "Click through to download the STL file.",
        "Visit ForgeFiles for the print-ready download.",
        "Get this design at forgefiles.com",
    ],
    "reddit": [],  # Reddit hates CTAs — keep it organic
}


# ============================================================================
# TIKTOK CAPTIONS (25+ variants across styles)
# ============================================================================

def generate_tiktok_captions(model_name, mode="turntable", specs_short="", variant_count=3):
    """Generate TikTok caption variants with hooks and hashtags."""
    display = _display_name(model_name)
    tags_str = " ".join(_pick_hashtags(["core_3dprint", "engagement", "platform_tiktok"], 12))
    cta = random.choice(CTAS["tiktok"])

    templates_turntable = [
        "{hook}\n\n{name} — now available as an STL.\n{cta}\n\n{tags}",
        "{hook} {name}\n\n{specs}\n\n{cta}\n\n{tags}",
        "{name}\n\nWould you print this? Drop your filament color below.\n{cta}\n\n{tags}",
        "New drop: {name}\n\nDesigned for FDM and resin.\n{cta}\n\n{tags}",
        "{hook}\n\n{name} STL file. Print it yourself.\n{cta}\n\n{tags}",
        "This {name} hits different in person.\n{cta}\n\n{tags}",
        "Can't stop watching this {name} spin.\n\n{cta}\n\n{tags}",
        "Tell me this {name} isn't fire. I'll wait.\n\n{cta}\n\n{tags}",
        "Your printer wants this {name}.\n\n{cta}\n\n{tags}",
        "Just finished this {name} design. Thoughts?\n\n{cta}\n\n{tags}",
        "{name} — print ready, no supports needed.\n{cta}\n\n{tags}",
        "Which color would you print this {name} in?\n\n{cta}\n\n{tags}",
    ]

    templates_dramatic = [
        "{hook}\n\nDramatic reveal: {name}\n{cta}\n\n{tags}",
        "When the lighting hits right...\n\n{name}\n{cta}\n\n{tags}",
        "Cinema mode: ON.\n\n{name} reveal.\n{cta}\n\n{tags}",
        "{name} — the cinematic showcase.\n{cta}\n\n{tags}",
    ]

    templates_wireframe = [
        "From wireframe to reality.\n\n{name}\n{cta}\n\n{tags}",
        "Watch this {name} build itself.\n\n{cta}\n\n{tags}",
        "The design process of {name}, visualized.\n{cta}\n\n{tags}",
    ]

    templates_material = [
        "Same {name}, different vibes.\n\nWhich finish would you pick?\n{cta}\n\n{tags}",
        "{name} in every material.\n\nWhich one wins?\n{cta}\n\n{tags}",
    ]

    template_map = {
        "turntable": templates_turntable,
        "dramatic": templates_dramatic,
        "wireframe": templates_wireframe,
        "material": templates_material,
    }
    templates = template_map.get(mode, templates_turntable)

    variants = []
    used = set()
    for _ in range(variant_count):
        for attempt in range(20):
            tmpl = random.choice(templates)
            hook = _pick_hook()
            tags_str = " ".join(_pick_hashtags(["core_3dprint", "engagement", "platform_tiktok"], 12))
            caption = tmpl.format(
                hook=hook, name=display, cta=cta,
                specs=specs_short, tags=tags_str
            )
            if caption not in used:
                used.add(caption)
                variants.append(caption)
                break

    return variants


# ============================================================================
# INSTAGRAM REELS CAPTIONS
# ============================================================================

def generate_reels_captions(model_name, mode="turntable", specs_short="", variant_count=3):
    """Generate Instagram Reels caption variants."""
    display = _display_name(model_name)
    cta = random.choice(CTAS["reels"])

    templates = [
        "{name} — Now available as a printable STL file.\n\nDesigned for FDM & resin printers.\n{cta}\n\n{tags}",
        "New design drop: {name}\n\n360-degree showcase of every detail.\nPrint-ready STL available now.\n\n{cta}\n\n{tags}",
        "Introducing: {name}\n\nEvery angle. Every detail. Ready for your printer.\n\n{specs}\n\n{cta}\n\n{tags}",
        "{name} — designed, tested, print-ready.\n\nAvailable as an STL download.\n{cta}\n\n{tags}",
        "We designed {name} for the detail-obsessed.\n\nZoom in. Every surface is intentional.\n\n{cta}\n\n{tags}",
        "The newest addition to our catalog:\n\n{name}\n\nFDM or resin — your choice.\n{cta}\n\n{tags}",
        "This {name} looks even better off the build plate.\n\n{specs}\n\n{cta}\n\n{tags}",
        "{name}\n\nFrom concept to print file.\nGrab the STL and make it yours.\n\n{cta}\n\n{tags}",
        "Built for printers. Designed for detail.\n\n{name} is live.\n{cta}\n\n{tags}",
        "What your next weekend project should look like:\n\n{name}\n\n{cta}\n\n{tags}",
    ]

    variants = []
    used = set()
    for _ in range(variant_count):
        for attempt in range(20):
            tmpl = random.choice(templates)
            tags_str = " ".join(_pick_hashtags(["core_3dprint", "maker_community", "platform_reels"], 20))
            caption = tmpl.format(name=display, cta=cta, specs=specs_short, tags=tags_str)
            if caption not in used:
                used.add(caption)
                variants.append(caption)
                break

    return variants


# ============================================================================
# YOUTUBE METADATA
# ============================================================================

def generate_youtube_metadata(model_name, mode="turntable", print_specs="", variant_count=2):
    """Generate YouTube title, description, and tags with SEO optimization."""
    display = _display_name(model_name)

    title_templates = [
        "{name} | 3D Printable STL File | 360 Showcase",
        "{name} — Premium 3D Print Design | ForgeFiles",
        "{name} | Print-Ready STL | Full Turntable",
        "3D Printable {name} | STL Download | ForgeFiles",
        "{name} | Detailed 3D Print Showcase",
        "NEW: {name} — 3D Printable Design Showcase",
    ]

    description_template = """{name} — Full 360-degree showcase of this premium 3D printable design.

Download the print-ready STL file: [LINK]

---

PRINT SPECIFICATIONS:
{specs}

---

ABOUT THIS DESIGN:
The {name} is a high-detail 3D printable model designed for both FDM and resin printers. Every surface has been optimized for clean prints with minimal supports.

ABOUT FORGEFILES:
ForgeFiles creates premium 3D printable designs — every file is tested and print-ready. Browse our full catalog at forgefiles.com

---

TIMESTAMPS:
0:00 Intro
0:03 Full rotation
0:15 Detail closeups

---

{tags_text}"""

    tags_base = [
        "3d printing", "stl files", "3d printable", display.lower(),
        "forgefiles", "3d printer", "3d models", "printable designs",
        "maker", "3d design", "stl download", "print ready",
        "fdm printing", "resin printing", "3d print showcase",
        "3d printing ideas", "3d printer projects", "cool 3d prints",
    ]

    variants = []
    for i in range(variant_count):
        title = random.choice(title_templates).format(name=display)
        tags_shuffled = tags_base.copy()
        random.shuffle(tags_shuffled)
        tags_text = "#" + " #".join(t.replace(" ", "") for t in tags_shuffled[:15])
        desc = description_template.format(
            name=display,
            specs=print_specs or _default_print_specs(),
            tags_text=tags_text,
        )
        variants.append({
            "title": title,
            "description": desc,
            "tags": ", ".join(tags_shuffled),
            "category": "Science & Technology",
        })

    return variants


# ============================================================================
# PINTEREST METADATA
# ============================================================================

def generate_pinterest_metadata(model_name, mode="turntable", print_specs="", variant_count=2):
    """Generate Pinterest pin titles and descriptions with SEO keywords."""
    display = _display_name(model_name)

    title_templates = [
        "{name} | 3D Printable STL File",
        "{name} — Download & Print | ForgeFiles",
        "3D Printable {name} | Ready to Print STL",
        "{name} | Premium 3D Print Design",
    ]

    desc_templates = [
        "Download this stunning {name} 3D printable STL file from ForgeFiles. "
        "Compatible with all FDM and resin 3D printers. "
        "High-detail design, tested and print-ready. "
        "Perfect for your next 3D printing project!\n\n"
        "{specs}\n\n"
        "{tags}",

        "{name} — a premium 3D printable design available for instant download. "
        "Works with PLA, ABS, PETG, and resin. "
        "Every detail optimized for beautiful prints. "
        "Visit ForgeFiles for the STL file.\n\n"
        "{tags}",

        "Looking for your next print project? The {name} is a show-stopping design "
        "available as an instant-download STL file. "
        "Designed for home 3D printers — no modifications needed.\n\n"
        "{specs}\n\n"
        "{tags}",
    ]

    variants = []
    for _ in range(variant_count):
        tags_str = " ".join(_pick_hashtags(["core_3dprint", "maker_community", "platform_pinterest", "product_showcase"], 10))
        title = random.choice(title_templates).format(name=display)
        desc = random.choice(desc_templates).format(
            name=display, specs=print_specs, tags=tags_str
        )
        variants.append({"title": title, "description": desc})

    return variants


# ============================================================================
# REDDIT CAPTIONS (community-native, NO brand voice)
# ============================================================================

def generate_reddit_captions(model_name, mode="turntable", print_specs="", variant_count=3):
    """Generate Reddit post titles that sound like a real community member.
    No hashtags, no CTAs, no brand language. Just genuine hobbyist tone.
    """
    display = _display_name(model_name)

    title_templates = [
        "Just finished designing this {name} — what do you think?",
        "{name} — feedback welcome!",
        "Designed a {name}, pretty happy with how it turned out",
        "New design: {name}. Would love to hear your thoughts.",
        "My latest design: {name}. Took longer than expected but worth it.",
        "Finally done with this {name} design. Any suggestions before I publish?",
        "{name} — been working on this one for a while. Thoughts?",
        "First attempt at a {name} design. How'd I do?",
        "Sharing my {name} design — tried to maximize detail while keeping supports minimal",
        "{name} printed in gray PLA. STL available if anyone's interested.",
        "This {name} printed way better than I expected",
        "Designed and printed this {name} over the weekend. Pretty stoked.",
        "My {name} design — tried to balance detail with printability",
        "Would you change anything about this {name}?",
        "Spent the week on this {name}. Here's the turntable.",
    ]

    body_templates = [
        "Designed this in Fusion 360. Prints well on FDM with {layer}mm layers. "
        "Happy to share the STL if anyone wants it.",

        "Here's a 360 of my latest design. {specs}\n\n"
        "Let me know if you see any issues — always looking to improve.",

        "Printed on my {printer} with {material}. "
        "No supports needed on most orientations. "
        "STL is free if anyone wants to try it.",

        "{specs}\n\nLet me know what you think. "
        "I'll drop the file link if there's interest.",
    ]

    printers = ["Ender 3 V3", "Bambu A1 Mini", "Prusa MK4", "Bambu P1S", "Ender 5 S1"]
    materials = ["PLA", "PETG", "gray PLA", "matte black PLA", "silk silver PLA"]
    layers = ["0.16", "0.20", "0.12"]

    variants = []
    used_titles = set()
    for _ in range(variant_count):
        for attempt in range(20):
            title = random.choice(title_templates).format(name=display)
            if title not in used_titles:
                used_titles.add(title)
                break

        body = random.choice(body_templates).format(
            name=display,
            specs=print_specs or _default_print_specs(),
            printer=random.choice(printers),
            material=random.choice(materials),
            layer=random.choice(layers),
        )

        variants.append({
            "title": title,
            "body": body,
            "subreddits": ["r/3Dprinting", "r/functionalprint", "r/3dprintingdms"],
            "flair": "Design",
        })

    return variants


# ============================================================================
# YOUTUBE SHORTS / TIKTOK-STYLE SHORT CAPTIONS
# ============================================================================

def generate_shorts_captions(model_name, mode="turntable", specs_short="", variant_count=3):
    """Generate YouTube Shorts caption variants."""
    display = _display_name(model_name)

    templates = [
        "{hook} {name} #3dprinting #stl #shorts",
        "{name} — print-ready STL. Full video on the channel. #3dprinting #shorts",
        "New drop: {name}. Link in description. #3dprinting #shorts #maker",
        "{hook}\n\n{name} #3dprinting #stlfiles #shorts",
        "Would you print this? {name} #3dprinting #shorts",
    ]

    variants = []
    for _ in range(variant_count):
        hook = _pick_hook()
        caption = random.choice(templates).format(hook=hook, name=display)
        variants.append(caption)

    return variants


# ============================================================================
# VOICEOVER SCRIPT GENERATION
# ============================================================================

def generate_voiceover_script(model_name, print_specs="", duration_seconds=15,
                               sequence_name=None):
    """Generate a natural-sounding narration script for YouTube videos.
    Output as text ready to feed into ElevenLabs or similar TTS.

    When sequence_name is provided, the script is timed to match the
    shot sequence for cinematic videos.
    """
    display = _display_name(model_name)

    # Sequence-aware scripts that match cinematic shot timings
    if sequence_name == "showcase_short":
        # 15-20s: dramatic reveal → turntable → close-up → hero spin
        scripts = [
            f"Check out the {display}. "
            f"Designed for home 3D printers, every detail optimized for clean prints. "
            f"Look at that surface quality. "
            f"Grab the STL — link in bio.",

            f"Introducing the {display}. "
            f"Let's take a closer look at this print-ready design. "
            f"The detail speaks for itself. "
            f"Download link below.",

            f"The {display} — our latest drop. "
            f"Watch how the light catches every surface. "
            f"This one prints beautifully in PLA or resin. "
            f"STL available now.",
        ]
        return random.choice(scripts)

    if sequence_name == "showcase_full":
        # 30-45s: reveal → orbital → wireframe → close-ups → pedestal → hero
        scripts = [
            f"Welcome to ForgeFiles. This is the {display}. "
            f"Let's start with the full design, rotating around to see every angle. "
            f"Notice the geometry — optimized for both form and printability. "
            f"Here's how the wireframe translates to the finished surface. "
            f"Up close, you can see the level of detail in every curve. "
            f"{print_specs or _default_print_specs_spoken()} "
            f"The STL file is ready for download — link in the description.",

            f"Today we're showcasing the {display}. "
            f"As we orbit around, pay attention to how the surfaces catch the light. "
            f"From wireframe to solid — every polygon serves a purpose. "
            f"The detail holds up even at this close range. "
            f"Print settings: {print_specs or _default_print_specs_spoken()} "
            f"Grab the file from the link below. Happy printing.",
        ]
        return random.choice(scripts)

    if sequence_name == "hero_video":
        # 60-90s: reveal → slow turntable → close-ups → wireframe → material carousel → orbital → hero
        scripts = [
            f"Welcome to ForgeFiles. Today we're diving deep into the {display}. "
            f"Let's take a slow 360 to appreciate the full design. "
            f"Every surface has been carefully sculpted and tested for printing. "
            f"\n\n"
            f"Now let's get up close. Notice the fine detail work here — "
            f"the ridges, the curves, the texture. All designed to print cleanly "
            f"with minimal supports. "
            f"\n\n"
            f"Watch how the wireframe reveals the underlying geometry. "
            f"This is what makes a good print — clean topology and intentional edges. "
            f"\n\n"
            f"And here's one of our favorite parts — the material showcase. "
            f"See how this design looks in different finishes. "
            f"Standard PLA, silk silver, and crystal clear resin. "
            f"Each one brings out different aspects of the design. "
            f"\n\n"
            f"Print specifications: {print_specs or _default_print_specs_spoken()} "
            f"\n\n"
            f"The STL file is available for download right now. "
            f"Link is in the description below. "
            f"If you print this, tag us — we'd love to see your results. "
            f"Subscribe for new designs every week.",

            f"This is the {display} from ForgeFiles. "
            f"Let me walk you through every detail of this design. "
            f"Starting with the overall form — you can see the proportions "
            f"are balanced for both aesthetics and printability. "
            f"\n\n"
            f"Zooming in now — the surface detail is something we're really proud of. "
            f"It's designed to look great at any layer height. "
            f"\n\n"
            f"The wireframe view shows the clean topology underneath. "
            f"No wasted geometry, no problematic overhangs. "
            f"\n\n"
            f"Let's see the material options. "
            f"Gray PLA gives you the classic maker look. "
            f"Silk silver adds that premium metallic finish. "
            f"And clear resin really makes the details pop. "
            f"\n\n"
            f"Here are the recommended settings: "
            f"{print_specs or _default_print_specs_spoken()} "
            f"\n\n"
            f"Download the STL from the link in the description. "
            f"Happy printing!",
        ]
        return random.choice(scripts)

    # Non-sequence scripts (original behavior)
    short_scripts = [
        f"Take a look at the {display}. "
        f"Every detail is designed with printing in mind. "
        f"This model works great with both FDM and resin printers. "
        f"Download the STL and see for yourself.",

        f"Here's our latest design — the {display}. "
        f"Rotating it around so you can see every angle. "
        f"This one prints clean with minimal supports. "
        f"Link in the description if you want the file.",

        f"The {display}. "
        f"A detailed, print-ready design built for home printers. "
        f"Whether you're running PLA, PETG, or resin, this file is optimized for quality. "
        f"Check the description for the download.",
    ]

    long_scripts = [
        f"Welcome to ForgeFiles. Today we're showcasing the {display}. "
        f"Let's take a full 360-degree look at this design. "
        f"You can see the level of detail in every surface — "
        f"this isn't just a model, it's built specifically for 3D printing. "
        f"\n\n"
        f"Here are the recommended print settings: "
        f"{print_specs or _default_print_specs_spoken()} "
        f"\n\n"
        f"The STL file is available for download — "
        f"link is in the description below. "
        f"If you print this, we'd love to see your results. "
        f"Drop a photo in the comments. Happy printing.",

        f"This is the {display} from ForgeFiles. "
        f"Let me walk you through this design. "
        f"As you can see from the turntable, every angle has been considered. "
        f"The geometry is optimized for clean FDM prints "
        f"and the detail really shines in resin. "
        f"\n\n"
        f"Print specs are in the description, "
        f"but here's the quick version: "
        f"{print_specs or _default_print_specs_spoken()} "
        f"\n\n"
        f"Grab the file from the link below. "
        f"Subscribe for more designs every week.",
    ]

    if duration_seconds <= 20:
        script = random.choice(short_scripts)
    else:
        script = random.choice(long_scripts)

    return script


# ============================================================================
# UTM / TRACKING
# ============================================================================

def generate_tracking_links(model_name, base_url="https://forgefiles.com", platforms=None):
    """Generate UTM-tagged links for each platform and content variant."""
    if platforms is None:
        platforms = ["tiktok", "reels", "youtube", "pinterest", "reddit"]

    slug = model_name.lower().replace(" ", "-").replace("_", "-")
    content_id = hashlib.md5(f"{model_name}{datetime.now().isoformat()}".encode()).hexdigest()[:8]

    links = {}
    for platform in platforms:
        links[platform] = (
            f"{base_url}/designs/{slug}"
            f"?utm_source={platform}"
            f"&utm_medium=social"
            f"&utm_campaign=content_pipeline"
            f"&utm_content={content_id}"
        )

    return {"links": links, "content_id": content_id}


# ============================================================================
# POSTING SCHEDULE METADATA
# ============================================================================

BEST_POSTING_TIMES = {
    "tiktok": {
        "monday": ["07:00", "10:00", "22:00"],
        "tuesday": ["09:00", "12:00", "17:00"],
        "wednesday": ["07:00", "11:00", "20:00"],
        "thursday": ["09:00", "12:00", "19:00"],
        "friday": ["05:00", "13:00", "15:00"],
        "saturday": ["11:00", "19:00", "20:00"],
        "sunday": ["07:00", "10:00", "16:00"],
    },
    "reels": {
        "monday": ["06:00", "10:00", "20:00"],
        "tuesday": ["09:00", "12:00", "14:00"],
        "wednesday": ["07:00", "11:00", "19:00"],
        "thursday": ["09:00", "12:00", "19:00"],
        "friday": ["05:00", "13:00", "15:00"],
        "saturday": ["09:00", "17:00", "20:00"],
        "sunday": ["07:00", "10:00", "16:00"],
    },
    "youtube": {
        "weekday": ["14:00", "16:00", "17:00"],
        "weekend": ["09:00", "12:00", "15:00"],
    },
    "pinterest": {
        "daily": ["20:00", "21:00", "22:00"],
        "best_days": ["saturday", "sunday"],
    },
    "reddit": {
        "daily": ["06:00", "10:00", "14:00"],
        "best_days": ["monday", "wednesday", "saturday"],
        "note": "Post early morning EST for maximum US visibility",
    },
}


def generate_schedule_metadata():
    """Return best posting times per platform."""
    return BEST_POSTING_TIMES


# ============================================================================
# A/B VARIANT MASTER GENERATOR
# ============================================================================

def generate_all_captions(model_name, mode="turntable", print_specs="",
                          print_specs_short="", platforms=None, variant_count=3,
                          sequence_name=None):
    """Master function: generate all caption variants for all platforms.
    Returns a structured dict ready for the pipeline manifest.

    Args:
        sequence_name: If provided, generates sequence-aware voiceover scripts
                       matching the shot sequence timing.
    """
    if platforms is None:
        platforms = ["tiktok", "reels", "youtube", "shorts", "pinterest", "reddit"]

    result = {
        "model": model_name,
        "display_name": _display_name(model_name),
        "generated": datetime.now().isoformat(),
        "platforms": {},
        "tracking": generate_tracking_links(model_name, platforms=platforms),
        "voiceover": {
            "script": generate_voiceover_script(
                model_name, print_specs, sequence_name=sequence_name
            ),
            "voice_recommendations": {
                "elevenlabs": {"voice": "George", "voice_id": "JBFqnCBsd6RMkjVDRZzb",
                               "stability": 0.5, "similarity_boost": 0.75},
                "style": "conversational, warm, confident — not salesy",
            },
            "sequence": sequence_name,
        },
        "schedule": generate_schedule_metadata(),
    }

    for platform in platforms:
        if platform == "tiktok":
            result["platforms"]["tiktok"] = {
                "variants": generate_tiktok_captions(model_name, mode, print_specs_short, variant_count),
                "type": "caption_text",
            }
        elif platform == "reels":
            result["platforms"]["reels"] = {
                "variants": generate_reels_captions(model_name, mode, print_specs_short, variant_count),
                "type": "caption_text",
            }
        elif platform == "youtube":
            result["platforms"]["youtube"] = {
                "variants": generate_youtube_metadata(model_name, mode, print_specs, variant_count),
                "type": "structured_metadata",
            }
        elif platform == "shorts":
            result["platforms"]["shorts"] = {
                "variants": generate_shorts_captions(model_name, mode, print_specs_short, variant_count),
                "type": "caption_text",
            }
        elif platform == "pinterest":
            result["platforms"]["pinterest"] = {
                "variants": generate_pinterest_metadata(model_name, mode, print_specs, variant_count),
                "type": "structured_metadata",
            }
        elif platform == "reddit":
            result["platforms"]["reddit"] = {
                "variants": generate_reddit_captions(model_name, mode, print_specs, variant_count),
                "type": "structured_post",
            }

    return result


# ============================================================================
# COLLECTION / SERIES DETECTION
# ============================================================================

def detect_collections(model_names):
    """Detect content series from naming patterns.
    E.g., dragon_head, dragon_body, dragon_tail → "Dragon" collection.
    """
    # Strategy: split on common separators and find shared prefixes
    prefix_groups = {}

    for name in model_names:
        clean = name.lower().replace("-", "_")
        parts = clean.split("_")
        if len(parts) >= 2:
            # Try prefixes of decreasing length
            for length in range(len(parts) - 1, 0, -1):
                prefix = "_".join(parts[:length])
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(name)

    # Filter to groups with 2+ members and meaningful prefix length
    collections = {}
    for prefix, members in prefix_groups.items():
        if len(members) >= 2 and len(prefix) >= 3:
            # Check this isn't a subset of a larger collection
            display = prefix.replace("_", " ").title()
            if display not in collections or len(members) > len(collections[display]):
                collections[display] = {
                    "prefix": prefix,
                    "models": sorted(set(members)),
                    "display_name": f"{display} Collection",
                    "count": len(set(members)),
                }

    # Remove subsets
    final = {}
    sorted_cols = sorted(collections.items(), key=lambda x: -x[1]["count"])
    used_models = set()
    for name, col in sorted_cols:
        new_models = [m for m in col["models"] if m not in used_models]
        if len(new_models) >= 2:
            col["models"] = new_models
            col["count"] = len(new_models)
            final[name] = col
            used_models.update(new_models)

    return final


# ============================================================================
# HELPERS
# ============================================================================

def _display_name(model_name):
    """Convert filename to display name."""
    return model_name.replace("_", " ").replace("-", " ").title()


def _default_print_specs():
    """Default print specs when STL analysis isn't available."""
    return (
        "Layer Height: 0.2mm\n"
        "Infill: 15-20%\n"
        "Supports: As needed\n"
        "Compatible: FDM and Resin printers"
    )


def _default_print_specs_spoken():
    """Default print specs in spoken form for voiceover."""
    return (
        "point two millimeter layers, "
        "fifteen to twenty percent infill, "
        "and supports as needed"
    )


# ============================================================================
# CLI
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python caption_engine.py <model_name> [--mode turntable] [--json]")
        sys.exit(1)

    model_name = sys.argv[1]
    mode = "turntable"
    output_json = "--json" in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]

    result = generate_all_captions(model_name, mode)

    if output_json:
        print(json.dumps(result, indent=2))
    else:
        for platform, data in result["platforms"].items():
            print(f"\n{'=' * 50}")
            print(f"  {platform.upper()}")
            print(f"{'=' * 50}")
            variants = data.get("variants", [])
            for i, variant in enumerate(variants):
                print(f"\n--- Variant {i + 1} ---")
                if isinstance(variant, str):
                    print(variant)
                elif isinstance(variant, dict):
                    for k, v in variant.items():
                        if isinstance(v, str) and len(v) > 100:
                            print(f"  {k}: {v[:100]}...")
                        else:
                            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
