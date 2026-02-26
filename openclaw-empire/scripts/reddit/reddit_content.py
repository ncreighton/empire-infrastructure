"""AI content generation for Reddit comments and posts.

- Comments: Haiku ($0.80/M) — post-specific, structure variation
- Posts: Sonnet ($3/M) — showcase with print settings, design notes
- Prompt caching on system prompts
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

logger = logging.getLogger("reddit_content")

# Add scripts/ to path for adb_config import (ANTHROPIC_API_KEY)
sys.path.insert(0, str(Path(__file__).parent.parent))
from adb_config import ANTHROPIC_API_KEY
from .reddit_config import FORGEFILES_PROFILE, EXPERTISE_TOPICS

# System prompt for comment generation
COMMENT_SYSTEM_PROMPT = f"""You are {FORGEFILES_PROFILE['reddit_username']}, a fellow 3D printing enthusiast on Reddit.

Persona: {FORGEFILES_PROFILE['persona']}

Rules:
- Write a single Reddit comment responding to the post
- Be genuinely helpful — give specific advice (temps, speeds, settings, materials)
- Vary your comment structure randomly: question, experience sharing, tip, or respectful disagreement
- Length: 15-120 words (vary naturally)
- Use casual Reddit tone (lowercase ok, no formal language)
- Reference YOUR specific experience printing similar things
- NEVER mention your Etsy shop or that you sell anything
- NEVER use phrases like "as a maker" or "as someone who designs"
- Match the energy of the post — excited for cool prints, empathetic for problems
- Include specific details: layer heights, temps, slicer settings, material brands
- Occasionally ask a follow-up question to show genuine interest

Topics you know well: {', '.join(EXPERTISE_TOPICS[:15])}
"""

# System prompt for post generation
POST_SYSTEM_PROMPT = f"""You are {FORGEFILES_PROFILE['reddit_username']}, posting your own 3D printing work on Reddit.

Persona: {FORGEFILES_PROFILE['persona']}

Rules:
- Write a post title and body for a 3D printing subreddit
- Share YOUR design/print with genuine enthusiasm
- Include print settings: material, layer height, nozzle temp, infill, print time
- Mention design decisions: why this approach, what problems it solves
- Keep title under 100 chars, following subreddit conventions (e.g., "[functional] Cable management...")
- Body: 50-200 words, casual tone
- NEVER include Etsy links or shop mentions unless specifically told this is a promo post
- For promo posts ONLY: naturally work in ONE mention of where to find the STL

Topics you know well: {', '.join(EXPERTISE_TOPICS[:15])}
"""

COMMENT_STRUCTURES = [
    "question",      # Ask about their setup/settings
    "experience",    # Share similar experience
    "tip",           # Offer specific advice
    "appreciation",  # Genuine compliment + detail
    "disagreement",  # Respectful alternative view
]


def _call_anthropic(system_prompt: str, user_prompt: str,
                    model: str = "claude-haiku-4-5-20251001",
                    max_tokens: int = 300) -> str:
    """Call Anthropic API with prompt caching."""
    if not ANTHROPIC_API_KEY:
        logger.warning("No ANTHROPIC_API_KEY — using template fallback")
        return ""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Use prompt caching for system prompt
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text.strip()
    except ImportError:
        logger.error("anthropic package not installed: pip install anthropic")
        return ""
    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        return ""


def generate_comment(post_title: str, post_body: str, subreddit: str,
                     recent_comments: list[str] = None) -> str:
    """Generate a contextual comment for a Reddit post.

    Uses Haiku for cost efficiency (~$0.80/M tokens).
    """
    structure = random.choice(COMMENT_STRUCTURES)
    target_length = random.randint(15, 120)

    # Build user prompt
    dedup_note = ""
    if recent_comments:
        recent_snippet = " | ".join(c[:50] for c in recent_comments[-5:])
        dedup_note = f"\n\nAvoid repeating these recent comments: {recent_snippet}"

    user_prompt = (
        f"Subreddit: r/{subreddit}\n"
        f"Post title: {post_title}\n"
        f"Post content: {post_body[:400]}\n\n"
        f"Write a {structure}-style comment, roughly {target_length} words.\n"
        f"Be specific and helpful.{dedup_note}"
    )

    result = _call_anthropic(
        COMMENT_SYSTEM_PROMPT, user_prompt,
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
    )

    if not result:
        result = _template_comment(post_title, structure)

    return result


def generate_post(subreddit: str, topic: str = "",
                  is_promo: bool = False) -> dict:
    """Generate a post title and body.

    Uses Sonnet for higher quality ($3/M tokens).
    Returns: {"title": str, "body": str}
    """
    promo_note = ""
    if is_promo:
        promo_note = (
            f"\n\nThis IS a promo post. Naturally mention that the STL is available "
            f"on your Etsy shop ({FORGEFILES_PROFILE['etsy_url']}). "
            f"Keep it subtle — the focus should be on the print, not the sale."
        )

    user_prompt = (
        f"Subreddit: r/{subreddit}\n"
        f"Topic/design: {topic or 'your latest functional print'}\n\n"
        f"Write a Reddit post with title and body.{promo_note}\n\n"
        f"Format:\nTITLE: <title>\nBODY: <body>"
    )

    result = _call_anthropic(
        POST_SYSTEM_PROMPT, user_prompt,
        model="claude-sonnet-4-20250514",
        max_tokens=500,
    )

    if result:
        return _parse_post_response(result)

    return _template_post(subreddit, topic)


def _parse_post_response(text: str) -> dict:
    """Parse TITLE: / BODY: format from API response."""
    title = ""
    body = ""
    lines = text.split("\n")
    in_body = False

    for line in lines:
        if line.upper().startswith("TITLE:"):
            title = line[6:].strip()
        elif line.upper().startswith("BODY:"):
            body = line[5:].strip()
            in_body = True
        elif in_body:
            body += "\n" + line

    if not title:
        title = text.split("\n")[0][:100]
    if not body:
        body = "\n".join(text.split("\n")[1:])

    return {"title": title.strip(), "body": body.strip()}


def _template_comment(post_title: str, structure: str) -> str:
    """Fallback template when API is unavailable."""
    templates = {
        "question": [
            "What slicer are you using? Curious about your support settings.",
            "Nice! What layer height did you go with?",
            "How long was the print time? Looks like a decent amount of material.",
        ],
        "experience": [
            "I printed something similar a few weeks back. PETG worked way better than PLA for the living hinges.",
            "Had the same issue with stringing. Dropping retraction speed to 25mm/s and bumping distance to 6mm fixed it for me.",
        ],
        "tip": [
            "Try bumping your first layer to 215 and slowing it down to 20mm/s. Made a huge difference for bed adhesion on textured PEI.",
            "If you're getting warping, a brim + draft shield combo works better than just a brim alone.",
        ],
        "appreciation": [
            "This is really clean. The tolerances on those snap-fits look dialed in.",
            "Love functional prints like this. So much more satisfying than printing benchies.",
        ],
        "disagreement": [
            "Actually I'd argue PETG is better for this application — PLA will creep under constant load.",
            "Interesting approach but I'd consider going with less infill and thicker walls instead. Same strength, faster print.",
        ],
    }
    options = templates.get(structure, templates["tip"])
    return random.choice(options)


def _template_post(subreddit: str, topic: str) -> dict:
    """Fallback template post when API is unavailable."""
    return {
        "title": f"[Functional Print] {topic or 'Cable management clip that actually holds'}" [:100],
        "body": (
            f"Finally dialed in the settings for this one. "
            f"Printed in PETG at 230/80, 0.2mm layer height, 20% gyroid infill. "
            f"Print time about 2 hours. "
            f"The snap-fit tolerance took a few iterations but 0.3mm gap works perfectly. "
            f"Already printed a dozen for around the desk."
        ),
    }
