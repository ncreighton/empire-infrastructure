"""How-to article generation pipeline."""

import re

from content.models import Article, Section, Difficulty
from content.writer import ContentWriter
from content.voice import get_voice_prompt


def write_article(
    writer: ContentWriter,
    topic: str,
    keywords: list[str] | None = None,
    difficulty: str = "beginner",
    voice_profile: str = "maker_mentor",
) -> Article:
    """Generate a full how-to article.

    Returns an Article model with parsed sections.
    """
    if not keywords:
        keywords = [topic.lower()]

    voice_prompt = get_voice_prompt(voice_profile)

    # Classify difficulty if not specified
    if difficulty not in ("beginner", "intermediate", "advanced"):
        difficulty = writer.classify(
            topic, ["beginner", "intermediate", "advanced"]
        )

    # Generate the article
    raw_md = writer.write_article(topic, keywords, difficulty, voice_prompt)

    # Generate tags
    tags = writer.generate_tags(topic, "article", keywords)

    # Parse the markdown into sections
    article = _parse_article(raw_md, topic, difficulty, keywords, tags)
    return article


def _parse_article(
    raw_md: str,
    topic: str,
    difficulty: str,
    keywords: list[str],
    tags: list[str],
) -> Article:
    """Parse raw markdown into an Article model."""
    article = Article(
        title=topic,
        difficulty=Difficulty(difficulty),
        keywords=keywords,
        tags=tags,
    )
    article.to_slug()

    # Split on ## headings
    sections_raw = re.split(r"^## (.+)$", raw_md, flags=re.MULTILINE)

    # First chunk is the intro (before any ## heading)
    if sections_raw:
        # Remove any # title at the start
        intro = re.sub(r"^# .+\n*", "", sections_raw[0]).strip()
        article.intro = intro

    # Parse heading/body pairs
    i = 1
    while i < len(sections_raw) - 1:
        heading = sections_raw[i].strip()
        body = sections_raw[i + 1].strip()
        # Check if it's the conclusion
        if any(
            kw in heading.lower()
            for kw in ("wrapping up", "conclusion", "final thoughts", "next steps")
        ):
            article.conclusion = body
        else:
            article.sections.append(Section(heading=heading, body=body))
        i += 2

    article.compute_word_count()
    return article
