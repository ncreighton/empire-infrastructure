"""Community post generation."""

from content.models import Post
from content.writer import ContentWriter
from content.voice import get_voice_prompt


def write_post(
    writer: ContentWriter,
    topic: str,
    keywords: list[str] | None = None,
    voice_profile: str = "community_voice",
) -> Post:
    """Generate a community post for Printables.

    Short, conversational, ends with an engagement question.
    """
    voice_prompt = get_voice_prompt(voice_profile)

    # Generate the post
    raw_md = writer.write_post(topic, voice_prompt)

    # Generate tags
    if not keywords:
        keywords = [topic.lower()]
    tags = writer.generate_tags(topic, "post", keywords)

    # Extract title from first line if it's a heading, otherwise use topic
    lines = raw_md.strip().split("\n")
    title = topic
    body = raw_md
    if lines and lines[0].startswith("# "):
        title = lines[0].lstrip("# ").strip()
        body = "\n".join(lines[1:]).strip()

    post = Post(
        title=title,
        body=body,
        keywords=keywords,
        tags=tags,
    )
    post.word_count = len(body.split())
    return post
