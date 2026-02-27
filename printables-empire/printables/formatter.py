"""Format content for Printables.com descriptions.

Printables renders descriptions as Markdown but handles headings poorly.
This module converts article-style markdown (with # headings) into
Printables-friendly format using **Bold:** headers and bullet lists.
"""

import re


def format_for_printables(
    markdown: str,
    content_type: str = "article",
    include_stl_note: bool = True,
) -> str:
    """Convert article markdown into Printables description format.

    Printables description best practices (from 3d-print-forge):
    - Use **Bold Header:** for section titles (not # headings)
    - Use bullet lists (- item) for specs and lists
    - Keep paragraphs short
    - End with print settings and license info

    Args:
        markdown: Full article/review/post markdown
        content_type: Type of content being formatted
        include_stl_note: Whether to add companion STL note

    Returns:
        Printables-friendly description string.
    """
    lines = markdown.split("\n")
    output_lines = []
    skip_title = True  # Skip the first # heading (redundant with model name)

    for line in lines:
        stripped = line.strip()

        # Skip the first H1 title (Printables shows it as the model name)
        if skip_title and stripped.startswith("# "):
            skip_title = False
            continue

        # Convert ## headings to **Bold** section headers
        if stripped.startswith("## "):
            heading_text = stripped.lstrip("#").strip()
            output_lines.append("")
            output_lines.append(f"**{heading_text}**")
            continue

        # Convert ### subheadings to bold text
        if stripped.startswith("### "):
            heading_text = stripped.lstrip("#").strip()
            output_lines.append(f"**{heading_text}**")
            continue

        # Strip any remaining # headers
        if stripped.startswith("#"):
            heading_text = stripped.lstrip("#").strip()
            output_lines.append(f"**{heading_text}**")
            continue

        # Keep everything else as-is
        output_lines.append(line)

    result = "\n".join(output_lines).strip()

    # Clean up excessive blank lines (max 2 consecutive)
    result = re.sub(r"\n{3,}", "\n\n", result)

    # Add companion STL info and footer
    footer_parts = []

    if include_stl_note:
        footer_parts.append(_stl_note(content_type))

    footer_parts.append(_print_settings_note())
    footer_parts.append(_license_note())

    result = result + "\n\n" + "\n\n".join(footer_parts)

    return result


def format_summary(description: str, title: str = "") -> str:
    """Create a clean summary from the description.

    Printables shows this as a preview. Max ~200 chars, plain text only.
    """
    # Strip markdown formatting for summary
    text = description
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # Remove bold
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)  # Remove headings
    text = re.sub(r"^- ", "", text, flags=re.MULTILINE)  # Remove bullets

    # Get first meaningful paragraph (skip title, skip empty lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in paragraphs:
        # Skip if it looks like a heading or very short
        if len(para) > 30 and not para.startswith("#"):
            summary = para[:200]
            if len(para) > 200:
                # Truncate at last word boundary
                summary = summary[:summary.rfind(" ")] + "..."
            return summary

    return title[:200] if title else description[:200]


def _stl_note(content_type: str) -> str:
    """Add a note about the included companion STL."""
    notes = {
        "article": (
            "**Included Test Print**\n"
            "This guide includes a companion STL test piece so you can "
            "practice the techniques described above. Print it with the "
            "recommended settings to test your results."
        ),
        "review": (
            "**Included Test Print**\n"
            "Includes a companion test block STL. Use it to verify your "
            "printer settings before starting larger projects."
        ),
        "post": (
            "**Included File**\n"
            "Includes a small companion STL test piece — print it to "
            "try the tip yourself!"
        ),
        "listing": (
            "**What's Included**\n"
            "- STL file (FDM optimized)\n"
            "- Compatible with all major slicers"
        ),
    }
    return notes.get(content_type, notes["article"])


def _print_settings_note() -> str:
    """Standard print settings footer."""
    return (
        "**Recommended Print Settings**\n"
        "- Layer Height: 0.2mm\n"
        "- Infill: 15-20%\n"
        "- Material: PLA\n"
        "- Supports: None needed"
    )


def _license_note() -> str:
    """Standard license footer."""
    return "Personal use license included. Print and enjoy!"
