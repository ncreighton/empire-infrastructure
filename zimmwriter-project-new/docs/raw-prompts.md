# Bulk RAW Prompts in ZimmWriter's Custom Outlines

## Overview
Raw prompts are advanced features for custom outlines in ZimmWriter's bulk writer that provide "extreme flexibility" to generate specific content by replacing standard settings like literary devices and voice with custom instructions.

## Key Characteristics

**Scope and Application:**
- Only function with custom outlines in the bulk writer
- Apply to subheadings (excluding intro, FAQ, and conclusion)
- Replace initial content generation settings while preserving secondary passes

**Template Structure:**
The first raw prompt uses a framework including:
- `[KNOWLEDGE]` - global background information
- `[FACTS]` - subheading-specific background
- `[ARTICLE TITLE]` - article name
- `[CURRENT SUBTOPIC]` - heading hierarchy
- `[DIRECTIONS]` - your custom prompt

## Creating and Saving Raw Prompts

**Naming Convention:** `{raw_name}` where name contains only English letters, numbers, or underscores

**Configuration Options:**
- Allow global background application
- Allow subheading background application
- Option to prevent WordPress upload (content appears only in text files)

## Advanced Features

**Stacking:** Up to three raw prompts per subheading apply sequentially left-to-right, with stacked prompts using modified templates referencing `[TEXT TO MODIFY]`.

**Master Raw Prompt:** A single raw prompt without outline specification applies to all AI-generated subheadings, enabling structure flexibility while maintaining custom prompting.

## Technical Notes
- Secondary processing (nuke AI words, bold words, skinny paragraphs, custom prompts, translation) still applies to raw prompt outputs
- Stacked prompts need not be adjacent in outline notation
- Works alongside other custom outline variables like `{img}` and `{auto_h3_#}`

---
*Source: https://www.rankingtactics.com/zimmwriter-raw-prompts/*
