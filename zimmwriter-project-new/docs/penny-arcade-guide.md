# ZimmWriter Penny Arcade Exhaustive Guide

> Source: https://www.rankingtactics.com/zimmwriter-penny-arcade-exhaustive-guide/

## Overview

The Penny Arcade is ZimmWriter's bulk content generation feature allowing users to queue up to 1,000 URLs for AI-powered article creation. Using GPT 3.5 Turbo, it produces articles at approximately 2-10 cents each under default settings.

## Two Content Generation Methods

### Method 1: URL Batch Processing
Users paste multiple URLs (one per line) and the tool visits each URL, summarizes the content, then generates original articles.

### Method 2: Direct Text Input
Paste article text or content up to 10,000 words for single-article generation.

## Unique Configuration Options

### H2 Subheading Settings
The "Use # of H2 as Suggestion" checkbox operates as a dynamic limiter rather than fixed requirement. A 1,000-word summary might generate 5-15 subheadings while a 300-word summary generates roughly 3.

### Listicle Detection
When both "Use Original Title" and "Use # of H2 as Suggestion" checkboxes are enabled, the system identifies numbered titles (e.g., "10 Tricks to Lose Weight") and matches subheadings to detected numbers, with a 15-item maximum cap.

### Automatic Voice
Default voice setting uses AI analysis to match writing tone to summarized content. Manual voice selection required for non-English inputs.

## Shared Settings
The tool includes standard options documented elsewhere:
- Voice
- Literary Devices
- Lists
- Tables
- Key Takeaways
- FAQ
- Audience Personality
- Style emulation
- Auto-styling
- Keyword automation
- Multilingual output

## Known Limitations

- Product roundup articles aren't well-suited for processing
- Articles average 350-1,000 words based on generated summaries
- Failed URL scrapes may result from site structure; adding API keys improves success rates
- Successful URLs automatically remove from input queue
