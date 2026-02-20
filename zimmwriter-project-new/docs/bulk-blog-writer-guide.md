# ZimmWriter Bulk Blog Writer Exhaustive Guide

> Source: https://www.rankingtactics.com/zimmwriter-bulk-blog-writer-exhaustive-guide/

## Core Functionality

The Bulk Blog Writer enables queuing up to 1,000 blog posts simultaneously for generation at minimal cost. It shares most configuration options with the SEO Blog Writer, which are documented separately.

## Primary Input: Blog Post Titles

Users can input up to 1,000 titles (70,000 character limit, approximately 63 characters per title) using either:
- Semicolon-separated format on one line
- One title per line

A preview function confirms proper parsing before generation begins.

## Blog Post Title Variables

Custom variables in curly brackets modify individual article generation:

### {profile=}
Loads a saved profile for specific articles; applies to all subsequent titles until another profile is loaded.

### {category=}
Assigns WordPress category; auto-creates if it doesn't exist.

### {outline_focus=}
Provides directional guidance (up to 500 characters) for outline generation when titles are ambiguous.

### {slug=}
Defines SEO-friendly URL slug distinct from title.

### {cgb_your_custom_global_background}
Applies custom global background to specific articles.

### {author=}
Specifies WordPress username for post attribution.

### {lp_name_of_your_link_pack}
Applies link pack settings per title.

### {research=up to 500 characters}
Provides additional research directions for outline generation.

### {fileid=number}
Prepends number with double underscore to generated files for database syncing.

## URL Merge Feature

Allows sourcing single articles from multiple URLs (up to 5 per title). Format example:

Article title, followed by URLs on subsequent lines, separated by blank lines between articles. Disables SERP scraping for merged URLs but maintains it for remaining titles. Supports YouTube URL summarization with appropriate scraping APIs. The input box accommodates 1,000 lines total.

## Configuration Toggles

### Use # of H2 as Suggestion
- **When unchecked:** Generates exact specified H2 count
- **When checked:** Applies sophisticated logic detecting listicles:
  - Below 5 specified H2s: uses that exact amount
  - 6+ specified H2s: randomly selects between 5 and specified amount per article
  - Listicles <=15: writes detected number
  - Listicles >15: renames to 15 and writes 15 H2s

**Caution:** Place listicle numbers first in titles to avoid false detection.

### Enable H3
Provides 20% probability of H3 items within each H2 (2-3 items per occurrence).

## SEO CSV Upload

Alternative to manual title input using specially formatted CSV files specifying:
- Titles
- Outline focus
- Custom backgrounds
- Complete outlines
- SEO keywords
- Categories

Uploaded outlines override # of H2 and Enable H3 settings for affected rows. Google Sheets template provided.

## Troubleshooting & Optimization

### Incomplete Batch Processing
If generation stops before 1,000 titles, sort output by date, identify last successful post, delete processed titles from input, and resume.

### Adjusting Word Count
Length affected by:
- Number of H2s
- Section length settings
- Lists/tables inclusion
- FAQ enablement
- H3 usage
- Automatic keyword application

### Cost Reduction
Recommends GPT-3.5 Turbo (quality matches newer models); reduce word counts through shorter sections and fewer H2s.
