# ZimmWriter SEO Blog Writer Exhaustive Guide

> Source: https://www.rankingtactics.com/zimmwriter-seo-blog-writer-guide/

## Core Purpose

ZimmWriter's SEO Blog Writer enables surgical, customized content creation designed to rank higher in search results, distinguishing itself from the bulk writing alternative.

## Essential Settings & Configuration

### Blog Post Title
The title should be straightforward and keyword-focused. Examples include "How to Wash a Dog" or "Best Places to Vacation in Canada." Avoid excessive filler like "Ultimate Guide to..." or "2023 Tips for..." as these confuse the AI. Titles with numbers (e.g., "15 Tips") won't automatically generate matching subsections without corresponding H2 settings.

### Global Background (Optional)
Accepts approximately 1,000 words of contextual information or up to three URLs for scraping. This feature works best for:
- Niche subjects
- Product information
- Current events
- Business-specific data

**Important consideration:** Larger backgrounds with many subheadings increase API costs substantially. A 1,000-word background with 60 subheadings using advanced models can exceed $5-15.

The AI may repeat itself when forced to write extensively on unfamiliar topics, so content familiarity impacts output quality.

### H2 Subheading Configuration
Three generation methods exist:
1. **Manual entry** directly into fields
2. **AI-only generation** based on title and count
3. **Enhanced generation** incorporating global background and competitor headings

For Option 3, gather H2/H3 titles from 5-20 ranking competitors, removing irrelevant entries before submission.

### Subheading Nesting
H2, H3, and H4 subheadings are supported. Adjust the "+H3" field next to each H2 to create nesting levels. H4 functions similarly.

### Subheading Background Features
When enabled, each subheading receives individual options:
- **Product Layout Trigger:** Formats output for product reviews with pros, cons, and recommendations
- **Scraping URL:** Automatically summarizes webpage content (limited to 500 words per subheading)
- **Image URL:** Assigns images to sections; auto-populates from scraping if Scraping Surgeon is configured
- **Background Info:** Provides topic-specific knowledge (500-word limit)

### Length Settings
Options include:
- Tiny
- Short
- Medium
- Long

Default: **short**

Note: Enabling lists, tables, auto-styling, keyword features, or style-writing options may override length specifications.

### Voice Options
Five choices available:
- **Second person** (you/your)
- **First person singular** (I/me)
- **First person plural** (we/us)
- **Third person** (he/she/they)
- **Professional** (minimal pronouns)

Subject matter and existing titles may override voice selections.

## Advanced Features

### Literary Devices & Formatting
- **Lists:** Toggleable feature for bulleted/numbered content
- **Tables:** Creates structured data presentations
- **Skinny Paragraphs:** Can be disabled to maintain longer paragraph blocks
- **Active Voice:** Toggleable; disabling allows passive constructions
- **Literary Devices:** Optional stylistic enhancement

### Content Elements
- **FAQ Section:** Auto-generates frequently asked questions
- **Key Takeaways:** Summarizes main points
- **Image Prompts:** Generates prompts for each H2 section
- **Image API Integration:** Connects to external image services
- **YouTube Videos:** Requires scraping API integration
- **Conclusion:** Can be disabled if preferred

### Keyword Optimization
- **Automatic keywords:** Generates from global background and titles
- **Manual keywords:** Specify exact keywords with per-section frequency
- **Configurable density:** Adjust keyword mentions per section

### Audience & Styling
- **Audience Personality:** Targets specific reader demographics
- **Style Emulation:** "Write in the Style of [Author]" option
- **Auto-Style:** Analyzes competitor content for consistency
- **Language Support:** Non-English language generation available

### Advanced Variables (Subheading-Level)
Special syntax enables specific formatting:
- `{list}` - forces list generation
- `{table}` - forces table creation
- `{yt}` - embeds YouTube content
- `{voice_1ps}` - first person singular voice override
- `{voice_1pp}` - first person plural voice override
- `{voice_2p}` - second person voice override
- `{voice_3p}` - third person voice override
- `{voice_pro}` - professional voice override
- `{cta}` - call-to-action button (requires scraping URL)
- `{cp_[name]}` - custom prompt integration

## AI Model Selection
Users can choose different models for outline generation versus article writing (e.g., GPT-4 for outlines, GPT-3.5 for content).

## Execution Options
- **Start SEO Writer (with Scraping):** Scrapes all URLs and begins writing
- **Start SEO Writer (no Scraping):** Proceeds without URL processing
- **Only Scrape URLs:** Scrapes and summarizes without writing, enabling review before content generation
- **Erase Settings:** Clears all menu configurations

## WordPress Integration
Direct publishing capability allows automatic post creation and featured image assignment.
