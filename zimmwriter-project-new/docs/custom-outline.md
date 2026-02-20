# ZimmWriter Custom Outline Feature - Complete Guide

## Core Functionality

ZimmWriter's custom outline feature enables users to create reusable structural templates for bulk article writing. This is particularly useful when producing multiple pieces sharing similar organizational patterns, such as recipe blog posts.

## Basic Structure

Custom outlines use simple formatting:
- Single lines become H2 subheadings (default)
- Single dash prefix "-" creates H3 subheadings
- Double dash prefix "--" creates H4 subheadings

## Available Variables

### Content Variables
- `{list}` - Prompts AI to include a bulleted or numbered list
- `{table}` - Requests tabular data presentation
- `{optimize_title}` - Rewrites subheadings contextually based on article title and background information
- `{auto_h3_#}` - Generates specified number of H3 subheadings automatically

### Media Variables
- `{yt}` - Attempts to embed relevant YouTube videos
- `{img}` - Inserts AI-generated images
- `{img_a_custom_image_prompt}` - Uses specific image generation prompts

### Advanced Variables
- `{model=name}` - Switches AI models for specific sections
- `{voice_1ps/1pp/2p/3p/pro}` - Overrides article tone for particular subsections
- `{cta}` - Adds call-to-action buttons or links
- `{save_text}` and `{load_text}` - Stores content for reuse in image prompts
- `{normal}` - Generates entire article, then appends custom sections
- `{cp_your_custom_prompt}` - Applies saved custom prompts
- `{raw_your_raw_prompt}` - Uses specialized prompt frameworks

### Research Variables
- `{research}` - Overrides Deep Research disabled settings for specific subheadings

### Link Pack Variables
- `[lp_packname]` - Loads a specific link pack for the subheading

## Key Limitations

Users should note that list/table insertion may fail approximately 5% of the time with lower-powered models, and tables/lists cannot appear within the same subsection.

---
*Source: https://www.rankingtactics.com/zimmwriter-custom-outline-feature/*
