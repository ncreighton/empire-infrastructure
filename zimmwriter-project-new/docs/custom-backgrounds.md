# Custom Backgrounds in ZimmWriter's Bulk Writer

## Overview

ZimmWriter offers two types of custom backgrounds for content generation:

1. **Custom Global Background** -- applies to entire articles
2. **Custom Background** -- applies to specific subheadings

Both features feed proprietary data to the AI that it wouldn't otherwise know, enabling more accurate and tailored content creation.

## Custom Global Background

### Creation Process

Access this feature by clicking "Custom Outline," then "Set Custom Background." Select "Global" from the dropdown menu.

The naming convention requires: start with `{cgb_`, end with `}`, and use only letters, numbers, and underscores. Examples include `{cgb_company_bio_taylortoyota}` or `{cgb_law_jacobs}`.

**Important caveat:** This background feeds into 50-100+ AI requests during article generation, so its impact on output quality -- positive or negative -- is substantial.

### Implementation

Rather than requiring a custom outline, you load global backgrounds directly in the bulk writer by appending the background name to blog post titles.

## Custom Background (Non-Global)

### Purpose

This feature targets specific subheadings within custom outlines. It **replaces SERP data** for that particular section with your proprietary information.

### Naming Convention

Uses the format `{cb_name}` where name contains only letters, numbers, and underscores.

### Practical Application

For example, an agency generating car accident law articles across multiple states can create separate statute-of-limitations backgrounds for each jurisdiction using the naming format `{cb_sol_oh_caraccidents}`, then bulk-generate localized content simultaneously.

## Key Advantage

This combination enables "Bulk Local Buffet generation" -- scaling localized content production across hundreds or thousands of locations while maintaining precise formatting and jurisdiction-specific accuracy.

---
*Source: https://www.rankingtactics.com/zimmwriter-custom-backgrounds/*
