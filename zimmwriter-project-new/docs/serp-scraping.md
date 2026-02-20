# ZimmWriter SERP Scraping Guide

## Overview
This guide explains how ZimmWriter's SERP (Search Engine Results Page) scraping feature automatically seeds AI content generation with relevant, detailed knowledge about topics.

## Key Functionality

**Behind-the-Scenes Process:**
The system performs a Google search of your blog title, receives ranking webpages and "people also ask" questions, then scrapes up to 5 relevant sites. For each webpage, it collects subheadings, extracts SEO keywords, and generates AI summaries. This data creates article outlines and discussion points for each section.

## Bulk Writer Options

The feature includes several configuration checkboxes:
- Overwrite existing cache for matching titles
- Overwrite webpage cache for matching URLs
- Disable SEO keyword extraction from SERP
- Disable using "people also ask" questions for FAQ sections

SERP scraping works alongside SEO CSV uploads and custom outlines, though it's disabled when using Deep Research mode.

## SEO Writer Implementation

**Critical Timing Rule:** AI summaries from scraped websites only generate discussion points for subheadings lacking supplied URLs or background information.

**Global Background Recommendation:** "Use a short 75 word global background when having the AI write about a narrow or niche topic" and "Use no global background when having the AI write about a broad topic."

This prevents repetition while providing targeted, subheading-specific discussion points.

## Two Operating Modes

1. **Pre-scrape method:** Click "Scrape the SERP Now" to populate background, competitor subheadings, and keywords before writing
2. **Automatic method:** SERP scrapes during article generation without pre-scraping

---
*Source: https://www.rankingtactics.com/zimmwriter-serp-scraping/*
