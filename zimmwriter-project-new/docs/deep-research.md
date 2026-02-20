# ZimmWriter Deep Research Feature - Complete Guide

## Overview
ZimmWriter's Deep Research is a powerful citation-based research tool that "produces content that combines real-time researched data with citations from recognized authorities on each topic."

## Key Availability & Requirements
- **Exclusive to Bulk Writer** - Deep Research only functions within ZimmWriter's Bulk Writer module
- **Mutually exclusive with SERP scraping** - users must choose one research method
- **Requires OpenRouter API key** for accessing online AI models

## Available AI Models
Five models support Deep Research with online access and citations:
- Perplexity 70b Online (OR)
- Perplexity 405b Online (OR)
- Perplexity Sonar Online (OR)
- Sonar Online (PP)
- Sonar Online Pro (PP)

## Core Configuration Options

### Citation Settings
- **Links per article (max)** - controls maximum citations across entire piece
- **Links per subheading (max)** - limits citations within individual sections
- Setting either to zero disables citation generation but retains research for outline/content creation

### Domain Management
Users can exclude up to 200 domains from appearing as citations, maintaining separate lists per profile in the Bulk Writer.

### Research Toggles
- Disable Deep Research for outline creation
- Disable Deep Research for subheading background
- Both settings can be selectively overridden using {research} variables in custom outlines

### Reference Display Options
- Display references within subheadings for fact-checking
- Generate end-of-article references section (customizable maximum)

## How It Functions

The system operates through a three-stage process:

1. **Initial Research Request** - ZimmWriter queries the AI model about your topic or subheading
2. **Citation Retrieval** - AI returns factual data paired with source URLs
3. **URL Scraping & Integration** - Citations are scraped, summarized, and injected with contextual anchor text into content

The feature automatically extracts optimal anchor text from source material and integrates both Deep Research and Link Pack citations within subheadings.

## Performance Considerations

"Selecting research links per subheading (max) above zero and research links per article (max) above zero can dramatically increase article generation time." Each citation undergoes full-page scraping and summarization, potentially extending generation significantly.

---
*Source: https://www.rankingtactics.com/zimmwriter-deep-research/*
