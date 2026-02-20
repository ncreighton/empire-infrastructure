# ZimmWriter SERP Discombobulator

> Source: https://www.rankingtactics.com/zimmwriter-serp-discombobulator/

## Overview

The SERP Discombobulator is a tool that processes search query data through custom AI prompts to extract and generate new content from merged SERP summaries.

## Core Workflow

### Input Requirements
- Search queries or blog post titles (one per line)
- Previously SERP-scraped data or data you want scraped
- Custom prompts created in Bulk Writer, SEO Writer, or Penny Arcade

### Processing
The tool takes merged summaries from up to five URLs and applies your custom prompts to extract information or create new content.

## Key Settings

### Custom Prompts 1 & 2

The system uses a unique prompt structure:

```
[DATA TO PROCESS]: merged summaries | [INSTRUCTIONS]: your custom prompt | [RESULT]:
```

This differs from standard rewriting -- it is designed for data extraction and creation.

### Prompt Stacking

When enabled, Prompt 2 operates on Prompt 1's output rather than the original merged summary, allowing sequential processing.

### Language Output

Input prompts must be in English, but output can be translated to non-English languages. With stacking enabled, only Prompt 2 is translated.

### Job Configuration
- Name your output CSV file
- Select GPT model (SERP scraping always uses GPT 3.5 regardless)

## Technical Notes

- The merged summary comes from up to five URLs from your SERP data
- Testing prompts in the OpenAI playground before deployment is recommended for optimal results
