# ZimmWriter Text Discombobulator

> Source: https://www.rankingtactics.com/zimmwriter-text-discombobulator/

## Overview

The Text Discombobulator processes up to 500 lines of text through custom AI prompts to generate bulk content outputs in CSV format, with optional AI image generation.

## Left Panel Features

### Input Parameters
- **Text Input**: Accepts up to 500 lines, each up to 5,000 characters
- **Special Syntax**: Prefix lines with "@" to skip processing while keeping them in output
- **Custom Prompts**: Uses two separate custom prompts created in other ZimmWriter tools
- **Prompt Stacking**: When enabled, Prompt 2 processes Prompt 1's output rather than the original text
- **Language Output**: Non-English languages supported for final CSV (English required for prompts)
- **Job Naming**: Customizable CSV filename
- **Model Selection**: Choose preferred AI model for generation

### System Prompt Structure

The tool implements a three-part processing format:

```
[DATA TO PROCESS]: line of text up to 5,000 characters
[INSTRUCTIONS]: your custom prompt
[RESULT]:
```

This differs from standard prompts -- it extracts and transforms data rather than rewriting existing content.

## Right Panel - AI Image Generation

### Capabilities
- Generates one image per input line
- Supports three placeholder types: `{input}`, `{output1}`, `{output2}`
- Character limits: 500 chars from input, 3,000 from each output
- Uses dynamic replacement during processing

### Use Cases
- Social media campaigns with paired text and images
- Recipe collections with contextual illustrations
- Bulk content creation without full blog posts

### Key Advantage
Feeding processed output into image prompts ensures visual accuracy -- recipe descriptions can generate relevant food imagery.

## Comparison: Bulk AI Image Generator vs. Text Discombobulator

| Feature | Bulk AI Image Generator | Text Discombobulator |
|---------|------------------------|---------------------|
| Input | Titles/URLs | Lines of text (up to 5,000 chars) |
| Placeholders | `{title}`, `{subheading}` | `{input}`, `{output1}`, `{output2}` |
| Images per item | Multiple | Single |
| Processing | Title/URL based | Line-by-line with dynamic substitution |
