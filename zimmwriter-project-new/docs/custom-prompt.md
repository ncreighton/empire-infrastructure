# ZimmWriter Custom Prompt Guide

## What Custom Prompts Do

Custom prompts let you apply personalized instructions to specific content sections in ZimmWriter. They're inserted as an additional processing step between the initial content generation and optional features like Skinny Paragraphs or translation.

## How They Work

ZimmWriter processes content section-by-section. When you enable custom prompts, the workflow becomes:

1. Generate section content
2. Apply your custom prompt
3. Apply Skinny Paragraphs (if enabled)
4. Translate to desired language (if enabled)

## Naming Requirements

All custom prompts must follow this format: `{cp_name_here}` with these rules:
- Start with "cp_"
- Use only English letters, numbers, and underscores
- Include curly brackets

Examples: `{cp_amazon}`, `{cp_fix_spelling}`, `{cp_aida_intro_1}`

## Where to Apply Them

You can assign custom prompts to:
- Introductions
- Conclusions
- Normal subheadings
- Transition subheadings (where next heading goes deeper)
- Product layouts
- Key takeaways
- FAQs

## Integration Points

Custom prompt variables work in:
- Custom outlines (append to subheadings)
- SEO Writer (use variable name)

## Cost Considerations

Using custom prompts incurs additional API calls to OpenAI, increasing both time and expense. You can select a different OpenAI model specifically for custom prompt processing versus article generation.

## Design Tips

When crafting prompts, remember the backend structure that gets sent to OpenAI:

```
[CONTEXT]: [article/subheading title]
[TEXT TO MODIFY]: [generated content]
[MODIFY THE TEXT ACCORDING TO THESE RULES]: [your prompt]
```

You can reference the content as "TEXT" in your instructions. Start simple and iterate based on results.

---
*Source: https://www.rankingtactics.com/zimmwriter-custom-prompt/*
