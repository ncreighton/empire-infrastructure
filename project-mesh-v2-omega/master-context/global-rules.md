# EMPIRE GLOBAL RULES
> These rules apply to EVERY project. No exceptions.

## Core Principles
1. **Never hardcode API keys, webhook URLs, or secrets** — Use environment variables
2. **Always use shared-core systems when available** — Check the registry first
3. **All API calls must use retry logic** — Use the api-retry shared system
4. **Images must be optimized before upload** — Use image-optimization system
5. **Content must pass SEO validation** — Use seo-toolkit system
6. **Never reference deprecated methods** — Check BLACKLIST.md below

## Technical Standards
- All WordPress sites use Blocksy or Astra themes on Hostinger
- All automation runs through n8n (ZimmWriter is DEPRECATED)
- All content generation uses Claude API (never GPT)
- All sites use LiteSpeed cache
- All affiliate links use affiliate-link-manager system

## Quality Standards
- Content demonstrates E-E-A-T signals
- Target featured snippets where applicable
- Proper schema markup on every page
- Images have alt text with target keywords
- Internal linking follows topical cluster strategy

## Brand Voice
- Each site has its own voice (see category context below)
- Never use generic AI-sounding language
- Never reference being AI-generated
- Content must feel human-written and authentic

## n8n Automation
- All content pipelines run through n8n workflows
- Use Steel.dev for browser automation with 10min keep-alive pings
- BrowserUse as fallback when Steel.dev fails
- All webhooks use environment variables, never hardcoded URLs
