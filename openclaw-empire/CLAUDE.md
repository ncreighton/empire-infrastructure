<!-- MESH:START -->

# -----------------------------------------------------------
# PROJECT MESH v2.0   AUTO-GENERATED CONTEXT
# Project: OpenClaw Empire
# Category: content-tools
# Priority: high
# Compiled: 2026-03-06 22:29
# -----------------------------------------------------------

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

# DEPRECATED METHODS — NEVER USE THESE

> This file is auto-included in every project's CLAUDE.md.
> Updated: 2026-02-28

## Content Generation
### ❌ NEVER use ZimmWriter or ZimmWriter API
- **Replacement**: n8n content pipeline + Claude API
- **Reason**: Deprecated in favor of Claude-native workflows
- **Stage**: REMOVED

### ❌ NEVER use GPT/OpenAI for content generation
- **Replacement**: Claude API (Anthropic)
- **Reason**: All content uses Claude for consistency and quality

## API Patterns
### ❌ NEVER hardcode webhook URLs
- **Replacement**: Use environment variables or config.get('webhooks.name')
- **Reason**: Security risk and maintenance nightmare

### ❌ NEVER make API calls without retry logic
- **Replacement**: Use shared-core/api-retry system
- **Reason**: APIs fail. Always retry with exponential backoff.

### ❌ NEVER use fetch() directly for external APIs
- **Replacement**: Use the api-retry wrapper which handles retries, timeouts, and error logging
- **Reason**: Raw fetch has no retry, no timeout, no error handling

## WordPress
### ❌ NEVER use Yoast SEO plugin
- **Replacement**: RankMath
- **Reason**: Standardized across all sites on RankMath

### ❌ NEVER edit theme files directly
- **Replacement**: Use child theme or Blocksy customizer
- **Reason**: Updates will overwrite direct edits

## Substack
### ❌ NEVER use witchcraftforbeginners.substack.com
- **Replacement**: witchcraftb.substack.com
- **Reason**: Correct URL is witchcraftb.substack.com

## Browser Automation
### ❌ NEVER use Puppeteer directly
- **Replacement**: Steel.dev with BrowserUse fallback
- **Reason**: Standardized on Steel.dev for session management
- **Note**: Steel.dev sessions expire after 15min idle — implement keep-alive pings

## Content Vertical Context
- **Voice**: Scholarly wonder (mythology) / Creative organizer (journals)
- **Sites**: MythicalArchives, BulletJournals
- **Content pillars**: mythology deep dives, journaling techniques, templates, supplies


## Shared Systems (Current Versions)

| System | Version | Criticality | Usage |
|--------|---------|-------------|-------|
| api-retry | 1.1.0 [OK] | high | hourly |


## Relevant Knowledge Base Entries

- AI Vertical Context
- Content Guidelines Summary
- Content Vertical Context
- FORGE + AMPLIFY Intelligence System (INSTALLED)
- Family Vertical Context
- API Cost Optimization Rules
- COMMANDS
- AVOID: [X] NEVER use Yoast SEO plugin
- API Patterns

## Self-Check Before Starting Work
Before writing any code or content for OpenClaw Empire:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for content-tools vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

# OpenClaw Empire   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.


# ═══════════════════════════════════════════════════════════════════════════════
# EMPIRE ARSENAL (Auto-Injected)
# ═══════════════════════════════════════════════════════════════════════════════
# ALWAYS read the Empire Arsenal skill at C:\Claude Code Projects\_SHARED\skills\empire-arsenal\SKILL.md
# before starting any task. It contains:
# - 60+ API keys and credentials
# - 24 tool categories with integration matrix
# - Anti-Generic Quality Enforcer (mandatory depth/uniqueness gates)
# - Workflow patterns and pipeline templates
# - MCP ecosystem and marketplace directory
# - Digital product sales channels
#
# QUALITY RULES:
# - Never produce generic/surface-level output
# - Every result passes: uniqueness test, empire context, depth check, multiplication
# - Use Nick's specific tools (check tool-registry.md), not generic suggestions
# - Branch every output into 3+ revenue/impact streams
# - Go Layer 3+ deep (niche-specific, cross-empire, competitor-blind)
# ═══════════════════════════════════════════════════════════════════════════════
