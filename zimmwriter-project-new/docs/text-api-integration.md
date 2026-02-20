# ZimmWriter Text API Integration Guide

> Source: https://www.rankingtactics.com/zimmwriter-text-api-integration/

## Overview

ZimmWriter supports multiple AI text generation providers. This guide covers how to integrate each provider's API for article generation.

## API Providers & Setup

### OpenAI (OA)

OpenAI is the primary platform for ZimmWriter text generation.

**Setup Steps:**
1. Create an OpenAI account at https://platform.openai.com
2. Add a credit card and deposit at least **$50** to your account
3. Navigate to the OpenAI dashboard
4. Generate an API key
5. Enter the API key in ZimmWriter's settings

**Configuration in ZimmWriter:**
- Navigate to ZimmWriter settings
- Enter your OpenAI API key in the designated field

### Anthropic (ANT)

**Setup Steps:**
1. Create an account at Anthropic
2. Start on the free tier
3. Adding **$50** typically triggers immediate promotion to **Tier 2**
4. Access the API key generator on Anthropic's console (https://console.anthropic.com)
5. Generate and copy your API key

**Important Notes:**
- Anthropic enforces usage and rate limits
- Higher tiers unlock higher rate limits

**Configuration in ZimmWriter:**
- Navigate to ZimmWriter settings
- Select **"Setup Text API (Non-OpenAI)"**
- Input your Anthropic API credentials

### Groq (GROQ)

**Setup Steps:**
1. Register at Groq's API page
2. Obtain your API credentials
3. Currently offering **free API access** (paid model anticipated in the future)

**Important Notes:**
- The only model capable of running ZimmWriter prompts is **LLaMA3 70b**
- Free tier may have rate limits

**Configuration in ZimmWriter:**
- Navigate to ZimmWriter settings
- Select **"Setup Text API (Non-OpenAI)"**
- Input your Groq API credentials

### OpenRouter (OR)

OpenRouter acts as an aggregator supporting multiple AI models from various providers.

**Setup Steps:**
1. Create an account at OpenRouter
2. Fund your account with credits
3. Generate an API key

**Important Notes:**
- Some models like **Perplexity Online** charge per request ($5/1,000 requests) beyond standard token pricing
- Review model-specific pricing before use
- Acts as a proxy to multiple model providers

**Configuration in ZimmWriter:**
- Navigate to ZimmWriter settings
- Select **"Setup Text API (Non-OpenAI)"**
- Input your OpenRouter API credentials

## ZimmWriter Settings Navigation

- **OpenAI:** Enter directly in the main API settings
- **Non-OpenAI providers:** Use the **"Setup Text API (Non-OpenAI)"** option in settings

## Quality Advisory

The author advises reviewing generated articles carefully, noting: "You might still run into glitches in the blog posts when using these models." Quality can vary between providers and models.

## Provider Comparison

| Provider | Label | Min Deposit | Key Model | Notes |
|----------|-------|-------------|-----------|-------|
| OpenAI | OA | $50 | GPT-4, GPT-3.5 | Primary platform |
| Anthropic | ANT | $50 (for Tier 2) | Claude models | Rate limits enforced |
| Groq | GROQ | Free | LLaMA3 70b | Only compatible model |
| OpenRouter | OR | Variable | Multiple | Aggregator, per-model pricing |

## Recommended Initial Funding

- **OpenAI:** $50 minimum deposit
- **Anthropic:** $50 to reach Tier 2 (higher rate limits)
- **Groq:** Free (for now)
- **OpenRouter:** Variable based on model usage
