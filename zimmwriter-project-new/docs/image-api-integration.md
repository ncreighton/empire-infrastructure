# ZimmWriter Image API Integration Guide

> Source: https://www.rankingtactics.com/zimmwriter-image-api-integration/

## Overview

ZimmWriter integrates multiple third-party image APIs for automated image sourcing and WordPress uploads. Access setup through the ZimmWriter options menu by clicking **"Setup Image API"**.

## Available Image API Providers

### Pexels API (Free Stock Images)

**Pricing:**
- Free tier: 200 requests/hour, 25,000 calls/month
- One API request per blog post

**Setup Steps:**
1. Sign up at https://pexels.com/api/key/
2. Obtain your API key
3. Enter the key in ZimmWriter's Image API settings

**Notes:**
- The verification button displays account balance and credit refresh timing
- Free stock photos -- no AI generation cost

### DALL-E (OpenAI)

**Pricing:**
- Higher cost compared to Stability alternatives

**Setup:**
- Uses existing OpenAI API credentials
- No separate configuration required if OpenAI is already configured for text generation

**Notes:**
- Leverages the same OpenAI API key used for text generation

### Stability AI

**Models and Pricing:**

| Model | Approximate Cost per Image |
|-------|---------------------------|
| Stable Diffusion XL | ~$0.003/image |
| Stable Diffusion Core | ~$0.03/image |
| Stable Diffusion 3 | ~$0.065/image |
| Stable Diffusion Ultra | ~$0.08/image |

**Setup Steps:**
1. Register at https://platform.stability.ai
2. Add credit ($10 minimum suggested)
3. Obtain API key from the account/keys section
4. Enter the key in ZimmWriter's Image API settings

**Features:**
- Supports **negative keywords** (e.g., "hands, fingers") to refine outputs and avoid common AI image artifacts

### Flux Models (via Fal or Replicate)

**Models and Pricing:**

| Model | Approximate Cost per Image | Platform |
|-------|---------------------------|----------|
| Flux Pro | ~$0.05/image | Fal / Replicate |
| Flux Realistic | ~$0.035/image | Fal only |
| Flux Dev | ~$0.025/image | Fal / Replicate |
| Flux Schnell | ~$0.003/image | Fal / Replicate |

**Benefits:**
- Superior hand rendering compared to other AI image generators
- Uncensored output
- Improved overall quality versus Stability AI models

**Notes:**
- Fal supports credit purchase for regions with credit card restrictions
- Flux Realistic is exclusive to Fal platform

### Ideogram

**Models:**
- Four models offered with varying capabilities

**Pricing:**
- Initial account charge: **$40 in credits** (billed immediately upon account creation)

**Strengths:**
- Crisp text rendering in images
- Customizable fonts

**Limitations:**
- Hand rendering limitations noted (less capable than Flux models)

## Usage in ZimmWriter Modes

Images activate automatically in the following modes when **"Enable Image API"** is checked:

- **Bulk Writer** -- automatic image generation per article
- **SEO Writer** -- automatic image generation per article
- **Penny Arcade** -- automatic image generation (Scraping Surgeon images take precedence when detected)

## Cost Comparison Summary

| Provider | Cheapest Option | Most Expensive Option |
|----------|----------------|----------------------|
| Pexels | Free | Free |
| DALL-E | Higher cost | Higher cost |
| Stability AI | ~$0.003 (SDXL) | ~$0.08 (Ultra) |
| Flux | ~$0.003 (Schnell) | ~$0.05 (Pro) |
| Ideogram | $40 upfront credits | $40 upfront credits |

## Recommended Budget Picks

- **Free images:** Pexels API (stock photos, no AI generation)
- **Cheapest AI images:** Stable Diffusion XL (~$0.003) or Flux Schnell (~$0.003)
- **Best quality AI images:** Flux Pro (~$0.05) or Flux Realistic (~$0.035)
- **Best text in images:** Ideogram (crisp text rendering)
