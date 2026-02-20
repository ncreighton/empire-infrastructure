# ZimmWriter Secure Mode Setup Guide

## Overview
Secure Mode protects your lifetime license key when sharing ZimmWriter with a virtual assistant. It prevents the VA from accessing your license key while allowing you to revoke their access by disabling their API keys.

## Key Recommendation
The creator suggests getting your VA a subscription instead, as this allows simple cancellation on Gumroad when parting ways. Each license (subscription or lifetime) is a single-seat license.

## Step-by-Step Setup

**Steps 1-3:** Download the latest version from the secret training page, unzip it, and launch with your email and license key.

**Step 4:** Create API keys at your AI provider (e.g., OpenAI, Groq, Anthropic, OpenRouter) specifically for your VA, and then enter them into ZimmWriter.

**Step 5:** Add additional text APIs through the options menu's "Setup Text API" feature (optional but recommended to prevent later additions).

**Steps 6-9:** Enable Secure Mode via the options menu, configure other settings (ScrapeOwl, Stability API), then exit.

**Step 10 - Distribution Options:**
- **Option 1:** Zip the entire ZimmWriter directory and send to your VA
- **Option 2:** Share only the settings.ini file for them to copy into their installation

## Access Revocation
When terminating the VA's access, revoke their text API keys from each provider (OpenAI, Groq, Anthropic, OpenRouter), and their ZimmWriter copy stops functioning.

## Critical Warning
"Tell your VA not to mess with the API keys once you've set this up" to avoid breaking their installation and requiring the entire process to repeat.

---
*Source: https://www.rankingtactics.com/zimmwriter-secure-mode/*
