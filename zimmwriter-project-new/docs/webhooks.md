# ZimmWriter Webhooks Guide

## Overview
ZimmWriter webhooks enable integration with external platforms like Zapier, Pabbly, N8N, and Make. When you complete an article using the Bulk Writer, SEO Writer, or Penny Arcade, ZimmWriter sends the content data to your configured webhook URL.

## Setup Process
1. Generate a webhook URL on your chosen platform
2. Paste the URL into ZimmWriter
3. Enter a webhook name
4. Click "Save New Webhook"

## Data Payload Structure
When triggered, ZimmWriter transmits JSON data containing:

| Field | Description |
|-------|-------------|
| `webhook_name` | Identifier for the webhook |
| `title` | Article headline |
| `markdown` | Formatted content in markdown |
| `html` | Formatted content in HTML |
| `excerpt` | Meta description |
| `image_base64` | Featured image (base64 encoded, if generated) |
| `category` | Post category assignment (if set) |
| `image_url` | WordPress image URL (if set) |
| `slug` | Post URL slug (if set) |
| `tags` | Array of post tags (if set) |

Optional fields return `false` when not populated.

## Key Features

**Profile Integration**: Save webhooks within Bulk Writer profiles for automatic activation when profiles load.

**Storage Capacity**: System supports up to 1,000 saved webhooks, though only one loads at a time.

**Platform Compatibility**: Works with any service accepting webhook URLs and processing JSON data. You must parse the data according to your platform's specifications.

## Limitations

Incoming webhooks aren't currently supported due to ZimmWriter's single-threaded architecture.

---
*Source: https://www.rankingtactics.com/zimmwriter-webhooks/*
