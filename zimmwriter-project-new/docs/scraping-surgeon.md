# ZimmWriter Scraping Surgeon - Complete Guide

## Overview
Scraping Surgeon is a ZimmWriter feature enabling surgical precision in web scraping by designating specific webpage sections via CSS selectors for AI processing.

## Core Purpose
The tool filters extraneous webpage content, ensuring AI summarization focuses only on relevant information. For example, scraping an Amazon product page targets only the specific iPhone model rather than competitor products or unrelated recommendations.

## Technical Foundation

**HTML Basics:** Webpage markup using tags like `<h1>`, `<p>`, `<div>`. Tags contain opening markers, content, and closing markers.

**CSS Selectors:** Two primary types -- ID selectors (single use, marked with `#`) and class selectors (multiple use, marked with `.`) that link HTML to styling rules.

**Nested Tags:** Parent tags contain child elements. Scraping a parent tag captures all nested content within it.

## Configuration Fields

| Field | Description |
|-------|-------------|
| **Domain** | Must contain a period, exclude "www" and "https" |
| **Title Selector** | CSS class/ID or HTML tags (h1, h2) |
| **Image Selector** | Optional; downloads images for Penny Arcade or SEO Writer |
| **Text Selectors** | Up to five CSS classes/IDs or HTML tags (p, h2, ul, ol, li, section, span) |
| **URL Append String** | Affiliate codes or custom URL parameters |
| **Treat as Review** | Instructs AI to include "review" language in titles |
| **Enable/Disable** | Toggle domain activation |
| **Storage** | Supports up to 50 saved domains |

## Operational Details
Scraping Surgeon activates automatically when ZimmWriter detects matching domains during scraping, provided the domain is saved and enabled.

---
*Source: https://www.rankingtactics.com/zimmwriter-scraping-surgeon/*
