# ZimmWriter Link Packs: Complete Feature Overview

## Core Functionality

ZimmWriter's link pack feature enables creation of unlimited "packs" containing up to 1,000 URLs for internal and external linking. Each pack functions as a database file stored in the `/database/linkpacks/` folder, allowing users to designate link candidates when articles are generated.

## Link Toolbox Operations

**Creation Process:**
1. Input URLs into the Links box
2. Name the pack (e.g., "dog_care")
3. Press "Save New Pack"
4. ZimmWriter scrapes and summarizes each URL using the AI Ancillary model

Processing time varies significantly -- a 200-link pack may require approximately one hour. The system leverages a built-in scrape cache storing the last 5,000 URLs, dramatically accelerating creation when packs share common links.

**The Pipe Symbol Technique:**
Users can append "|" to any URL followed by a custom one-line summary: `https://example.com|summary text`. This bypasses scraping for URLs that don't exist yet or lack sufficient page information.

## Link Pack Menu Settings

**Insertion Controls:**
- **Non-Link Pack Links:** Input up to 5 standalone URLs without using a pack
- **Per-Article Limit:** Set maximum links per article (insertion isn't guaranteed)
- **Per-Subheading Limit:** Cap links to individual subheadings while respecting article-level limits
- **H3/H4 Insertion:** Enable linking beyond standard H2 subheadings

**Link Attributes:**
- Dofollow/Nofollow configuration via domain whitelist in Link Toolbox
- New tab opening option
- Bold styling capability
- Custom CSS class assignment

**Advanced Features:**

Linknado Mode forces links into approximately 90% of subheadings regardless of relevance detection, overriding the default behavior that skips sections without suitable candidates.

## Additional Capabilities

Users can specify link packs directly within bulk writer titles using the format `[lp_packname]` or within custom outlines at subheading endings. Link packs are shareable -- valid database files can be transferred between users by placing them in the linkpacks folder.

Regular backup of the database folder is recommended given the time investment in pack creation.

---
*Source: https://www.rankingtactics.com/zimmwriter-link-packs/*
