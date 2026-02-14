# KDP Publisher — OpenClaw Skill

## Purpose
End-to-end Amazon KDP book creation pipeline. Generate outlines, write chapters, create covers, format manuscripts, and prepare upload packages across all niches in the empire.

## Trigger Phrases
- "create a book about [topic]", "KDP pipeline", "generate book outline"
- "write chapter [n] for [book]", "create book cover", "format manuscript"
- "KDP status", "book pipeline", "publish to KDP"

## Book Pipeline Stages

```
IDEATION → OUTLINE → CHAPTERS → EDIT → COVER → FORMAT → REVIEW → UPLOAD
```

### Stage Details

1. **IDEATION**: Research profitable niches, analyze competition, identify gaps
2. **OUTLINE**: Generate chapter structure with SEO-aware titles
3. **CHAPTERS**: Write each chapter matching niche voice (2,000-4,000 words each)
4. **EDIT**: Proofread, fact-check, consistency pass
5. **COVER**: Generate cover via fal.ai or Midjourney (2560x1600 for paperback)
6. **FORMAT**: Convert to KDP-ready formats (EPUB, PDF with proper margins)
7. **REVIEW**: Final quality check, metadata preparation
8. **UPLOAD**: Package for KDP upload (manuscript + cover + metadata)

## Book Project Structure
```
~/.openclaw/workspace/kdp/
├── projects/
│   ├── crystal-healing-101/
│   │   ├── meta.json          # Title, subtitle, categories, keywords
│   │   ├── outline.md         # Chapter outline
│   │   ├── chapters/
│   │   │   ├── 01-introduction.md
│   │   │   ├── 02-what-are-crystals.md
│   │   │   └── ...
│   │   ├── manuscript.md      # Combined manuscript
│   │   ├── manuscript.epub    # Formatted EPUB
│   │   ├── cover-front.png    # Front cover (2560x1600)
│   │   ├── cover-full.png     # Full wrap cover
│   │   └── kdp-metadata.json  # Upload metadata
│   └── ...
└── templates/
    ├── nonfiction-outline.md
    ├── journal-template.md
    └── coloring-book-template.md
```

## Book Metadata Template
```json
{
  "title": "Crystal Healing for Beginners",
  "subtitle": "A Modern Witch's Guide to Harnessing Crystal Energy",
  "author": "Nick Creighton",
  "series": "Witchcraft for Beginners",
  "description": "...",
  "categories": ["Religion & Spirituality > New Age > Crystals", "Body, Mind & Spirit > Healing > Energy"],
  "keywords": ["crystal healing", "crystals for beginners", "healing crystals", "crystal magic", "crystal witchcraft", "gemstone healing", "crystal energy"],
  "language": "English",
  "price_ebook": 4.99,
  "price_paperback": 12.99,
  "page_count_estimate": 150,
  "isbn": null,
  "asin": null
}
```

## Niche-Specific Book Ideas

### Witchcraft Vertical
- Crystal Healing for Beginners
- Moon Phase Magic Workbook
- Kitchen Witch Recipe Journal
- Herbal Grimoire: 100 Plants and Their Magical Uses
- Sabbat Celebration Guide
- Tarot Journal with Spread Templates

### Smart Home Vertical
- Smart Home Setup Guide for Non-Techies
- Home Security Automation Handbook
- Voice Assistant Mastery

### AI Vertical
- AI Side Hustle Blueprint
- Prompt Engineering Playbook
- Automate Your Business with AI

### Low-Content / Journals
- Moon Phase Tracker Journal
- Spell Record Grimoire (lined)
- Tarot Reading Log
- Crystal Collection Catalog
- Bullet Journal Dot Grid Notebooks
- Gratitude Journal for Families

## Commands
```bash
# Start new book project
kdp new --title "Crystal Healing 101" --niche witchcraft --type nonfiction

# Generate outline
kdp outline --project crystal-healing-101 --chapters 12

# Write a chapter
kdp write --project crystal-healing-101 --chapter 3

# Write all remaining chapters
kdp write --project crystal-healing-101 --all

# Generate cover
kdp cover --project crystal-healing-101 --style "mystical purple crystals on dark background"

# Format manuscript
kdp format --project crystal-healing-101 --output epub,pdf

# Full pipeline (end to end)
kdp pipeline --title "Moon Magic Workbook" --niche witchcraft --chapters 10 --auto

# Status of all projects
kdp status
```

## Quality Standards
- Minimum 20,000 words for nonfiction (target 30,000-50,000)
- Minimum 100 pages for journals/low-content
- Cover must be 300 DPI, RGB, no bleed issues
- All content must pass AI detection (humanize pass)
- Unique voice matching niche brand
- Proper front/back matter (TOC, copyright, about author, resources)
- Cross-promote other books in series
