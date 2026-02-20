"""
ZimmWriter Desktop Controller
Full programmatic control of ZimmWriter via Windows UI Automation.

Modules:
  controller        — Core ZimmWriter UI automation (pywinauto)
  api               — FastAPI REST server
  site_presets      — All 14 site preset configurations
  image_prompts     — Topic-adaptive image meta-prompts
  image_options     — Per-model image option configs
  csv_generator     — SEO CSV generation utilities
  article_types     — Article type classifier (how-to, listicle, review, etc.)
  outline_templates — ZimmWriter outline template library with rotation
  campaign_engine   — Dynamic campaign planning + SEO CSV generation
  link_pack_builder — WordPress URL scraper + link pack generator
  screen_navigator  — Multi-screen detection & navigation
  orchestrator      — Multi-site job & campaign orchestration
  monitor           — Progress monitoring & notifications
"""

__version__ = "1.2.0"
__author__ = "Nick Creighton"
