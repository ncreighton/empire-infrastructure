"""Seed Brain with critical learnings from accumulated empire knowledge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.brain_db import BrainDB
from forge.brain_codex import BrainCodex

db = BrainDB()
codex = BrainCodex(db)

learnings = [
    ("Creatomate base64 data URIs do NOT work for audio sources - must use hosted URLs like catbox.moe", "videoforge-engine", "api_quirk", 0.95),
    ("Pixabay CDN URLs are hotlink-protected (403) - need to re-host music files for Creatomate", "videoforge-engine", "api_quirk", 0.95),
    ("ElevenLabs TTS provider requires integration in Creatomate project settings before use", "videoforge-engine", "api_quirk", 0.9),
    ("Revid.ai text-to-video requires hasToSearchMedia: true (otherwise black screen)", "revid-forge", "api_quirk", 0.95),
    ("Revid.ai video slug is faceless-video not text-to-video", "revid-forge", "api_quirk", 0.95),
    ("Revid.ai script field is inputText not text", "revid-forge", "api_quirk", 0.9),
    ("Revid.ai publish titles must be in nested platform objects: youtube: {title: ...}", "revid-forge", "api_quirk", 0.95),
    ("bun.exe must be next to screenpipe.exe - PATH lookup unreliable on Windows", "empire-infrastructure", "gotcha", 0.9),
    ("vision-agent v0.2.36 requires Python <=3.12 - av compilation fails on 3.14", "geelark-automation", "gotcha", 0.95),
    (r"Must use Git GNU tar (C:\Program Files\Git\usr\bin\tar.exe) for VPS deploy - Windows bsdtar corrupts streams", "empire-infrastructure", "gotcha", 0.95),
    (r"ADB binary is at C:\Users\ncreighton\AppData\Roaming\GeeLark\adb\adb.exe - SDK path does NOT exist", "openclaw-empire", "gotcha", 0.95),
    ("ADB fixed port 5555 via adb tcpip 5555 survives sleep/wake but resets on reboot", "openclaw-empire", "gotcha", 0.9),
    ("Phone Tailscale and ADB need battery optimization disabled", "openclaw-empire", "gotcha", 0.85),
    ("All images MUST upload to WordPress media library - no ngrok or external URLs on aged sites", "empire-master", "decision", 0.95),
    ("Rate limit aged sites to 3 updates per day maximum", "empire-master", "decision", 0.9),
    ("Default to claude-sonnet-4-20250514 for most tasks, Haiku for classification, Opus only for complex reasoning", "empire-infrastructure", "optimization", 0.95),
    ("Always enable prompt caching (cache_control ephemeral) for system prompts > 2048 tokens", "empire-infrastructure", "optimization", 0.95),
    ("VPS deploy uses tar+ssh single connection per directory", "empire-infrastructure", "pattern", 0.85),
    ("Windows startup pattern: VBS launcher (hidden window) -> PowerShell script -> service binary", "empire-infrastructure", "pattern", 0.9),
    ("Grimoire Intelligence has zero AI API costs - all intelligence is algorithmic", "grimoire-intelligence", "optimization", 0.9),
    ("VideoForge cost per video: ~$0.46-0.58 (all scenes get AI visuals + ElevenLabs voice)", "videoforge-engine", "optimization", 0.9),
    ("Systeme.io browser automation uses Browserbase MCP + Stagehand as primary method", "openclaw-empire", "integration", 0.85),
    ("Vision service compare endpoint fields are before/after (not image_before/image_after)", "empire-infrastructure", "api_quirk", 0.9),
    ("Dashboard /api/screenpipe/search proxies to localhost:3030 to fix CORS", "empire-dashboard", "pattern", 0.85),
    ("Chrome CDP posting needs port conflict fix and missing tab navigation", "empire-infrastructure", "gotcha", 0.85),
    ("Screenpipe binary at C:\\Users\\ncreighton\\screenpipe\\bin\\screenpipe.exe (v0.3.135)", "empire-infrastructure", "pattern", 0.85),
    ("n8n webhooks are at http://217.216.84.245:5678/webhook/ - hostname DNS may not resolve from Windows", "empire-infrastructure", "gotcha", 0.9),
    ("Witchcraft is FLAGSHIP site (site 4 of 16) - maximum protection, rate limits apply", "witchcraftforbeginners", "decision", 0.95),
    ("16 niche ElevenLabs voice profiles (Turbo v2.5) configured - Drew for witchcraft, Dave for mythology, Brian for smart home", "videoforge-engine", "pattern", 0.85),
    ("RenderScript uses content-hash-based animation selection for deterministic variety", "videoforge-engine", "pattern", 0.8),
]

count = 0
for content, source, category, confidence in learnings:
    lid = codex.learn(content, source, category, confidence)
    count += 1

print(f"Seeded {count} learnings into BrainCodex")
print(f"Total learnings: {db.stats()['learnings']}")
