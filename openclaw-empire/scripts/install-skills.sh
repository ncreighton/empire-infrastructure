#!/usr/bin/env bash
# ============================================================
# OpenClaw — Batch Skill Installer
# Installs recommended ClawHub skills for the empire
# ============================================================
set -euo pipefail

echo "🔧 Installing ClawHub skills for Nick's Empire..."
echo "=================================================="

# Essential Skills
ESSENTIAL_SKILLS=(
  "gmail"
  "google-calendar"
  "wordpress"
  "github"
  "docker"
  "ssh-remote"
)

# Content & Media Skills
CONTENT_SKILLS=(
  "fal-ai"
  "ffmpeg-video-editor"
  "rss-reader"
  "image-editor"
)

# Research Skills
RESEARCH_SKILLS=(
  "tavily"
  "perplexity"
  "serp-analysis"
)

# Productivity Skills
PRODUCTIVITY_SKILLS=(
  "google-sheets"
  "google-drive"
  "notion"
  "todoist"
)

# Utility Skills
UTILITY_SKILLS=(
  "qr-code"
  "screenshot"
  "pdf-tools"
  "json-tools"
)

install_skills() {
  local category="$1"
  shift
  local skills=("$@")
  
  echo ""
  echo "📦 Installing ${category} skills..."
  echo "---"
  
  for skill in "${skills[@]}"; do
    echo -n "  → ${skill}... "
    if openclaw skills install "${skill}" 2>/dev/null; then
      echo "✓"
    else
      echo "⚠ (may not exist on ClawHub yet — skip)"
    fi
  done
}

# Install all categories
install_skills "Essential" "${ESSENTIAL_SKILLS[@]}"
install_skills "Content & Media" "${CONTENT_SKILLS[@]}"
install_skills "Research" "${RESEARCH_SKILLS[@]}"
install_skills "Productivity" "${PRODUCTIVITY_SKILLS[@]}"
install_skills "Utility" "${UTILITY_SKILLS[@]}"

echo ""
echo "=================================================="
echo "✅ Skill installation complete!"
echo ""
echo "Installed skills:"
openclaw skills list 2>/dev/null || echo "(Run 'openclaw skills list' to verify)"
echo ""
echo "Custom empire skills (copy manually from ./skills/):"
echo "  → wordpress-empire-manager"
echo "  → content-calendar"
echo "  → kdp-publisher"
echo "  → etsy-pod-manager"
echo "  → revenue-tracker"
echo "  → brand-voice-library"
echo "  → n8n-empire-webhook"
