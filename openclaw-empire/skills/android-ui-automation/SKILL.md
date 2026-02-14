---
name: android-ui-automation
description: Full Android screen control via ADB/Shizuku — launch apps, tap, type, swipe, vision-guided automation
version: 1.0.0
requires: [termux-api, shizuku]
---

# Android UI Automation — Full Phone Screen Control

Control any Android app remotely. Launch apps, tap buttons, type text, scroll, navigate — all guided by AI vision that sees and understands the phone screen in real-time.

## Architecture

```
Command ("Post to Facebook about Valentine's Day")
    │
    ▼
FORGE Intelligence (pre-scan phone state, predict risks)
    │
    ▼
AMPLIFY Pipeline (enhance task, add retries, edge cases)
    │
    ▼
Task Executor (break into steps)
    │
    ├── Step 1: Launch Facebook (am start)
    ├── Step 2: Screenshot → Vision → Find "What's on your mind?"
    ├── Step 3: Tap coordinates from vision
    ├── Step 4: Type content (input text)
    ├── Step 5: Screenshot → Vision → Find "Post" button
    ├── Step 6: Tap Post
    └── Step 7: Screenshot → Vision → Verify posted
    │
    ▼
FORGE Codex (record outcome, learn for next time)
```

## Commands

### App Control
```bash
# Launch any app by package name
phone app launch com.facebook.katana
phone app launch com.instagram.android
phone app launch com.twitter.android

# Launch by friendly name (resolved via app-registry.json)
phone app launch facebook
phone app launch instagram
phone app launch chrome

# Force stop an app
phone app stop com.facebook.katana

# List running apps
phone app list-running

# Check if app is installed
phone app check com.facebook.katana
```

### Screen Interaction
```bash
# Tap at coordinates
phone tap 540 1200

# Tap element by description (vision-guided)
phone tap-element "Post button"
phone tap-element "search bar"
phone tap-element "hamburger menu"

# Type text (into focused field)
phone type "Hello world! Happy Valentine's Day"

# Type with field clear first
phone type --clear "New text replacing old"

# Swipe / Scroll
phone swipe up          # Scroll down
phone swipe down        # Scroll up
phone swipe left        # Next page/tab
phone swipe right       # Previous page/tab
phone swipe 100 500 900 500 300   # Custom: x1 y1 x2 y2 duration_ms

# Press buttons
phone press back
phone press home
phone press recent
phone press enter
phone press volume-up
phone press volume-down
phone press power
```

### Screen Analysis
```bash
# Take screenshot (returns base64 or saves to path)
phone screenshot
phone screenshot --save /tmp/screen.png

# Get full UI element hierarchy
phone ui-dump

# Find element by text or description
phone find-element "Create Post"
phone find-element --type button "Submit"
phone find-element --type input "Search"

# Wait for element to appear (with timeout)
phone wait-for "Success" --timeout 10
phone wait-for "Loading" --gone --timeout 30

# Read text from screen region
phone read-text --region "0,0,1080,200"    # Status bar area
phone read-text --region "0,1800,1080,2400" # Bottom nav area

# Identify current screen
phone identify-screen
```

### Vision-Guided Tasks (High-Level)
```bash
# Execute a natural language task with full vision guidance
phone task "Open Facebook and create a post saying Happy Valentine's Day"
phone task "Go to Instagram, search for #witchcraft, and save the first 3 posts"
phone task "Open Chrome, go to witchcraftforbeginners.com, check if latest post is live"
phone task "Open Gmail and reply to the latest email from Amazon with 'Thank you'"
phone task "Take a selfie and share it to WhatsApp status"

# Task with explicit app target
phone task --app facebook "Create a post with photo from gallery"
phone task --app chrome "Navigate to google.com and search for 'OpenClaw'"

# Task with safety confirmation (for irreversible actions)
phone task --confirm "Delete the draft post on WordPress"
```

### Device Management
```bash
# Screen control
phone screen on
phone screen off
phone screen unlock         # Requires known PIN/pattern in config
phone screen brightness 128 # 0-255
phone screen rotate portrait
phone screen rotate landscape

# Device info
phone battery
phone wifi-info
phone storage-info
phone device-info

# Notifications
phone notifications list
phone notifications clear
phone notifications dismiss 3  # Dismiss notification #3
```

## Vision-Guided Automation Loop

Every action follows this cycle:

1. **Screenshot** — Capture current phone screen
2. **Analyze** — Send to Vision Service (Claude Haiku) with SENTINEL-optimized prompt
3. **Plan** — Determine next action based on task goal + current screen state
4. **Execute** — Run ADB command (tap, type, swipe, etc.)
5. **Verify** — Screenshot again, confirm action succeeded
6. **Learn** — FORGE Codex records outcome for future improvement

### Verification Strategies
- **Visual diff**: Compare before/after screenshots
- **Text presence**: Check if expected text appeared
- **UI dump**: Verify element state changed
- **Screen identity**: Confirm we're on the expected screen

## App Registry

Known apps with pre-mapped navigation patterns are in `configs/app-registry.json`. Each entry includes:
- Package name and launch activity
- Typical load time
- Common screens and how to reach them
- Known UI quirks and workarounds
- Authentication requirements

### Supported App Categories
| Category | Apps |
|----------|------|
| Social Media | Facebook, Instagram, Twitter/X, TikTok, Pinterest, LinkedIn |
| Messaging | WhatsApp, Telegram, Discord, Slack, SMS |
| Browsers | Chrome, Firefox, Samsung Internet |
| Productivity | Gmail, Calendar, Google Drive, Notes, Files |
| Empire Tools | WordPress, Google Analytics, AdSense, Etsy, Amazon KDP |
| Media | YouTube, Spotify, Camera, Gallery |
| System | Settings, Play Store, File Manager |

## FORGE + AMPLIFY Integration

Every task automatically passes through:
1. **FORGE SCOUT** — Is the phone ready? Screen on? WiFi connected? App installed?
2. **FORGE ORACLE** — Risk prediction. How likely is this to fail? What to watch for?
3. **AMPLIFY ENRICH** — Add app-specific context and known patterns
4. **AMPLIFY EXPAND** — Handle edge cases (popups, notifications, updates)
5. **AMPLIFY FORTIFY** — Add retry logic with vision-verified recovery
6. **AMPLIFY ANTICIPATE** — Predict what each screen should look like after each step
7. **AMPLIFY OPTIMIZE** — Use learned timing data for this specific app
8. **AMPLIFY VALIDATE** — Safety gate before irreversible actions
9. **Execution** — Run the enhanced task
10. **FORGE CODEX** — Record everything learned

## Prerequisites

1. **Termux** + **Termux:API** installed from F-Droid
2. **Shizuku** installed and running (provides ADB-level access)
3. Wireless debugging enabled in Developer Options
4. Shizuku authorized for Termux
5. Phone paired as OpenClaw node to gateway

## Limitations

- First-run for any app requires vision learning (slower initially, faster each time)
- Captchas and 2FA may require user intervention
- Some banking/security apps block overlay and screenshot
- Actions are sequential (one screen interaction at a time)
- Performance depends on phone hardware and network speed
