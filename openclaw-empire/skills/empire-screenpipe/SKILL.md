---
name: empire-screenpipe
description: Passive monitoring via Screenpipe OCR — activity tracking, error detection, progress monitoring
version: 1.0.0
requires: [screenpipe]
---

# Empire Screenpipe — Passive Intelligence Layer

Leverages Screenpipe's continuous OCR capture and audio transcription to provide passive monitoring, historical context, and ambient awareness for all automation tasks.

## What Screenpipe Provides

Screenpipe runs continuously on the Windows PC (port 3030), capturing:
- **Screen OCR**: Text visible on any window, every 1-2 seconds
- **Audio transcription**: Microphone and system audio
- **UI events**: Clicks, keystrokes, app switches, clipboard operations

This skill taps into that data stream for:
1. **Monitoring automation progress** without touching the UI
2. **Historical context** — what happened in the last hour/day
3. **Error detection** — catch errors across any app
4. **Activity intelligence** — understand usage patterns
5. **Cross-device awareness** — if phone is mirrored via scrcpy, Screenpipe sees it

## Commands

### Real-Time Monitoring
```
screenpipe watch "Processing article"     # Alert when text appears on screen
screenpipe watch --gone "Loading..."      # Alert when text disappears
screenpipe watch --app "Chrome" "Error"   # Watch specific app for errors
screenpipe monitor --duration 30m         # Monitor all activity for 30 minutes
```

### Search & History
```
screenpipe search "witchcraftforbeginners"                    # Find mentions
screenpipe search "error" --last 1h                           # Errors in last hour
screenpipe search "published" --app "Chrome" --last 24h       # Published articles
screenpipe search --audio "meeting notes" --last 2h           # Audio transcriptions
```

### Activity Intelligence
```
screenpipe activity --last 1h          # What apps were used, what text was visible
screenpipe timeline --last 30m         # Minute-by-minute activity log
screenpipe apps --last 4h              # App usage breakdown
screenpipe typing --last 1h            # What was typed (keyboard events)
screenpipe clicks --last 30m           # Click activity by app
```

### Phone Monitoring (via scrcpy mirror)
When the Android phone screen is mirrored to PC via scrcpy, Screenpipe captures it:
```
screenpipe phone-state                 # Current phone screen text (via OCR)
screenpipe phone-search "Success"      # Search phone screen history
screenpipe phone-errors --last 15m     # Errors on phone screen
screenpipe phone-progress "uploading"  # Track upload/download progress
```

### Integration with FORGE
```
# Feed Screenpipe observations into FORGE Codex
screenpipe feed-forge --last 1h        # Send last hour of observations to learning
screenpipe patterns --app "Facebook"   # Observed patterns for app
screenpipe errors --summarize          # AI summary of recent errors
```

## How It Enhances Phone Automation

### Before Task (SCOUT support)
- Check if any errors appeared recently on the PC or mirrored phone
- Verify the phone mirror is active (scrcpy window visible)
- Review recent activity for context (what was the user doing?)

### During Task (Progress monitoring)
- Watch for progress indicators via OCR ("Uploading...", "1 of 5", "Processing")
- Detect errors without taking screenshots (faster, non-invasive)
- Monitor task duration against expected time

### After Task (CODEX learning)
- Feed all observed text patterns to CODEX
- Record what screens were visible during successful vs failed tasks
- Track timing data from OCR timestamps

## Screenpipe API Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `GET /search` | Search OCR text, audio, UI events |
| `GET /health` | Check Screenpipe is running |

### Search Parameters
- `q` — Search query (text)
- `content_type` — `ocr`, `audio`, `ui`, `all`
- `app_name` — Filter by application
- `start_time` / `end_time` — Time range (ISO 8601)
- `limit` — Max results
- `min_length` / `max_length` — Content length filter

## Configuration

Screenpipe must be running on the Windows PC:
- **Binary**: `C:\Users\ncreighton\screenpipe\bin\screenpipe.exe`
- **Port**: 3030
- **MCP**: Registered via `claude mcp add` in per-project config

### For Phone Screen Monitoring
Install and run scrcpy to mirror Android screen to PC:
```bash
# On Windows:
scoop install scrcpy
scrcpy --window-title "Android Phone" --stay-awake

# Or via ADB wireless:
adb connect <phone-ip>:5555
scrcpy
```

Screenpipe will automatically capture the scrcpy window via OCR.

## Data Flow

```
Screenpipe (continuous capture)
    │
    ├── OCR Text Stream ──▶ Empire Screenpipe Agent
    │                         ├── Pattern Matching (errors, progress)
    │                         ├── Activity Timeline
    │                         └── FORGE Codex Feed
    │
    ├── Audio Stream ──▶ Transcription Search
    │                     └── Meeting notes, voice commands
    │
    └── UI Events ──▶ Usage Analytics
                       ├── App switching patterns
                       ├── Click/type activity
                       └── Workflow reconstruction
```

## Limitations

- Screenpipe runs on Windows PC only (not on the Android phone directly)
- Phone monitoring requires scrcpy mirror (adds ~100ms latency to OCR)
- OCR accuracy depends on font size and contrast
- Audio transcription may miss quiet or overlapping speech
- UI events only captured on the PC, not the phone
