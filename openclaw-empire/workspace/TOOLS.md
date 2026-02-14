# Empire Tools Registry

## Built-in Tools (OpenClaw Core)
These are available automatically:
- `read` / `write` / `edit` — File operations on gateway
- `bash` / `process` — Shell commands
- `browser` — Chrome/Chromium via CDP
- `canvas` — Visual workspace with A2UI
- `sessions` — Multi-session management
- `cron` — Scheduled tasks

## Android Phone — Full UI Automation (via Shizuku + ADB)
Requires paired Android node with Shizuku enabled. Two levels of control:

### Level 1: Screen Control (ADB commands via Shizuku)
Direct phone screen interaction — tap, type, swipe, launch apps.

| Tool | Command | Description |
|------|---------|-------------|
| Tap | `phone tap {x} {y}` | Tap at screen coordinates |
| Tap Element | `phone tap-element "{desc}"` | Vision-guided tap (finds element) |
| Type | `phone type "{text}"` | Type text into focused field |
| Swipe | `phone swipe {direction}` | Swipe up/down/left/right |
| Launch App | `phone app launch {name}` | Open any app by name or package |
| Stop App | `phone app stop {package}` | Force stop an app |
| Back | `phone press back` | Press back button |
| Home | `phone press home` | Press home button |
| Screenshot | `phone screenshot` | Capture phone screen |
| UI Dump | `phone ui-dump` | Get full UI element hierarchy |
| Find Element | `phone find-element "{text}"` | Find element by text/description |
| Wait For | `phone wait-for "{text}"` | Wait for element to appear |
| Screen On/Off | `phone screen on/off` | Wake or sleep device |
| Brightness | `phone screen brightness {0-255}` | Set screen brightness |
| Rotate | `phone screen rotate portrait/landscape` | Set orientation |

### Level 2: Vision-Guided Tasks (AI-powered)
Natural language tasks executed with vision verification at every step.

```
phone task "Open Facebook and create a post saying Hello World"
phone task "Go to Instagram, search for #witchcraft, save the first 3 posts"
phone task "Open Chrome, navigate to witchcraftforbeginners.com"
phone task "Take a selfie and share it to WhatsApp status"
phone task --app pinterest "Create a pin for Moon Ritual article"
phone task --confirm "Delete the draft post on WordPress"
```

### Level 3: Termux:API (Headless)
No screen interaction needed. Invoke via `nodes invoke --node "android" --command <tool>`.

| Tool | Command | Description |
|------|---------|-------------|
| Camera | `camera.snap` | Take photo (params: camera=front/back) |
| Video | `camera.clip` | Record video (params: duration=seconds) |
| Screen | `screen.record` | Record phone screen |
| Location | `location.get` | GPS coordinates |
| SMS Send | `termux.sms-send` | Send text message |
| SMS List | `termux.sms-list` | Read inbox |
| Call | `termux.telephony-call` | Make phone call |
| Notify | `termux.notification` | Create notification |
| Clipboard | `termux.clipboard-get` | Read clipboard |
| TTS | `termux.tts-speak` | Text to speech |
| Torch | `termux.torch` | Toggle flashlight |
| Battery | `termux.battery-status` | Battery info |
| WiFi | `termux.wifi-connectioninfo` | WiFi details |
| Contacts | `termux.contact-list` | Read contacts |
| Audio | `termux.microphone-record` | Record audio |

## Intelligence API (port 8765)
FastAPI server exposing FORGE + AMPLIFY + Vision + Screenpipe.

### Task Execution
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/task/execute` | POST | Execute a natural language task with full intelligence |
| `/task/analyze` | POST | Pre-flight analysis only (don't execute) |
| `/task/{id}/status` | GET | Monitor running task progress |
| `/task/{id}/complete` | POST | Record task completion for learning |

### Phone Control API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/phone/screenshot` | POST | Take screenshot + analyze |
| `/phone/state` | GET | Comprehensive phone state |
| `/phone/tap` | POST | Tap at coordinates |
| `/phone/type` | POST | Type text |
| `/phone/swipe` | POST | Swipe gesture |
| `/phone/launch` | POST | Launch app |

### FORGE Intelligence
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/forge/pre-flight` | POST | Full pre-task analysis |
| `/forge/stats` | GET | FORGE engine statistics |
| `/forge/codex/app/{name}` | GET | What CODEX knows about an app |
| `/forge/codex/patterns/{name}` | GET | Failure patterns for an app |
| `/forge/codex/learn` | POST | Feed learning data |
| `/forge/vision-prompt` | POST | Get SENTINEL-optimized prompt |

### AMPLIFY Pipeline
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/amplify/process` | POST | Run full 6-stage pipeline |
| `/amplify/stats/{app}` | GET | App performance statistics |
| `/amplify/record` | POST | Record execution timing |

### Screenpipe
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/screenpipe/state` | GET | Current screen state via OCR |
| `/screenpipe/errors` | GET | Recent errors detected |
| `/screenpipe/timeline` | GET | Activity timeline |
| `/screenpipe/monitor` | POST | Watch for a text pattern |

### Vision
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vision/analyze` | POST | Analyze a screenshot |
| `/vision/find-element` | POST | Find UI element by description |
| `/vision/detect-state` | POST | Detect app state |
| `/vision/detect-errors` | POST | Detect error dialogs |

## n8n Webhooks
Trigger n8n workflows from OpenClaw:

| Workflow | Webhook Path | Payload |
|----------|-------------|---------|
| Content Pipeline | `/webhook/openclaw-content` | `{site, topic, keywords, voice}` |
| Publish | `/webhook/openclaw-publish` | `{site, post_id, channels}` |
| KDP Generate | `/webhook/openclaw-kdp` | `{title, niche, outline}` |
| Site Monitor | `/webhook/openclaw-monitor` | `{site, check_type}` |
| Revenue Alert | `/webhook/openclaw-revenue` | `{source, amount, period}` |
| Design Audit | `/webhook/openclaw-audit` | `{site, audit_type}` |

## WordPress REST API
All sites accessible via REST API with application passwords:

```bash
# Example: Create post on WitchcraftForBeginners
curl -X POST "https://witchcraftforbeginners.com/wp-json/wp/v2/posts" \
  -H "Authorization: Basic $(echo -n 'nick:APP_PASSWORD' | base64)" \
  -H "Content-Type: application/json" \
  -d '{"title":"New Post","content":"...","status":"draft"}'
```

### Key Endpoints
- `GET /wp-json/wp/v2/posts` — List posts
- `POST /wp-json/wp/v2/posts` — Create post
- `PUT /wp-json/wp/v2/posts/{id}` — Update post
- `GET /wp-json/wp/v2/categories` — List categories
- `POST /wp-json/wp/v2/media` — Upload media
- `GET /wp-json/rankmath/v1/getHead` — SEO data (RankMath)

## External APIs
| Service | Purpose | Env Var |
|---------|---------|---------|
| Anthropic | Primary AI model | `ANTHROPIC_API_KEY` |
| ElevenLabs | Voice TTS | `ELEVENLABS_API_KEY` |
| fal.ai | Image generation | Via ClawHub skill |
| Brave Search | Web research | `BRAVE_API_KEY` |
| Exa | Deep research | `EXA_API_KEY` |
| Printify | POD fulfillment | Via Etsy integration |
| Vision Service | Phone screen analysis | `http://localhost:8002` |
| Screenpipe | OCR monitoring | `http://localhost:3030` |

## App Registry
Known Android apps with pre-mapped UI patterns are in `configs/app-registry.json`. Includes:
- 20+ apps across 5 categories (social, messaging, browsers, productivity, empire tools)
- Package names, load times, auth types, screen indicators
- Known quirks and workarounds per app
- Global edge case handlers (permissions, popups, crashes, updates)
