---
name: amplify-pipeline
description: 6-stage task enhancement — enriches, hardens, and optimizes every automation before execution
version: 1.0.0
stages: [enrich, expand, fortify, anticipate, optimize, validate]
---

# AMPLIFY Pipeline — Make Every Task Bulletproof

AMPLIFY takes a raw automation task and runs it through 6 enhancement stages. The task that comes out the other end is enriched with context, hardened against failures, optimized for speed, and gated for safety.

## The 6 Stages

```
Raw Task ──▶ ENRICH ──▶ EXPAND ──▶ FORTIFY ──▶ ANTICIPATE ──▶ OPTIMIZE ──▶ VALIDATE ──▶ Enhanced Task
             (context)   (edges)   (retries)   (predictions)  (timing)     (safety)
```

### Stage 1: ENRICH — App & Context Awareness

Adds rich context about the target app, device state, and user preferences.

**App Profiles** (loaded from app-registry.json + CODEX):
| Category | Apps | Context Added |
|----------|------|---------------|
| Social Media | Facebook, Instagram, Twitter/X, TikTok, Pinterest, LinkedIn | Post formats, character limits, media requirements, algorithm timing |
| Messaging | WhatsApp, Telegram, Discord, Slack, SMS | Message types, group behavior, media sharing, end-to-end encryption notes |
| Browsers | Chrome, Firefox, Samsung Internet | Tab management, download behavior, cookie handling |
| Productivity | Gmail, Calendar, Drive, Notes | Account context, workspace vs personal, sharing defaults |
| Empire Tools | WordPress, Analytics, AdSense, Etsy, KDP | Site-specific configs, dashboard layouts, common workflows |

**What gets enriched**:
- App package name and launch intent
- Expected load time (from CODEX history or defaults)
- Authentication requirements and last known auth state
- Common screens and navigation paths
- Known quirks ("Facebook app sometimes shows 'Rate Us' dialog on launch")
- Device-specific adjustments (screen size, Android version)

### Stage 2: EXPAND — Edge Case Armor

Handles all the annoying Android things that break automation.

**Edge Cases Handled**:
| Edge Case | Detection | Response |
|-----------|-----------|----------|
| Notification shade down | Vision: dark overlay at top | Swipe up to dismiss |
| Permission dialog | Vision: "Allow" / "Deny" buttons | Grant if safe, deny if suspicious |
| App update dialog | Vision: "Update" / "Later" | Tap "Later" (don't interrupt task) |
| ANR dialog | Vision: "App not responding" | Wait 3s, then force close + relaunch |
| Screen rotation | UI dump: orientation changed | Lock to portrait, retry |
| Keyboard blocking target | Vision: keyboard visible + target below fold | Scroll target into view first |
| Cookie/GDPR banner | Vision: consent overlay | Accept and continue |
| Captcha | Vision: captcha detected | Pause, alert user, wait for manual solve |
| Low storage warning | System: <500MB free | Warn user, continue if possible |
| Split screen / PiP | UI dump: multi-window mode | Exit to single app mode |

**Safety Limits**:
- Max 50 steps per task (prevent infinite loops)
- Max 30 seconds wait per step (prevent hanging)
- Max 5 retries per action (prevent thrashing)
- Max 10 minutes total task duration
- Max 20 screenshots per task (prevent storage abuse)

### Stage 3: FORTIFY — Retry & Recovery

Adds intelligent retry logic with escalating fallback strategies.

**Retry Policies**:
```
tap_element:    3 attempts │ 0.5s delay │ 1.5x backoff
                Fallbacks: coordinates → accessibility ID → text search → vision re-scan

type_text:      2 attempts │ 0.3s delay │ 1.0x backoff
                Fallbacks: input text → clipboard paste → key-by-key events

swipe_scroll:   3 attempts │ 0.5s delay │ 1.5x backoff
                Fallbacks: adjust coordinates → try larger swipe → try fling gesture

launch_app:     3 attempts │ 2.0s delay │ 2.0x backoff
                Fallbacks: am start → force stop + relaunch → home + app drawer search

find_element:   5 attempts │ 1.0s delay │ 1.5x backoff
                Fallbacks: text match → content-desc → class+index → vision scan

navigate:       3 attempts │ 1.0s delay │ 1.5x backoff
                Fallbacks: intent → back+retry → home+relaunch → deep link
```

**Recovery Patterns**:
- **Soft recovery**: Retry the same action with adjusted parameters
- **Medium recovery**: Go back one step and try alternative path
- **Hard recovery**: Go home, relaunch app, start task from beginning
- **Abort**: Alert user, save progress, provide manual instructions

### Stage 4: ANTICIPATE — UI State Prediction

Predicts what the screen should look like after each action. If reality doesn't match prediction, triggers recovery.

**State Transition Map**:
```
Action: launch_app
  → Expected: App splash screen OR main feed (within 3-8s)
  → Failure indicators: still on home screen, crash dialog, "Unfortunately stopped"

Action: tap_button("Post")
  → Expected: Loading indicator OR success confirmation (within 2-5s)
  → Failure indicators: button still visible, error message, unchanged screen

Action: type_text("Hello world")
  → Expected: Text visible in focused field
  → Failure indicators: field still empty, different text, cursor not in field

Action: swipe_up (scroll)
  → Expected: New content visible below previous viewport
  → Failure indicators: same content (hit bottom), app switched, dialog appeared

Action: press_back
  → Expected: Previous screen in navigation stack
  → Failure indicators: app closed entirely, same screen, unexpected screen
```

**Verification Methods** (selected per action):
1. **Vision check**: Screenshot + AI analysis (most accurate, slowest)
2. **UI dump check**: Parse accessibility tree (fast, works for element states)
3. **Text presence**: Check if specific text appeared (fast, simple)
4. **Screen identity**: Verify we're on the expected screen name/activity

### Stage 5: OPTIMIZE — Performance Learning

Learns from history to make everything faster and more reliable.

**What gets optimized**:
- **Wait times**: Use actual measured load times instead of conservative defaults
- **Screenshot frequency**: Skip screenshots when recent one is <2s old
- **Action grouping**: Batch sequential same-app actions (no context switch overhead)
- **Verification level**: Use lighter checks for known-reliable actions, heavier for unknown
- **Timing windows**: Learn when apps are fastest (avoid peak hours for slow servers)

**Data tracked** (in `data/amplify/timing_data.json`):
```json
{
  "com.facebook.katana": {
    "launch": {"p50": 2.1, "p90": 4.3, "p99": 8.1, "samples": 47},
    "navigate_to_create_post": {"p50": 1.2, "p90": 2.5, "samples": 23},
    "upload_photo": {"p50": 3.5, "p90": 7.2, "samples": 12}
  }
}
```

### Stage 6: VALIDATE — Safety Gate

Final checkpoint before execution. Blocks dangerous tasks, confirms irreversible ones.

**Preflight Checks by Risk Level**:

| Risk Level | Checks | User Action |
|------------|--------|-------------|
| Low (reading, browsing) | App installed, screen ready | Auto-proceed |
| Medium (typing, navigating) | Above + correct account, correct app | Auto-proceed with logging |
| High (posting, sending) | Above + content review, recipient confirm | Request user confirmation |
| Critical (deleting, purchasing) | Above + explicit approval, backup check | REQUIRE user confirmation |

**Irreversible Action Detection**:
- Posting content (social media, blog)
- Sending messages
- Making purchases/payments
- Deleting content/accounts
- Changing passwords/security settings
- Uninstalling apps

For any irreversible action, VALIDATE will:
1. Summarize what will happen
2. Show the content/action for review
3. Wait for explicit user approval
4. Create a checkpoint (screenshot) before proceeding

## Usage

AMPLIFY runs automatically on every task. The full pipeline:

```
amplify run "Post a photo to Instagram with caption 'Sunset vibes'"
# Returns enhanced task with all 6 stages applied:
# {
#   "original_task": "Post a photo to Instagram...",
#   "enriched": {app_context, load_time, auth_state, ...},
#   "edge_cases": ["cookie_banner", "update_dialog"],
#   "retry_policies": {per_action_retries},
#   "predictions": {per_step_expected_states},
#   "timing": {optimized_waits},
#   "validation": {safety_level: "high", requires_confirmation: true},
#   "estimated_duration": "35-50 seconds",
#   "risk_level": "medium"
# }
```

## Combined with FORGE

AMPLIFY and FORGE work together:
- FORGE **analyzes** (is the task doable? what could go wrong?)
- AMPLIFY **enhances** (make it more reliable, faster, safer)
- FORGE **learns** from the result (CODEX records everything)
- Next time AMPLIFY **uses** that knowledge (OPTIMIZE stage)

The more tasks you run, the better both systems get.
