---
name: forge-intelligence
description: Adaptive learning engine — predicts failures, optimizes actions, remembers everything
version: 1.0.0
modules: [scout, sentinel, oracle, smith, codex]
---

# FORGE Intelligence Engine — Adaptive Phone Automation Brain

FORGE makes every automation task smarter. It scans before acting, predicts failures before they happen, optimizes vision analysis, auto-fixes problems, and remembers everything for next time.

## The 5 Modules

### SCOUT — Pre-Task Environment Scanner
Scans the phone's state BEFORE any task begins. Catches problems early.

```
SCOUT checks:
├── Screen state (on/off, locked, orientation)
├── Active app (is something blocking?)
├── Notifications (any covering the screen?)
├── WiFi/Network connectivity
├── Battery level (sufficient for task?)
├── Storage space (for screenshots, downloads)
├── Target app (installed? updated? permissions granted?)
├── Recent errors (anything crashed recently?)
└── Custom rules (user-defined, persist across sessions)
```

**Output**: Readiness score (0-100), blocking issues, warnings, recommended pre-actions

**Usage**:
```
forge scout                          # Full phone scan
forge scout --app facebook           # Check readiness for specific app
forge scout --quick                  # Fast check (screen + network only)
```

### SENTINEL — Vision Prompt Optimizer
Makes the AI vision system see phone screens more accurately by continuously optimizing prompts.

**8 Prompt Templates** (each continuously refined):
1. `identify_screen` — What app and screen is showing?
2. `find_element` — Where exactly is this button/field?
3. `read_text` — What text is in this screen region?
4. `detect_state` — Is the app loading? Error? Logged out?
5. `verify_action` — Did the last tap/type actually work?
6. `detect_errors` — Any popup, crash dialog, permission request?
7. `compare_states` — What changed between these two screenshots?
8. `navigation_check` — Are we on the expected screen?

**How it learns**:
- Every vision analysis gets a quality score (0-1)
- Prompts that score well get used more
- Context enrichment: adds app name, expected state, last action, device info
- Tracks last 100 scores per prompt variant
- Persists to `data/forge/sentinel_scores.json`

### ORACLE — Failure Prediction Engine
Predicts how likely a task is to fail BEFORE execution. Suggests preventive actions.

**Risk Factors** (weighted):
| Factor | Weight | Example |
|--------|--------|---------|
| App complexity | 0.15 | Facebook=high, Calculator=low |
| Network dependency | 0.12 | Posting needs internet, Notes doesn't |
| Auth required | 0.15 | Login flows are failure-prone |
| Multi-step depth | 0.18 | 20 steps = riskier than 3 steps |
| First run for app | 0.15 | Never automated this app before |
| Time sensitivity | 0.10 | Scheduled post vs casual browse |
| Irreversible action | 0.15 | Posting/sending can't be undone |

**Risk Levels**:
- **Low** (<0.15) — Proceed confidently
- **Medium** (<0.35) — Proceed with extra verification
- **High** (<0.60) — Add checkpoints, confirm with user before irreversible steps
- **Critical** (>=0.60) — Warn user, suggest manual assistance

**Usage**:
```
forge oracle "Open Facebook and delete my last 5 posts"
# → Risk: CRITICAL (0.72) — irreversible + multi-step + auth
# → Suggestions: Require user confirmation at each delete

forge oracle "Check battery level"
# → Risk: LOW (0.05) — single step, no network, no auth
```

### SMITH — Auto-Fix Generator
When SCOUT finds problems, SMITH generates and applies fixes automatically.

**Fix Strategies**:
| Problem | Fix | Confidence |
|---------|-----|------------|
| Screen off | Wake device + unlock | 0.95 |
| Notification blocking | Swipe to dismiss | 0.85 |
| App not installed | Report to user | 0.99 |
| Dialog/popup blocking | Identify and dismiss | 0.75 |
| Wrong screen/app | Press back or home, relaunch | 0.80 |
| Network disconnected | Toggle WiFi, wait 5s, retry | 0.70 |
| Permission denied | Guide user to Settings | 0.60 |
| Low battery | Reduce brightness, warn user | 0.90 |
| App crashed | Force stop, relaunch | 0.85 |
| Keyboard covering element | Scroll target into view | 0.80 |

**Codex-informed**: SMITH prefers fixes that worked in the past for this specific situation.

### CODEX — Persistent Learning Memory
The brain that remembers everything. Every task, every success, every failure — stored and used to make future tasks smarter.

**What CODEX stores**:
```
data/forge/
├── codex_tasks.json          # Last 500 task executions
│   └── {task_id, app, steps, outcome, duration, errors}
├── codex_patterns.json       # Failure patterns per app (100/app)
│   └── {app: [{pattern, frequency, fix, success_rate}]}
├── codex_app_knowledge.json  # Learned app behaviors
│   └── {app: {load_time, common_screens, quirks, auth_flow}}
├── codex_preferences.json    # Optimal settings per context
│   └── {action: {preferred_method, timing, retries}}
└── codex_vision_tips.json    # Vision analysis learnings (20/type)
    └── {task_type: [{tip, quality_score, context}]}
```

**How CODEX improves over time**:
1. First run of any app: CODEX has no data → relies on app-registry defaults
2. After 5 runs: CODEX knows load times, common failure points
3. After 20 runs: CODEX has accurate timing, knows all screen transitions
4. After 50+ runs: CODEX anticipates problems before they happen

**Usage**:
```
forge codex stats                    # Overall learning summary
forge codex app facebook             # What we know about Facebook automation
forge codex patterns facebook        # Known failure patterns
forge codex success-rate             # Success rates by app
forge codex reset                    # Clear all learning data (careful!)
forge codex export                   # Export knowledge base
```

## Integration

FORGE runs automatically on every task. You can also invoke it directly:

```
# Full pre-task analysis
forge analyze "Open Instagram and post a story"

# Returns:
# {
#   "scout": {"readiness": 85, "warnings": ["battery at 22%"]},
#   "oracle": {"risk": 0.28, "level": "medium"},
#   "sentinel": {"prompt_quality": 0.92},
#   "smith": {"fixes_applied": 0},
#   "codex": {"app_runs": 15, "success_rate": 0.87}
# }
```

## Learning Feedback

After every task, feed results back:
```
forge learn --task-id abc123 --success true --duration 45
forge learn --task-id abc123 --success false --error "App crashed during upload"
```

This closes the loop — CODEX records the outcome, ORACLE adjusts predictions, SENTINEL refines prompts.
