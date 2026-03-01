# PROJECT MESH v2.0: OMEGA ARCHITECTURE — Part 2

## Enhanced Layers 4-6 & New Systems 7-11

---

# LAYER 4: THE PULSE v2.0 — Enhanced Sync Protocol

## What's New

### 4.1 Transactional Sync with Rollback

Every sync operation is now ATOMIC — it either fully succeeds or fully rolls back:

```
SYNC TRANSACTION: content-pipeline v2.0 → smart-home-wizards

[1/6] CREATE SNAPSHOT
  → Saved rollback point: rollback/2025-02-28T091500_smarthomewizards.tar.gz
  → Includes: CLAUDE.md, .project-mesh/, affected code files

[2/6] RESOLVE DEPENDENCIES
  → content-pipeline v2.0 requires wp-api-wrapper ≥1.0.0 → ✅ satisfied
  → content-pipeline v2.0 requires queue-manager utility → ⚠️ syncing now...
  → queue-manager v1.0.0 synced successfully

[3/6] SYNC SYSTEM FILES
  → Copying content-pipeline v2.0 to .project-mesh/shared-systems/
  → 14 files updated, 3 new, 1 removed

[4/6] APPLY MIGRATIONS
  → Running migration: content-pipeline-v1-to-v2
  → Auto-fix: publishPost → queueContent (3 occurrences)
  → Auto-fix: config format upgrade (1 file)
  → Manual review needed: 1 file flagged (scripts/custom-publisher.js)

[5/6] VALIDATE
  → Config schema validation: ✅ passed
  → Smoke tests: ✅ 4/4 passed
  → Integration tests: ✅ 2/2 passed
  → CLAUDE.md recompile: ✅ valid

[6/6] COMMIT
  → Sync committed successfully
  → Rollback snapshot retained for 30 days
  → Sync log updated

RESULT: ✅ SUCCESS — content-pipeline v2.0 now active in smart-home-wizards
```

If ANY step fails:
```
[4/6] APPLY MIGRATIONS
  → Auto-fix: publishPost → queueContent → ❌ FAILED (file locked)

ROLLBACK INITIATED
  → Restoring from: rollback/2025-02-28T091500_smarthomewizards.tar.gz
  → .project-mesh/ restored
  → CLAUDE.md restored
  → Code files restored
  → ✅ Rollback complete — project unchanged

RESULT: ❌ FAILED — See sync-log for details. Fix and retry.
```

### 4.2 Intelligent Sync Scheduling

Not all syncs need to happen immediately:

```json
// sync-schedule.json
{
  "rules": [
    {
      "trigger": "security-patch",
      "mode": "emergency",
      "delay": "0",
      "description": "Security fixes sync immediately to all projects"
    },
    {
      "trigger": "bugfix",
      "mode": "auto",
      "delay": "5m",
      "description": "Bug fixes sync after 5-minute cooling period"
    },
    {
      "trigger": "minor-update",
      "mode": "auto",
      "delay": "1h",
      "description": "Minor updates batch and sync hourly"
    },
    {
      "trigger": "major-update",
      "mode": "staged",
      "delay": "manual",
      "description": "Major updates require manual approval"
    },
    {
      "trigger": "breaking-change",
      "mode": "staged",
      "delay": "manual",
      "priority_order": ["lowest-traffic-first"],
      "description": "Breaking changes roll out starting from least critical projects"
    }
  ],
  "batch_window": {
    "enabled": true,
    "collect_period_minutes": 60,
    "sync_at": "next-idle",
    "description": "Collect minor changes for 60 min, sync when system is idle"
  },
  "quiet_hours": {
    "enabled": true,
    "start": "22:00",
    "end": "06:00",
    "timezone": "America/New_York",
    "exception": ["security-patch"],
    "description": "No syncs during quiet hours except security patches"
  }
}
```

### 4.3 Sync Chain Orchestration

When syncing triggers cascading updates:

```
SYNC CHAIN DETECTED:

Step 1: wp-api-wrapper v2.0.0 published to shared-core
  ↓ triggers
Step 2: content-pipeline depends on wp-api-wrapper → needs compatibility check
  ↓ check passed, triggers
Step 3: content-pipeline consumers (12 projects) → need CLAUDE.md recompile
  ↓ also triggers
Step 4: seo-optimizer uses wp-api-wrapper internally → needs compatibility check
  ↓ check passed, triggers
Step 5: seo-optimizer consumers (14 projects) → need CLAUDE.md recompile

OPTIMIZATION: Steps 3 and 5 overlap (10 projects consume both)
  → Deduplicating: 16 unique projects need recompile (not 26)
  → Executing single batch recompile...
  → ✅ 16 CLAUDE.md files recompiled in 3.8 seconds
```

### 4.4 Sync Analytics & History

Complete history of every sync operation:

```json
// sync-log.json (rolling 90-day log)
{
  "entries": [
    {
      "id": "sync-20250228-001",
      "timestamp": "2025-02-28T09:15:00Z",
      "type": "auto",
      "trigger": "minor-update",
      "system": "image-optimization",
      "from_version": "2.9.0",
      "to_version": "3.0.0",
      "projects_affected": 12,
      "projects_synced": 12,
      "duration_seconds": 8.5,
      "status": "success",
      "rollback_available": true,
      "rollback_expires": "2025-03-30T09:15:00Z",
      "migrations_applied": 2,
      "auto_fixes": 15,
      "manual_reviews_needed": 2,
      "tests_passed": 48,
      "tests_failed": 0
    }
  ],
  "statistics": {
    "total_syncs_30d": 47,
    "success_rate": 0.98,
    "avg_duration_seconds": 6.2,
    "most_synced_system": "content-pipeline",
    "most_stable_system": "seo-optimizer",
    "rollbacks_30d": 1,
    "total_auto_fixes_30d": 230
  }
}
```

---

# LAYER 5: THE BRIDGE v2.0 — Enhanced Knowledge Sharing

## What's New

### 5.1 Auto-Capture from Claude Conversations

When working in any project, discoveries are auto-flagged for knowledge base capture:

```markdown
## Auto-Capture Protocol

When Claude discovers something during a project session that could benefit
other projects, it should create an entry in:
  .project-mesh/discoveries/pending/[timestamp].json

Format:
{
  "discovered_in": "current-project-name",
  "timestamp": "2025-02-28T14:30:00Z",
  "category": "api-quirk|lesson-learned|gotcha|performance|cost-optimization",
  "finding": "Description of what was discovered",
  "context": "What was happening when this was discovered",
  "affects": ["system-names", "project-names", "technologies"],
  "confidence": "verified|measured|observed|suspected",
  "solution": "What to do about it",
  "tags": ["relevant", "tags"],
  "auto_captured": true,
  "needs_human_review": true
}

On next sync, pending discoveries are:
  1. Moved to _empire-hub/knowledge-base/auto-captured/
  2. Added to pending-review.json
  3. Tagged for relevance scoring
  4. Available for CLAUDE.md compilation (with [UNVERIFIED] flag)
```

### 5.2 Knowledge Graph with Cross-References

Entries link to each other and to systems/projects:

```json
{
  "nodes": [
    {
      "id": "kn-001",
      "type": "api-quirk",
      "title": "Steel.dev Session Timeout",
      "system": "browser-automation"
    },
    {
      "id": "kn-002", 
      "type": "solution",
      "title": "Keep-Alive Ping Implementation",
      "system": "browser-automation"
    },
    {
      "id": "kn-003",
      "type": "gotcha",
      "title": "Hostinger Rate Limiting",
      "system": "wp-api-wrapper"
    },
    {
      "id": "kn-004",
      "type": "solution",
      "title": "WP-CLI Batch Operations",
      "system": "wp-api-wrapper"
    }
  ],
  "edges": [
    {"from": "kn-001", "to": "kn-002", "relationship": "solved-by"},
    {"from": "kn-003", "to": "kn-004", "relationship": "solved-by"},
    {"from": "kn-003", "to": "kn-001", "relationship": "similar-pattern"}
  ]
}
```

### 5.3 Knowledge Expiration & Review Cycle

Knowledge entries can go stale. The system tracks freshness:

```json
{
  "review_schedule": [
    {
      "entry_id": "kn-001",
      "title": "Steel.dev Session Timeout",
      "last_verified": "2025-02-28",
      "review_interval_days": 90,
      "next_review": "2025-05-29",
      "status": "current",
      "auto_verify_method": "test-steel-session-timeout"
    },
    {
      "entry_id": "kn-005",
      "title": "ZimmWriter CSV Headers",
      "last_verified": "2024-12-01",
      "review_interval_days": 60,
      "next_review": "2025-01-30",
      "status": "OVERDUE",
      "auto_verify_method": null,
      "note": "ZimmWriter deprecated — mark entry as historical?"
    }
  ]
}
```

### 5.4 Smart Knowledge Compilation

When compiling CLAUDE.md, the knowledge base entries are scored for relevance:

```
Selecting knowledge for: smart-home-wizards

SCORING ENTRIES:
  [Score: 95] Steel.dev Session Timeout → Directly uses browser-automation ✅
  [Score: 90] Hostinger Rate Limiting → All sites on Hostinger ✅
  [Score: 85] n8n Webhook Reliability → Uses n8n webhooks ✅
  [Score: 40] Substack Comment Rate Limit → Does NOT use Substack ❌
  [Score: 35] Etsy Moon Phase Timing → Does NOT use Etsy ❌
  [Score: 20] ZimmWriter CSV Headers → ZimmWriter deprecated ❌

INCLUDED (score ≥ 60): 3 entries
EXCLUDED (score < 60): 3 entries
BUDGET USED: 1,200 / 4,000 chars allocated
```

### 5.5 Searchable Knowledge Index

Full-text search across ALL knowledge:

```
$ python search/search.py "rate limit"

RESULTS (3 matches):

[1] API Quirks: Hostinger Rate Limiting (Score: 0.95)
    "Hostinger API starts returning 503 after ~50 rapid sequential requests"
    → Solution: Batch operations with 500ms delays
    → Affects: All 16 sites on Hostinger

[2] API Quirks: Substack Comment Rate Limit (Score: 0.88)
    "Substack API returns 429 if you post comments faster than every 2 seconds"
    → Solution: Add 2-second delay between comment posts
    → Affects: witchcraft-for-beginners, ai-discovery-digest

[3] Decisions Log: Rate Limiter Utility (Score: 0.72)
    "Decided to build shared-core/utilities/rate-limiter to handle all rate limiting"
    → Applied to: shared-core utilities
    → Date: 2025-02-20
```

---

# LAYER 6: THE GUARDIAN v2.0 — Enhanced Anti-Hallucination

## What's New

### 6.1 Graduated Deprecation Lifecycle

Methods don't just get deprecated overnight. They go through stages:

```
LIFECYCLE: publishPost() in content-pipeline

Stage 1: SOFT DEPRECATION (announced)
  Date: 2025-02-01
  Action: Warning in CLAUDE.md — "publishPost() will be deprecated. Start using queueContent()."
  Code behavior: Still works, logs deprecation warning
  
Stage 2: HARD DEPRECATION (enforced)  
  Date: 2025-02-15
  Action: Error in CLAUDE.md — "❌ NEVER use publishPost(). Use queueContent()."
  Code behavior: Still works, logs warning + increments violation counter
  Guardian: Flags any project still using it

Stage 3: SUNSET (removal countdown)
  Date: 2025-03-01
  Action: Removal warning — "publishPost() will be REMOVED on 2025-03-15"
  Code behavior: Works but emits critical warning
  Guardian: Blocks sync for projects that haven't migrated

Stage 4: REMOVED
  Date: 2025-03-15
  Action: Function removed from shared-core
  Code behavior: Throws error
  Guardian: Auto-migration offered for any remaining consumers
```

### 6.2 Deprecation Exception System

Sometimes a project needs extra time. The exception system handles this cleanly:

```json
// deprecated/exceptions/exceptions-registry.json
{
  "exceptions": [
    {
      "id": "exc-001",
      "project": "wealth-from-ai",
      "deprecated_item": "publishPost()",
      "reason": "Complex custom integration needs refactoring time",
      "granted_date": "2025-02-15",
      "expires_date": "2025-03-15",
      "granted_by": "nick",
      "status": "active",
      "migration_plan": "Scheduled for Week 2 March sprint",
      "reminder_at": ["2025-03-01", "2025-03-10"]
    }
  ]
}
```

In that project's compiled CLAUDE.md:
```markdown
## ⚠️ Active Deprecation Exceptions

This project has a TEMPORARY exception to use the following deprecated methods:
- `publishPost()` — Exception expires: 2025-03-15
  Reason: Complex custom integration needs refactoring time
  Plan: Scheduled for Week 2 March sprint
  
  ⚠️ DO NOT build new code using publishPost(). Only maintain existing code.
  All NEW code must use queueContent().
```

### 6.3 Pattern Detection Engine

The Guardian doesn't just list deprecated methods — it can DETECT them in code:

```json
// deprecated/patterns/code-patterns.json
{
  "patterns": [
    {
      "name": "direct-wp-rest-post",
      "description": "Direct WordPress REST API post creation",
      "severity": "high",
      "regex": "fetch\\(['\"].*\\/wp-json\\/wp\\/v2\\/posts['\"]",
      "file_types": [".js", ".ts", ".py"],
      "replacement": "Use shared-core/content-pipeline queueContent()",
      "auto_fixable": false
    },
    {
      "name": "hardcoded-webhook-url",
      "description": "Hardcoded n8n webhook URL",
      "severity": "high",
      "regex": "https?://[\\w.-]+/webhook/[a-f0-9-]+",
      "file_types": [".js", ".ts", ".py", ".json"],
      "replacement": "Use config-driven webhook URLs via n8n-webhook-handler",
      "auto_fixable": true,
      "auto_fix_transform": "Replace with config.get('webhooks.{detected_path}')"
    },
    {
      "name": "sharp-default-settings",
      "description": "Using sharp with default (unoptimized) settings",
      "severity": "medium",
      "regex": "sharp\\(.*\\)\\.(resize|jpeg|png|webp)\\(\\)",
      "file_types": [".js", ".ts"],
      "replacement": "Use shared-core/image-optimization with project config",
      "auto_fixable": false
    },
    {
      "name": "console-log-production",
      "description": "Using console.log instead of structured logging",
      "severity": "low",
      "regex": "console\\.(log|warn|error|info)\\(",
      "file_types": [".js", ".ts"],
      "replacement": "Use shared-core/utilities/structured-logging",
      "auto_fixable": true,
      "auto_fix_transform": "Replace console.{method} with logger.{method}"
    }
  ]
}
```

### 6.4 Compliance Scoring

Every project gets a compliance score:

```json
{
  "project": "smart-home-wizards",
  "compliance_score": 82,
  "breakdown": {
    "deprecated_usage": {
      "score": 70,
      "violations": [
        {"pattern": "console-log-production", "count": 12, "severity": "low"},
        {"pattern": "hardcoded-webhook-url", "count": 2, "severity": "high"}
      ]
    },
    "shared_core_adoption": {
      "score": 85,
      "available_systems": 14,
      "adopted_systems": 12,
      "missing": ["analytics-tracker", "lead-magnet-engine"]
    },
    "version_currency": {
      "score": 90,
      "outdated": [
        {"system": "image-optimization", "current": "2.9.0", "latest": "3.0.0"}
      ]
    },
    "documentation": {
      "score": 95,
      "claude_md_fresh": true,
      "manifest_complete": true,
      "local_md_exists": true
    },
    "testing": {
      "score": 78,
      "smoke_tests_passing": true,
      "integration_tests_passing": true,
      "config_valid": true,
      "coverage_gap": "No performance tests"
    }
  },
  "trend": "improving",
  "score_30d_ago": 75,
  "recommendations": [
    "Replace 2 hardcoded webhook URLs (high priority)",
    "Upgrade image-optimization to v3.0.0",
    "Replace 12 console.log calls with structured logging"
  ]
}
```

### 6.5 Auto-Fix Engine

For fixable violations, the Guardian can auto-repair:

```
AUTO-FIX REPORT: smart-home-wizards

FIXABLE VIOLATIONS:
  [1] Hardcoded webhook URL in: config/webhooks.js (line 15)
      Before: const url = "https://n8n.contabo.example/webhook/abc-123"
      After:  const url = config.get('webhooks.content-pipeline')
      → ✅ Fixed

  [2] Hardcoded webhook URL in: scripts/publish.js (line 42)
      Before: fetch("https://n8n.contabo.example/webhook/def-456", ...)
      After:  fetch(config.get('webhooks.seo-optimizer'), ...)
      → ✅ Fixed

  [3-14] console.log → logger migration (12 instances)
      → ✅ Fixed (added import { logger } from 'shared-core/utilities/structured-logging')

UNFIXABLE (needs manual review):
  [15] Direct WP REST API call in: lib/custom-importer.js (line 88)
      Reason: Custom logic too complex for auto-transform
      → Flagged for manual migration

RESULT: 14/15 violations auto-fixed. Compliance score: 82 → 94
```

---

# SYSTEM 7: THE FORGE — Auto-Extraction & System Evolution

## Purpose

The Forge automatically detects code in satellite projects that SHOULD be in shared-core, tracks how systems evolve over time, and scaffolds new shared systems.

### 7.1 Auto-Extraction Detection

The Forge scans all projects for shareable code:

```
FORGE SCAN REPORT — 2025-02-28

HIGH-CONFIDENCE EXTRACTION CANDIDATES:

[1] smart-home-device-api-wrapper
    Found in: smart-home-wizards/lib/device-api/
    Lines: 230 | Complexity: Medium | Tests: Yes
    Similar code found in: ai-in-action-hub/utils/iot-api.js (68% similarity)
    RECOMMENDATION: Extract to shared-core, merge both implementations
    Benefit: 2 projects immediately, any future IoT projects
    Effort: Low (well-structured, documented)

[2] markdown-to-html-converter
    Found in: ai-discovery-digest/utils/md-converter.js
    Lines: 85 | Complexity: Low | Tests: No
    Similar code found in: 
      - mythical-archives/lib/content-formatter.js (72% similarity)
      - bullet-journals/utils/md-parser.js (61% similarity)
    RECOMMENDATION: Extract to shared-core/utilities
    Benefit: 3+ projects
    Effort: Low

[3] social-media-share-formatter
    Found in: witchcraft-for-beginners/lib/social/
    Lines: 180 | Complexity: Medium | Tests: Partial
    No duplicates found, but universal applicability detected
    RECOMMENDATION: Extract to shared-core after adding tests
    Benefit: All content-generating projects (12+)
    Effort: Medium (needs test coverage)

MEDIUM-CONFIDENCE CANDIDATES:

[4] custom-cron-scheduler
    Found in: family-flourish/lib/scheduler.js
    Lines: 120 | Complexity: Low
    Overlaps with: shared-core/utilities/queue-manager (40%)
    RECOMMENDATION: Merge into queue-manager rather than separate system
    
DRIFT DETECTED:

[5] retry-logic-divergence
    Project A (witchcraft-for-beginners): Exponential backoff, max 3 retries
    Project B (ai-discovery-digest): Linear backoff, max 5 retries, with jitter
    Project C (smart-home-wizards): Exponential backoff, max 3 retries, with circuit breaker
    RECOMMENDATION: Merge ALL into shared-core/utilities/api-retry v2.0
      → Include: exponential backoff + jitter + circuit breaker + configurable max retries
      → This would be the "best of all three" merged implementation
```

### 7.2 System Scaffolder

Create new shared systems from a template in seconds:

```
$ python forge/scaffolder.py --name "social-media-formatter" --type system

Creating shared-core/systems/social-media-formatter/

├── src/
│   └── index.js          ← Entry point (template)
├── tests/
│   ├── unit/
│   │   └── test_basic.py ← Basic test template
│   ├── integration/
│   ├── smoke/
│   │   └── test_smoke.py ← Smoke test template
│   └── performance/
├── examples/
│   └── basic-usage.js    ← Usage example template
├── VERSION               ← "0.1.0"
├── CHANGELOG.md          ← Template
├── README.md             ← Template with sections
├── config.schema.json    ← Empty schema template
├── config.defaults.json  ← Empty defaults
├── DEPENDENCIES.json     ← Empty deps
├── CONSUMERS.md          ← Empty
├── MIGRATION.md          ← Template
└── meta.json             ← Pre-filled metadata

✅ System scaffolded! Next steps:
  1. Implement in src/
  2. Write tests
  3. Document in README.md
  4. Set version to 1.0.0 when ready
  5. Register consumers via their manifests
```

### 7.3 Evolution Tracker

How systems change over time:

```json
{
  "system": "content-pipeline",
  "evolution": [
    {
      "version": "0.1.0",
      "date": "2024-09-01",
      "origin": "witchcraft-for-beginners (manual extraction)",
      "lines_of_code": 80,
      "consumers": 1,
      "features": ["basic-post-creation"]
    },
    {
      "version": "1.0.0",
      "date": "2024-11-15",
      "changes": "Added SEO integration, bulk posting",
      "lines_of_code": 220,
      "consumers": 5,
      "features": ["basic-post-creation", "seo-integration", "bulk-posting"]
    },
    {
      "version": "2.0.0",
      "date": "2025-02-15",
      "changes": "Queue-based architecture, n8n integration, A/B testing",
      "lines_of_code": 440,
      "consumers": 12,
      "features": ["queue-based", "n8n-integration", "ab-testing", "source-tagging"],
      "breaking_changes": ["publishPost renamed to queueContent", "config format changed"]
    }
  ],
  "velocity": {
    "releases_per_month": 0.8,
    "avg_lines_changed_per_release": 120,
    "trend": "accelerating",
    "predicted_next_release": "2025-03-15"
  }
}
```

---

# SYSTEM 8: THE SENTINEL — Monitoring & Alerting

## Purpose

Real-time monitoring of mesh health with automated alerts when things go wrong.

### 8.1 Alert Rules

```json
{
  "alerts": [
    {
      "name": "stale-project",
      "condition": "project.staleness_days > 14",
      "severity": "warning",
      "message": "Project {project.name} hasn't been synced in {staleness_days} days",
      "action": "notify"
    },
    {
      "name": "critical-system-outdated",
      "condition": "consumed_system.criticality == 'critical' AND consumed_system.outdated == true",
      "severity": "critical",
      "message": "{project.name} is using outdated CRITICAL system: {system.name} v{current} → v{latest}",
      "action": "notify + auto-sync"
    },
    {
      "name": "compliance-drop",
      "condition": "project.compliance_score < project.compliance_score_7d_ago - 10",
      "severity": "warning",
      "message": "{project.name} compliance dropped from {old_score} to {new_score}",
      "action": "notify + generate-report"
    },
    {
      "name": "sync-failure",
      "condition": "sync.status == 'failed'",
      "severity": "critical",
      "message": "Sync failed for {project.name}: {error}",
      "action": "notify + retain-rollback"
    },
    {
      "name": "knowledge-review-overdue",
      "condition": "knowledge_entry.review_status == 'OVERDUE'",
      "severity": "info",
      "message": "{count} knowledge entries are overdue for review",
      "action": "notify-weekly-digest"
    },
    {
      "name": "drift-detected",
      "condition": "forge.drift_detected == true",
      "severity": "warning",
      "message": "Code drift detected: {description}",
      "action": "notify + add-to-extraction-candidates"
    },
    {
      "name": "deprecation-exception-expiring",
      "condition": "exception.expires_date - today <= 7 days",
      "severity": "warning",
      "message": "Deprecation exception for {project.name}/{item} expires in {days} days",
      "action": "notify"
    }
  ]
}
```

### 8.2 Anomaly Detection

Detect unusual patterns that might indicate problems:

```
ANOMALY REPORT — 2025-02-28

[WARNING] Sync frequency spike: content-pipeline synced 8 times today (avg: 1.2/day)
  → Possible cause: Rapid iteration cycle or unstable changes
  → Recommendation: Consider batching changes

[INFO] New dependency pattern: 3 projects added browser-automation in the last week
  → Possible cause: New automation initiative
  → Recommendation: Ensure all new consumers follow keep-alive pattern

[WARNING] Compliance score declining: ai-in-action-hub dropped 15 points in 7 days
  → Possible cause: New code not following shared patterns
  → Recommendation: Run full compliance scan and auto-fix

[OK] No critical anomalies detected. Empire health: GOOD
```

---

# SYSTEM 9: THE NEXUS — Deep Integration Layer

## Purpose

Connects Project Mesh to n8n, Git, and CI/CD for fully automated operations.

### 9.1 n8n Integration Workflows

**Mesh Sync Pipeline (mesh-sync-pipeline.json)**
```
Trigger: Webhook (from file watcher or git hook)
  → Node 1: Parse change event (what changed, where)
  → Node 2: Query registry (who's affected)
  → Node 3: Run impact analysis
  → Branch: 
      If breaking → Queue for manual review, notify via Discord
      If non-breaking → Continue
  → Node 4: Execute sync for each affected project
  → Node 5: Run smoke tests
  → Node 6: Recompile CLAUDE.md files
  → Node 7: Update health dashboard
  → Node 8: Send summary notification
  → Error handler: Rollback + alert
```

**Scheduled Health Check (health-check-scheduled.json)**
```
Trigger: Cron (daily at 8:00 AM ET)
  → Node 1: Run full health check across all projects
  → Node 2: Calculate compliance scores
  → Node 3: Check knowledge base review schedule
  → Node 4: Run Forge extraction scan
  → Node 5: Generate daily digest
  → Branch:
      If any critical alerts → Send immediate notification
      If only info/warnings → Include in digest
  → Node 6: Update dashboard data
  → Node 7: Send daily digest email/Discord
```

**Knowledge Auto-Capture (knowledge-capture.json)**
```
Trigger: Webhook (from Claude Code hook on-project-close)
  → Node 1: Receive discoveries from project session
  → Node 2: Deduplicate against existing knowledge
  → Node 3: Score relevance and confidence
  → Node 4: If high confidence → Auto-add to knowledge base
  → Node 5: If medium confidence → Add to pending-review
  → Node 6: If low confidence → Log and discard
  → Node 7: Trigger CLAUDE.md recompile for affected projects
```

### 9.2 Git Hook Integration

**Pre-Commit Hook (deprecated pattern detection)**
```python
# nexus/git-hooks/pre-commit.py
# Runs BEFORE every git commit in any satellite project

# 1. Load deprecated patterns from hub
# 2. Scan staged files for violations
# 3. Block commit if critical violations found
# 4. Warn (but allow) for non-critical violations

STAGED FILES SCAN:
  ✅ src/app.js — No deprecated patterns
  ❌ lib/publisher.js — BLOCKED: Uses publishPost() (deprecated)
  ⚠️  utils/logger.js — WARNING: Uses console.log (soft deprecated)

COMMIT BLOCKED — Fix 1 critical violation:
  lib/publisher.js:42 — Replace publishPost() with queueContent()
```

**Post-Commit Hook (auto-sync trigger)**
```python
# nexus/git-hooks/post-commit.py
# Runs AFTER every successful commit

# 1. Check if committed files affect shared systems
# 2. If in _empire-hub/shared-core → trigger sync pipeline
# 3. If in satellite project → check for extraction candidates
# 4. Update project's last_human_touch timestamp
```

---

# SYSTEM 10: THE ORACLE — Predictive Intelligence

## Purpose

Analyze mesh data to predict problems before they happen and recommend proactive actions.

### 10.1 Predictive Analysis

```markdown
## Oracle Weekly Forecast — 2025-02-28

### Systems Likely to Need Updates
1. **browser-automation** (confidence: 85%)
   - Velocity: 2 updates/month
   - Last update: 18 days ago
   - Predicted next update: ~5 days
   - Reason: Steel.dev API has been releasing weekly changes

2. **content-pipeline** (confidence: 70%)
   - Consumer growth: 3 new projects in 30 days
   - Current bottleneck: queue throughput at 80% capacity
   - Predicted: Performance update needed within 2 weeks

### Projects at Risk of Drift
1. **wealth-from-ai** (risk: HIGH)
   - Sync health: 40%
   - Active deprecation exceptions: 1 (expiring in 15 days)
   - No commits in 21 days
   - RECOMMENDATION: Schedule dedicated maintenance session

2. **ai-in-action-hub** (risk: MEDIUM)
   - Compliance declining (dropped 5 points/week for 3 weeks)
   - Possible cause: Rapid development without shared-core adoption
   - RECOMMENDATION: Run auto-fix engine, review new code patterns

### Optimization Opportunities
1. **Merge candidate detected:**
   - smart-home-wizards/lib/device-api (230 lines)
   - ai-in-action-hub/utils/iot-api.js (180 lines)
   - 68% similarity → Merge into shared-core for ~30% code reduction

2. **Adoption opportunity:**
   - lead-magnet-engine used by 2 projects but applicable to 8+
   - Estimated productivity gain: 4 hours/week if universally adopted

### Health Trend
Empire health score: 87/100 (↑ 3 from last week)
Trend: IMPROVING
Projected next week: 89/100 (if recommended actions taken)
```

### 10.2 Recommendations Engine

```json
{
  "recommendations": [
    {
      "priority": "high",
      "type": "maintenance",
      "title": "Sync wealth-from-ai (3 systems outdated)",
      "effort": "30 minutes",
      "impact": "Empire health +4 points",
      "deadline": "2025-03-05 (before exception expires)"
    },
    {
      "priority": "medium",
      "type": "extraction",
      "title": "Extract device-api-wrapper to shared-core",
      "effort": "2 hours",
      "impact": "Eliminates 410 lines of duplicate code, benefits 2+ projects"
    },
    {
      "priority": "medium",
      "type": "adoption",
      "title": "Roll out lead-magnet-engine to 6 more projects",
      "effort": "1 hour per project",
      "impact": "Consistent lead capture across empire, ~4 hrs/week saved"
    },
    {
      "priority": "low",
      "type": "optimization",
      "title": "Upgrade console.log to structured logging (42 remaining instances)",
      "effort": "Auto-fixable (5 minutes)",
      "impact": "Better debugging, compliance score +5 across 4 projects"
    }
  ]
}
```

---

# SYSTEM 11: THE COMMAND CENTER — Live Dashboard

## Purpose

A single-page HTML dashboard that shows the entire mesh state at a glance with one-click operations.

### Dashboard Features

| Section | What It Shows | Actions Available |
|---------|---------------|-------------------|
| **Empire Overview** | Total projects, systems, health score, sync status | One-click full sync |
| **Project Cards** | Per-project health, compliance, versions, sync status | Sync, compile, audit per project |
| **Dependency Graph** | Interactive visualization of all project relationships | Zoom, filter, highlight paths |
| **System Registry** | All shared systems with versions, consumers, health | View details, trigger update |
| **Alert Feed** | Real-time alerts from Sentinel | Acknowledge, snooze, resolve |
| **Knowledge Browser** | Searchable knowledge base with relevance | Add, edit, review entries |
| **Oracle Insights** | Predictions, recommendations, trends | Accept/dismiss recommendations |
| **Sync Log** | Complete sync history with filters | Rollback, retry failed syncs |
| **Compliance Heatmap** | Visual compliance scores across all projects | Run auto-fix, view violations |
| **Evolution Timeline** | How systems have evolved over time | Compare versions, view changelogs |

---

# CROSS-PROJECT SEARCH ENGINE

## Purpose

Find ANYTHING across the entire empire in milliseconds.

### Search Capabilities

```
$ mesh search "retry logic"

CODE RESULTS (4 matches):
  [1] shared-core/utilities/api-retry/src/retry.js (canonical)
      → The official retry implementation
  [2] smart-home-wizards/lib/custom-retry.js (DRIFT WARNING)
      → Local implementation — should migrate to shared-core
  [3] ai-discovery-digest/utils/retry-helper.js (DRIFT WARNING)
      → Local implementation — should migrate to shared-core

KNOWLEDGE RESULTS (2 matches):
  [1] API Quirks: "n8n webhooks need retry logic due to Contabo restart drops"
  [2] Decisions Log: "Chose exponential backoff with jitter as retry strategy"

DOCUMENTATION RESULTS (3 matches):
  [1] shared-core/utilities/api-retry/README.md
  [2] master-context/global-rules.md (section: "All API calls include retry logic")
  [3] deprecated/BLACKLIST.md (section: "Single-retry on webhook failure")

CONFIG RESULTS (1 match):
  [1] shared-core/configs/base.n8n.json (retry settings)
```

### Search Modes

| Mode | Command | Searches |
|------|---------|----------|
| **All** | `mesh search "query"` | Everything |
| **Code** | `mesh search --code "query"` | Source files only |
| **Knowledge** | `mesh search --kb "query"` | Knowledge base only |
| **Config** | `mesh search --config "query"` | Config files only |
| **Manifest** | `mesh search --manifest "query"` | Manifests/registry |
| **Deprecated** | `mesh search --deprecated "query"` | Deprecated items only |

---

# CLAUDE CODE AUTO-HOOKS

## Purpose

Automatically run mesh operations when you open/close/work in projects.

### On Project Open

```python
# hooks/on-project-open.py
# Triggered when you open a Claude Code project

1. CHECK CLAUDE.md FRESHNESS
   → Compare compiled CLAUDE.md hash with source hashes
   → If stale: auto-recompile and notify
   → "CLAUDE.md was 3 days stale. Recompiled with latest context."

2. CHECK SYNC STATUS
   → Are all consumed systems at latest version?
   → If outdated: show summary and offer quick-sync
   → "2 systems have updates available. Sync now? [Y/n]"

3. CHECK COMPLIANCE
   → Quick compliance scan of recently changed files
   → If violations: show summary
   → "3 deprecated pattern violations detected. Run auto-fix? [Y/n]"

4. LOAD RELEVANT KNOWLEDGE
   → Pull latest knowledge entries relevant to this project
   → Inject into session context

5. CHECK ALERTS
   → Any active alerts for this project?
   → Display if present
```

### On Project Close

```python
# hooks/on-project-close.py
# Triggered when you close/exit a Claude Code project

1. EXTRACT DISCOVERIES
   → Scan conversation/session for new knowledge
   → Auto-create discovery entries in .project-mesh/discoveries/pending/

2. UPDATE METRICS
   → Record session duration, files touched, systems used
   → Update project manifest last_human_touch

3. FLAG EXTRACTION CANDIDATES
   → If new code was written that matches shared patterns
   → Flag for Forge review

4. SYNC BACK
   → If project modified any shared system configs
   → Push changes to hub for review
```

### On File Save

```python
# hooks/on-file-save.py
# Triggered on every file save (lightweight, fast)

1. PATTERN CHECK (< 100ms)
   → Quick scan of saved file for deprecated patterns
   → If found: inline warning (non-blocking)

2. DRIFT CHECK (< 200ms)
   → Compare saved file against shared-core equivalents
   → If drift detected: log for Forge review (non-blocking)
```

---

# TESTING MESH

## Purpose

Ensure syncs don't break things and shared systems work across all consumers.

### Test Levels

```
Level 1: UNIT TESTS (per system)
  → Test individual system functions
  → Run: On system change
  → Speed: Fast (< 5 seconds per system)

Level 2: SMOKE TESTS (per project)
  → Test that synced systems work in project context
  → Run: After every sync
  → Speed: Medium (< 30 seconds per project)

Level 3: INTEGRATION TESTS (cross-system)
  → Test that systems work together
  → Run: On major version changes
  → Speed: Slow (< 2 minutes)

Level 4: CONFIG VALIDATION (per project)
  → Validate project configs against system schemas
  → Run: After every sync
  → Speed: Fast (< 2 seconds per project)

Level 5: FULL REGRESSION (empire-wide)
  → Run ALL tests across ALL projects
  → Run: Weekly scheduled + before major releases
  → Speed: Several minutes
```

### Config Validation

```python
# Validates that a project's override config matches the system's schema

VALIDATING: smart-home-wizards/overrides/image-optimization.config.json

Against schema: shared-core/systems/image-optimization/config.schema.json

  ✅ quality: 85 (valid: integer, range 1-100)
  ✅ format: "webp" (valid: enum ["jpeg", "png", "webp", "avif"])
  ✅ max_width: 1200 (valid: integer, min 100)
  ❌ deprecated_field: "resize_mode" (removed in v3.0 — use "scaling_strategy")
  ⚠️  missing optional: "avif_quality" (default will be used: 80)

RESULT: 1 error, 1 warning. Fix deprecated_field before next sync.
```

---

# IMPLEMENTATION ROADMAP v2.0

### Phase 1: Core Foundation (Days 1-3)
1. Create _empire-hub with full v2.0 directory structure
2. Write enhanced manifests for top 5 projects
3. Build CLAUDE.md compiler v2.0 (with conditionals, validation, diffing)
4. Build sync engine v2.0 (with rollback, dependency resolution)
5. Set up deprecated blacklist with pattern detection
6. Create global-rules, category contexts, conditional blocks

### Phase 2: Shared Core Migration (Week 1)
1. Identify and extract top 10 shared systems
2. Create meta.json, DEPENDENCIES.json, tests for each
3. Build config inheritance chains (base → category → project)
4. Run first full sync across all projects
5. Generate initial compliance scores

### Phase 3: Intelligence Layer (Week 2)
1. Deploy Forge auto-extraction scanner
2. Set up Sentinel monitoring with alert rules
3. Build cross-project search index
4. Populate knowledge base from existing conversations
5. Create Claude Code auto-hooks

### Phase 4: Automation & Integration (Week 3)
1. Build n8n mesh-sync-pipeline workflow
2. Install git hooks across all projects
3. Deploy scheduled health check workflow
4. Set up knowledge auto-capture workflow
5. Build notification pipeline (Discord/email)

### Phase 5: Analytics & Prediction (Week 4)
1. Build Oracle predictive analysis engine
2. Deploy Command Center dashboard
3. Set up evolution tracking for all systems
4. Generate first weekly Oracle forecast
5. Review and refine all systems based on first month of data

### Phase 6: Continuous Evolution (Ongoing)
1. Weekly Oracle reviews → take recommended actions
2. Monthly Forge scans → extract new shared systems
3. Quarterly architecture reviews → evolve the mesh itself
4. Continuous knowledge capture and curation
5. Adapt alert thresholds based on operational data
