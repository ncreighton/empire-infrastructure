# PROJECT MESH v2.0: OMEGA ARCHITECTURE

## The System That Makes 16 Projects Think, Learn, and Evolve as One Organism

---

## ARCHITECTURE OVERVIEW

Project Mesh v2.0 expands from 6 layers to **11 interconnected systems**. Every original layer has been massively enhanced, and 5 entirely new systems have been added to cover every gap.

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM 11: COMMAND CENTER                    │
│          Live Dashboard, Analytics, One-Click Operations          │
├─────────────────────────────────────────────────────────────────┤
│                     SYSTEM 10: ORACLE                            │
│          Predictive Intelligence & Trend Analysis                │
├─────────────────────────────────────────────────────────────────┤
│                     SYSTEM 9: NEXUS                              │
│          n8n + Git + CI/CD Deep Integration                      │
├─────────────────────────────────────────────────────────────────┤
│                     SYSTEM 8: SENTINEL                           │
│          Real-Time Monitoring, Alerts & Anomaly Detection        │
├─────────────────────────────────────────────────────────────────┤
│                     SYSTEM 7: FORGE                              │
│          Auto-Extraction, System Creation & Evolution Engine      │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 6: GUARDIAN    │  LAYER 5: BRIDGE    │  LAYER 4: PULSE   │
│  Anti-Hallucination   │  Knowledge Sharing   │  Sync Protocol    │
│  + Pattern Detection  │  + Auto-Capture      │  + Rollback       │
│  + Compliance Score   │  + Expiration/Review  │  + File Watching  │
│  + Auto-Fix Engine    │  + Knowledge Graph    │  + Transaction    │
├───────────────────────┴─────────────────────┴───────────────────┤
│                     LAYER 3: BRAIN v2.0                          │
│          Conditional Compiler, Validation, Hot-Reload            │
├─────────────────────────────────────────────────────────────────┤
│                     LAYER 2: NERVE v2.0                          │
│          Auto-Discovery, Impact Analysis, Conflict Detection     │
├─────────────────────────────────────────────────────────────────┤
│                     LAYER 1: SPINE v2.0                          │
│          Shared Core + Internal Dependencies + Testing           │
└─────────────────────────────────────────────────────────────────┘
```

---

## ENHANCED DIRECTORY STRUCTURE

```
C:\Claude Code Projects\
├── _empire-hub\                              ← THE HUB (COMMAND CENTER)
│   ├── CLAUDE.md                              ← Hub instructions
│   ├── mesh.config.json                       ← Master mesh configuration
│   │
│   ├── shared-core\                           ← LAYER 1: The Spine v2.0
│   │   ├── systems\                           ← Complete shared systems
│   │   │   ├── image-optimization\
│   │   │   │   ├── src\                       ← Source code
│   │   │   │   ├── tests\                     ← System-level tests
│   │   │   │   ├── examples\                  ← Usage examples
│   │   │   │   ├── benchmarks\                ← Performance benchmarks
│   │   │   │   ├── VERSION                    ← Semantic version
│   │   │   │   ├── CHANGELOG.md               ← Full change history
│   │   │   │   ├── README.md                  ← Comprehensive docs
│   │   │   │   ├── config.schema.json         ← JSON schema for config
│   │   │   │   ├── config.defaults.json       ← Default configuration
│   │   │   │   ├── DEPENDENCIES.json          ← Internal system deps
│   │   │   │   ├── CONSUMERS.md               ← Auto-generated: who uses this
│   │   │   │   ├── MIGRATION.md               ← Upgrade path from all prev versions
│   │   │   │   └── meta.json                  ← System metadata (below)
│   │   │   ├── content-pipeline\
│   │   │   ├── seo-optimizer\
│   │   │   ├── wp-api-wrapper\
│   │   │   ├── n8n-webhook-handler\
│   │   │   ├── browser-automation\
│   │   │   ├── affiliate-link-manager\
│   │   │   ├── ai-content-generator\
│   │   │   ├── podcast-pipeline\
│   │   │   ├── youtube-pipeline\
│   │   │   ├── etsy-pod-manager\
│   │   │   ├── kdp-publisher\
│   │   │   ├── email-automation\
│   │   │   ├── analytics-tracker\
│   │   │   └── lead-magnet-engine\
│   │   │
│   │   ├── utilities\
│   │   │   ├── error-handling\
│   │   │   ├── structured-logging\
│   │   │   ├── api-retry\
│   │   │   ├── config-loader\
│   │   │   ├── rate-limiter\
│   │   │   ├── cache-manager\
│   │   │   ├── queue-manager\
│   │   │   ├── file-watcher\
│   │   │   ├── health-check-lib\
│   │   │   └── credential-manager\
│   │   │
│   │   ├── patterns\
│   │   │   ├── wordpress-plugin-scaffold\
│   │   │   ├── n8n-workflow-template\
│   │   │   ├── api-integration-pattern\
│   │   │   ├── content-hub-pattern\
│   │   │   ├── affiliate-funnel-pattern\
│   │   │   ├── landing-page-pattern\
│   │   │   └── automation-workflow-pattern\
│   │   │
│   │   ├── configs\                           ← NEW: Shared config templates
│   │   │   ├── base.wordpress.json            ← Base WP config all sites share
│   │   │   ├── base.n8n.json                  ← Base n8n workflow config
│   │   │   ├── base.seo.json                  ← Base SEO settings
│   │   │   └── categories\                    ← Category-level config overrides
│   │   │       ├── witchcraft.overrides.json
│   │   │       ├── ai-sites.overrides.json
│   │   │       └── tech-sites.overrides.json
│   │   │
│   │   ├── DEPENDENCY-TREE.json               ← Internal system dependency tree
│   │   └── VERSION                            ← Global shared-core version
│   │
│   ├── registry\                              ← LAYER 2: The Nerve v2.0
│   │   ├── manifests\
│   │   │   ├── witchcraft-for-beginners.manifest.json
│   │   │   ├── smart-home-wizards.manifest.json
│   │   │   └── [one per project...]
│   │   ├── dependency-graph.json              ← Auto-generated
│   │   ├── reverse-dependency-graph.json      ← NEW: "What depends on X?"
│   │   ├── capability-index.json              ← Searchable capability catalog
│   │   ├── impact-matrix.json                 ← NEW: Change impact predictions
│   │   ├── conflict-map.json                  ← NEW: Detected conflicts
│   │   ├── adoption-tracker.json              ← NEW: System usage analytics
│   │   └── REGISTRY.md                        ← Human-readable summary
│   │
│   ├── master-context\                        ← LAYER 3: The Brain v2.0
│   │   ├── global-rules.md
│   │   ├── brand-voices.md
│   │   ├── api-credentials.md
│   │   ├── tech-stack.md
│   │   ├── coding-standards.md                ← NEW: Empire-wide code standards
│   │   ├── naming-conventions.md              ← NEW: Consistent naming rules
│   │   ├── error-handling-policy.md           ← NEW: How errors are handled
│   │   ├── security-policy.md                 ← NEW: Security standards
│   │   ├── categories\
│   │   │   ├── witchcraft-sites.md
│   │   │   ├── ai-sites.md
│   │   │   ├── tech-sites.md
│   │   │   ├── lifestyle-sites.md
│   │   │   ├── productivity-sites.md
│   │   │   ├── mythology-sites.md
│   │   │   ├── publishing.md
│   │   │   └── agency.md
│   │   ├── conditionals\                      ← NEW: Conditional context blocks
│   │   │   ├── has-substack.md                ← Include if project has Substack
│   │   │   ├── has-etsy.md                    ← Include if project has Etsy
│   │   │   ├── has-youtube.md
│   │   │   ├── has-podcast.md
│   │   │   ├── has-kdp.md
│   │   │   └── has-newsletter.md
│   │   ├── compiler-config.json
│   │   └── validation-rules.json              ← NEW: Rules to validate compiled output
│   │
│   ├── deprecated\                            ← LAYER 6: The Guardian v2.0
│   │   ├── BLACKLIST.md
│   │   ├── patterns\                          ← NEW: Regex patterns for detection
│   │   │   ├── code-patterns.json             ← Code patterns to catch
│   │   │   └── config-patterns.json           ← Config anti-patterns
│   │   ├── migrations\
│   │   │   └── [migration guides...]
│   │   ├── exceptions\                        ← NEW: Temporary exceptions
│   │   │   └── exceptions-registry.json       ← "Project X can use Y until date Z"
│   │   ├── compliance\                        ← NEW: Compliance tracking
│   │   │   ├── scores.json                    ← Per-project compliance scores
│   │   │   └── violations-log.json            ← Historical violations
│   │   ├── auto-fix\                          ← NEW: Automated fix scripts
│   │   │   ├── transforms.json                ← Code transformations
│   │   │   └── fix-engine.py                  ← Auto-fix runner
│   │   └── reasons.md
│   │
│   ├── knowledge-base\                        ← LAYER 5: The Bridge v2.0
│   │   ├── api-quirks.md
│   │   ├── lessons-learned.md
│   │   ├── decisions-log.md
│   │   ├── gotchas.md
│   │   ├── performance-notes.md               ← NEW: Performance discoveries
│   │   ├── cost-optimizations.md              ← NEW: Money-saving discoveries
│   │   ├── search-index.json                  ← NEW: Full-text search index
│   │   ├── knowledge-graph.json               ← NEW: Relationships between entries
│   │   ├── review-schedule.json               ← NEW: When entries need re-verification
│   │   └── auto-captured\                     ← NEW: Auto-captured from conversations
│   │       ├── [timestamped entries...]
│   │       └── pending-review.json            ← Entries awaiting human review
│   │
│   ├── sync\                                  ← LAYER 4: The Pulse v2.0
│   │   ├── claude_md_compiler.py              ← Enhanced compiler
│   │   ├── sync_engine.py                     ← Enhanced sync engine
│   │   ├── rollback\                          ← NEW: Rollback snapshots
│   │   │   └── [timestamped snapshots...]
│   │   ├── sync-log.json                      ← NEW: Complete sync history
│   │   ├── sync-schedule.json                 ← NEW: Automated sync schedule
│   │   └── sync-status.json
│   │
│   ├── forge\                                 ← SYSTEM 7: The Forge
│   │   ├── extractor.py                       ← Auto-extraction engine
│   │   ├── scaffolder.py                      ← System template generator
│   │   ├── evolution-tracker.json             ← System evolution history
│   │   ├── extraction-candidates.json         ← Detected shareable code
│   │   └── templates\                         ← System scaffolding templates
│   │       ├── system-template\
│   │       ├── utility-template\
│   │       └── pattern-template\
│   │
│   ├── sentinel\                              ← SYSTEM 8: The Sentinel
│   │   ├── monitor.py                         ← Monitoring engine
│   │   ├── alerts-config.json                 ← Alert rules
│   │   ├── anomaly-detector.py                ← Anomaly detection
│   │   ├── alerts-log.json                    ← Alert history
│   │   └── thresholds.json                    ← Warning/critical thresholds
│   │
│   ├── nexus\                                 ← SYSTEM 9: The Nexus
│   │   ├── git-hooks\                         ← Git integration
│   │   │   ├── pre-commit.py                  ← Deprecation check
│   │   │   ├── post-commit.py                 ← Auto-sync trigger
│   │   │   ├── pre-push.py                    ← Full validation
│   │   │   └── install-hooks.py               ← Hook installer
│   │   ├── n8n-workflows\                     ← n8n integration
│   │   │   ├── mesh-sync-pipeline.json        ← Full sync workflow
│   │   │   ├── health-check-scheduled.json    ← Scheduled health check
│   │   │   ├── knowledge-capture.json         ← Auto-capture workflow
│   │   │   └── alert-dispatcher.json          ← Alert routing workflow
│   │   └── ci\                                ← CI/CD integration
│   │       ├── validate-mesh.py               ← Full mesh validation
│   │       └── deploy-sync.py                 ← Deploy + sync pipeline
│   │
│   ├── oracle\                                ← SYSTEM 10: The Oracle
│   │   ├── analyzer.py                        ← Predictive analysis engine
│   │   ├── trends.json                        ← Historical trend data
│   │   ├── predictions.json                   ← Current predictions
│   │   ├── recommendations.json               ← Action recommendations
│   │   └── evolution-forecast.md              ← Human-readable forecast
│   │
│   ├── command-center\                        ← SYSTEM 11: Command Center
│   │   ├── dashboard.html                     ← Interactive dashboard
│   │   ├── api-server.py                      ← Local API for dashboard
│   │   └── assets\
│   │
│   ├── search\                                ← Cross-Project Search Engine
│   │   ├── indexer.py                         ← Full-text indexer
│   │   ├── search-index.json                  ← Search index
│   │   └── search.py                          ← Search CLI
│   │
│   ├── testing\                               ← Testing Mesh
│   │   ├── test-runner.py                     ← Cross-project test runner
│   │   ├── smoke-tests\                       ← Per-system smoke tests
│   │   ├── integration-tests\                 ← Cross-system integration tests
│   │   ├── config-validator.py                ← Config schema validation
│   │   └── test-results.json                  ← Latest test results
│   │
│   ├── hooks\                                 ← Claude Code Auto-Hooks
│   │   ├── on-project-open.py                 ← Runs when opening a project
│   │   ├── on-project-close.py                ← Runs when closing
│   │   ├── on-file-save.py                    ← Runs on file save
│   │   └── hooks-config.json                  ← Hook configuration
│   │
│   └── scripts\
│       ├── init-project.py
│       ├── publish-system.py
│       ├── migrate.py
│       ├── full-audit.py                      ← NEW: Complete mesh audit
│       ├── emergency-rollback.py              ← NEW: Emergency rollback
│       └── mesh-doctor.py                     ← NEW: Diagnose mesh issues
```

---

# LAYER 1: THE SPINE v2.0 — Enhanced Shared Core

## What's New

### 1.1 System Metadata (meta.json)

Every shared system now carries rich metadata:

```json
{
  "name": "image-optimization",
  "version": "3.0.0",
  "created": "2024-11-15",
  "last_updated": "2025-02-28",
  "author": "extracted-from:smart-home-wizards",
  "status": "production",
  "stability": "stable",
  
  "metrics": {
    "consumers_count": 12,
    "avg_daily_invocations": 450,
    "error_rate_30d": 0.02,
    "last_breaking_change": "2025-01-10",
    "avg_response_time_ms": 120,
    "lines_of_code": 340,
    "complexity_score": "medium",
    "test_coverage_pct": 78
  },
  
  "health": {
    "score": 92,
    "factors": {
      "documentation": 95,
      "test_coverage": 78,
      "update_freshness": 100,
      "consumer_satisfaction": 95,
      "error_rate": 98
    },
    "last_health_check": "2025-02-28T09:00:00Z"
  },
  
  "compatibility": {
    "min_node_version": "18.0.0",
    "min_python_version": "3.10",
    "required_utilities": ["error-handling", "config-loader", "structured-logging"],
    "optional_utilities": ["cache-manager", "rate-limiter"],
    "wp_min_version": "6.0",
    "incompatible_with": []
  },
  
  "lifecycle": {
    "stage": "mature",
    "next_planned_version": "3.1.0",
    "next_version_features": ["WebP auto-conversion", "AVIF support"],
    "deprecation_date": null,
    "replacement": null,
    "breaking_changes_policy": "semver-strict"
  },
  
  "tags": ["images", "performance", "core-web-vitals", "media"],
  "category": "media-processing"
}
```

### 1.2 Internal Dependency Resolution

Systems can now depend on OTHER shared-core systems and utilities:

```json
// shared-core/systems/content-pipeline/DEPENDENCIES.json
{
  "requires": {
    "systems": [
      {"name": "seo-optimizer", "min_version": "1.5.0"},
      {"name": "image-optimization", "min_version": "3.0.0"},
      {"name": "wp-api-wrapper", "min_version": "1.0.0"}
    ],
    "utilities": [
      {"name": "error-handling", "min_version": "1.0.0"},
      {"name": "structured-logging", "min_version": "1.0.0"},
      {"name": "queue-manager", "min_version": "1.0.0"},
      {"name": "rate-limiter", "min_version": "1.0.0"}
    ]
  },
  "optional": {
    "systems": [
      {"name": "ai-content-generator", "min_version": "1.0.0", "purpose": "AI-enhanced content rewriting"}
    ]
  }
}
```

The sync engine resolves dependencies BEFORE syncing:
```
Content-pipeline v2.0 requested
  → Requires seo-optimizer ≥1.5.0 → ✅ v1.5.0 present
  → Requires image-optimization ≥3.0.0 → ✅ v3.0.0 present
  → Requires wp-api-wrapper ≥1.0.0 → ❌ v0.9.0 present → AUTO-UPGRADE
  → Requires queue-manager utility → ❌ not synced → AUTO-SYNC
  → All dependencies resolved → Proceed with sync
```

### 1.3 Config Inheritance Chains

Three-tier configuration that eliminates duplication while allowing customization:

```
BASE CONFIG (shared-core/configs/base.wordpress.json)
    ↓ inherits + overrides
CATEGORY CONFIG (shared-core/configs/categories/witchcraft.overrides.json)
    ↓ inherits + overrides
PROJECT CONFIG (.project-mesh/overrides/wordpress.config.json)
```

```json
// Base config: applies to ALL sites
{
  "wordpress": {
    "post_status": "draft",
    "comment_status": "open",
    "ping_status": "closed",
    "image_quality": 85,
    "max_image_width": 1200,
    "seo": {
      "auto_meta": true,
      "schema_markup": true,
      "sitemap": true
    },
    "performance": {
      "lazy_load_images": true,
      "minify_css": true,
      "minify_js": true,
      "cache_ttl_hours": 24
    }
  }
}

// Category override: witchcraft sites get specific tweaks
{
  "$extends": "base.wordpress.json",
  "wordpress": {
    "comment_status": "open",
    "post_categories_default": ["witchcraft-basics"],
    "seo": {
      "focus_keywords_strategy": "beginner-intent",
      "schema_types": ["HowTo", "FAQ", "Article"]
    },
    "content": {
      "default_voice": "mystical-warmth",
      "safety_disclaimers": true,
      "cultural_sensitivity_check": true
    }
  }
}

// Project override: WitchcraftForBeginners.com specific
{
  "$extends": "witchcraft.overrides.json",
  "wordpress": {
    "site_url": "https://witchcraftforbeginners.com",
    "amazon_tag": "witchforbegin-20",
    "substack_integration": {
      "url": "https://witchcraftb.substack.com",
      "cross_post": true
    }
  }
}
```

The config loader resolves the full chain at runtime:
```python
final_config = merge(base_config, category_override, project_override)
```

### 1.4 System Testing Framework

Every shared system includes standardized tests:

```
shared-core/systems/image-optimization/
├── tests/
│   ├── unit/
│   │   ├── test_resize.py
│   │   ├── test_compress.py
│   │   └── test_format_convert.py
│   ├── integration/
│   │   ├── test_with_wp_api.py          ← Works with wp-api-wrapper?
│   │   └── test_with_content_pipeline.py ← Works in pipeline context?
│   ├── smoke/
│   │   └── test_basic_operation.py      ← "Does it work at all?"
│   ├── performance/
│   │   └── test_benchmark.py            ← Performance regression check
│   └── config/
│       └── test_schema_validation.py    ← Config format validation
```

After every sync, smoke tests run automatically:
```
Syncing image-optimization v3.0.0 to smart-home-wizards...
  ✅ Smoke test: basic operation passed
  ✅ Config validation: project config matches schema
  ✅ Dependency check: all deps satisfied
  ⚠️  Performance: 15% slower than baseline (acceptable threshold: 20%)
  ✅ Sync complete
```

### 1.5 Auto-Generated Consumer Documentation

Each system automatically generates a CONSUMERS.md showing who uses it and how:

```markdown
# image-optimization — Consumer Report

## Active Consumers (12 projects)

| Project | Version | Config Overrides | Daily Calls | Error Rate |
|---------|---------|-----------------|-------------|------------|
| witchcraft-for-beginners | v3.0.0 | quality=90, webp=true | 85 | 0.01% |
| smart-home-wizards | v3.0.0 | quality=85, avif=true | 120 | 0.02% |
| ai-discovery-digest | v3.0.0 | default | 45 | 0.00% |
| family-flourish | v2.0.0 ⚠️ | quality=80 | 30 | 0.05% |
| ...

## Pending Migrations
- family-flourish: v2.0.0 → v3.0.0 (migration guide available)

## Usage Patterns
- Most common operation: resize + compress (78%)
- Most used format: WebP (65%), JPEG (30%), PNG (5%)
- Average file size reduction: 72%
```

---

# LAYER 2: THE NERVE v2.0 — Enhanced Capability Registry

## What's New

### 2.1 Enhanced Manifest Schema v2

```json
{
  "$schema": "project-mesh-manifest-v2",
  "project": {
    "name": "witchcraft-for-beginners",
    "category": "witchcraft-sites",
    "path": "C:\\Claude Code Projects\\witchcraft-for-beginners",
    "description": "Flagship witchcraft/spirituality publishing site",
    "urls": {
      "production": "https://witchcraftforbeginners.com",
      "staging": null,
      "substack": "https://witchcraftb.substack.com",
      "etsy": "https://etsy.com/shop/...",
      "youtube": null,
      "podcast": null
    },
    "tech_stack": {
      "cms": "wordpress",
      "theme": "blocksy",
      "hosting": "hostinger",
      "cdn": "cloudflare",
      "email": "mailchimp",
      "analytics": "google-analytics"
    },
    "revenue_streams": ["affiliate", "digital-products", "substack", "etsy-pod"],
    "priority": "critical",
    "active_development": true,
    "last_human_touch": "2025-02-28"
  },
  
  "provides": {
    "systems": [
      {
        "name": "substack-engagement-automation",
        "version": "2.1.0",
        "description": "Coven Keeper: Automated Substack engagement system",
        "entryPoint": "systems/substack-automation/",
        "exportable": true,
        "maturity": "production",
        "maturity_score": 8.5,
        "documentation_quality": "high",
        "test_coverage": "medium",
        "adaptation_effort": "low",
        "tags": ["substack", "engagement", "automation", "comments"]
      }
    ],
    "patterns": [
      {
        "name": "spiritual-content-voice",
        "description": "Brand voice pattern for witchcraft/spiritual content",
        "type": "content-pattern",
        "reusable_by": ["any-spiritual-vertical"]
      },
      {
        "name": "moon-phase-scheduling",
        "description": "Content and product scheduling aligned with lunar cycles",
        "type": "scheduling-pattern",
        "reusable_by": ["witchcraft-sites", "astrology-sites"]
      }
    ],
    "discoveries": [
      {
        "finding": "Substack API requires 2-second delay between comment posts",
        "date": "2025-02-15",
        "confidence": "verified",
        "affects": ["substack-integration"],
        "tags": ["substack", "rate-limit"]
      },
      {
        "finding": "Etsy listings convert 3x better when timed to moon phases",
        "date": "2025-02-20",
        "confidence": "measured",
        "data_source": "etsy-analytics-90-days",
        "affects": ["etsy-pod"],
        "tags": ["etsy", "conversion", "moon-phases"]
      }
    ],
    "expertise": [
      "witchcraft-content-creation",
      "substack-growth",
      "etsy-pod-optimization",
      "spiritual-seo"
    ]
  },
  
  "consumes": {
    "shared-core": [
      {
        "system": "image-optimization",
        "version": "3.0.0",
        "overrides": "overrides/image-optimization.config.json",
        "criticality": "high",
        "usage_frequency": "daily"
      },
      {
        "system": "content-pipeline",
        "version": "2.0.0",
        "overrides": "overrides/content-pipeline.config.json",
        "criticality": "critical",
        "usage_frequency": "hourly"
      }
    ],
    "from-projects": [
      {
        "project": "ai-discovery-digest",
        "system": "ai-trend-detector",
        "version": "1.0.0",
        "usage": "Detecting trending spiritual/witchcraft topics",
        "criticality": "medium",
        "usage_frequency": "daily"
      }
    ]
  },
  
  "context": {
    "inherits": ["global-rules", "witchcraft-sites"],
    "conditionals": ["has-substack", "has-etsy"],
    "localContext": "CLAUDE.local.md",
    "context_budget_override": null
  },
  
  "health": {
    "last_sync": "2025-02-28T09:15:00Z",
    "sync_health_pct": 100,
    "compliance_score": 95,
    "test_pass_rate": 98,
    "staleness_days": 0
  },
  
  "automation": {
    "auto_sync": true,
    "auto_compile": true,
    "sync_schedule": "on-change",
    "notification_channel": "discord",
    "auto_test_on_sync": true
  }
}
```

### 2.2 Auto-Discovery Engine

The registry can SCAN a project and auto-detect undeclared dependencies:

```
Scanning: smart-home-wizards

DECLARED in manifest:
  ✅ image-optimization v3.0.0
  ✅ content-pipeline v2.0.0
  ✅ seo-optimizer v1.5.0

DETECTED but NOT declared:
  ⚠️  Found pattern matching shared-core/api-retry in: lib/utils/retry.js
      → RECOMMEND: Add api-retry utility to consumes
  ⚠️  Found pattern matching shared-core/error-handling in: lib/error-handler.js
      → RECOMMEND: Replace local implementation with shared utility
  ⚠️  Found code similar to ai-discovery-digest/trend-detector in: scripts/trend-scan.js
      → RECOMMEND: Add cross-project dependency or extract to shared-core

POTENTIAL EXPORTS (code unique to this project but useful elsewhere):
  💡 smart-home-device-api-wrapper (230 lines, well-documented)
      → Could benefit: ai-in-action-hub, any future IoT projects
  💡 z-wave-protocol-handler (180 lines)
      → Could benefit: any smart home vertical expansion
```

### 2.3 Impact Analysis Matrix

Before making ANY change, the system can predict impact:

```
IMPACT ANALYSIS: Upgrading content-pipeline from v1.0 to v2.0

Direct Impact (projects consuming content-pipeline):
  ├── witchcraft-for-beginners [CRITICAL] — v2.0.0 (already current)
  ├── smart-home-wizards [CRITICAL] — v1.0.0 → needs migration
  ├── ai-discovery-digest [HIGH] — v1.0.0 → needs migration
  ├── ai-in-action-hub [HIGH] — v1.0.0 → needs migration
  ├── family-flourish [MEDIUM] — v1.0.0 → needs migration
  └── mythical-archives [LOW] — v1.0.0 → needs migration

Indirect Impact (projects consuming systems that depend on content-pipeline):
  └── None — content-pipeline has no downstream dependents

Breaking Changes:
  1. publishPost() → queueContent() [FUNCTION RENAME]
     Affected files: ~2-3 per project
     Auto-fixable: YES (find & replace)
  2. Config format change [SCHEMA CHANGE]
     Affected files: 1 per project (config.json)
     Auto-fixable: YES (migration script available)
  3. Direct WP API calls removed [BEHAVIOR CHANGE]
     Affected files: varies (0-5 per project)
     Auto-fixable: PARTIAL (needs review)

Total Estimated Migration:
  - Automated: 80%
  - Manual review: 20%
  - Time per project: ~15 minutes
  - Total empire migration: ~1.5 hours

RECOMMENDATION: Stage migration over 2 days. Start with lowest-criticality projects.
```

### 2.4 Conflict Detection System

Before publishing a new version to shared-core, detect conflicts:

```
CONFLICT SCAN: Publishing updated n8n-webhook-handler v2.5.0

Scanning all consumers for compatibility...

✅ witchcraft-for-beginners — No conflicts
⚠️  smart-home-wizards — POTENTIAL CONFLICT
    └─ Local override sets retry_count=1 (new version minimum=3)
    └─ Resolution: Update override or accept new minimum
✅ ai-discovery-digest — No conflicts
❌ wealth-from-ai — BREAKING CONFLICT
    └─ Uses webhook_url format "http://" (new version requires "https://")
    └─ Resolution: Update webhook URLs to HTTPS (migration script available)
✅ family-flourish — No conflicts

SUMMARY: 1 potential, 1 breaking conflict. Resolve before sync.
```

---

# LAYER 3: THE BRAIN v2.0 — Enhanced CLAUDE.md Compiler

## What's New

### 3.1 Conditional Compilation

Include context blocks ONLY when conditions are met:

```json
// compiler-config.json
{
  "conditionals": {
    "has-substack": {
      "condition": "project.urls.substack != null",
      "include": "conditionals/has-substack.md",
      "priority": "high"
    },
    "has-etsy": {
      "condition": "project.revenue_streams contains 'etsy-pod'",
      "include": "conditionals/has-etsy.md",
      "priority": "medium"
    },
    "has-youtube": {
      "condition": "project.urls.youtube != null",
      "include": "conditionals/has-youtube.md",
      "priority": "medium"
    },
    "high-traffic": {
      "condition": "project.metrics.daily_visits > 10000",
      "include": "conditionals/high-traffic-optimization.md",
      "priority": "high"
    },
    "revenue-critical": {
      "condition": "project.priority == 'critical'",
      "include": "conditionals/revenue-critical-caution.md",
      "priority": "critical"
    }
  }
}
```

A project that has Substack AND Etsy gets those context blocks. A project with neither doesn't get bloated with irrelevant instructions.

### 3.2 Template Variables & Interpolation

The compiler supports dynamic variable insertion:

```markdown
<!-- In global-rules.md -->
This project ({{project.name}}) is in the {{project.category}} vertical.
Production URL: {{project.urls.production}}
Amazon affiliate tag: {{project.amazon_tag}}

## Content Voice
Use the **{{category.voice_name}}** voice profile:
{{category.voice_description}}

## Current Shared Systems
{{#each consumed_systems}}
- **{{name}}** v{{version}} — {{description}}
{{/each}}
```

### 3.3 Compiled Output Validation

After compilation, the validator checks:

```
Validating CLAUDE.md for: witchcraft-for-beginners

STRUCTURE CHECKS:
  ✅ Global rules section present
  ✅ Deprecated methods section present
  ✅ Version table present
  ✅ Self-check rules present
  ✅ Project-specific section present

CONTENT CHECKS:
  ✅ No references to deprecated methods in positive context
  ✅ All consumed systems documented
  ✅ No hardcoded credentials detected
  ✅ No conflicting instructions detected
  ✅ Brand voice correctly referenced

SIZE CHECKS:
  ✅ Total size: 8,200 chars (~2,050 tokens) — within budget
  ✅ No section exceeds its budget allocation
  ✅ Critical sections fully included (not truncated)

FRESHNESS CHECKS:
  ✅ All referenced system versions match current shared-core
  ✅ Deprecated blacklist matches current hub version
  ✅ Knowledge base entries are current (oldest: 3 days)

DIFF FROM PREVIOUS:
  📝 12 lines changed (3 added, 2 removed, 7 modified)
  📝 Changes: Updated image-optimization from v2.9 → v3.0, added new API quirk

RESULT: ✅ VALID — Ready to deploy
```

### 3.4 Diff Preview Before Overwrite

Never blindly overwrite — always show what's changing:

```diff
Compiling CLAUDE.md for: smart-home-wizards
Showing diff from current → new:

--- CLAUDE.md (current)
+++ CLAUDE.md (compiled)
@@ Section: Current System Versions @@
 | image-optimization | v2.0.0 | 2025-02-20 |
-| content-pipeline   | v1.0.0 | 2025-02-15 |
+| content-pipeline   | v2.0.0 | 2025-02-28 |
 | seo-optimizer      | v1.5.0 | 2025-02-22 |

@@ Section: Deprecated Methods @@
+### Content Pipeline v1.0 Methods
+- ❌ NEVER: publishPost() — renamed to queueContent()
+- ❌ NEVER: Direct wp-json API calls from scripts

@@ Section: Cross-Project Knowledge @@
+## [2025-02-28] n8n Webhook Reliability
+- n8n webhooks occasionally drop payloads during server restart

Apply changes? [Y/n/review]:
```

### 3.5 Hot-Reload Detection

A file watcher detects when source files change and triggers recompilation:

```
WATCHING: _empire-hub/master-context/, _empire-hub/deprecated/, _empire-hub/knowledge-base/

[09:15:32] Changed: master-context/global-rules.md
  → Affected: ALL projects
  → Auto-recompiling 16 CLAUDE.md files...
  → ✅ Done (2.3 seconds)

[09:22:45] Changed: deprecated/BLACKLIST.md
  → Affected: ALL projects
  → Auto-recompiling 16 CLAUDE.md files...
  → ✅ Done (2.1 seconds)

[09:45:01] Changed: knowledge-base/api-quirks.md
  → Affected: Projects consuming browser-automation (8 projects)
  → Auto-recompiling 8 CLAUDE.md files...
  → ✅ Done (1.2 seconds)
```

### 3.6 Multi-Format Output

The compiler can output more than just CLAUDE.md:

| Format | Purpose | When Used |
|--------|---------|-----------|
| `CLAUDE.md` | Claude Code project instructions | Primary — always generated |
| `PROJECT-WIKI.md` | Human-readable project documentation | For your reference |
| `CONTEXT-SUMMARY.json` | Machine-readable context for n8n/APIs | For automation workflows |
| `ONBOARDING.md` | New collaborator quick-start guide | If you ever hire help |
| `AUDIT-REPORT.md` | Compliance and health report | Weekly reviews |
