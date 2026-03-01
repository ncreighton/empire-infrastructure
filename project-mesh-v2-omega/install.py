#!/usr/bin/env python3
"""
PROJECT MESH v3.0 ULTIMATE   INSTALLER
========================================
Run from the project-mesh-v2-omega directory.
Bootstraps directory structure, manifests, and shared systems.

Usage:
  cd "D:\\Claude Code Projects\\project-mesh-v2-omega"
  python install.py

That's it. Everything else is automatic.
"""

import json, os, shutil, sys
from pathlib import Path
from datetime import datetime

# ============================================================================
# YOUR 16 SITES   Edit these if you add/remove sites
# ============================================================================

SITES = {
    # =========================================================================
    # WORDPRESS SITES (16 total)
    # =========================================================================

    # WITCHCRAFT VERTICAL
    "witchcraft-for-beginners": {
        "name": "Witchcraft for Beginners",
        "category": "witchcraft-sites",
        "priority": "critical",
        "url": "https://witchcraftforbeginners.com",
        "substack": "https://witchcraftb.substack.com/",
        "etsy": True,
        "revenue": ["affiliate", "digital-products", "substack", "etsy-pod"],
        "description": "Flagship witchcraft & spirituality site"
    },
    "moon-ritual-library": {
        "name": "Moon Ritual Library",
        "category": "witchcraft-sites",
        "priority": "normal",
        "url": "https://moonrituallibrary.com",
        "revenue": ["affiliate", "digital-products"],
        "description": "Moon phases, rituals, and lunar magic"
    },
    "manifest-and-align": {
        "name": "Manifest and Align",
        "category": "witchcraft-sites",
        "priority": "normal",
        "url": "https://manifestandalign.com",
        "revenue": ["affiliate", "digital-products"],
        "description": "Manifestation, law of attraction, and alignment"
    },

    # SMART HOME / TECH
    "smart-home-wizards": {
        "name": "Smart Home Wizards",
        "category": "tech-sites",
        "priority": "high",
        "url": "https://smarthomewizards.com",
        "revenue": ["affiliate", "digital-products"],
        "description": "Smart home technology guides and reviews"
    },

    # AI SITES
    "ai-in-action-hub": {
        "name": "AI in Action Hub",
        "category": "ai-sites",
        "priority": "high",
        "url": "https://aiinactionhub.com",
        "revenue": ["affiliate", "digital-products"],
        "description": "AI tools, tutorials, and practical applications"
    },
    "ai-discovery-digest": {
        "name": "AI Discovery Digest",
        "category": "ai-sites",
        "priority": "high",
        "url": "https://aidiscoverydigest.com",
        "revenue": ["affiliate", "digital-products"],
        "description": "AI news, discoveries, and trend analysis"
    },
    "wealth-from-ai": {
        "name": "Wealth From AI",
        "category": "ai-sites",
        "priority": "high",
        "url": "https://wealthfromai.com",
        "revenue": ["affiliate", "digital-products"],
        "description": "AI-powered wealth building and investment insights"
    },
    "clear-ai-news": {
        "name": "Clear AI News",
        "category": "ai-sites",
        "priority": "normal",
        "url": "https://clearainews.com",
        "revenue": ["affiliate"],
        "description": "Clear, concise AI news and analysis"
    },

    # FAMILY
    "family-flourish": {
        "name": "Family Flourish",
        "category": "family-sites",
        "priority": "high",
        "url": "https://family-flourish.com",
        "revenue": ["affiliate", "digital-products"],
        "amazon_tag": "familyflourish-20",
        "description": "Family wellness, parenting, and lifestyle"
    },
    "sprout-and-spruce": {
        "name": "Sprout and Spruce",
        "category": "family-sites",
        "priority": "normal",
        "url": "https://sproutandspruce.com",
        "revenue": ["affiliate"],
        "description": "Gardening, homesteading, and sustainable living"
    },
    "celebration-season": {
        "name": "Celebration Season",
        "category": "family-sites",
        "priority": "normal",
        "url": "https://celebrationseason.com",
        "revenue": ["affiliate"],
        "description": "Holiday planning, parties, and celebrations"
    },
    "the-connected-haven": {
        "name": "The Connected Haven",
        "category": "family-sites",
        "priority": "normal",
        "url": "https://theconnectedhaven.com",
        "revenue": ["affiliate"],
        "description": "Family connection, relationships, and togetherness"
    },

    # CONTENT SITES
    "mythical-archives": {
        "name": "Mythical Archives",
        "category": "content-sites",
        "priority": "normal",
        "url": "https://mythicalarchives.com",
        "revenue": ["affiliate", "digital-products"],
        "description": "Mythology, legends, and ancient wisdom"
    },
    "bullet-journals": {
        "name": "Bullet Journals",
        "category": "content-sites",
        "priority": "normal",
        "url": "https://bulletjournals.net",
        "revenue": ["affiliate", "digital-products", "etsy-pod"],
        "etsy": True,
        "description": "Bullet journaling guides, templates, and supplies"
    },

    # REVIEW SITES
    "pulse-gear-reviews": {
        "name": "Pulse Gear Reviews",
        "category": "review-sites",
        "priority": "normal",
        "url": "https://pulsegearreviews.com",
        "revenue": ["affiliate"],
        "description": "Fitness and health gear reviews"
    },
    "wearable-gear-reviews": {
        "name": "Wearable Gear Reviews",
        "category": "review-sites",
        "priority": "normal",
        "url": "https://wearablegearreviews.com",
        "revenue": ["affiliate"],
        "description": "Wearable tech and smartwatch reviews"
    },
    "smart-home-gear-reviews": {
        "name": "Smart Home Gear Reviews",
        "category": "review-sites",
        "priority": "normal",
        "url": "https://smarthomegearreviews.com",
        "revenue": ["affiliate"],
        "description": "Smart home product reviews and comparisons"
    },

    # =========================================================================
    # INTELLIGENCE SYSTEMS (non-WordPress, code projects)
    # =========================================================================

    "grimoire-intelligence": {
        "name": "Grimoire Intelligence",
        "category": "intelligence-systems",
        "priority": "high",
        "project_type": "api-service",
        "port": 8080,
        "path": "grimoire-intelligence",
        "revenue": [],
        "description": "Witchcraft practice companion   12 FastAPI endpoints, FORGE+AMPLIFY"
    },
    "videoforge-engine": {
        "name": "VideoForge Engine",
        "category": "intelligence-systems",
        "priority": "high",
        "project_type": "api-service",
        "port": 8090,
        "path": "videoforge-engine",
        "revenue": [],
        "description": "Video creation pipeline   12-step FORGE+AMPLIFY, Creatomate render"
    },
    "velvetveil-printables": {
        "name": "VelvetVeil Printables",
        "category": "intelligence-systems",
        "priority": "normal",
        "project_type": "code-project",
        "path": "VelvetVeilPrintables-ClaudeCode",
        "revenue": ["digital-products"],
        "description": "PDF/eBook creation   FORGE+AMPLIFY for printable products"
    },
    "3d-print-forge": {
        "name": "3D Print Forge",
        "category": "intelligence-systems",
        "priority": "normal",
        "project_type": "code-project",
        "path": "3d-print-forge",
        "revenue": ["digital-products"],
        "description": "3D model generation and STL processing pipeline"
    },
    "forgefiles-pipeline": {
        "name": "ForgeFiles Pipeline",
        "category": "intelligence-systems",
        "priority": "normal",
        "project_type": "code-project",
        "path": "forgefiles-pipeline",
        "revenue": [],
        "description": "3D brand generation, rendering, and composition pipeline"
    },

    # =========================================================================
    # DASHBOARD & INFRASTRUCTURE
    # =========================================================================

    "empire-dashboard": {
        "name": "Empire Dashboard",
        "category": "infrastructure",
        "priority": "critical",
        "project_type": "api-service",
        "port": 8000,
        "path": "empire-dashboard",
        "revenue": [],
        "description": "Central monitoring hub   27 API routers, alerts, health checks"
    },
    "geelark-automation": {
        "name": "GeeLark Automation",
        "category": "infrastructure",
        "priority": "high",
        "project_type": "api-service",
        "port": 8002,
        "path": "geelark-automation",
        "revenue": [],
        "description": "Cloud phone automation, vision service, ADB management"
    },
    "bmc-witchcraft": {
        "name": "BMC Webhook Handler",
        "category": "infrastructure",
        "priority": "normal",
        "project_type": "api-service",
        "port": 8095,
        "path": "bmc-witchcraft",
        "revenue": ["memberships", "tips"],
        "description": "Buy Me a Coffee webhook handler, membership tracking"
    },

    # =========================================================================
    # CONTENT & SEO TOOLS
    # =========================================================================

    "nick-seo-content-engine": {
        "name": "Nick SEO Content Engine",
        "category": "content-tools",
        "priority": "high",
        "project_type": "code-project",
        "path": "nick-seo-content-engine",
        "revenue": [],
        "description": "SEO content generation, keyword research, schema injection"
    },
    "article-audit-system": {
        "name": "Article Audit System",
        "category": "content-tools",
        "priority": "high",
        "project_type": "code-project",
        "path": "article-audit-system",
        "revenue": [],
        "description": "ZIMM visual robot   article quality audits and scoring"
    },
    "zimmwriter-project-new": {
        "name": "ZimmWriter Pipeline",
        "category": "content-tools",
        "priority": "high",
        "project_type": "code-project",
        "path": "zimmwriter-project-new",
        "revenue": [],
        "description": "Content pipeline controller   batch campaigns, model routing"
    },
    "openclaw-empire": {
        "name": "OpenClaw Empire",
        "category": "content-tools",
        "priority": "high",
        "project_type": "code-project",
        "path": "openclaw-empire",
        "revenue": [],
        "description": "LinkedIn + Substack automation, ADB-driven engagement"
    },

    # =========================================================================
    # E-COMMERCE
    # =========================================================================

    "etsy-agent-v2": {
        "name": "Etsy Agent v2",
        "category": "ecommerce",
        "priority": "high",
        "project_type": "code-project",
        "path": "etsy-agent-v2",
        "revenue": ["digital-products", "etsy-pod"],
        "description": "Etsy listing automation, SEO optimization, order management"
    },
    "printables-empire": {
        "name": "Printables Empire",
        "category": "ecommerce",
        "priority": "normal",
        "project_type": "code-project",
        "path": "printables-empire",
        "revenue": ["digital-products"],
        "description": "Digital printables creation and distribution"
    },
    "pod-automation-system": {
        "name": "POD Automation System",
        "category": "ecommerce",
        "priority": "normal",
        "project_type": "code-project",
        "path": "pod-automation-system",
        "revenue": ["etsy-pod"],
        "description": "Print-on-demand automation and fulfillment"
    },
    "pinflux-engine": {
        "name": "PinFlux Engine",
        "category": "ecommerce",
        "priority": "normal",
        "project_type": "code-project",
        "path": "pinflux-engine",
        "revenue": [],
        "description": "Pinterest pin generation and scheduling engine"
    },

    # =========================================================================
    # VIDEO
    # =========================================================================

    "revid-forge": {
        "name": "Revid Forge",
        "category": "video",
        "priority": "normal",
        "project_type": "code-project",
        "path": "revid-forge",
        "revenue": [],
        "description": "Revid.ai integration for faceless video generation"
    },

    # =========================================================================
    # EMAIL
    # =========================================================================

    "empire-email-system": {
        "name": "Empire Email System",
        "category": "email",
        "priority": "normal",
        "project_type": "code-project",
        "path": "empire-email-system",
        "revenue": [],
        "description": "Multi-site email campaigns and newsletter management"
    },
}

# Shared systems that every WordPress site should consume
DEFAULT_WORDPRESS_SYSTEMS = [
    {"system": "content-pipeline", "version": "1.0.0", "criticality": "critical", "usage_frequency": "daily"},
    {"system": "image-optimization", "version": "1.0.0", "criticality": "high", "usage_frequency": "daily"},
    {"system": "seo-toolkit", "version": "1.0.0", "criticality": "critical", "usage_frequency": "daily"},
    {"system": "api-retry", "version": "1.0.0", "criticality": "high", "usage_frequency": "hourly"},
    {"system": "wordpress-automation", "version": "1.0.0", "criticality": "high", "usage_frequency": "daily"},
    {"system": "affiliate-link-manager", "version": "1.0.0", "criticality": "high", "usage_frequency": "weekly"},
]


# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

DIRS = [
    # Original shared systems
    "shared-core/systems/content-pipeline/src",
    "shared-core/systems/content-pipeline/tests",
    "shared-core/systems/image-optimization/src",
    "shared-core/systems/image-optimization/tests",
    "shared-core/systems/seo-toolkit/src",
    "shared-core/systems/seo-toolkit/tests",
    "shared-core/systems/api-retry/src",
    "shared-core/systems/api-retry/tests",
    "shared-core/systems/wordpress-automation/src",
    "shared-core/systems/wordpress-automation/tests",
    "shared-core/systems/affiliate-link-manager/src",
    "shared-core/systems/affiliate-link-manager/tests",
    # New shared systems (v3.0)
    "shared-core/systems/forge-amplify-pipeline/src",
    "shared-core/systems/forge-amplify-pipeline/tests",
    "shared-core/systems/elevenlabs-tts/src",
    "shared-core/systems/elevenlabs-tts/tests",
    "shared-core/systems/fal-image-gen/src",
    "shared-core/systems/fal-image-gen/tests",
    "shared-core/systems/creatomate-render/src",
    "shared-core/systems/creatomate-render/tests",
    "shared-core/systems/openrouter-llm/src",
    "shared-core/systems/openrouter-llm/tests",
    "shared-core/systems/fastapi-service/src",
    "shared-core/systems/fastapi-service/tests",
    "shared-core/systems/sqlite-codex/src",
    "shared-core/systems/sqlite-codex/tests",
    "shared-core/systems/brand-config/src",
    "shared-core/systems/brand-config/tests",
    "shared-core/configs",
    # Registry
    "registry/manifests",
    # Context
    "master-context/categories",
    "master-context/conditionals",
    # Deprecated
    "deprecated/patterns",
    "deprecated/exceptions",
    "deprecated/migrations",
    "deprecated/compliance",
    # Knowledge (v3.0   graph-powered)
    "knowledge-base/auto-captured",
    "knowledge-base/pending-review",
    "knowledge",
    # Sync
    "sync/rollback",
    # Core (v3.0)
    "core",
    # Scripts
    "scripts",
    "search",
    # Testing
    "testing/smoke-tests",
    "testing/integration-tests",
    # Nexus
    "nexus/git-hooks",
    "nexus/n8n-workflows",
    "nexus/ci",
    # Legacy dirs
    "forge",
    "sentinel",
    "oracle",
    "command-center",
    # v3.0 additions
    "hooks",
    "events",
    "config",
    "dashboard",
    "dashboard/static",
]


# ============================================================================
# INSTALLER
# ============================================================================

def main():
    # Hub IS this directory (project-mesh-v2-omega)
    hub = Path(__file__).parent.resolve()
    projects_root = hub.parent  # D:\Claude Code Projects

    print(f"\n Installing Project Mesh v3.0 ULTIMATE")
    print(f"   Hub: {hub}")
    print(f"   Projects root: {projects_root}\n")
    
    # 1. Create directory structure
    print("[1/6] Creating directory structure...")
    for d in DIRS:
        (hub / d).mkdir(parents=True, exist_ok=True)
    print(f"  [OK] {len(DIRS)} directories created")
    
    # 2. Verify core scripts exist (no copy needed   we ARE the hub)
    print("[2/6] Verifying core scripts...")
    core_scripts = [
        "sync/claude_md_compiler_v2.py", "sync/sync_engine_v2.py",
        "scripts/forge.py", "scripts/sentinel.py", "scripts/oracle.py",
        "scripts/knowledge_harvester.py", "scripts/project_bootstrapper.py",
        "search/search.py", "testing/test_runner.py", "mesh_cli.py",
        "quick_compile.py", "mesh_daemon.py",
        "knowledge/graph_engine.py", "knowledge/code_scanner.py",
        "knowledge/search_engine.py", "knowledge/dna_profiler.py",
        "core/event_bus.py", "core/service_monitor.py",
        "dashboard/api.py",
    ]
    found = sum(1 for s in core_scripts if (hub / s).exists())
    missing = [s for s in core_scripts if not (hub / s).exists()]
    if missing:
        for m in missing:
            print(f"  [WARN] Missing: {m}")
    print(f"  [OK] {found}/{len(core_scripts)} scripts present")
    
    # 3. Verify hub paths are correct
    print("[3/6] Verifying hub paths...")
    hub_path_str = str(hub)
    fixed = 0
    for py_file in hub.rglob("*.py"):
        try:
            content = py_file.read_text("utf-8")
            if r"D:\Claude Code Projects\project-mesh-v2-omega" in content:
                content = content.replace(r"D:\Claude Code Projects\project-mesh-v2-omega", hub_path_str)
                py_file.write_text(content, "utf-8")
                fixed += 1
        except:
            pass
    print(f"  Hub path: {hub}" + (f" (fixed {fixed} files)" if fixed else " (all correct)"))
    
    # 4. Create initial shared systems
    print("[4/6] Scaffolding shared systems...")
    systems = [
        "content-pipeline", "image-optimization", "seo-toolkit",
        "api-retry", "wordpress-automation", "affiliate-link-manager",
        # v3.0 new systems
        "forge-amplify-pipeline", "elevenlabs-tts", "fal-image-gen",
        "creatomate-render", "openrouter-llm", "fastapi-service",
        "sqlite-codex", "brand-config",
    ]
    
    for sys_name in systems:
        sys_dir = hub / "shared-core" / "systems" / sys_name
        sys_dir.mkdir(parents=True, exist_ok=True)
        
        # VERSION
        (sys_dir / "VERSION").write_text("1.0.0", "utf-8")
        
        # meta.json
        meta = {
            "name": sys_name,
            "description": f"Shared {sys_name.replace('-', ' ')} system",
            "version": "1.0.0",
            "status": "production",
            "owner": "Nick Creighton",
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "maturity_score": 7.0,
            "documentation_quality": "medium",
            "test_coverage": "low",
            "tags": []
        }
        (sys_dir / "meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
        
        # DEPENDENCIES.json
        deps = {"requires": {"systems": [], "external": []}, "optional": []}
        (sys_dir / "DEPENDENCIES.json").write_text(json.dumps(deps, indent=2), "utf-8")
        
        # CHANGELOG.md
        (sys_dir / "CHANGELOG.md").write_text(
            f"# {sys_name} Changelog\n\n## 1.0.0 ({datetime.now().strftime('%Y-%m-%d')})\n- Initial shared system creation\n",
            "utf-8"
        )
        
        # README.md
        (sys_dir / "README.md").write_text(
            f"# {sys_name}\n\nShared system for the empire.\n\n## Usage\n\nConsumed via Project Mesh manifest.\n",
            "utf-8"
        )
    
    print(f"  [OK] {len(systems)} systems scaffolded")
    
    # 5. Generate manifests for all sites/projects
    print("[5/6] Generating project manifests...")
    for slug, site in SITES.items():
        project_type = site.get("project_type", "wordpress")
        is_wp = project_type == "wordpress" or site.get("url", "").startswith("https://")

        conditionals = []
        if is_wp and not site.get("project_type"):
            conditionals.append("is-wordpress")
        if site.get("substack"):
            conditionals.append("has-substack")
        if site.get("etsy"):
            conditionals.append("has-etsy")
        if site.get("priority") == "critical":
            conditionals.append("is-revenue-critical")
        if site.get("port"):
            conditionals.append("is-api-service")

        # Tech stack varies by project type
        if is_wp and not site.get("project_type"):
            tech_stack = {
                "cms": "WordPress",
                "theme": "Blocksy",
                "hosting": "Hostinger",
                "cdn": "LiteSpeed",
                "analytics": "RankMath"
            }
            consumed = DEFAULT_WORDPRESS_SYSTEMS.copy()
        else:
            tech_stack = {"language": "Python", "framework": "FastAPI" if site.get("port") else "CLI"}
            consumed = [
                {"system": "api-retry", "version": "1.0.0", "criticality": "high", "usage_frequency": "hourly"},
            ]

        manifest = {
            "schema_version": "3.0.0",
            "project": {
                "name": site["name"],
                "slug": slug,
                "category": site["category"],
                "description": site.get("description", ""),
                "priority": site.get("priority", "normal"),
                "project_type": project_type,
                "active_development": True,
                "last_human_touch": datetime.now().strftime("%Y-%m-%d"),
                "urls": {
                    "production": site.get("url", ""),
                    "substack": site.get("substack"),
                    "etsy": "etsy-shop-url" if site.get("etsy") else None,
                },
                "tech_stack": tech_stack,
                "port": site.get("port"),
                "path": site.get("path", slug),
                "revenue_streams": site.get("revenue", [])
            },
            "provides": {
                "systems": [],
                "discoveries": []
            },
            "consumes": {
                "shared-core": consumed,
                "from-projects": []
            },
            "context": {
                "conditionals": conditionals,
                "context_budget_override": None
            },
            "health": {
                "sync_health_pct": 100,
                "compliance_score": 100,
                "staleness_days": 0
            },
            "automation": {
                "auto_sync": True,
                "sync_schedule": "on-change",
                "auto_compile": True
            }
        }

        mf_path = hub / "registry" / "manifests" / f"{slug}.manifest.json"
        mf_path.write_text(json.dumps(manifest, indent=2), "utf-8")

    print(f"  {len(SITES)} manifests generated")
    
    # 6. Create template files
    print("[6/6] Creating context files...")
    create_context_files(hub)
    create_deprecated_files(hub)
    create_n8n_templates(hub)
    
    # 7. Create mesh.bat launcher
    bat_content = f'@echo off\npython "{hub / "mesh_cli.py"}" %*\n'
    (hub / "mesh.bat").write_text(bat_content, "utf-8")
    
    # Also create at parent level for easy access
    (hub.parent / "mesh.bat").write_text(bat_content, "utf-8")
    
    print(f"\n{'='*60}")
    print(f"  PROJECT MESH v3.0 ULTIMATE INSTALLED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"\n  Hub location: {hub}")
    print(f"  Sites registered: {len(SITES)}")
    print(f"  Shared systems: {len(systems)}")
    print(f"\n  QUICK START:")
    print(f"  -------------")
    print(f'  cd "{hub}"')
    print(f"  python mesh_cli.py check       # Health dashboard")
    print(f"  python mesh_cli.py compile --all  # Build all CLAUDE.md")
    print(f"  python mesh_cli.py forecast     # Oracle predictions")
    print(f"\n  START LIVE SYNC (keeps everything synced all day):")
    print(f"  ---------------------------------------------------")
    print(f"  python mesh_daemon.py --background   # Start daemon")
    print(f"  python mesh_daemon.py --status        # Check it's running")
    print(f"\n  v3.0 NEW COMMANDS:")
    print(f"  ---------------------------------------------------")
    print(f"  python -m knowledge.code_scanner --scan-all  # Index empire")
    print(f"  python -m knowledge.search_engine \"retry\"     # Search graph")
    print(f"  python -m core.service_monitor --check        # Service health")
    print(f"  python -m uvicorn dashboard.api:app --port 8100  # Dashboard")


def create_context_files(hub):
    """Create all master-context template files."""
    
    # Global rules
    (hub / "master-context" / "global-rules.md").write_text("""# EMPIRE GLOBAL RULES
> These rules apply to EVERY project. No exceptions.

## Core Principles
1. **Never hardcode API keys, webhook URLs, or secrets**   Use environment variables
2. **Always use shared-core systems when available**   Check the registry first
3. **All API calls must use retry logic**   Use the api-retry shared system
4. **Images must be optimized before upload**   Use image-optimization system
5. **Content must pass SEO validation**   Use seo-toolkit system
6. **Never reference deprecated methods**   Check BLACKLIST.md below

## Technical Standards
- All WordPress sites use Blocksy or Astra themes on Hostinger
- All automation runs through n8n (ZimmWriter is DEPRECATED)
- All content generation uses Claude API (never GPT)
- All sites use LiteSpeed cache
- All affiliate links use affiliate-link-manager system

## Quality Standards
- Content demonstrates E-E-A-T signals
- Target featured snippets where applicable
- Proper schema markup on every page
- Images have alt text with target keywords
- Internal linking follows topical cluster strategy

## Brand Voice
- Each site has its own voice (see category context below)
- Never use generic AI-sounding language
- Never reference being AI-generated
- Content must feel human-written and authentic

## n8n Automation
- All content pipelines run through n8n workflows
- Use Steel.dev for browser automation with 10min keep-alive pings
- BrowserUse as fallback when Steel.dev fails
- All webhooks use environment variables, never hardcoded URLs
""", "utf-8")
    
    # Categories
    categories = {
        "witchcraft-sites.md": """## Witchcraft Vertical Context
- **Voice**: Mystical warmth   knowledgeable yet approachable
- **Tone**: Wise mentor speaking to curious seekers
- **Avoid**: Dismissive of spiritual practices, overly academic language
- **Embrace**: Practical guidance, seasonal awareness, ethical foundations
- **Sub-niches**: cosmic witch, cottage witch, green witch, sea witch
- **Content pillars**: spellwork, herbalism, divination, seasonal celebrations, crystals
- **Substack**: witchcraftb.substack.com (NEVER use witchcraftforbeginners.substack.com)
- **POD**: Witchcraft-themed Etsy products across sub-niches
- **Automation**: Coven Keeper agent for Substack engagement
""",
        "ai-sites.md": """## AI Vertical Context
- **Voice**: Forward analyst   data-driven, insightful, cutting-edge
- **Tone**: Expert briefing an informed professional
- **Avoid**: Hype without substance, fear-mongering about AI
- **Embrace**: Practical applications, real-world impact, actionable intelligence
- **Sites**: AIinActionHub, AIDiscoveryDigest, WealthFromAI
- **Content pillars**: AI tools reviews, automation tutorials, industry analysis, investment insights
""",
        "tech-sites.md": """## Tech Vertical Context
- **Voice**: Tech authority   practical, detailed, trustworthy
- **Tone**: Knowledgeable friend who knows all the gear
- **Sites**: SmartHomeWizards
- **Content pillars**: product reviews, setup guides, comparisons, troubleshooting
""",
        "family-sites.md": """## Family Vertical Context
- **Voice**: Nurturing guide   warm, supportive, practical
- **Tone**: Experienced parent sharing what works
- **Sites**: Family-Flourish
- **Amazon tag**: familyflourish-20
- **Content pillars**: parenting, wellness, activities, product recommendations
""",
        "content-sites.md": """## Content Vertical Context
- **Voice**: Scholarly wonder (mythology) / Creative organizer (journals)
- **Sites**: MythicalArchives, BulletJournals
- **Content pillars**: mythology deep dives, journaling techniques, templates, supplies
""",
        "review-sites.md": """## Review Sites Vertical Context
- **Voice**: Trusted expert   data-backed, hands-on, unbiased
- **Tone**: Experienced user sharing real test results
- **Sites**: PulseGearReviews, WearableGearReviews, SmartHomeGearReviews
- **Content pillars**: product reviews, comparison tables, buyer guides, deal alerts
- **Revenue**: Primarily Amazon affiliate (use correct per-site tags)
- **Standards**: Include pros/cons, real specs, comparison tables, verdict scores
""",
        "intelligence-systems.md": """## Intelligence Systems Context
- **Pattern**: FORGE+AMPLIFY pipeline (scout, enrich, expand, validate)
- **Common stack**: Python, FastAPI, SQLite knowledge codex, OpenRouter LLM
- **Projects**: Grimoire (witchcraft), VideoForge (video), VelvetVeil (printables)
- **Key principle**: Algorithmic intelligence first, LLM only for generation tasks
- **Testing**: Every system must have unit tests for all FORGE modules
""",
        "infrastructure.md": """## Infrastructure Context
- **Services**: Dashboard (8000), Vision (8002), Grimoire (8080), VideoForge (8090), BMC (8095)
- **Startup**: VBS launchers -> PowerShell -> service binary via Task Scheduler
- **Monitoring**: Empire dashboard checks all service health every 30s
- **Deployment**: VPS at 217.216.84.245, Docker compose for remote services
""",
    }
    
    for fname, content in categories.items():
        (hub / "master-context" / "categories" / fname).write_text(content, "utf-8")
    
    # Conditionals
    conditionals = {
        "has-substack.md": """## Substack Integration Rules
- Substack URL: {{project.urls.substack}}
- Rate limit: 10 API calls/minute on free tier
- Always implement exponential backoff for Substack API
- Newsletter content complements (not duplicates) site content
- Cross-promote between site and Substack
- For witchcraft vertical: Use Coven Keeper automation for engagement
""",
        "has-etsy.md": """## Etsy POD Integration
- Etsy shop: {{project.urls.etsy}}
- Product listings need SEO-optimized titles and tags (13 tags max)
- ForgeFiles workflow for 3D printable models (NEVER disclose AI origin)
- Seasonal collections align with site content calendar
- Product photography must be original or properly licensed
""",
        "is-revenue-critical.md": """## Revenue-Critical Project
[WARN] This project directly generates significant revenue. Extra care required:
- Test ALL changes on staging before production
- Never modify affiliate links without verification
- Content changes need SEO impact assessment
- Downtime directly impacts revenue   minimize deploy risks
- Monitor analytics after any major change for 48 hours
- Backup before any theme/plugin updates
""",
        "is-wordpress.md": """## WordPress Standards
- Hostinger hosting with LiteSpeed cache enabled
- Blocksy or Astra theme
- RankMath for SEO (never Yoast)
- AI Engine plugin for Claude integration
- WP-CLI available for bulk operations
- Wordfence or Sucuri for security
- Regular backup schedule via Hostinger
""",
    }
    
    for fname, content in conditionals.items():
        (hub / "master-context" / "conditionals" / fname).write_text(content, "utf-8")
    
    print("  [OK] Context files created")


def create_deprecated_files(hub):
    """Create deprecated blacklist and patterns."""
    
    # BLACKLIST.md
    (hub / "deprecated" / "BLACKLIST.md").write_text("""# DEPRECATED METHODS   NEVER USE THESE

> This file is auto-included in every project's CLAUDE.md.
> Updated: """ + datetime.now().strftime("%Y-%m-%d") + """

## Content Generation
### [FAIL] NEVER use ZimmWriter or ZimmWriter API
- **Replacement**: n8n content pipeline + Claude API
- **Reason**: Deprecated in favor of Claude-native workflows
- **Stage**: REMOVED

### [FAIL] NEVER use GPT/OpenAI for content generation
- **Replacement**: Claude API (Anthropic)
- **Reason**: All content uses Claude for consistency and quality

## API Patterns
### [FAIL] NEVER hardcode webhook URLs
- **Replacement**: Use environment variables or config.get('webhooks.name')
- **Reason**: Security risk and maintenance nightmare

### [FAIL] NEVER make API calls without retry logic
- **Replacement**: Use shared-core/api-retry system
- **Reason**: APIs fail. Always retry with exponential backoff.

### [FAIL] NEVER use fetch() directly for external APIs
- **Replacement**: Use the api-retry wrapper which handles retries, timeouts, and error logging
- **Reason**: Raw fetch has no retry, no timeout, no error handling

## WordPress
### [FAIL] NEVER use Yoast SEO plugin
- **Replacement**: RankMath
- **Reason**: Standardized across all sites on RankMath

### [FAIL] NEVER edit theme files directly
- **Replacement**: Use child theme or Blocksy customizer
- **Reason**: Updates will overwrite direct edits

## Substack
### [FAIL] NEVER use witchcraftforbeginners.substack.com
- **Replacement**: witchcraftb.substack.com
- **Reason**: Correct URL is witchcraftb.substack.com

## Browser Automation
### [FAIL] NEVER use Puppeteer directly
- **Replacement**: Steel.dev with BrowserUse fallback
- **Reason**: Standardized on Steel.dev for session management
- **Note**: Steel.dev sessions expire after 15min idle   implement keep-alive pings
""", "utf-8")
    
    # Code patterns for scanning
    patterns = {
        "version": "1.0.0",
        "patterns": [
            {
                "name": "hardcoded-webhook-url",
                "regex": "https?://[\\\\w.-]+/webhook/[a-f0-9-]+",
                "severity": "high",
                "description": "Hardcoded webhook URLs should use config values",
                "replacement": "Use config.get('webhooks.{name}') or environment variable",
                "auto_fixable": False
            },
            {
                "name": "hardcoded-api-key",
                "regex": "(?:api[_-]?key|apikey|secret|token)\\\\s*[:=]\\\\s*['\"][a-zA-Z0-9_-]{20,}['\"]",
                "severity": "high",
                "description": "API keys must never be hardcoded",
                "replacement": "Use environment variables or secure config",
                "auto_fixable": False
            },
            {
                "name": "zimm-reference",
                "regex": "zimm(?:writer)?[_-]?(?:pipeline|factory|api)",
                "severity": "medium",
                "description": "ZimmWriter is deprecated",
                "replacement": "Use n8n content pipeline or direct Claude API",
                "auto_fixable": False
            },
            {
                "name": "raw-fetch-no-retry",
                "regex": "(?:fetch|axios\\\\.get|axios\\\\.post)\\\\([^)]+\\\\)(?!.*retry)",
                "severity": "medium",
                "description": "API calls should use retry wrapper",
                "replacement": "Use shared-core/api-retry system",
                "auto_fixable": False
            },
            {
                "name": "console-log-debug",
                "regex": "console\\\\.log\\\\(['\"]debug",
                "severity": "low",
                "description": "Debug console.log should be removed",
                "replacement": "Remove or use proper logging",
                "auto_fixable": True
            },
            {
                "name": "wrong-substack-url",
                "regex": "witchcraftforbeginners\\\\.substack",
                "severity": "high",
                "description": "Wrong Substack URL",
                "replacement": "Use witchcraftb.substack.com",
                "auto_fixable": True
            }
        ]
    }
    (hub / "deprecated" / "patterns" / "code-patterns.json").write_text(
        json.dumps(patterns, indent=2), "utf-8"
    )
    
    print("  [OK] Deprecated blacklist + patterns created")


def create_n8n_templates(hub):
    """Create n8n workflow template references."""
    templates = {
        "workflows": {
            "mesh-sync-pipeline": {
                "name": "Mesh Sync Pipeline",
                "trigger": "webhook POST /mesh-sync",
                "flow": "Webhook  Impact Analysis  Sync  Test  Compile  Dashboard  Notify",
                "description": "Auto-triggered when shared-core systems change"
            },
            "daily-health-check": {
                "name": "Daily Health Check",
                "trigger": "cron 0 8 * * *",
                "flow": "Health Check  Sentinel  Oracle  Digest Email",
                "description": "Morning health digest with alerts"
            },
            "knowledge-capture": {
                "name": "Knowledge Auto-Capture",
                "trigger": "webhook POST /mesh-knowledge",
                "flow": "Receive  Deduplicate  Score  Route (auto-add/review/discard)",
                "description": "Captures discoveries from Claude sessions"
            },
            "weekly-forge-scan": {
                "name": "Weekly Forge Scan",
                "trigger": "cron 0 9 * * 1",
                "flow": "Forge Scan  Drift Report  Email Summary",
                "description": "Finds extractable code across projects"
            }
        }
    }
    (hub / "nexus" / "n8n-workflows" / "workflow-templates.json").write_text(
        json.dumps(templates, indent=2), "utf-8"
    )
    print("  [OK] n8n workflow templates created")


if __name__ == "__main__":
    main()
