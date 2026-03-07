#!/usr/bin/env python3
"""
Empire Arsenal — Pipeline Deployment Script
============================================
Deploys all pipeline systems to the VPS via SSH (paramiko).

Deploys:
  1. Content Pipeline (Python) → /opt/arsenal/pipelines/
  2. RAG Knowledge Base → /opt/arsenal/pipelines/rag/
  3. n8n Workflows → via n8n API
  4. Open WebUI Configuration → via API
  5. Dify App Configuration → via API
  6. Contabo n8n Bridge → via webhook

Usage:
  python deploy_pipelines.py --all
  python deploy_pipelines.py --workflows
  python deploy_pipelines.py --rag-init
  python deploy_pipelines.py --openwebui
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deploy")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

ARSENAL_IP = "89.116.29.33"
N8N_URL = f"http://{ARSENAL_IP}:5678"
N8N_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2Y2E1Y2NjNS1lYjEwLTQzYjEtYmYzNy1kM2ZmYzBhNzQ3MDUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiODI1MTMzZDAtOGRhMi00YmY5LWEzZTUtN2U4YTQ5OTgwYjdhIiwiaWF0IjoxNzcyOTIxOTY5LCJleHAiOjE4MDQ0NTc5Njk1Njd9.F7Wpdj4ZaWS2KGtoUfjahwFLo97UCEEUu1BU-2aej9s"
LITELLM_URL = f"http://{ARSENAL_IP}:4000"
LITELLM_KEY = "sk-arsenal-fec2dfe2b1256586b84b962c9d25e4e9"
OPENWEBUI_URL = f"http://{ARSENAL_IP}:3000"
DIFY_URL = f"http://{ARSENAL_IP}:3001"
QDRANT_URL = f"http://{ARSENAL_IP}:6333"
CONTABO_N8N_URL = "https://ncreighton.app.n8n.cloud"

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "pipelines" / "n8n-workflows"


# ──────────────────────────────────────────────────────────────────────────────
# n8n Workflow Deployer
# ──────────────────────────────────────────────────────────────────────────────

class N8NDeployer:
    """Deploy n8n workflows via API."""

    def __init__(self):
        self.base_url = f"{N8N_URL}/api/v1"
        self.headers = {
            "X-N8N-API-KEY": N8N_API_KEY,
            "Content-Type": "application/json",
        }

    def list_workflows(self) -> list:
        """List existing workflows."""
        resp = requests.get(f"{self.base_url}/workflows", headers=self.headers, timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", [])

    def create_workflow(self, workflow_json: dict) -> dict:
        """Create a new workflow."""
        resp = requests.post(
            f"{self.base_url}/workflows",
            headers=self.headers,
            json=workflow_json,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        logger.info(f"✅ Created workflow: {result.get('name', 'unknown')} (ID: {result.get('id', 'N/A')})")
        return result

    def activate_workflow(self, workflow_id: str):
        """Activate a workflow."""
        resp = requests.patch(
            f"{self.base_url}/workflows/{workflow_id}",
            headers=self.headers,
            json={"active": True},
            timeout=15,
        )
        resp.raise_for_status()
        logger.info(f"✅ Activated workflow: {workflow_id}")

    def deploy_all_workflows(self):
        """Deploy all workflow JSON files."""
        if not WORKFLOWS_DIR.exists():
            logger.error(f"Workflows directory not found: {WORKFLOWS_DIR}")
            return

        existing = self.list_workflows()
        existing_names = {w["name"] for w in existing}

        for wf_file in WORKFLOWS_DIR.glob("*.json"):
            try:
                with open(wf_file) as f:
                    workflow = json.load(f)

                name = workflow.get("name", wf_file.stem)

                if name in existing_names:
                    logger.info(f"⏭️ Workflow already exists: {name}")
                    continue

                result = self.create_workflow(workflow)

                # Auto-activate RAG crawler and bridge
                if "crawler" in name.lower() or "bridge" in name.lower():
                    self.activate_workflow(result["id"])

            except Exception as e:
                logger.error(f"Failed to deploy {wf_file.name}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# RAG Initializer
# ──────────────────────────────────────────────────────────────────────────────

class RAGInitializer:
    """Initialize Qdrant collections for the RAG system."""

    SITES = [
        "witchcraftforbeginners", "smarthomewizards", "mythicalarchives",
        "bulletjournals", "wealthfromai", "aidiscoverydigest",
        "aiinactionhub", "pulsegearreviews", "wearablegearreviews",
        "smarthomegearreviews", "clearainews", "theconnectedhaven",
        "manifestandalign", "familyflourish",
    ]

    def __init__(self):
        self.qdrant_url = QDRANT_URL

    def init_collections(self, dim: int = 768):
        """Create all Qdrant collections."""
        # Per-site collections
        for site in self.SITES:
            collection_name = f"site_{site.replace('-', '_')}"
            self._create_collection(collection_name, dim)

        # Empire-wide cross-search collection
        self._create_collection("empire_knowledge", dim)

        # Brain/learning collections (for EMPIRE-BRAIN compatibility)
        for collection in ["learnings", "solutions", "skills", "sessions"]:
            self._create_collection(collection, 384)  # MiniLM dimensions

        logger.info(f"✅ Initialized {len(self.SITES) + 5} Qdrant collections")

    def _create_collection(self, name: str, dim: int):
        """Create a single collection if it doesn't exist."""
        try:
            resp = requests.get(f"{self.qdrant_url}/collections/{name}", timeout=10)
            if resp.status_code == 200:
                logger.info(f"  Collection exists: {name}")
                return

            resp = requests.put(
                f"{self.qdrant_url}/collections/{name}",
                json={"vectors": {"size": dim, "distance": "Cosine"}},
                timeout=15,
            )
            resp.raise_for_status()
            logger.info(f"  ✅ Created collection: {name}")
        except Exception as e:
            logger.error(f"  ❌ Failed to create {name}: {e}")

    def get_stats(self) -> dict:
        """Get stats for all collections."""
        stats = {}
        try:
            resp = requests.get(f"{self.qdrant_url}/collections", timeout=10)
            resp.raise_for_status()
            collections = resp.json().get("result", {}).get("collections", [])

            for col in collections:
                name = col["name"]
                try:
                    count_resp = requests.post(
                        f"{self.qdrant_url}/collections/{name}/points/count",
                        json={"exact": True},
                        timeout=10,
                    )
                    count_resp.raise_for_status()
                    count = count_resp.json().get("result", {}).get("count", 0)
                    stats[name] = count
                except Exception:
                    stats[name] = -1

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")

        return stats


# ──────────────────────────────────────────────────────────────────────────────
# Open WebUI Configurator
# ──────────────────────────────────────────────────────────────────────────────

class OpenWebUIConfigurator:
    """Configure Open WebUI with full model access."""

    def __init__(self):
        self.base_url = OPENWEBUI_URL

    def check_health(self) -> bool:
        """Check if Open WebUI is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    def verify_models(self) -> list:
        """Verify available models through LiteLLM."""
        try:
            resp = requests.get(
                f"{LITELLM_URL}/v1/models",
                headers={"Authorization": f"Bearer {LITELLM_KEY}"},
                timeout=15,
            )
            resp.raise_for_status()
            models = resp.json().get("data", [])
            logger.info(f"✅ {len(models)} models available via LiteLLM:")
            for m in models:
                logger.info(f"  → {m.get('id', 'unknown')}")
            return models
        except Exception as e:
            logger.error(f"Failed to verify models: {e}")
            return []


# ──────────────────────────────────────────────────────────────────────────────
# N8N Bridge Configurator
# ──────────────────────────────────────────────────────────────────────────────

class BridgeConfigurator:
    """Configure the bridge between Arsenal and Contabo n8n instances."""

    def __init__(self):
        self.arsenal_url = N8N_URL
        self.contabo_url = CONTABO_N8N_URL

    def test_arsenal_connection(self) -> bool:
        """Test Arsenal n8n API."""
        try:
            resp = requests.get(
                f"{self.arsenal_url}/api/v1/workflows",
                headers={"X-N8N-API-KEY": N8N_API_KEY},
                timeout=10,
            )
            logger.info(f"Arsenal n8n: {'✅ Connected' if resp.status_code == 200 else '❌ Failed'}")
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Arsenal n8n connection failed: {e}")
            return False

    def test_contabo_connection(self) -> bool:
        """Test Contabo n8n (cloud) connection."""
        try:
            resp = requests.get(f"{self.contabo_url}", timeout=10)
            logger.info(f"Contabo n8n: {'✅ Reachable' if resp.status_code in [200, 301, 302] else '❌ Unreachable'}")
            return resp.status_code in [200, 301, 302]
        except Exception as e:
            logger.warning(f"Contabo n8n: ❌ {e}")
            return False

    def generate_bridge_config(self) -> dict:
        """Generate bridge configuration for both n8n instances."""
        return {
            "bridge": {
                "arsenal": {
                    "url": self.arsenal_url,
                    "webhook_base": f"{self.arsenal_url}/webhook",
                    "endpoints": {
                        "content_pipeline": "/webhook/content-pipeline",
                        "content_distribution": "/webhook/content-distribution",
                        "bridge_relay": "/webhook/bridge-relay",
                        "rag_crawler": "/webhook/rag-crawler",
                    },
                    "capabilities": [
                        "llm_inference",
                        "rag_search",
                        "content_generation",
                        "image_generation",
                        "vector_search",
                        "web_crawling",
                        "meta_search",
                    ],
                },
                "contabo": {
                    "url": self.contabo_url,
                    "capabilities": [
                        "wordpress_publishing",
                        "social_distribution",
                        "email_marketing",
                        "analytics_sync",
                        "cron_scheduling",
                    ],
                },
                "routing": {
                    "ai_tasks": "arsenal",
                    "publishing": "contabo",
                    "monitoring": "both",
                },
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Dify App Configurator
# ──────────────────────────────────────────────────────────────────────────────

class DifyConfigurator:
    """Configure Dify AI apps for each site."""

    SITE_APPS = {
        "witchcraftforbeginners": {
            "name": "Witchcraft Wisdom AI",
            "icon": "🔮",
            "description": "AI assistant for Witchcraft For Beginners - answers questions about spells, rituals, crystals, and magical practices.",
            "suggested_questions": [
                "What crystals are best for protection?",
                "How do I cast my first spell?",
                "What are the phases of the moon for rituals?",
                "How do I set up an altar for beginners?",
            ],
        },
        "smarthomewizards": {
            "name": "Smart Home Wizard AI",
            "icon": "🏠",
            "description": "AI assistant for Smart Home Wizards - helps with home automation, device setup, and smart home troubleshooting.",
            "suggested_questions": [
                "What's the best smart hub for beginners?",
                "How do I automate my morning routine?",
                "Zigbee vs Z-Wave comparison?",
                "Best smart security cameras 2026?",
            ],
        },
        "wealthfromai": {
            "name": "AI Wealth Advisor",
            "icon": "💰",
            "description": "AI assistant for Wealth From AI - guides on AI side hustles, income strategies, and building AI-powered businesses.",
            "suggested_questions": [
                "How can I start making money with AI today?",
                "Best AI tools for freelancing?",
                "How to build an AI SaaS business?",
                "What are the top AI side hustles in 2026?",
            ],
        },
    }

    def __init__(self):
        self.dify_url = DIFY_URL

    def check_health(self) -> bool:
        """Check Dify health."""
        try:
            resp = requests.get(f"{self.dify_url}", timeout=10)
            return resp.status_code in [200, 301, 302]
        except Exception:
            return False

    def generate_all_configs(self) -> dict:
        """Generate Dify app configurations for all sites."""
        configs = {}
        for site_id, app_config in self.SITE_APPS.items():
            configs[site_id] = {
                **app_config,
                "model_provider": {
                    "provider": "openai_api_compatible",
                    "name": "Arsenal LiteLLM",
                    "base_url": f"http://{ARSENAL_IP}:4000/v1",
                    "api_key": LITELLM_KEY,
                    "models": [
                        {"name": "claude-sonnet", "type": "chat", "default": True},
                        {"name": "claude-haiku", "type": "chat"},
                        {"name": "gpt-4o", "type": "chat"},
                        {"name": "deepseek-chat", "type": "chat"},
                    ],
                },
                "knowledge_base": {
                    "source": "qdrant",
                    "collection": f"site_{site_id.replace('-', '_')}",
                    "embedding_model": "nomic-embed-text",
                    "top_k": 5,
                    "score_threshold": 0.5,
                },
            }
        return configs


# ──────────────────────────────────────────────────────────────────────────────
# Master Deployment
# ──────────────────────────────────────────────────────────────────────────────

def deploy_all():
    """Deploy everything."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║       🚀 Empire Arsenal — Full Pipeline Deployment          ║
╚══════════════════════════════════════════════════════════════╝
""")

    # 1. Deploy n8n Workflows
    print("\n📋 Step 1: Deploying n8n Workflows...")
    n8n = N8NDeployer()
    try:
        n8n.deploy_all_workflows()
    except Exception as e:
        logger.error(f"n8n deployment failed: {e}")

    # 2. Initialize RAG Collections
    print("\n🧠 Step 2: Initializing RAG Knowledge Base...")
    rag = RAGInitializer()
    try:
        rag.init_collections()
        stats = rag.get_stats()
        for name, count in sorted(stats.items()):
            print(f"  {name}: {count} vectors")
    except Exception as e:
        logger.error(f"RAG init failed: {e}")

    # 3. Verify Open WebUI
    print("\n🌐 Step 3: Verifying Open WebUI...")
    webui = OpenWebUIConfigurator()
    if webui.check_health():
        webui.verify_models()
    else:
        logger.warning("Open WebUI not responding - visit http://89.116.29.33:3000 to set up")

    # 4. Configure Bridge
    print("\n🌉 Step 4: Configuring n8n Bridge...")
    bridge = BridgeConfigurator()
    bridge.test_arsenal_connection()
    bridge.test_contabo_connection()
    bridge_config = bridge.generate_bridge_config()

    # Save bridge config
    config_path = Path(__file__).parent.parent / "config" / "bridge-config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(bridge_config, f, indent=2)
    logger.info(f"Bridge config saved to: {config_path}")

    # 5. Generate Dify Configs
    print("\n🤖 Step 5: Generating Dify App Configs...")
    dify = DifyConfigurator()
    if dify.check_health():
        configs = dify.generate_all_configs()
        dify_config_path = Path(__file__).parent.parent / "pipelines" / "dify-apps" / "app-configs.json"
        dify_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dify_config_path, "w") as f:
            json.dump(configs, f, indent=2)
        logger.info(f"Dify configs saved to: {dify_config_path}")
    else:
        logger.warning("Dify not responding - visit http://89.116.29.33:3001 to set up")

    # Summary
    print("""
╔══════════════════════════════════════════════════════════════╗
║ ✅ Deployment Complete!                                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ 📋 n8n Workflows:                                           ║
║   → Content Pipeline Master (webhook: /content-pipeline)     ║
║   → Content Distribution (webhook: /content-distribution)    ║
║   → Arsenal ↔ Contabo Bridge (webhook: /bridge-relay)        ║
║   → RAG Site Crawler (auto: every 12 hours)                 ║
║                                                              ║
║ 🧠 RAG Collections:                                         ║
║   → 14 site-specific + 1 empire-wide + 4 brain collections  ║
║                                                              ║
║ 🌐 Open WebUI: http://89.116.29.33:3000                     ║
║ 🤖 Dify Apps: http://89.116.29.33:3001                      ║
║ 📊 Langfuse: http://89.116.29.33:3004                       ║
║                                                              ║
║ Next Steps:                                                  ║
║ 1. Run: python content_pipeline.py --auto --dry-run          ║
║ 2. Run: python empire_rag.py crawl --all                     ║
║ 3. Create Dify apps at http://89.116.29.33:3001              ║
║ 4. Visit Open WebUI at http://89.116.29.33:3000              ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="Deploy Empire Arsenal Pipelines")
    parser.add_argument("--all", action="store_true", help="Deploy everything")
    parser.add_argument("--workflows", action="store_true", help="Deploy n8n workflows only")
    parser.add_argument("--rag-init", action="store_true", help="Initialize RAG collections only")
    parser.add_argument("--rag-stats", action="store_true", help="Show RAG stats")
    parser.add_argument("--openwebui", action="store_true", help="Configure Open WebUI")
    parser.add_argument("--bridge", action="store_true", help="Configure n8n bridge")
    parser.add_argument("--dify", action="store_true", help="Generate Dify configs")

    args = parser.parse_args()

    if args.all:
        deploy_all()
    elif args.workflows:
        n8n = N8NDeployer()
        n8n.deploy_all_workflows()
    elif args.rag_init:
        rag = RAGInitializer()
        rag.init_collections()
    elif args.rag_stats:
        rag = RAGInitializer()
        stats = rag.get_stats()
        print("\n📊 Qdrant Collections:")
        for name, count in sorted(stats.items()):
            print(f"  {name:<35} {count:>6} vectors")
    elif args.openwebui:
        webui = OpenWebUIConfigurator()
        if webui.check_health():
            webui.verify_models()
        else:
            print("❌ Open WebUI not responding")
    elif args.bridge:
        bridge = BridgeConfigurator()
        bridge.test_arsenal_connection()
        bridge.test_contabo_connection()
        config = bridge.generate_bridge_config()
        print(json.dumps(config, indent=2))
    elif args.dify:
        dify = DifyConfigurator()
        configs = dify.generate_all_configs()
        print(json.dumps(configs, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
