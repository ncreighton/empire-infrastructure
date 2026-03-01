"""Deploy n8n Workflows — Import brain workflows to n8n server

Tries multiple auth methods. If API fails, outputs curl commands for manual import.

Usage:
    python scripts/deploy_workflows.py
    python scripts/deploy_workflows.py --api-key YOUR_KEY
    python scripts/deploy_workflows.py --manual  # Just output curl commands
"""
import json
import argparse
import sys
from pathlib import Path

import httpx

WORKFLOWS_DIR = Path(__file__).parent.parent / "workflows"
N8N_BASE = "http://217.216.84.245:5678"

WORKFLOW_FILES = [
    "brain-data-receiver.json",
    "brain-pattern-detector.json",
    "brain-opportunity-finder.json",
    "brain-morning-briefing.json",
]


def deploy_via_api(api_key: str) -> list[dict]:
    """Deploy workflows via n8n REST API."""
    results = []
    headers = {"X-N8N-API-KEY": api_key, "Content-Type": "application/json"}

    for wf_file in WORKFLOW_FILES:
        path = WORKFLOWS_DIR / wf_file
        if not path.exists():
            results.append({"file": wf_file, "status": "error", "detail": "File not found"})
            continue

        workflow = json.loads(path.read_text(encoding="utf-8"))

        try:
            resp = httpx.post(
                f"{N8N_BASE}/api/v1/workflows",
                headers=headers,
                json=workflow,
                timeout=30.0,
            )
            if resp.status_code < 400:
                data = resp.json()
                results.append({
                    "file": wf_file,
                    "status": "deployed",
                    "id": data.get("id"),
                    "name": data.get("name"),
                })
                print(f"  [OK] {wf_file} -> ID: {data.get('id')}")
            else:
                results.append({
                    "file": wf_file,
                    "status": "error",
                    "code": resp.status_code,
                    "detail": resp.text[:200],
                })
                print(f"  [FAIL] {wf_file}: HTTP {resp.status_code}")
        except Exception as e:
            results.append({"file": wf_file, "status": "error", "detail": str(e)})
            print(f"  [ERROR] {wf_file}: {e}")

    return results


def output_manual_instructions():
    """Output instructions for manual import via n8n UI."""
    print("\n" + "=" * 60)
    print("  MANUAL IMPORT INSTRUCTIONS")
    print("=" * 60)
    print(f"\nThe n8n API key returned 401. Import workflows manually:\n")
    print(f"1. Open n8n UI: {N8N_BASE}")
    print(f"2. Go to Settings > API > Create new API key")
    print(f"3. Then run: python scripts/deploy_workflows.py --api-key YOUR_NEW_KEY")
    print(f"\nOR import manually via the n8n UI:")
    print(f"4. Click 'Add workflow' > 'Import from file'")
    print(f"5. Import these files from D:\\Claude Code Projects\\EMPIRE-BRAIN\\workflows\\:")
    for wf_file in WORKFLOW_FILES:
        path = WORKFLOWS_DIR / wf_file
        if path.exists():
            workflow = json.loads(path.read_text(encoding="utf-8"))
            print(f"   - {wf_file}  ({workflow.get('name', 'Unknown')})")
    print(f"\n6. After import, configure the PostgreSQL credential in each workflow")
    print(f"   (Use your existing 'Empire Architect Postgres' credential)")
    print(f"7. Activate each workflow")
    print()

    # Also output the webhook URLs they'll need
    print("After activation, these webhook URLs will be available:")
    print(f"  POST {N8N_BASE}/webhook/brain/projects")
    print(f"  POST {N8N_BASE}/webhook/brain/skills")
    print(f"  POST {N8N_BASE}/webhook/brain/patterns")
    print(f"  POST {N8N_BASE}/webhook/brain/learnings")
    print(f"  POST {N8N_BASE}/webhook/brain/query")
    print()


def main():
    parser = argparse.ArgumentParser(description="Deploy n8n Workflows")
    parser.add_argument("--api-key", type=str, help="n8n API key")
    parser.add_argument("--manual", action="store_true", help="Output manual import instructions")
    parser.add_argument("--url", type=str, default=N8N_BASE, help="n8n base URL")
    args = parser.parse_args()

    global N8N_BASE
    N8N_BASE = args.url

    print("=== EMPIRE-BRAIN: Deploy n8n Workflows ===\n")

    if args.manual:
        output_manual_instructions()
        return

    if args.api_key:
        # Try with provided key
        print(f"Deploying to {N8N_BASE}...")
        results = deploy_via_api(args.api_key)
        success = sum(1 for r in results if r["status"] == "deployed")
        print(f"\nDeployed: {success}/{len(results)} workflows")
        if success < len(results):
            output_manual_instructions()
        return

    # Try default key from env
    import os
    api_key = os.environ.get("N8N_API_KEY", "")
    if api_key:
        print(f"Using N8N_API_KEY from environment...")
        results = deploy_via_api(api_key)
        success = sum(1 for r in results if r["status"] == "deployed")
        if success == len(results):
            print(f"\nAll {success} workflows deployed successfully!")
            return

    # Fallback to manual instructions
    output_manual_instructions()


if __name__ == "__main__":
    main()
