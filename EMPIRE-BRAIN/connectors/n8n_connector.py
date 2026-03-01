"""n8n Connector — Interface to n8n automation server

Capabilities:
- List workflows
- Trigger workflow executions
- Check execution status
- Push data to webhooks
- Manage workflow state (activate/deactivate)
"""
import json
from datetime import datetime
from typing import Optional

import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import N8N_BASE_URL, N8N_API_KEY


class N8NConnector:
    """Interface to n8n REST API."""

    def __init__(self, base_url: str = "", api_key: str = ""):
        self.base_url = (base_url or N8N_BASE_URL).rstrip("/")
        self.api_key = api_key or N8N_API_KEY
        self.headers = {
            "X-N8N-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

    def list_workflows(self, active_only: bool = False) -> list[dict]:
        """List all workflows."""
        try:
            resp = httpx.get(f"{self.base_url}/api/v1/workflows", headers=self.headers, timeout=15.0)
            resp.raise_for_status()
            workflows = resp.json().get("data", [])
            if active_only:
                workflows = [w for w in workflows if w.get("active")]
            return workflows
        except Exception as e:
            return [{"error": str(e)}]

    def get_workflow(self, workflow_id: str) -> dict:
        """Get workflow details."""
        try:
            resp = httpx.get(f"{self.base_url}/api/v1/workflows/{workflow_id}", headers=self.headers, timeout=15.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def activate_workflow(self, workflow_id: str) -> dict:
        """Activate a workflow."""
        try:
            resp = httpx.patch(
                f"{self.base_url}/api/v1/workflows/{workflow_id}",
                headers=self.headers,
                json={"active": True},
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def deactivate_workflow(self, workflow_id: str) -> dict:
        """Deactivate a workflow."""
        try:
            resp = httpx.patch(
                f"{self.base_url}/api/v1/workflows/{workflow_id}",
                headers=self.headers,
                json={"active": False},
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def create_workflow(self, workflow_json: dict) -> dict:
        """Import/create a workflow."""
        try:
            resp = httpx.post(
                f"{self.base_url}/api/v1/workflows",
                headers=self.headers,
                json=workflow_json,
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def trigger_webhook(self, path: str, data: dict) -> dict:
        """Send data to an n8n webhook."""
        url = f"{self.base_url}/webhook/{path}"
        try:
            resp = httpx.post(url, json=data, timeout=30.0)
            return {
                "status_code": resp.status_code,
                "response": resp.text[:500],
                "url": url,
            }
        except Exception as e:
            return {"error": str(e), "url": url}

    def get_executions(self, workflow_id: Optional[str] = None, limit: int = 20) -> list[dict]:
        """Get recent workflow executions."""
        params = {"limit": limit}
        if workflow_id:
            params["workflowId"] = workflow_id
        try:
            resp = httpx.get(
                f"{self.base_url}/api/v1/executions",
                headers=self.headers,
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            return [{"error": str(e)}]

    # --- Brain-specific webhook helpers ---

    def push_projects(self, projects: list[dict]) -> dict:
        """Push project data to brain webhook."""
        return self.trigger_webhook("brain/projects", {
            "projects": projects,
            "timestamp": datetime.now().isoformat(),
            "source": "empire-brain",
        })

    def push_skills(self, skills: list[dict]) -> dict:
        """Push skill data to brain webhook."""
        return self.trigger_webhook("brain/skills", {
            "skills": skills,
            "timestamp": datetime.now().isoformat(),
            "source": "empire-brain",
        })

    def push_patterns(self, patterns: list[dict]) -> dict:
        """Push pattern data to brain webhook."""
        return self.trigger_webhook("brain/patterns", {
            "patterns": patterns,
            "timestamp": datetime.now().isoformat(),
            "source": "empire-brain",
        })

    def push_learnings(self, learnings: list[dict]) -> dict:
        """Push learnings to brain webhook."""
        return self.trigger_webhook("brain/learnings", {
            "learnings": learnings,
            "timestamp": datetime.now().isoformat(),
            "source": "empire-brain",
        })

    def query_brain(self, query: str) -> dict:
        """Query the brain via webhook."""
        return self.trigger_webhook("brain/query", {
            "query": query,
            "timestamp": datetime.now().isoformat(),
        })
