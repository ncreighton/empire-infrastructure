"""CodeSmith — template-based code generation.

Provides boilerplate templates for common patterns (FastAPI endpoints,
dataclasses, tests, CLI commands, Docker configs). The executor uses
these templates for algorithmic code gen ($0 cost) before falling back
to LLM-based generation.

All logic is algorithmic — zero LLM cost.
Part of the VibeCoder FORGE intelligence layer.
"""

from __future__ import annotations

from typing import Any


# ─── Template Registry ───────────────────────────────────────────────────────

_TEMPLATES: dict[str, str] = {
    # ─── Python ───
    "python_module": '''"""{description}"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


{body}
''',

    "python_class": '''class {class_name}:
    """{description}"""

    def __init__(self{init_params}):
{init_body}

{methods}
''',

    "python_dataclass": '''from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class {class_name}:
    """{description}"""
{fields}
''',

    "python_function": '''def {func_name}({params}) -> {return_type}:
    """{description}"""
    {body}
''',

    "python_test": '''"""Tests for {module_name}."""

import pytest

from {import_path} import {class_name}


class Test{class_name}:
    """Tests for {class_name}."""

    def setup_method(self):
        self.instance = {class_name}({setup_args})

{test_methods}
''',

    # ─── FastAPI ───
    "fastapi_endpoint": '''@app.{method}("{path}")
async def {func_name}({params}):
    """{description}"""
    {body}
''',

    "fastapi_router": '''"""API routes for {module_name}."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/{prefix}", tags=["{tag}"])


{routes}
''',

    "pydantic_model": '''class {class_name}(BaseModel):
    """{description}"""
{fields}
''',

    # ─── Docker ───
    "dockerfile_python": '''FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE {port}

CMD ["python", "-m", "uvicorn", "{app_module}:app", "--host", "0.0.0.0", "--port", "{port}"]
''',

    "docker_compose_service": '''  {service_name}:
    build: ./{build_path}
    container_name: {container_name}
    restart: unless-stopped
    ports:
      - "{host_port}:{container_port}"
    environment:
{env_vars}
    volumes:
      - ./{data_path}:/app/data
''',

    # ─── Config ───
    "pyproject_toml": '''[project]
name = "{name}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.10"

[tool.pytest.ini_options]
asyncio_mode = "strict"
testpaths = ["tests"]

[tool.ruff]
target-version = "py310"
line-length = 100
''',

    # ─── Git ───
    "gitignore_python": '''__pycache__/
*.pyc
.env
.venv/
venv/
*.egg-info/
dist/
build/
.mypy_cache/
.pytest_cache/
.ruff_cache/
data/*.db
''',

    # ─── Shell ───
    "bash_script": '''#!/usr/bin/env bash
set -euo pipefail

# {description}

{body}
''',

    # ─── Init ───
    "init_py": '''"""{package_name} — {description}."""

{exports}
''',
}


class CodeSmith:
    """Template-based code generation for common patterns.

    Usage::

        smith = CodeSmith()
        code = smith.render("python_class", class_name="MyClass", ...)
        templates = smith.list_templates()
    """

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Render a named template with keyword arguments."""
        template = _TEMPLATES.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        # Fill defaults for optional placeholders
        defaults = {
            "description": "",
            "body": "pass",
            "methods": "",
            "fields": "",
            "params": "",
            "return_type": "None",
            "init_params": "",
            "init_body": "        pass",
            "test_methods": "",
            "setup_args": "",
            "routes": "",
            "env_vars": "",
            "exports": "",
        }
        merged = {**defaults, **kwargs}

        try:
            return template.format(**merged)
        except KeyError as e:
            raise ValueError(f"Missing template parameter: {e}")

    def list_templates(self) -> list[dict[str, str]]:
        """List all available templates with descriptions."""
        result = []
        for name, tmpl in _TEMPLATES.items():
            # Extract first line as description
            first_line = tmpl.strip().split("\n")[0]
            result.append({
                "name": name,
                "preview": first_line[:80],
                "params": self._extract_params(tmpl),
            })
        return result

    def has_template(self, template_name: str) -> bool:
        """Check if a template exists."""
        return template_name in _TEMPLATES

    def get_template(self, template_name: str) -> str | None:
        """Get raw template string."""
        return _TEMPLATES.get(template_name)

    @staticmethod
    def _extract_params(template: str) -> list[str]:
        """Extract {param} placeholders from a template."""
        import re
        params = re.findall(r"\{(\w+)\}", template)
        return sorted(set(params))

    def suggest_template(self, description: str) -> str | None:
        """Suggest a template based on natural language description."""
        desc = description.lower()

        if "test" in desc:
            return "python_test"
        if "endpoint" in desc or "route" in desc or "api" in desc:
            return "fastapi_endpoint"
        if "class" in desc:
            return "python_class"
        if "dataclass" in desc:
            return "python_dataclass"
        if "function" in desc or "def " in desc:
            return "python_function"
        if "docker" in desc and "compose" in desc:
            return "docker_compose_service"
        if "dockerfile" in desc:
            return "dockerfile_python"
        if "module" in desc or "file" in desc:
            return "python_module"
        if "script" in desc or "bash" in desc or "shell" in desc:
            return "bash_script"
        return None
