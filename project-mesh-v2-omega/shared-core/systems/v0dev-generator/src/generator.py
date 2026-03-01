"""
v0.dev Component Generator — Shared System v1.0.0
==================================================
Generates React/Tailwind components via v0.dev API.
Extracted from 16 identical copies across WordPress sites.

Usage:
    from mesh.shared.v0dev_generator import V0Generator, generate_site_components

    gen = V0Generator()
    result = gen.generate_component("Create a hero section...", "hero")
"""

import os
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

# Configuration
V0_API_KEY = os.getenv("V0_API_KEY", "")
V0_API_BASE = "https://api.v0.dev"

STANDARD_COMPONENTS = [
    "hero",
    "navigation",
    "content-cards",
    "secondary-cards",
    "newsletter",
    "sidebar-widgets",
    "footer",
    "comparison-table",
    "resource-cards",
    "social-proof",
]


class V0Generator:
    """Generates React/Tailwind components via v0.dev API."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or V0_API_KEY
        if not requests:
            raise ImportError("requests package required: pip install requests")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def generate_component(self, prompt: str, component_name: str) -> Dict:
        """Generate a single component from a prompt."""
        print(f"  Generating: {component_name}...")

        try:
            response = self.session.post(
                f"{V0_API_BASE}/chat/completions",
                json={
                    "model": "v0-1.0-md",
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )

            if response.status_code == 200:
                data = response.json()
                code = (data.get("choices", [{}])[0]
                        .get("message", {}).get("content", ""))
                return {"success": True, "code": code, "component": component_name}
            return {
                "success": False,
                "error": f"API error: {response.status_code}",
                "component": component_name,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "component": component_name}

    def extract_react_code(self, response_content: str) -> str:
        """Extract React/TSX code from API response markdown."""
        for lang in ("tsx", "jsx"):
            marker = f"```{lang}"
            if marker in response_content:
                start = response_content.find(marker) + len(marker)
                end = response_content.find("```", start)
                if end != -1:
                    return response_content[start:end].strip()
        if "```" in response_content:
            start = response_content.find("```") + 3
            end = response_content.find("```", start)
            if end != -1:
                return response_content[start:end].strip()
        return response_content


def load_site_prompts(site_name: str, prompts_dir: Path) -> Dict[str, str]:
    """Load component prompts from site starter markdown file."""
    starter_file = prompts_dir / f"{site_name}-starter.md"
    if not starter_file.exists():
        print(f"Error: Starter file not found: {starter_file}")
        return {}

    content = starter_file.read_text("utf-8", errors="ignore")
    prompts = {}
    sections = content.split("### COMPONENT")

    component_map = {
        "hero": ["hero"],
        "navigation": ["navigation", "header"],
        "content-cards": ["article", "content"],
        "newsletter": ["newsletter"],
        "sidebar-widgets": ["sidebar"],
        "footer": ["footer"],
        "comparison-table": ["comparison", "table"],
    }

    for section in sections[1:]:
        lines = section.strip().split("\n")
        header = lines[0].lower() if lines else ""

        if "**v0.dev Prompt:**" in section:
            prompt_start = section.find("```", section.find("**v0.dev Prompt:**"))
            if prompt_start != -1:
                prompt_start += 3
                prompt_end = section.find("```", prompt_start)
                if prompt_end != -1:
                    prompt = section[prompt_start:prompt_end].strip()
                    component_name = "unknown"
                    for name, keywords in component_map.items():
                        if any(kw in header for kw in keywords):
                            component_name = name
                            break
                    prompts[component_name] = prompt

    return prompts


def save_component(code: str, component_name: str, output_dir: Path):
    """Save generated component to .tsx file with header."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{component_name}.tsx"
    header = f"""/**
 * {component_name.replace('-', ' ').title()} Component
 * Generated via v0.dev API
 * Generated: {datetime.now().isoformat()}
 *
 * NEXT STEPS:
 * 1. Review and enhance this component
 * 2. Add animations and micro-interactions
 * 3. Convert to WordPress block pattern
 * 4. Implement in Blocksy/Astra theme
 */

"""
    filepath.write_text(header + code, "utf-8")
    print(f"  Saved: {filepath}")


def generate_site_components(
    site_name: str,
    components: Optional[List[str]] = None,
    prompts_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """Generate all components for a site."""
    print(f"\n{'='*60}")
    print(f"Generating v0.dev components for: {site_name}")
    print(f"{'='*60}\n")

    prompts = load_site_prompts(site_name, prompts_dir or Path("prompts"))
    if not prompts:
        print("No prompts found.")
        return []

    generator = V0Generator()
    components_to_generate = components or list(prompts.keys())
    results = []

    for component in components_to_generate:
        if component in prompts:
            result = generator.generate_component(prompts[component], component)
            results.append(result)
            if result["success"]:
                code = generator.extract_react_code(result["code"])
                save_component(code, component, output_dir or Path("v0-components"))
            else:
                print(f"  Failed: {component} - {result.get('error')}")
        else:
            print(f"  No prompt found for: {component}")

    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"Generation complete: {successful}/{len(results)} components")
    print(f"{'='*60}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate v0.dev components")
    parser.add_argument("--site", required=True, help="Site name")
    parser.add_argument("--component", help="Specific component to generate")
    parser.add_argument("--all", action="store_true", help="Generate all")
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--output-dir", default="v0-components")
    args = parser.parse_args()

    components = None
    if args.component:
        components = [args.component]
    elif args.all:
        components = STANDARD_COMPONENTS

    generate_site_components(
        args.site, components, Path(args.prompts_dir), Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
