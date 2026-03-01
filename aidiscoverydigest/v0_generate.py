#!/usr/bin/env python3
"""
v0.dev API Component Generator

Canonical source: project-mesh-v2-omega/shared-core/systems/v0dev-generator/src/generator.py
Import: from mesh.shared.v0dev_generator import V0Generator, generate_site_components

Generates React/Tailwind components via v0.dev API and saves them
for Claude Code implementation.

Usage:
    python v0_generate.py --site aidiscoverydigest --component hero
    python v0_generate.py --site aidiscoverydigest --all
"""

import os
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime

# Configuration
V0_API_KEY = os.getenv("V0_API_KEY", "v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4")
V0_API_BASE = "https://api.v0.dev"

# Component list for each site
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
    "social-proof"
]


class V0Generator:
    """Generates components via v0.dev API"""
    
    def __init__(self, api_key: str = V0_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def generate_component(self, prompt: str, component_name: str) -> dict:
        """Generate a single component from a prompt"""
        print(f"  Generating: {component_name}...")
        
        try:
            response = self.session.post(
                f"{V0_API_BASE}/chat/completions",
                json={
                    "model": "v0-1.0-md",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                code = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "success": True,
                    "code": code,
                    "component": component_name
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "component": component_name
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "component": component_name
            }
    
    def extract_react_code(self, response_content: str) -> str:
        """Extract React/TSX code from API response"""
        # Look for code blocks
        if "```tsx" in response_content:
            start = response_content.find("```tsx") + 6
            end = response_content.find("```", start)
            return response_content[start:end].strip()
        elif "```jsx" in response_content:
            start = response_content.find("```jsx") + 6
            end = response_content.find("```", start)
            return response_content[start:end].strip()
        elif "```" in response_content:
            start = response_content.find("```") + 3
            end = response_content.find("```", start)
            return response_content[start:end].strip()
        return response_content


def load_site_prompts(site_name: str, prompts_dir: Path) -> dict:
    """Load component prompts from site starter file"""
    starter_file = prompts_dir / f"{site_name}-starter.md"
    
    if not starter_file.exists():
        print(f"Error: Starter file not found: {starter_file}")
        return {}
    
    content = starter_file.read_text()
    prompts = {}
    
    # Parse v0.dev prompts from the starter file
    # Look for sections marked with "**v0.dev Prompt:**"
    sections = content.split("### COMPONENT")
    
    for section in sections[1:]:  # Skip first split (before COMPONENT 1)
        # Extract component number and name
        lines = section.strip().split("\n")
        header = lines[0] if lines else ""
        
        # Find the v0.dev prompt block
        if "**v0.dev Prompt:**" in section:
            prompt_start = section.find("```", section.find("**v0.dev Prompt:**"))
            if prompt_start != -1:
                prompt_start += 3
                prompt_end = section.find("```", prompt_start)
                if prompt_end != -1:
                    prompt = section[prompt_start:prompt_end].strip()
                    
                    # Determine component name from header
                    component_name = "unknown"
                    header_lower = header.lower()
                    if "hero" in header_lower:
                        component_name = "hero"
                    elif "navigation" in header_lower or "header" in header_lower:
                        component_name = "navigation"
                    elif "article" in header_lower or "content" in header_lower:
                        component_name = "content-cards"
                    elif "newsletter" in header_lower:
                        component_name = "newsletter"
                    elif "sidebar" in header_lower:
                        component_name = "sidebar-widgets"
                    elif "footer" in header_lower:
                        component_name = "footer"
                    elif "comparison" in header_lower or "table" in header_lower:
                        component_name = "comparison-table"
                    
                    prompts[component_name] = prompt
    
    return prompts


def save_component(code: str, component_name: str, output_dir: Path):
    """Save generated component to file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{component_name}.tsx"
    filepath = output_dir / filename
    
    # Add header comment
    header = f'''/**
 * {component_name.replace('-', ' ').title()} Component
 * Generated via v0.dev API
 * Generated: {datetime.now().isoformat()}
 * 
 * NEXT STEPS FOR CLAUDE CODE:
 * 1. Review and enhance this component
 * 2. Add animations and micro-interactions
 * 3. Convert to WordPress block pattern
 * 4. Implement in Blocksy/Astra theme
 */

'''
    
    filepath.write_text(header + code)
    print(f"  ✅ Saved: {filepath}")


def generate_site_components(site_name: str, components: list = None, 
                            prompts_dir: Path = None, output_dir: Path = None):
    """Generate all components for a site"""
    print(f"\n{'='*60}")
    print(f"Generating v0.dev components for: {site_name}")
    print(f"{'='*60}\n")
    
    # Load prompts
    prompts = load_site_prompts(site_name, prompts_dir or Path("prompts"))
    
    if not prompts:
        print("No prompts found. Using manual generation mode.")
        return
    
    # Initialize generator
    generator = V0Generator()
    
    # Generate each component
    components_to_generate = components or list(prompts.keys())
    results = []
    
    for component in components_to_generate:
        if component in prompts:
            result = generator.generate_component(prompts[component], component)
            results.append(result)
            
            if result["success"]:
                code = generator.extract_react_code(result["code"])
                save_component(
                    code, 
                    component, 
                    output_dir or Path("v0-components")
                )
            else:
                print(f"  ❌ Failed: {component} - {result.get('error')}")
        else:
            print(f"  ⚠️  No prompt found for: {component}")
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"Generation complete: {successful}/{len(results)} components")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate v0.dev components")
    parser.add_argument("--site", required=True, help="Site name (e.g., aidiscoverydigest)")
    parser.add_argument("--component", help="Specific component to generate")
    parser.add_argument("--all", action="store_true", help="Generate all components")
    parser.add_argument("--prompts-dir", default="prompts", help="Directory containing starter prompts")
    parser.add_argument("--output-dir", default="v0-components", help="Output directory for components")
    
    args = parser.parse_args()
    
    components = None
    if args.component:
        components = [args.component]
    elif args.all:
        components = STANDARD_COMPONENTS
    
    generate_site_components(
        args.site,
        components,
        Path(args.prompts_dir),
        Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
