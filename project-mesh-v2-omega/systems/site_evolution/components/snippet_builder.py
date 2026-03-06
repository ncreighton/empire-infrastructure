"""
Snippet Builder — Convert components into WPCode-ready snippet payloads.
Handles CSS, PHP, JS, and HTML snippet types with proper hooks.
"""

import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)


class SnippetBuilder:
    """Convert generated components into WPCode-ready payloads."""

    def build_css_snippet(self, site_slug: str, css_code: str,
                          name: str = "") -> Dict:
        """Build a WPCode CSS snippet payload."""
        snippet_name = name or f"{site_slug[:4]}-css-v1"
        return {
            "title": snippet_name,
            "code": css_code,
            "code_type": "css",
            "location": "site_wide_header",
            "priority": 10,
            "status": "active",
        }

    def build_php_snippet(self, site_slug: str, php_code: str,
                          name: str = "", hook: str = "wp_head",
                          priority: int = 10) -> Dict:
        """Build a WPCode PHP snippet with WordPress hook."""
        snippet_name = name or f"{site_slug[:4]}-php-v1"
        # Wrap in hook if not already wrapped
        if "add_action" not in php_code and "add_filter" not in php_code:
            func_name = snippet_name.replace("-", "_").replace(" ", "_")
            wrapped = (
                f"<?php\n"
                f"function {func_name}() {{\n"
                f"?>\n{php_code}\n<?php\n"
                f"}}\n"
                f"add_action('{hook}', '{func_name}', {priority});\n"
            )
        else:
            wrapped = php_code

        return {
            "title": snippet_name,
            "code": wrapped,
            "code_type": "php",
            "location": "everywhere",
            "priority": priority,
            "status": "active",
        }

    def build_html_snippet(self, site_slug: str, html_code: str,
                           name: str = "",
                           location: str = "site_wide_footer") -> Dict:
        """Build a WPCode HTML snippet."""
        snippet_name = name or f"{site_slug[:4]}-html-v1"
        return {
            "title": snippet_name,
            "code": html_code,
            "code_type": "html",
            "location": location,
            "priority": 10,
            "status": "active",
        }

    def build_js_snippet(self, site_slug: str, js_code: str,
                         name: str = "") -> Dict:
        """Build a WPCode JS snippet (loaded in footer)."""
        snippet_name = name or f"{site_slug[:4]}-js-v1"
        # Wrap in script tags if not already
        if "<script" not in js_code:
            js_code = f"<script>\n{js_code}\n</script>"
        return {
            "title": snippet_name,
            "code": js_code,
            "code_type": "html",
            "location": "site_wide_footer",
            "priority": 20,
            "status": "active",
        }

    def component_to_snippets(self, site_slug: str, component_type: str,
                              component: Dict) -> list:
        """Convert a component {html, css, js} into a list of WPCode payloads."""
        snippets = []
        prefix = f"{site_slug[:4]}-{component_type}"

        if component.get("css"):
            snippets.append(self.build_css_snippet(
                site_slug, component["css"], f"{prefix}-css-v1"
            ))

        if component.get("html"):
            location = component.get("location", "site_wide_footer")
            snippets.append(self.build_html_snippet(
                site_slug, component["html"], f"{prefix}-html-v1", location
            ))

        if component.get("js"):
            snippets.append(self.build_js_snippet(
                site_slug, component["js"], f"{prefix}-js-v1"
            ))

        if component.get("php"):
            snippets.append(self.build_php_snippet(
                site_slug, component["php"], f"{prefix}-php-v1"
            ))

        return snippets
