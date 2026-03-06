"""
WP Deployer — Enhanced WordPress client for deploying CSS, PHP, pages,
and WPCode snippets to any of the 14 empire sites.

Uses WP REST API + WPCode REST API.
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)

# Lazy import to avoid circular
_sites_cache = None


def _load_sites() -> Dict:
    global _sites_cache
    if _sites_cache is not None:
        return _sites_cache
    from pathlib import Path
    config_path = Path(r"D:\Claude Code Projects\config\sites.json")
    if config_path.exists():
        data = json.loads(config_path.read_text("utf-8"))
        _sites_cache = data.get("sites", data)
    else:
        _sites_cache = {}
    return _sites_cache


def _get_wp_client(site_slug: str):
    """Validate site credentials exist. Returns site config dict.

    Note: We use _wp_request() for all REST calls instead of WordPressClient
    to avoid import issues with hyphenated shared-core directory paths.
    """
    sites = _load_sites()
    site = sites.get(site_slug)
    if not site:
        raise ValueError(f"Unknown site: {site_slug}")

    domain = site.get("domain", "")
    wp = site.get("wordpress", {})
    user = wp.get("user", site.get("wp_user", ""))
    password = wp.get("app_password", site.get("wp_app_password", ""))

    if not all([domain, user, password]):
        raise ValueError(f"Missing WP credentials for {site_slug}")

    return site


def _wp_request(site_slug: str, method: str, endpoint: str,
                data: Optional[Dict] = None, timeout: int = 30,
                retries: int = 1) -> Any:
    """Generic authenticated WP REST request with retry and safe parsing."""
    import requests
    from base64 import b64encode

    sites = _load_sites()
    site = sites.get(site_slug)
    if not site:
        raise ValueError(f"Unknown site: {site_slug}")

    domain = site.get("domain", "")
    wp = site.get("wordpress", {})
    user = wp.get("user", site.get("wp_user", ""))
    password = wp.get("app_password", site.get("wp_app_password", ""))
    creds = b64encode(f"{user}:{password}".encode()).decode()

    url = f"https://{domain}/wp-json/{endpoint}"
    headers = {
        "Authorization": f"Basic {creds}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(1 + retries):
        try:
            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=timeout)
            elif method == "POST":
                resp = requests.post(url, headers=headers, json=data, timeout=timeout)
            elif method == "PUT":
                resp = requests.put(url, headers=headers, json=data, timeout=timeout)
            elif method == "DELETE":
                resp = requests.delete(url, headers=headers, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            resp.raise_for_status()

            if not resp.content:
                return {}
            try:
                return resp.json()
            except json.JSONDecodeError:
                log.warning("Non-JSON response from %s %s (status %d)", method, url, resp.status_code)
                return {"_raw": resp.text[:500], "_status": resp.status_code}

        except requests.Timeout as e:
            last_error = e
            if attempt < retries:
                log.warning("Timeout on %s %s, retrying... (%d/%d)", method, url, attempt + 1, retries)
                continue
        except requests.ConnectionError as e:
            last_error = e
            if attempt < retries:
                log.warning("Connection error on %s %s, retrying...", method, url)
                import time
                time.sleep(2)
                continue
        except requests.HTTPError as e:
            # Retry on 502/503/504 (transient server errors)
            if resp.status_code in (502, 503, 504) and attempt < retries:
                log.warning("HTTP %d on %s %s, retrying...", resp.status_code, method, url)
                import time
                time.sleep(2)
                continue
            raise
        except requests.RequestException:
            raise

    # All retries exhausted
    raise last_error


class WPDeployer:
    """Deploy CSS, snippets, pages to WordPress sites via REST API."""

    def get_site_client(self, site_slug: str):
        """Get a WordPressClient for a site."""
        return _get_wp_client(site_slug)

    def list_sites(self) -> List[str]:
        """List all available site slugs."""
        return list(_load_sites().keys())

    def test_connection(self, site_slug: str) -> bool:
        """Test if we can reach a site's WP API."""
        try:
            result = _wp_request(site_slug, "GET", "wp/v2/types")
            return bool(result)
        except Exception as e:
            log.error("Connection test failed for %s: %s", site_slug, e)
            return False

    # -- Snippet Management (WPCode + Code Snippets auto-detect) --

    _snippet_api_cache: Dict = {}  # site_slug -> "wpcode" | "code-snippets" | None

    def _detect_snippet_api(self, site_slug: str) -> Optional[str]:
        """Auto-detect which snippet plugin REST API is available."""
        if site_slug in self._snippet_api_cache:
            return self._snippet_api_cache[site_slug]

        # Try WPCode first
        try:
            _wp_request(site_slug, "GET", "wpcode/v1/snippets", retries=0)
            self._snippet_api_cache[site_slug] = "wpcode"
            return "wpcode"
        except Exception:
            pass

        # Try Code Snippets
        try:
            _wp_request(site_slug, "GET", "code-snippets/v1/snippets", retries=0)
            self._snippet_api_cache[site_slug] = "code-snippets"
            return "code-snippets"
        except Exception:
            pass

        self._snippet_api_cache[site_slug] = None
        return None

    def get_existing_snippets(self, site_slug: str) -> List[Dict]:
        """List snippets on a site (auto-detects WPCode or Code Snippets)."""
        api = self._detect_snippet_api(site_slug)
        if not api:
            return []
        try:
            endpoint = "wpcode/v1/snippets" if api == "wpcode" else "code-snippets/v1/snippets"
            return _wp_request(site_slug, "GET", endpoint)
        except Exception as e:
            log.warning("Could not list snippets on %s: %s", site_slug, e)
            return []

    def deploy_snippet(self, site_slug: str, name: str, code: str,
                       code_type: str = "css", location: str = "site_wide_header",
                       priority: int = 10, auto_activate: bool = True) -> Dict:
        """Create or update a snippet via WPCode or Code Snippets.

        Args:
            code_type: css, php, js, html, text
            location: site_wide_header, site_wide_footer, site_wide_body,
                      frontend_only, admin_only, everywhere
        """
        from systems.site_evolution import codex

        api = self._detect_snippet_api(site_slug)
        if not api:
            if code_type == "css":
                return self._deploy_css_fallback(site_slug, name, code)
            log.warning("No snippet API available on %s, cannot deploy %s snippet '%s'",
                        site_slug, code_type, name)
            return {"error": "No snippet plugin API available"}

        # Check for existing snippet with same name
        existing = self.get_existing_snippets(site_slug)
        existing_id = None
        previous_hash = ""
        for s in existing:
            if s.get("title") == name or s.get("name") == name:
                existing_id = s.get("id")
                previous_hash = hashlib.sha256(
                    s.get("code", "").encode()
                ).hexdigest()[:16]
                break

        content_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        try:
            if api == "wpcode":
                result = self._deploy_wpcode(site_slug, name, code, code_type,
                                              location, priority, auto_activate,
                                              existing_id)
            else:
                result = self._deploy_code_snippets(site_slug, name, code, code_type,
                                                     location, priority, auto_activate,
                                                     existing_id)

            # Record deployment
            codex.record_deployment(
                site_slug=site_slug,
                component_type=name.split("-")[0] if "-" in name else "snippet",
                deployment_type="snippet",
                snippet_name=name,
                content_hash=content_hash,
                previous_hash=previous_hash,
                details=json.dumps({"code_type": code_type, "location": location,
                                     "api": api})
            )
            return result

        except Exception as e:
            log.error("Failed to deploy snippet '%s' to %s: %s", name, site_slug, e)
            if code_type == "css":
                return self._deploy_css_fallback(site_slug, name, code)
            raise

    def _deploy_wpcode(self, site_slug, name, code, code_type, location,
                       priority, auto_activate, existing_id):
        """Deploy via WPCode REST API."""
        payload = {
            "title": name,
            "code": code,
            "code_type": code_type,
            "location": location,
            "priority": priority,
            "status": "active" if auto_activate else "inactive",
        }
        if existing_id:
            result = _wp_request(site_slug, "PUT",
                                  f"wpcode/v1/snippets/{existing_id}", payload)
            log.info("Updated WPCode snippet '%s' on %s (id=%s)", name, site_slug, existing_id)
        else:
            result = _wp_request(site_slug, "POST", "wpcode/v1/snippets", payload)
            log.info("Created WPCode snippet '%s' on %s", name, site_slug)
        return result

    def _deploy_code_snippets(self, site_slug, name, code, code_type, location,
                               priority, auto_activate, existing_id):
        """Deploy via Code Snippets REST API."""
        # Map code_type/location to Code Snippets scope
        scope = "global"  # Default: run everywhere
        if location == "frontend_only":
            scope = "front-end"
        elif location == "admin_only":
            scope = "admin"
        elif code_type == "css":
            scope = "front-end"
        elif code_type == "html":
            scope = "front-end"

        payload = {
            "name": name,
            "code": code,
            "scope": scope,
            "priority": priority,
            "active": auto_activate,
            "tags": ["evolution", code_type],
        }

        if existing_id:
            result = _wp_request(site_slug, "PUT",
                                  f"code-snippets/v1/snippets/{existing_id}", payload)
            log.info("Updated Code Snippet '%s' on %s (id=%s)", name, site_slug, existing_id)
        else:
            result = _wp_request(site_slug, "POST", "code-snippets/v1/snippets", payload)
            log.info("Created Code Snippet '%s' on %s", name, site_slug)

        # Activate if needed
        if auto_activate and result.get("id") and not result.get("active"):
            try:
                _wp_request(site_slug, "POST",
                            f"code-snippets/v1/snippets/{result['id']}/activate")
            except Exception:
                pass

        return result

    def _deploy_css_fallback(self, site_slug: str, name: str, css: str) -> Dict:
        """Fallback: inject CSS via WP Customizer additional CSS."""
        try:
            result = _wp_request(
                site_slug, "POST", "wp/v2/settings",
                {"custom_css": css}
            )
            log.info("Deployed CSS via settings API fallback on %s", site_slug)
            return result
        except Exception as e:
            log.error("CSS fallback also failed on %s: %s", site_slug, e)
            return {"error": str(e)}

    # -- Custom CSS --

    def deploy_custom_css(self, site_slug: str, css_code: str,
                          snippet_name: str = None) -> Dict:
        """Deploy custom CSS to a site (via WPCode or settings API)."""
        name = snippet_name or f"{site_slug}-evolution-css-v1"
        return self.deploy_snippet(
            site_slug, name, css_code,
            code_type="css", location="site_wide_header"
        )

    # -- Pages --

    def deploy_page(self, site_slug: str, title: str, content: str,
                    slug: str = "", template: str = "",
                    status: str = "publish") -> Dict:
        """Create or update a WordPress page."""
        from systems.site_evolution import codex

        # Check if page exists by slug
        page_slug = slug or title.lower().replace(" ", "-").replace("'", "")
        try:
            existing = _wp_request(
                site_slug, "GET",
                f"wp/v2/pages?slug={page_slug}&status=any"
            )
        except Exception:
            existing = []

        page_data = {
            "title": title,
            "content": content,
            "slug": page_slug,
            "status": status,
        }
        if template:
            page_data["template"] = template

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        if existing and len(existing) > 0:
            page_id = existing[0]["id"]
            result = _wp_request(
                site_slug, "POST", f"wp/v2/pages/{page_id}", page_data
            )
            log.info("Updated page '%s' on %s (id=%s)", title, site_slug, page_id)
        else:
            result = _wp_request(site_slug, "POST", "wp/v2/pages", page_data)
            log.info("Created page '%s' on %s", title, site_slug)

        codex.record_deployment(
            site_slug=site_slug,
            component_type="page",
            deployment_type="page",
            snippet_name=page_slug,
            content_hash=content_hash,
            details=json.dumps({"title": title, "slug": page_slug})
        )

        return result

    # -- Schema injection via post meta --

    def update_post_schema(self, site_slug: str, post_id: int,
                           schema_data: Dict) -> Dict:
        """Inject JSON-LD schema into a post via RankMath or custom field."""
        try:
            # Try RankMath REST endpoint first
            result = _wp_request(
                site_slug, "POST",
                f"rankmath/v1/updateMeta",
                {
                    "objectID": post_id,
                    "objectType": "post",
                    "meta": {
                        "rank_math_schema_Article": json.dumps(schema_data)
                    }
                }
            )
            return result
        except Exception:
            # Fallback: update via post meta
            try:
                result = _wp_request(
                    site_slug, "POST",
                    f"wp/v2/posts/{post_id}",
                    {"meta": {"_evolution_schema": json.dumps(schema_data)}}
                )
                return result
            except Exception as e:
                log.error("Schema injection failed on %s post %d: %s",
                          site_slug, post_id, e)
                return {"error": str(e)}

    # -- Rollback --

    def rollback(self, site_slug: str, deployment_id: int) -> bool:
        """Rollback a deployment by deactivating or reverting the snippet."""
        from systems.site_evolution import codex

        deployments = codex.get_deployments(site_slug, limit=100)
        target = None
        for d in deployments:
            if d["id"] == deployment_id:
                target = d
                break

        if not target:
            log.error("Deployment %d not found for %s", deployment_id, site_slug)
            return False

        snippet_name = target.get("snippet_name", "")
        if not snippet_name:
            log.warning("No snippet name for deployment %d", deployment_id)
            codex.rollback_deployment(deployment_id)
            return True

        # Find and deactivate the snippet
        existing = self.get_existing_snippets(site_slug)
        for s in existing:
            if s.get("title") == snippet_name or s.get("name") == snippet_name:
                try:
                    _wp_request(
                        site_slug, "PUT",
                        f"wpcode/v1/snippets/{s['id']}",
                        {"status": "inactive"}
                    )
                    log.info("Deactivated snippet '%s' on %s", snippet_name, site_slug)
                except Exception as e:
                    log.error("Rollback failed for snippet '%s': %s",
                              snippet_name, e)
                break

        codex.rollback_deployment(deployment_id)
        return True
