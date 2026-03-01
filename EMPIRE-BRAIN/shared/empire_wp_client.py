"""
Empire WordPress Client — Shared WordPress REST API wrapper.

Replaces 97 duplicate implementations (login_wordpress, create_post, clear_cache,
upload_media, set_featured_image, etc.) across 16+ projects with one canonical client.

Usage:
    from shared.empire_wp_client import EmpireWPClient

    wp = EmpireWPClient.from_site_config("witchcraftforbeginners")
    posts = wp.get_posts(per_page=10)
    media_id = wp.upload_media("/path/to/image.png", alt_text="Moon ritual")
    wp.set_featured_image(post_id=123, media_id=media_id)
    wp.clear_cache()
"""

import json
import logging
from base64 import b64encode
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

SITES_CONFIG_PATH = Path(r"D:\Claude Code Projects\config\sites.json")


class EmpireWPClient:
    """WordPress REST API client with application password auth.

    Consolidates login_wordpress, create_post, update_post, delete_post,
    upload_media, set_featured_image, clear_cache, get_posts, get_categories,
    get_tags, and find_or_create_category into a single class.
    """

    def __init__(self, domain: str, username: str, app_password: str,
                 scheme: str = "https", timeout: int = 30):
        self.domain = domain
        self.base_url = f"{scheme}://{domain}/wp-json/wp/v2"
        self.timeout = timeout
        creds = f"{username}:{app_password}"
        self._auth = b64encode(creds.encode()).decode()

    @classmethod
    def from_site_config(cls, site_id: str, config_path: str = None) -> "EmpireWPClient":
        """Create client from sites.json config."""
        path = Path(config_path) if config_path else SITES_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sites = data.get("sites", data)
        site = sites.get(site_id)
        if not site:
            raise ValueError(f"Site '{site_id}' not found in {path}")

        # Handle both flat and nested credential formats
        if "wordpress" in site:
            wp = site["wordpress"]
            username = wp.get("user", wp.get("username", ""))
            password = wp.get("app_password", "")
        else:
            username = site.get("wp_user", site.get("username", ""))
            password = site.get("wp_app_password", site.get("app_password", ""))

        domain = site.get("domain", "")
        return cls(domain=domain, username=username, app_password=password)

    # --- HTTP helpers ---

    def _headers(self, content_type: str = "application/json") -> dict:
        return {
            "Authorization": f"Basic {self._auth}",
            "Content-Type": content_type,
        }

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Any:
        import requests
        url = f"{self.base_url}/{endpoint}"
        resp = requests.get(url, headers=self._headers(), params=params,
                            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, data: Optional[dict] = None) -> Any:
        import requests
        url = f"{self.base_url}/{endpoint}"
        resp = requests.post(url, headers=self._headers(), json=data,
                             timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, endpoint: str, params: Optional[dict] = None) -> Any:
        import requests
        url = f"{self.base_url}/{endpoint}"
        resp = requests.delete(url, headers=self._headers(), params=params,
                               timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # --- Connection test ---

    def test_connection(self) -> bool:
        """Test if WordPress API is reachable and authenticated."""
        try:
            self._get("users/me")
            return True
        except Exception as e:
            log.error("Connection test failed for %s: %s", self.domain, e)
            return False

    # --- Posts ---

    def get_posts(self, per_page: int = 100, page: int = 1,
                  status: str = "publish", fields: Optional[str] = None) -> list[dict]:
        params = {"per_page": min(per_page, 100), "page": page, "status": status}
        if fields:
            params["_fields"] = fields
        try:
            return self._get("posts", params)
        except Exception as e:
            log.error("Error fetching posts from %s: %s", self.domain, e)
            return []

    def get_all_posts(self, status: str = "publish",
                      fields: Optional[str] = None, max_pages: int = 50) -> list[dict]:
        """Fetch ALL posts with auto-pagination."""
        all_posts = []
        default_fields = "id,title,slug,link,date,modified,categories,tags,featured_media"
        _fields = fields or default_fields
        for page in range(1, max_pages + 1):
            posts = self.get_posts(per_page=100, page=page, status=status, fields=_fields)
            if not posts:
                break
            all_posts.extend(posts)
        return all_posts

    def create_post(self, title: str, content: str, status: str = "draft",
                    categories: Optional[list[int]] = None,
                    tags: Optional[list[int]] = None,
                    featured_media: Optional[int] = None, **kwargs) -> dict:
        data = {"title": title, "content": content, "status": status}
        if categories:
            data["categories"] = categories
        if tags:
            data["tags"] = tags
        if featured_media:
            data["featured_media"] = featured_media
        data.update(kwargs)
        return self._post("posts", data)

    def update_post(self, post_id: int, **fields) -> dict:
        return self._post(f"posts/{post_id}", fields)

    def delete_post(self, post_id: int, force: bool = True) -> dict:
        return self._delete(f"posts/{post_id}", {"force": force})

    # --- Media ---

    def upload_media(self, file_path: str, title: Optional[str] = None,
                     alt_text: Optional[str] = None) -> int:
        """Upload media file, returns media ID."""
        import requests

        filepath = Path(file_path)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content_types = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
        }
        content_type = content_types.get(filepath.suffix.lower(), "application/octet-stream")

        headers = {
            "Authorization": f"Basic {self._auth}",
            "Content-Disposition": f'attachment; filename="{filepath.name}"',
            "Content-Type": content_type,
        }

        url = f"{self.base_url}/media"
        with open(file_path, "rb") as f:
            resp = requests.post(url, headers=headers, data=f, timeout=60)
        resp.raise_for_status()

        media = resp.json()
        media_id = media.get("id", 0)

        if alt_text and media_id:
            try:
                self._post(f"media/{media_id}", {"alt_text": alt_text})
            except Exception:
                pass

        log.info("Uploaded media to %s: %s -> ID %d", self.domain, filepath.name, media_id)
        return media_id

    def set_featured_image(self, post_id: int, media_id: int) -> bool:
        try:
            self.update_post(post_id, featured_media=media_id)
            log.info("Set featured image on %s: post %d -> media %d",
                     self.domain, post_id, media_id)
            return True
        except Exception as e:
            log.error("Failed to set featured image on %s: %s", self.domain, e)
            return False

    # --- Categories & Tags ---

    def get_categories(self, per_page: int = 100) -> dict[int, str]:
        try:
            cats = self._get("categories", {"per_page": per_page})
            return {c["id"]: c["name"] for c in cats}
        except Exception:
            return {}

    def get_tags(self, per_page: int = 100) -> dict[int, str]:
        try:
            tags = self._get("tags", {"per_page": per_page})
            return {t["id"]: t["name"] for t in tags}
        except Exception:
            return {}

    def find_or_create_category(self, name: str) -> int:
        cats = self.get_categories()
        for cat_id, cat_name in cats.items():
            if cat_name.lower() == name.lower():
                return cat_id
        result = self._post("categories", {"name": name})
        return result.get("id", 0)

    # --- Cache ---

    def clear_cache(self) -> bool:
        """Clear LiteSpeed cache via REST API."""
        import requests
        try:
            url = f"https://{self.domain}/wp-json/litespeed/v1/purge_all"
            resp = requests.post(url, headers=self._headers(), timeout=15)
            if resp.status_code == 200:
                log.info("Cache cleared on %s", self.domain)
                return True
            log.warning("Cache clear returned %d on %s", resp.status_code, self.domain)
            return False
        except Exception as e:
            log.warning("Cache clear failed on %s: %s", self.domain, e)
            return False
