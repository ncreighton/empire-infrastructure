"""
wordpress-automation -- WordPress REST API wrapper for post/media CRUD.
Extracted from wordpress_sync.py and article_images_pipeline.py.

Provides:
- WordPressClient: authenticated REST API wrapper
- Methods for posts, media, categories, tags, and featured images

Usage:
    wp = WordPressClient("example.com", "user", "xxxx xxxx xxxx xxxx")
    posts = wp.get_posts(per_page=10)
    media_id = wp.upload_media("/path/to/image.png", title="My Image")
    wp.set_featured_image(post_id=123, media_id=media_id)
"""

import logging
from base64 import b64encode
from pathlib import Path
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)


class WordPressClient:
    """WordPress REST API client with authentication.

    Supports application password authentication for the
    WP REST API v2.
    """

    def __init__(self, domain: str, username: str, app_password: str,
                 scheme: str = "https"):
        self.domain = domain
        self.base_url = f"{scheme}://{domain}/wp-json/wp/v2"
        creds = f"{username}:{app_password}"
        self._auth = b64encode(creds.encode()).decode()

    def _headers(self, content_type: str = "application/json") -> Dict:
        """Build standard request headers."""
        return {
            "Authorization": f"Basic {self._auth}",
            "Content-Type": content_type,
        }

    def _get(self, endpoint: str, params: Optional[Dict] = None,
             timeout: int = 30) -> Any:
        """Make an authenticated GET request."""
        import requests
        url = f"{self.base_url}/{endpoint}"
        resp = requests.get(url, headers=self._headers(),
                            params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, data: Optional[Dict] = None,
              timeout: int = 30) -> Any:
        """Make an authenticated POST request with JSON body."""
        import requests
        url = f"{self.base_url}/{endpoint}"
        resp = requests.post(url, headers=self._headers(),
                             json=data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # -- Posts --

    def get_posts(self, per_page: int = 100, page: int = 1,
                  status: str = "publish",
                  fields: Optional[str] = None) -> List[Dict]:
        """Fetch posts with pagination.

        Args:
            per_page: Posts per page (max 100)
            page: Page number
            status: Post status filter
            fields: Comma-separated fields to return

        Returns:
            List of post dicts.
        """
        params = {
            "per_page": min(per_page, 100),
            "page": page,
            "status": status,
        }
        if fields:
            params["_fields"] = fields
        try:
            return self._get("posts", params)
        except Exception as e:
            log.error("Error fetching posts: %s", e)
            return []

    def get_all_posts(self, status: str = "publish",
                      fields: Optional[str] = None,
                      max_pages: int = 50) -> List[Dict]:
        """Fetch ALL posts with auto-pagination.

        Returns:
            Complete list of post dicts across all pages.
        """
        all_posts = []
        default_fields = ("id,title,slug,link,date,modified,"
                          "categories,tags,featured_media")
        _fields = fields or default_fields

        for page in range(1, max_pages + 1):
            posts = self.get_posts(per_page=100, page=page,
                                   status=status, fields=_fields)
            if not posts:
                break
            all_posts.extend(posts)
        return all_posts

    def create_post(self, title: str, content: str,
                    status: str = "draft",
                    categories: Optional[List[int]] = None,
                    tags: Optional[List[int]] = None,
                    featured_media: Optional[int] = None,
                    **kwargs) -> Dict:
        """Create a new post.

        Returns the created post dict.
        """
        data = {
            "title": title,
            "content": content,
            "status": status,
        }
        if categories:
            data["categories"] = categories
        if tags:
            data["tags"] = tags
        if featured_media:
            data["featured_media"] = featured_media
        data.update(kwargs)
        return self._post("posts", data)

    def update_post(self, post_id: int, **fields) -> Dict:
        """Update an existing post.

        Args:
            post_id: WordPress post ID
            **fields: Fields to update (title, content, status, etc.)

        Returns:
            Updated post dict.
        """
        return self._post(f"posts/{post_id}", fields)

    # -- Media --

    def upload_media(self, file_path: str,
                     title: Optional[str] = None,
                     alt_text: Optional[str] = None) -> int:
        """Upload a media file (image/video).

        Args:
            file_path: Local path to the file
            title: Optional title for the media item
            alt_text: Optional alt text for accessibility

        Returns:
            Media ID (int) on success.

        Raises:
            FileNotFoundError: If file_path does not exist
            requests.HTTPError: On upload failure
        """
        import requests

        filepath = Path(file_path)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect content type
        suffix = filepath.suffix.lower()
        content_types = {
            ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".gif": "image/gif",
            ".webp": "image/webp", ".svg": "image/svg+xml",
        }
        content_type = content_types.get(suffix, "application/octet-stream")

        headers = {
            "Authorization": f"Basic {self._auth}",
            "Content-Disposition": (
                f'attachment; filename="{filepath.name}"'
            ),
            "Content-Type": content_type,
        }

        url = f"{self.base_url}/media"
        with open(file_path, "rb") as f:
            resp = requests.post(url, headers=headers, data=f, timeout=60)
        resp.raise_for_status()

        media = resp.json()
        media_id = media.get("id", 0)

        # Set alt text if provided
        if alt_text and media_id:
            try:
                self._post(f"media/{media_id}", {"alt_text": alt_text})
            except Exception:
                pass

        log.info("Uploaded media: %s -> ID %d", filepath.name, media_id)
        return media_id

    def set_featured_image(self, post_id: int, media_id: int) -> bool:
        """Set featured image on a post.

        Returns True on success.
        """
        try:
            self.update_post(post_id, featured_media=media_id)
            log.info("Set featured image: post %d -> media %d",
                     post_id, media_id)
            return True
        except Exception as e:
            log.error("Failed to set featured image: %s", e)
            return False

    # -- Categories & Tags --

    def get_categories(self, per_page: int = 100) -> Dict[int, str]:
        """Get category ID-to-name mapping."""
        try:
            cats = self._get("categories", {"per_page": per_page})
            return {c["id"]: c["name"] for c in cats}
        except Exception:
            return {}

    def get_tags(self, per_page: int = 100) -> Dict[int, str]:
        """Get tag ID-to-name mapping."""
        try:
            tags = self._get("tags", {"per_page": per_page})
            return {t["id"]: t["name"] for t in tags}
        except Exception:
            return {}

    def find_or_create_category(self, name: str) -> int:
        """Find a category by name or create it. Returns category ID."""
        cats = self.get_categories()
        for cat_id, cat_name in cats.items():
            if cat_name.lower() == name.lower():
                return cat_id
        # Create new
        result = self._post("categories", {"name": name})
        return result.get("id", 0)

    # -- Health --

    def test_connection(self) -> bool:
        """Test if the WordPress API is reachable and authenticated."""
        try:
            self._get("users/me")
            return True
        except Exception as e:
            log.error("Connection test failed: %s", e)
            return False
