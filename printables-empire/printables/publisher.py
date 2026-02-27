"""Publish content to Printables.com.

Creates rich content entries using the modelUpdate GraphQL mutation.
Printables "models" support rich Markdown descriptions and images
without requiring STL files.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from printables.client import PrintablesClient, CONTENT_CATEGORIES
from printables.formatter import format_for_printables, format_summary

log = logging.getLogger("printables.publisher")

DATA_DIR = Path(__file__).parent.parent / "data"

# Rate limiting — keep publishing low to avoid bans/flags
DAILY_PUBLISH_LIMIT = 2  # Max publishes per calendar day
MIN_PUBLISH_INTERVAL_SEC = 1800  # 30 minutes minimum between publishes


class Publisher:
    """Publishes content pieces to Printables.com."""

    def __init__(self, client: PrintablesClient):
        self.client = client
        self._db = None

    def _get_db(self) -> sqlite3.Connection:
        """Get or create SQLite tracking database."""
        if self._db is None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(DATA_DIR / "content.db"))
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS published_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    slug TEXT,
                    published_url TEXT,
                    printables_model_id TEXT,
                    score REAL DEFAULT 0,
                    cost_usd REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'published'
                )
            """)
            self._db.commit()
        return self._db

    async def publish(
        self,
        title: str,
        description: str,
        content_type: str,
        tags: list[str],
        image_paths: list[str | Path] | None = None,
        stl_path: str | Path | None = None,
        category: str = "guide",
        score: float = 0,
        cost_usd: float = 0,
    ) -> dict:
        """Publish content to Printables.

        Args:
            title: Content title (max 70 chars)
            description: Markdown content body
            content_type: article, review, listing, or post
            tags: List of tag strings
            image_paths: Optional list of image file paths to upload
            category: Printables category key
            score: Quality score
            cost_usd: Generation cost

        Returns:
            Dict with 'success', 'model_id', 'url', 'error'.
        """
        try:
            # 0. Rate limit check — block if daily limit reached
            today_count = self.get_today_publish_count()
            if today_count >= DAILY_PUBLISH_LIMIT:
                msg = (
                    f"Daily publish limit reached ({today_count}/{DAILY_PUBLISH_LIMIT}). "
                    f"Try again tomorrow to avoid being flagged."
                )
                log.warning(msg)
                return {"success": False, "error": msg, "rate_limited": True}

            # 0b. Enforce minimum interval between publishes
            last_publish_time = self._get_last_publish_time()
            if last_publish_time:
                elapsed = (datetime.utcnow() - last_publish_time).total_seconds()
                if elapsed < MIN_PUBLISH_INTERVAL_SEC:
                    wait_sec = int(MIN_PUBLISH_INTERVAL_SEC - elapsed)
                    log.info(f"Rate limit: waiting {wait_sec}s before next publish...")
                    await asyncio.sleep(wait_sec)

            # 1. Clean tag names — Printables resolves names to existing tags
            # Tags must be single lowercase words (no spaces/hyphens)
            tag_names = []
            for t in tags:
                clean = t.lower().replace(" ", "").replace("-", "").replace("_", "")
                if clean and clean not in tag_names:
                    tag_names.append(clean)

            # 2. Upload images
            image_ids = []
            if image_paths:
                for img_path in image_paths:
                    img_id = await self.client.upload_image(Path(img_path))
                    if img_id:
                        image_ids.append(img_id)
                    await self.client.delay()

            if not image_ids:
                log.error("No images uploaded successfully — at least one required")
                return {"success": False, "error": "No images uploaded"}

            # 2b. Upload companion STL file
            stl_ids = []
            if stl_path:
                stl_id = await self.client.upload_stl(Path(stl_path))
                if stl_id:
                    stl_ids.append(stl_id)
                await self.client.delay()

            if not stl_ids:
                # Generate a default companion STL
                from images.companion_stl import get_companion_stl
                import tempfile
                tmp_dir = tempfile.mkdtemp(prefix="stl-")
                fallback_stl = get_companion_stl(content_type, tmp_dir, title)
                stl_id = await self.client.upload_stl(Path(fallback_stl))
                if stl_id:
                    stl_ids.append(stl_id)
                await self.client.delay()

            # 3. Format description for Printables
            formatted_desc = format_for_printables(description, content_type)
            summary = format_summary(description, title)

            # Create the content entry via modelUpdate mutation
            category_id = CONTENT_CATEGORIES.get(category, "48")

            # Build the mutation — matches real Printables schema
            # Uses direct arguments (not input object) and output { id slug }
            mutation = """
            mutation CreateContent(
                $name: String, $description: String, $summary: String,
                $category: ID, $license: ID, $stls: [STLFileInputType],
                $images: [PrintImageInputType], $mainImage: ID, $tags: [ID],
                $aiGenerated: Boolean
            ) {
                modelUpdate(
                    name: $name
                    description: $description
                    summary: $summary
                    category: $category
                    license: $license
                    stls: $stls
                    images: $images
                    mainImage: $mainImage
                    tags: $tags
                    draft: false
                    published: true
                    aiGenerated: $aiGenerated
                    authorship: author
                    excludeCommercialUsage: false
                    nsfw: false
                ) {
                    ok
                    errors { field messages }
                    output { id slug }
                }
            }
            """

            # Format image IDs as PrintImageInputType: [{"id": "123"}, ...]
            image_inputs = [{"id": iid} for iid in image_ids]
            main_image_id = image_ids[0] if image_ids else None

            variables = {
                "name": title[:70],
                "description": formatted_desc,
                "summary": summary,
                "category": category_id,
                "license": "13",  # Standard Digital File License
                "stls": [{"id": sid} for sid in stl_ids],
                "images": image_inputs,
                "mainImage": main_image_id,
                "tags": tag_names[:10],
                "aiGenerated": True,
            }

            result = await self.client.gql(mutation, variables)
            log.info(f"GraphQL response: {json.dumps(result, indent=2)[:1000]}")
            data = (result or {}).get("data", {}).get("modelUpdate", {})

            if data.get("ok"):
                output = data.get("output", {})
                model_id = output.get("id", "")
                slug = output.get("slug", "")
                url = f"https://www.printables.com/model/{model_id}-{slug}" if model_id else ""

                # Log to database
                self._log_publish(content_type, title, slug, url, model_id, score, cost_usd)

                log.info(f"Published: {title} -> {url}")
                return {
                    "success": True,
                    "model_id": model_id,
                    "slug": slug,
                    "url": url,
                }
            else:
                errors = data.get("errors", [])
                error_msg = str(errors)
                log.error(f"Publish failed: {error_msg}")
                return {"success": False, "error": error_msg}

        except Exception as e:
            log.error(f"Publish error: {e}")
            return {"success": False, "error": str(e)}

    def _log_publish(
        self,
        content_type: str,
        title: str,
        slug: str,
        url: str,
        model_id: str,
        score: float,
        cost_usd: float,
    ):
        """Log published content to SQLite."""
        db = self._get_db()
        db.execute(
            """INSERT INTO published_content
               (content_type, title, slug, published_url, printables_model_id, score, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (content_type, title, slug, url, model_id, score, cost_usd),
        )
        db.commit()

    def get_today_publish_count(self) -> int:
        """Get number of publishes today (UTC)."""
        db = self._get_db()
        today = date.today().isoformat()
        row = db.execute(
            "SELECT COUNT(*) FROM published_content WHERE created_at >= ?",
            (today,),
        ).fetchone()
        return row[0] if row else 0

    def get_remaining_today(self) -> int:
        """Get how many more publishes are allowed today."""
        return max(0, DAILY_PUBLISH_LIMIT - self.get_today_publish_count())

    def _get_last_publish_time(self) -> datetime | None:
        """Get the timestamp of the most recent publish."""
        db = self._get_db()
        row = db.execute(
            "SELECT created_at FROM published_content ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row and row[0]:
            try:
                return datetime.fromisoformat(row[0].replace("Z", "+00:00").replace("+00:00", ""))
            except (ValueError, AttributeError):
                return None
        return None

    def get_published_count(self) -> dict:
        """Get count of published content by type."""
        db = self._get_db()
        rows = db.execute(
            "SELECT content_type, COUNT(*) FROM published_content GROUP BY content_type"
        ).fetchall()
        return dict(rows)

    def get_total_cost(self) -> float:
        """Get total API cost across all published content."""
        db = self._get_db()
        row = db.execute("SELECT SUM(cost_usd) FROM published_content").fetchone()
        return row[0] or 0.0

    def get_recent(self, limit: int = 10) -> list[dict]:
        """Get recently published content."""
        db = self._get_db()
        rows = db.execute(
            """SELECT content_type, title, published_url, score, cost_usd, created_at
               FROM published_content ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "type": r[0],
                "title": r[1],
                "url": r[2],
                "score": r[3],
                "cost": r[4],
                "date": r[5],
            }
            for r in rows
        ]
