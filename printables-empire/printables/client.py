"""Core Playwright + GraphQL client for Printables.com.

Extracted and adapted from 3d-print-forge/scripts/upload_printables.py.
Handles browser setup, session management, GraphQL queries, image upload,
and tag lookup with conservative anti-bot pacing.
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("printables.client")

# Pacing constants
MIN_ACTION_DELAY = 1.5  # seconds between browser actions
TYPING_DELAY = 30       # ms per keystroke

# Printables category IDs for content types
CONTENT_CATEGORIES = {
    "guide": "48",        # Hobby & Makers
    "review": "48",       # Hobby & Makers
    "article": "48",      # Hobby & Makers
    "tip": "48",          # Hobby & Makers
    "home_decor": "44",   # Home Decor
    "gaming": "30",       # Toys & Games
    "tools": "21",        # Gadgets
    "outdoor": "53",      # Outdoor & Garden
}


class PrintablesClient:
    """Playwright-based client for Printables.com."""

    def __init__(self, session_path: str | Path | None = None, headless: bool = False):
        self.session_path = Path(session_path) if session_path else (
            Path(__file__).parent.parent / "config" / "printables_session.json"
        )
        self.headless = headless
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._tag_cache: dict[str, str] = {}

    async def start(self):
        """Launch browser with saved session."""
        from playwright.async_api import async_playwright
        try:
            from playwright_stealth import stealth_async
        except ImportError:
            stealth_async = None

        self._pw = await async_playwright().start()

        self._browser = await self._pw.chromium.launch(
            headless=self.headless,
            slow_mo=50,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        context_args = {
            "viewport": {"width": 1440, "height": 900},
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        }

        if self.session_path.exists():
            context_args["storage_state"] = str(self.session_path)
            log.info("Loaded saved session")

        self._context = await self._browser.new_context(**context_args)
        self._page = await self._context.new_page()

        if stealth_async:
            await stealth_async(self._page)

        # Navigate to Printables so cookies are sent with GraphQL requests
        await self._page.goto("https://www.printables.com/", wait_until="domcontentloaded")
        await asyncio.sleep(2)

        log.info("Browser started")

    async def close(self):
        """Clean up browser resources."""
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        log.info("Browser closed")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def save_session(self):
        """Save browser session state."""
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        state = await self._context.storage_state()
        with open(self.session_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        log.info(f"Session saved to {self.session_path}")

    async def gql(self, query: str, variables: dict | None = None) -> dict:
        """Execute a GraphQL query via the authenticated browser session."""
        body = {"query": query}
        if variables:
            body["variables"] = variables
        return await self._page.evaluate("""async (body) => {
            const r = await fetch('https://api.printables.com/graphql/', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                credentials: 'include',
                body: JSON.stringify(body)
            });
            const text = await r.text();
            try { return JSON.parse(text); }
            catch(e) { return {raw: text.substring(0, 500), status: r.status}; }
        }""", body)

    async def check_auth(self) -> dict | None:
        """Check if currently authenticated. Returns user info or None."""
        result = await self.gql("{ me { id handle } }")
        me = (result or {}).get("data", {}).get("me")
        if me and me.get("handle"):
            return me
        return None

    async def login(self) -> bool:
        """Interactive login flow — opens browser for manual login."""
        log.info("Navigating to Printables login...")
        await self._page.goto(
            "https://www.printables.com/settings/account",
            wait_until="networkidle",
        )
        await asyncio.sleep(2)

        # Check if already logged in
        me = await self.check_auth()
        if me:
            log.info(f"Already logged in as {me['handle']}!")
            await self.save_session()
            return True

        log.info("Please log in manually in the browser window...")
        log.info("Waiting up to 10 minutes for login to complete...")

        for i in range(600):
            await asyncio.sleep(1)
            if i % 5 == 0:
                try:
                    me = await self.check_auth()
                    if me:
                        log.info(f"Login detected! Logged in as: {me['handle']}")
                        await self.save_session()
                        return True
                except Exception:
                    pass
            if i % 30 == 0 and i > 0:
                log.info(f"  Still waiting for login... ({i}s elapsed)")

        log.error("Login timed out after 10 minutes")
        return False

    async def lookup_tag_ids(self, tag_strings: list[str]) -> list[str]:
        """Look up Printables tag IDs for tag strings."""
        tag_ids = []

        for tag_str in tag_strings:
            clean = tag_str.lower().replace(" ", "").replace("-", "").replace("_", "")
            if not clean:
                continue

            if clean in self._tag_cache:
                tag_ids.append(self._tag_cache[clean])
                continue

            try:
                r = await self.gql(f'{{ tags(query: "{clean}", limit: 5) {{ id name }} }}')
                tags_found = (r or {}).get("data", {}).get("tags", [])
                exact = next((t for t in tags_found if t["name"] == clean), None)
                if exact:
                    self._tag_cache[clean] = exact["id"]
                    tag_ids.append(exact["id"])
                elif tags_found:
                    self._tag_cache[clean] = tags_found[0]["id"]
                    tag_ids.append(tags_found[0]["id"])
                else:
                    words = tag_str.lower().split()
                    if len(words) > 1:
                        for word in words:
                            if word in self._tag_cache:
                                tag_ids.append(self._tag_cache[word])
                                break
                            r2 = await self.gql(f'{{ tags(query: "{word}", limit: 1) {{ id name }} }}')
                            found2 = (r2 or {}).get("data", {}).get("tags", [])
                            if found2:
                                self._tag_cache[word] = found2[0]["id"]
                                tag_ids.append(found2[0]["id"])
                                break
                await asyncio.sleep(0.3)  # Small delay between tag lookups
            except Exception:
                pass

        # Deduplicate
        seen = set()
        return [tid for tid in tag_ids if not (tid in seen or seen.add(tid))]

    async def upload_image(self, image_path: Path) -> Optional[str]:
        """Upload an image via GraphQL + GCS signed URL.

        Returns the processed image ID, or None on failure.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            log.error(f"Image not found: {image_path}")
            return None

        # Step 1: Get signed upload URL
        mutation = """
        mutation {
            printFileUpload2(fileName: "%s", unzip: false) {
                ok
                signedUrl
                fileUpload { id name }
                errors { field messages }
            }
        }
        """ % image_path.name

        result = await self.gql(mutation)
        data = (result or {}).get("data", {}).get("printFileUpload2", {})

        if not data.get("ok"):
            log.error(f"Upload slot failed: {data.get('errors')}")
            return None

        signed_url = data["signedUrl"]
        upload_id = data["fileUpload"]["id"]

        # Step 2: Upload to GCS
        success = await self._upload_to_gcs(signed_url, image_path)
        if not success:
            return None

        # Step 3: Mark upload finished
        await self.gql(f'mutation {{ printFileUploadFinished(fileUploadId: "{upload_id}") {{ ok }} }}')

        # Step 4: Poll for processing
        for _ in range(30):
            await asyncio.sleep(1)
            poll = await self.gql(
                'query { modelFileUploads(ids: ["%s"]) { id isProcessed printimageSet { id name } } }' % upload_id
            )
            uploads = (poll or {}).get("data", {}).get("modelFileUploads", [])
            if uploads and uploads[0].get("isProcessed"):
                images = uploads[0].get("printimageSet", [])
                if images:
                    return images[0]["id"]
                return None

        log.warning(f"Image upload timed out for {image_path.name}")
        return None

    async def upload_stl(self, stl_path: Path) -> Optional[str]:
        """Upload an STL file via GraphQL + GCS signed URL.

        Returns the processed STL file ID, or None on failure.
        """
        stl_path = Path(stl_path)
        if not stl_path.exists():
            log.error(f"STL not found: {stl_path}")
            return None

        mutation = """
        mutation {
            printFileUpload2(fileName: "%s", unzip: false) {
                ok
                signedUrl
                fileUpload { id name }
                errors { field messages }
            }
        }
        """ % stl_path.name

        result = await self.gql(mutation)
        data = (result or {}).get("data", {}).get("printFileUpload2", {})

        if not data.get("ok"):
            log.error(f"STL upload slot failed: {data.get('errors')}")
            return None

        signed_url = data["signedUrl"]
        upload_id = data["fileUpload"]["id"]

        success = await self._upload_to_gcs(signed_url, stl_path)
        if not success:
            return None

        await self.gql(f'mutation {{ printFileUploadFinished(fileUploadId: "{upload_id}") {{ ok }} }}')

        # Poll for STL processing — uses stlSet instead of printimageSet
        for _ in range(60):
            await asyncio.sleep(1)
            poll = await self.gql(
                'query { modelFileUploads(ids: ["%s"]) { id isProcessed stlSet { id name } } }' % upload_id
            )
            uploads = (poll or {}).get("data", {}).get("modelFileUploads", [])
            if uploads and uploads[0].get("isProcessed"):
                stls = uploads[0].get("stlSet", [])
                if stls:
                    log.info(f"STL uploaded: {stl_path.name} -> ID {stls[0]['id']}")
                    return stls[0]["id"]
                return None

        log.warning(f"STL upload timed out for {stl_path.name}")
        return None

    async def _upload_to_gcs(self, signed_url: str, file_path: Path) -> bool:
        """Upload a file to GCS signed URL via curl multipart POST."""
        try:
            proc = subprocess.run(
                [
                    "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                    "-X", "POST", "-F", f"file=@{file_path}", signed_url,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            status = proc.stdout.strip()
            if status in ("200", "204"):
                return True
            log.warning(f"GCS upload returned {status}")
            return False
        except Exception as e:
            log.error(f"GCS upload failed: {e}")
            return False

    async def delay(self, seconds: float = MIN_ACTION_DELAY):
        """Conservative delay between actions."""
        await asyncio.sleep(seconds)
