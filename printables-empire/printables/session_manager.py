"""Session management for Printables.com.

Handles login, session save/load, and auth verification.
Wraps PrintablesClient for session-only operations.
"""

import asyncio
import logging
from pathlib import Path

from printables.client import PrintablesClient

log = logging.getLogger("printables.session")


async def login_interactive(
    session_path: str | Path | None = None,
    headless: bool = False,
) -> bool:
    """Run interactive login flow.

    Opens browser, waits for manual login, saves session.
    Returns True if login succeeded.
    """
    async with PrintablesClient(session_path=session_path, headless=headless) as client:
        return await client.login()


async def verify_session(session_path: str | Path | None = None) -> dict | None:
    """Verify saved session is still valid.

    Returns user info dict if valid, None otherwise.
    """
    async with PrintablesClient(session_path=session_path, headless=True) as client:
        await client._page.goto("https://www.printables.com", wait_until="networkidle")
        return await client.check_auth()


def login_sync(session_path: str | Path | None = None) -> bool:
    """Synchronous wrapper for login."""
    return asyncio.run(login_interactive(session_path))


def verify_sync(session_path: str | Path | None = None) -> dict | None:
    """Synchronous wrapper for session verification."""
    return asyncio.run(verify_session(session_path))
