"""
App Discovery — OpenClaw Empire App Discovery & Installation

Automates Play Store searches, app evaluation, installation, first-run
handling (permissions, onboarding), uninstall, updates, APK sideloading,
and app inventory scanning. Syncs discoveries to configs/app-registry.json.

Data persisted to: data/app_discovery/

Usage:
    from src.app_discovery import AppDiscovery, get_app_discovery

    discovery = get_app_discovery()
    results = await discovery.search_play_store("instagram")
    await discovery.install_app("com.instagram.android")
    inventory = await discovery.scan_inventory()

CLI:
    python -m src.app_discovery search --query "social media"
    python -m src.app_discovery install --package com.instagram.android
    python -m src.app_discovery inventory
    python -m src.app_discovery evaluate --package com.instagram.android
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("app_discovery")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "app_discovery"
REGISTRY_PATH = Path(__file__).parent.parent / "configs" / "app-registry.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(str(tmp), str(path))


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class AppCategory(str, Enum):
    SOCIAL = "social"
    PRODUCTIVITY = "productivity"
    ENTERTAINMENT = "entertainment"
    COMMUNICATION = "communication"
    SHOPPING = "shopping"
    FINANCE = "finance"
    EDUCATION = "education"
    HEALTH = "health"
    NEWS = "news"
    TOOLS = "tools"
    PHOTO_VIDEO = "photo_video"
    MUSIC = "music"
    GAMING = "gaming"
    BUSINESS = "business"
    TRAVEL = "travel"
    FOOD = "food"
    LIFESTYLE = "lifestyle"
    WEATHER = "weather"
    OTHER = "other"


class InstallStatus(str, Enum):
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    UPDATING = "updating"
    UNINSTALLING = "uninstalling"
    FAILED = "failed"
    SIDELOADED = "sideloaded"


class AppRating(str, Enum):
    EXCELLENT = "excellent"  # 4.5+
    GOOD = "good"            # 4.0-4.4
    AVERAGE = "average"      # 3.0-3.9
    POOR = "poor"            # 2.0-2.9
    TERRIBLE = "terrible"    # <2.0
    UNKNOWN = "unknown"


@dataclass
class PlayStoreResult:
    """A Play Store search result."""
    package: str = ""
    name: str = ""
    developer: str = ""
    rating: float = 0.0
    rating_count: str = ""
    downloads: str = ""
    price: str = "Free"
    description: str = ""
    category: AppCategory = AppCategory.OTHER
    size: str = ""
    position: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class AppEvaluation:
    """Evaluation of an app's suitability."""
    package: str = ""
    name: str = ""
    rating: float = 0.0
    rating_grade: AppRating = AppRating.UNKNOWN
    downloads: str = ""
    size: str = ""
    permissions: List[str] = field(default_factory=list)
    permission_risk: str = "low"  # low, medium, high
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    recommendation: str = ""
    evaluated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rating_grade"] = self.rating_grade.value
        return d


@dataclass
class InstalledApp:
    """Information about an installed app."""
    package: str = ""
    name: str = ""
    version: str = ""
    version_code: int = 0
    install_status: InstallStatus = InstallStatus.INSTALLED
    install_date: str = ""
    last_updated: str = ""
    size_mb: float = 0.0
    category: AppCategory = AppCategory.OTHER
    system_app: bool = False
    enabled: bool = True
    permissions: List[str] = field(default_factory=list)
    data_size_mb: float = 0.0
    first_run_completed: bool = False
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["install_status"] = self.install_status.value
        d["category"] = self.category.value
        return d


@dataclass
class FirstRunConfig:
    """Configuration for handling an app's first-run experience."""
    package: str = ""
    grant_permissions: bool = True
    skip_onboarding: bool = True
    accept_tos: bool = True
    skip_login: bool = True
    custom_steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# AppDiscovery
# ---------------------------------------------------------------------------

class AppDiscovery:
    """
    App discovery and installation manager for Android phones.

    Searches the Play Store, evaluates apps, handles installation
    and first-run setup, manages app inventory, and sideloads APKs.

    Usage:
        discovery = get_app_discovery()
        results = await discovery.search_play_store("social media")
        await discovery.install_app("com.instagram.android")
    """

    def __init__(
        self,
        controller: Any = None,
        vision: Any = None,
        learner: Any = None,
        data_dir: Optional[Path] = None,
    ):
        self._controller = controller
        self._vision = vision
        self._learner = learner
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._inventory: Dict[str, InstalledApp] = {}
        self._evaluations: Dict[str, AppEvaluation] = {}
        self._first_run_configs: Dict[str, FirstRunConfig] = {}
        self._search_history: List[Dict[str, Any]] = []

        self._load_state()
        logger.info("AppDiscovery initialized (%d apps in inventory)", len(self._inventory))

    # ── Property helpers ──

    @property
    def controller(self):
        if self._controller is None:
            try:
                from src.phone_controller import PhoneController
                self._controller = PhoneController()
            except ImportError:
                logger.error("PhoneController not available")
        return self._controller

    @property
    def vision(self):
        if self._vision is None:
            try:
                from src.vision_agent import VisionAgent
                self._vision = VisionAgent()
            except ImportError:
                logger.warning("VisionAgent not available")
        return self._vision

    @property
    def learner(self):
        if self._learner is None:
            try:
                from src.app_learner import get_app_learner
                self._learner = get_app_learner()
            except ImportError:
                logger.debug("AppLearner not available")
        return self._learner

    # ── Persistence ──

    def _load_state(self) -> None:
        state = _load_json(self._data_dir / "state.json")
        for pkg, data in state.get("inventory", {}).items():
            if isinstance(data, dict):
                self._inventory[pkg] = InstalledApp(**data)
        for pkg, data in state.get("evaluations", {}).items():
            if isinstance(data, dict):
                self._evaluations[pkg] = AppEvaluation(**data)
        for pkg, data in state.get("first_run_configs", {}).items():
            if isinstance(data, dict):
                self._first_run_configs[pkg] = FirstRunConfig(**data)
        self._search_history = state.get("search_history", [])[-100:]

    def _save_state(self) -> None:
        _save_json(self._data_dir / "state.json", {
            "inventory": {k: v.to_dict() for k, v in self._inventory.items()},
            "evaluations": {k: v.to_dict() for k, v in self._evaluations.items()},
            "first_run_configs": {k: v.to_dict() for k, v in self._first_run_configs.items()},
            "search_history": self._search_history[-100:],
            "updated_at": _now_iso(),
        })

    def _sync_registry(self) -> None:
        """Sync inventory to configs/app-registry.json."""
        registry = {
            "apps": {k: v.to_dict() for k, v in self._inventory.items()},
            "updated_at": _now_iso(),
        }
        _save_json(REGISTRY_PATH, registry)

    # ── ADB helpers ──

    async def _adb_shell(self, cmd: str) -> str:
        if self.controller is None:
            raise RuntimeError("PhoneController not available")
        return await self.controller._adb_shell(cmd)

    async def _take_screenshot(self) -> str:
        return await self.controller.screenshot()

    async def _analyze_screen(self, screenshot_path: str) -> dict:
        if self.vision is None:
            return {}
        result = await self.vision.analyze_screen(screenshot_path=screenshot_path)
        return result if isinstance(result, dict) else {"raw": str(result)}

    async def _find_element(self, description: str, screenshot_path: str = None) -> Optional[dict]:
        if self.vision is None:
            return None
        kwargs = {"description": description}
        if screenshot_path:
            kwargs["screenshot_path"] = screenshot_path
        result = await self.vision.find_element(**kwargs)
        if isinstance(result, dict) and result.get("x") is not None:
            return result
        return None

    # ── Play Store search ──

    async def search_play_store(
        self, query: str, max_results: int = 10
    ) -> List[PlayStoreResult]:
        """
        Search the Play Store and extract results via OCR.

        Opens the Play Store app, enters the search query, and
        extracts app names, ratings, and other info from the results.

        Args:
            query: Search query.
            max_results: Maximum number of results to extract.

        Returns:
            List of PlayStoreResult objects.
        """
        results: List[PlayStoreResult] = []

        try:
            # Open Play Store
            await self._adb_shell(
                f"am start -a android.intent.action.VIEW "
                f"'market://search?q={query.replace(' ', '+')}'"
            )
            await asyncio.sleep(3.0)

            # Extract results by scrolling and reading
            for scroll in range(5):
                screenshot = await self._take_screenshot()
                analysis = await self._analyze_screen(screenshot)

                new_results = self._parse_play_store_results(analysis, len(results))
                results.extend(new_results)

                if len(results) >= max_results:
                    break

                # Scroll for more results
                await self.controller.scroll_down(600)
                await asyncio.sleep(1.5)

        except Exception as exc:
            logger.error("Play Store search failed: %s", exc)

        results = results[:max_results]

        # Record search
        self._search_history.append({
            "query": query,
            "results": len(results),
            "timestamp": _now_iso(),
        })
        self._save_state()

        logger.info("Play Store search '%s': %d results", query, len(results))
        return results

    def _parse_play_store_results(
        self, analysis: dict, start_pos: int
    ) -> List[PlayStoreResult]:
        """Parse Play Store search results from vision analysis."""
        results = []
        if not isinstance(analysis, dict):
            return results

        visible = analysis.get("visible_text", "")
        if isinstance(visible, list):
            visible = "\n".join(visible)

        # Simple heuristic: look for rating patterns (e.g., "4.5★" or "4.5 stars")
        lines = visible.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            rating_match = re.search(r'(\d\.\d)\s*[★⭐]', line)
            if rating_match:
                result = PlayStoreResult(
                    rating=float(rating_match.group(1)),
                    position=start_pos + len(results) + 1,
                )
                # Previous line might be the app name
                if i > 0 and lines[i - 1].strip():
                    result.name = lines[i - 1].strip()
                # Current line might have more info
                result.description = line
                results.append(result)
            i += 1

        return results

    # ── App evaluation ──

    async def evaluate_app(self, package: str) -> AppEvaluation:
        """
        Evaluate an app by checking its Play Store listing, permissions,
        and reviews.

        Args:
            package: App package name.

        Returns:
            AppEvaluation with rating, permissions, pros/cons, etc.
        """
        evaluation = AppEvaluation(package=package)

        try:
            # Open the app's Play Store page
            await self._adb_shell(
                f"am start -a android.intent.action.VIEW "
                f"'market://details?id={package}'"
            )
            await asyncio.sleep(3.0)

            # Read the page
            screenshot = await self._take_screenshot()
            analysis = await self._analyze_screen(screenshot)

            if isinstance(analysis, dict):
                visible = analysis.get("visible_text", "")
                if isinstance(visible, list):
                    visible = "\n".join(visible)

                # Extract rating
                rating_match = re.search(r'(\d\.\d)\s*[★⭐]', visible)
                if rating_match:
                    evaluation.rating = float(rating_match.group(1))

                # Extract app name
                name_match = re.search(r'^(.+?)(?:\n|$)', visible)
                if name_match:
                    evaluation.name = name_match.group(1).strip()

                # Extract download count
                dl_match = re.search(r'([\d.]+[KMB]?\+?\s*downloads)', visible, re.IGNORECASE)
                if dl_match:
                    evaluation.downloads = dl_match.group(1)

                # Extract size
                size_match = re.search(r'([\d.]+\s*[KMG]B)', visible, re.IGNORECASE)
                if size_match:
                    evaluation.size = size_match.group(1)

            # Grade the rating
            if evaluation.rating >= 4.5:
                evaluation.rating_grade = AppRating.EXCELLENT
            elif evaluation.rating >= 4.0:
                evaluation.rating_grade = AppRating.GOOD
            elif evaluation.rating >= 3.0:
                evaluation.rating_grade = AppRating.AVERAGE
            elif evaluation.rating >= 2.0:
                evaluation.rating_grade = AppRating.POOR
            elif evaluation.rating > 0:
                evaluation.rating_grade = AppRating.TERRIBLE

            # Check permissions
            evaluation.permissions = await self._get_app_permissions(package)
            high_risk_perms = ["READ_CONTACTS", "READ_SMS", "SEND_SMS",
                               "READ_CALL_LOG", "ACCESS_FINE_LOCATION",
                               "CAMERA", "RECORD_AUDIO", "READ_EXTERNAL_STORAGE"]
            risk_count = sum(1 for p in evaluation.permissions if any(r in p for r in high_risk_perms))
            if risk_count >= 4:
                evaluation.permission_risk = "high"
            elif risk_count >= 2:
                evaluation.permission_risk = "medium"
            else:
                evaluation.permission_risk = "low"

            # Build recommendation
            if evaluation.rating >= 4.0 and evaluation.permission_risk != "high":
                evaluation.recommendation = "recommended"
                evaluation.pros.append(f"High rating ({evaluation.rating})")
            elif evaluation.rating >= 3.0:
                evaluation.recommendation = "acceptable"
            else:
                evaluation.recommendation = "not_recommended"
                evaluation.cons.append(f"Low rating ({evaluation.rating})")

            if evaluation.permission_risk == "high":
                evaluation.cons.append(f"High permission risk ({risk_count} sensitive permissions)")

        except Exception as exc:
            logger.error("App evaluation failed for %s: %s", package, exc)

        self._evaluations[package] = evaluation
        self._save_state()
        return evaluation

    async def _get_app_permissions(self, package: str) -> List[str]:
        """Get the permissions declared by an app."""
        try:
            output = await self._adb_shell(f"dumpsys package {package} | grep permission")
            permissions = []
            for line in output.split("\n"):
                line = line.strip()
                if "android.permission." in line:
                    perm_match = re.search(r'(android\.permission\.\w+)', line)
                    if perm_match:
                        perm = perm_match.group(1).replace("android.permission.", "")
                        if perm not in permissions:
                            permissions.append(perm)
            return permissions
        except Exception:
            return []

    # ── Installation ──

    async def install_app(
        self,
        package: str,
        handle_first_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Install an app from the Play Store.

        Opens the Play Store listing and clicks the Install button.
        Optionally handles the first-run experience (permissions, onboarding).

        Args:
            package: App package name.
            handle_first_run: Whether to handle first-run setup after install.

        Returns:
            Dict with installation result.
        """
        try:
            # Open Play Store listing
            await self._adb_shell(
                f"am start -a android.intent.action.VIEW "
                f"'market://details?id={package}'"
            )
            await asyncio.sleep(3.0)

            # Find and click Install button
            screenshot = await self._take_screenshot()
            install_btn = await self._find_element("Install button or Get button", screenshot)

            if not install_btn:
                # Maybe already installed — check for Open/Update button
                open_btn = await self._find_element("Open button", screenshot)
                if open_btn:
                    return {"success": True, "status": "already_installed", "package": package}
                update_btn = await self._find_element("Update button", screenshot)
                if update_btn:
                    return {"success": True, "status": "update_available", "package": package}
                return {"success": False, "error": "Install button not found"}

            # Click Install
            await self.controller.tap(install_btn["x"], install_btn["y"])
            await asyncio.sleep(2.0)

            # Handle "Accept" dialog if present
            screenshot2 = await self._take_screenshot()
            accept_btn = await self._find_element("Accept button or Continue button", screenshot2)
            if accept_btn:
                await self.controller.tap(accept_btn["x"], accept_btn["y"])
                await asyncio.sleep(2.0)

            # Wait for installation (poll for "Open" button)
            installed = False
            for _ in range(30):  # Max 60 seconds
                await asyncio.sleep(2.0)
                screenshot3 = await self._take_screenshot()
                open_btn = await self._find_element("Open button", screenshot3)
                if open_btn:
                    installed = True
                    break

            if installed:
                # Record in inventory
                app = InstalledApp(
                    package=package,
                    install_status=InstallStatus.INSTALLED,
                    install_date=_now_iso(),
                )
                self._inventory[package] = app
                self._save_state()
                self._sync_registry()

                # Handle first run
                if handle_first_run:
                    await self._handle_first_run(package)

                logger.info("Installed app: %s", package)
                return {"success": True, "status": "installed", "package": package}
            else:
                return {"success": False, "error": "Installation timeout"}

        except Exception as exc:
            logger.error("Install failed for %s: %s", package, exc)
            return {"success": False, "error": str(exc)}

    async def _handle_first_run(self, package: str) -> Dict[str, Any]:
        """Handle first-run experience: permissions, onboarding, TOS."""
        results = []

        try:
            # Launch the app
            await self.controller.launch_app(package)
            await asyncio.sleep(3.0)

            config = self._first_run_configs.get(package, FirstRunConfig(package=package))

            for attempt in range(10):  # Max 10 dialogs/screens
                screenshot = await self._take_screenshot()
                analysis = await self._analyze_screen(screenshot)
                visible = ""
                if isinstance(analysis, dict):
                    v = analysis.get("visible_text", "")
                    visible = " ".join(v) if isinstance(v, list) else str(v)
                visible_lower = visible.lower()

                # Handle permission dialogs
                if config.grant_permissions and any(
                    kw in visible_lower for kw in ["allow", "permission", "access"]
                ):
                    allow_btn = await self._find_element(
                        "Allow button or While using the app button", screenshot
                    )
                    if allow_btn:
                        await self.controller.tap(allow_btn["x"], allow_btn["y"])
                        await asyncio.sleep(1.0)
                        results.append({"action": "granted_permission", "attempt": attempt})
                        continue

                # Handle onboarding / tutorials
                if config.skip_onboarding and any(
                    kw in visible_lower for kw in ["skip", "next", "got it", "continue", "get started"]
                ):
                    skip_btn = await self._find_element(
                        "Skip button or Next button or Got it button or Continue button",
                        screenshot
                    )
                    if skip_btn:
                        await self.controller.tap(skip_btn["x"], skip_btn["y"])
                        await asyncio.sleep(1.0)
                        results.append({"action": "skipped_onboarding", "attempt": attempt})
                        continue

                # Handle TOS / privacy
                if config.accept_tos and any(
                    kw in visible_lower for kw in ["agree", "accept", "terms", "privacy"]
                ):
                    agree_btn = await self._find_element(
                        "Agree button or Accept button or I agree button", screenshot
                    )
                    if agree_btn:
                        await self.controller.tap(agree_btn["x"], agree_btn["y"])
                        await asyncio.sleep(1.0)
                        results.append({"action": "accepted_tos", "attempt": attempt})
                        continue

                # If nothing matches, we're likely past first-run
                break

            # Mark first run as completed
            if package in self._inventory:
                self._inventory[package].first_run_completed = True
                self._save_state()

        except Exception as exc:
            logger.warning("First-run handling failed for %s: %s", package, exc)
            results.append({"action": "error", "error": str(exc)})

        return {"package": package, "steps": results}

    async def uninstall_app(self, package: str) -> Dict[str, Any]:
        """Uninstall an app."""
        try:
            await self._adb_shell(f"pm uninstall {package}")
            if package in self._inventory:
                self._inventory[package].install_status = InstallStatus.NOT_INSTALLED
            self._save_state()
            self._sync_registry()
            logger.info("Uninstalled: %s", package)
            return {"success": True, "package": package}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def update_app(self, package: str) -> Dict[str, Any]:
        """Update an app via Play Store."""
        try:
            await self._adb_shell(
                f"am start -a android.intent.action.VIEW "
                f"'market://details?id={package}'"
            )
            await asyncio.sleep(3.0)

            screenshot = await self._take_screenshot()
            update_btn = await self._find_element("Update button", screenshot)
            if update_btn:
                await self.controller.tap(update_btn["x"], update_btn["y"])
                await asyncio.sleep(5.0)
                if package in self._inventory:
                    self._inventory[package].last_updated = _now_iso()
                self._save_state()
                return {"success": True, "package": package, "action": "updating"}
            else:
                return {"success": True, "package": package, "action": "up_to_date"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def sideload_apk(self, apk_path: str) -> Dict[str, Any]:
        """Install an APK file from the device's storage."""
        try:
            output = await self._adb_shell(f"pm install -r {apk_path}")
            success = "Success" in output
            if success:
                # Try to get package name
                pkg_match = re.search(r'package:\s*name=\'(\S+)\'', output)
                package = pkg_match.group(1) if pkg_match else apk_path.split("/")[-1]
                app = InstalledApp(
                    package=package,
                    install_status=InstallStatus.SIDELOADED,
                    install_date=_now_iso(),
                )
                self._inventory[package] = app
                self._save_state()
                self._sync_registry()
                return {"success": True, "package": package, "method": "sideload"}
            return {"success": False, "error": output}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Inventory ──

    async def scan_inventory(self) -> List[InstalledApp]:
        """Scan all installed apps on the device."""
        try:
            # Get list of installed packages
            output = await self._adb_shell("pm list packages -3")  # -3 = third-party only
            packages = []
            for line in output.split("\n"):
                line = line.strip()
                if line.startswith("package:"):
                    packages.append(line.replace("package:", ""))

            for pkg in packages:
                if pkg not in self._inventory:
                    self._inventory[pkg] = InstalledApp(
                        package=pkg,
                        install_status=InstallStatus.INSTALLED,
                    )

                # Get version info
                try:
                    ver_output = await self._adb_shell(
                        f"dumpsys package {pkg} | grep versionName"
                    )
                    ver_match = re.search(r'versionName=(\S+)', ver_output)
                    if ver_match:
                        self._inventory[pkg].version = ver_match.group(1)
                except Exception:
                    pass

            # Mark packages not found as uninstalled
            for pkg in list(self._inventory.keys()):
                if pkg not in packages and self._inventory[pkg].install_status == InstallStatus.INSTALLED:
                    self._inventory[pkg].install_status = InstallStatus.NOT_INSTALLED

            self._save_state()
            self._sync_registry()

            installed = [v for v in self._inventory.values() if v.install_status == InstallStatus.INSTALLED]
            logger.info("Inventory scan: %d apps installed", len(installed))
            return installed

        except Exception as exc:
            logger.error("Inventory scan failed: %s", exc)
            return []

    def get_inventory(self, category: str = "", tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get inventory, optionally filtered."""
        apps = list(self._inventory.values())
        if category:
            apps = [a for a in apps if a.category.value == category]
        if tags:
            apps = [a for a in apps if any(t in a.tags for t in tags)]
        return [a.to_dict() for a in apps]

    def tag_app(self, package: str, tags: List[str]) -> Dict[str, Any]:
        """Add tags to an app in the inventory."""
        app = self._inventory.get(package)
        if not app:
            return {"success": False, "error": f"App {package} not in inventory"}
        for tag in tags:
            if tag not in app.tags:
                app.tags.append(tag)
        self._save_state()
        return {"success": True, "package": package, "tags": app.tags}

    def categorize_app(self, package: str, category: AppCategory) -> Dict[str, Any]:
        """Set an app's category."""
        app = self._inventory.get(package)
        if not app:
            return {"success": False, "error": f"App {package} not in inventory"}
        app.category = category
        self._save_state()
        return {"success": True, "package": package, "category": category.value}

    def set_first_run_config(self, package: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set first-run handling configuration for an app."""
        frc = FirstRunConfig(package=package, **config)
        self._first_run_configs[package] = frc
        self._save_state()
        return {"success": True, "config": frc.to_dict()}

    # ── Statistics ──

    def stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        installed = [a for a in self._inventory.values() if a.install_status == InstallStatus.INSTALLED]
        categories = {}
        for app in installed:
            cat = app.category.value
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_tracked": len(self._inventory),
            "installed": len(installed),
            "evaluations": len(self._evaluations),
            "searches": len(self._search_history),
            "categories": categories,
        }

    # ── Sync wrappers ──

    def search_play_store_sync(self, query: str, **kwargs) -> List[PlayStoreResult]:
        return _run_sync(self.search_play_store(query, **kwargs))

    def install_app_sync(self, package: str, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.install_app(package, **kwargs))

    def uninstall_app_sync(self, package: str) -> Dict[str, Any]:
        return _run_sync(self.uninstall_app(package))

    def evaluate_app_sync(self, package: str) -> AppEvaluation:
        return _run_sync(self.evaluate_app(package))

    def scan_inventory_sync(self) -> List[InstalledApp]:
        return _run_sync(self.scan_inventory())

    def update_app_sync(self, package: str) -> Dict[str, Any]:
        return _run_sync(self.update_app(package))

    def sideload_apk_sync(self, apk_path: str) -> Dict[str, Any]:
        return _run_sync(self.sideload_apk(apk_path))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[AppDiscovery] = None


def get_app_discovery(
    controller: Any = None,
    vision: Any = None,
    learner: Any = None,
) -> AppDiscovery:
    """Get the singleton AppDiscovery instance."""
    global _instance
    if _instance is None:
        _instance = AppDiscovery(controller=controller, vision=vision, learner=learner)
    return _instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _cli_search(args: argparse.Namespace) -> None:
    discovery = get_app_discovery()
    results = discovery.search_play_store_sync(args.query, max_results=args.limit)
    _print_json([r.to_dict() for r in results])


def _cli_install(args: argparse.Namespace) -> None:
    discovery = get_app_discovery()
    result = discovery.install_app_sync(args.package, handle_first_run=not args.no_first_run)
    _print_json(result)


def _cli_uninstall(args: argparse.Namespace) -> None:
    discovery = get_app_discovery()
    result = discovery.uninstall_app_sync(args.package)
    _print_json(result)


def _cli_update(args: argparse.Namespace) -> None:
    discovery = get_app_discovery()
    result = discovery.update_app_sync(args.package)
    _print_json(result)


def _cli_evaluate(args: argparse.Namespace) -> None:
    discovery = get_app_discovery()
    evaluation = discovery.evaluate_app_sync(args.package)
    _print_json(evaluation.to_dict())


def _cli_inventory(args: argparse.Namespace) -> None:
    discovery = get_app_discovery()
    if args.scan:
        apps = discovery.scan_inventory_sync()
        _print_json([a.to_dict() for a in apps])
    else:
        apps = discovery.get_inventory(category=args.category or "")
        _print_json(apps)


def _cli_sideload(args: argparse.Namespace) -> None:
    discovery = get_app_discovery()
    result = discovery.sideload_apk_sync(args.apk)
    _print_json(result)


def _cli_tag(args: argparse.Namespace) -> None:
    discovery = get_app_discovery()
    tags = [t.strip() for t in args.tags.split(",")]
    result = discovery.tag_app(args.package, tags)
    _print_json(result)


def _cli_stats(args: argparse.Namespace) -> None:
    discovery = get_app_discovery()
    _print_json(discovery.stats())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="app_discovery",
        description="OpenClaw Empire — App Discovery & Installation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # search
    sr = sub.add_parser("search", help="Search Play Store")
    sr.add_argument("--query", required=True)
    sr.add_argument("--limit", type=int, default=10)
    sr.set_defaults(func=_cli_search)

    # install
    ins = sub.add_parser("install", help="Install an app")
    ins.add_argument("--package", required=True)
    ins.add_argument("--no-first-run", action="store_true")
    ins.set_defaults(func=_cli_install)

    # uninstall
    uni = sub.add_parser("uninstall", help="Uninstall an app")
    uni.add_argument("--package", required=True)
    uni.set_defaults(func=_cli_uninstall)

    # update
    upd = sub.add_parser("update", help="Update an app")
    upd.add_argument("--package", required=True)
    upd.set_defaults(func=_cli_update)

    # evaluate
    evl = sub.add_parser("evaluate", help="Evaluate an app")
    evl.add_argument("--package", required=True)
    evl.set_defaults(func=_cli_evaluate)

    # inventory
    inv = sub.add_parser("inventory", help="App inventory")
    inv.add_argument("--scan", action="store_true", help="Scan device for apps")
    inv.add_argument("--category", default="")
    inv.set_defaults(func=_cli_inventory)

    # sideload
    sl = sub.add_parser("sideload", help="Sideload an APK")
    sl.add_argument("--apk", required=True)
    sl.set_defaults(func=_cli_sideload)

    # tag
    tg = sub.add_parser("tag", help="Tag an app")
    tg.add_argument("--package", required=True)
    tg.add_argument("--tags", required=True, help="Comma-separated tags")
    tg.set_defaults(func=_cli_tag)

    # stats
    st = sub.add_parser("stats", help="Discovery statistics")
    st.set_defaults(func=_cli_stats)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
