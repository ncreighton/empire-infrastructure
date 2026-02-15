"""
App Learner — OpenClaw Empire Self-Teaching App Navigator

Autonomously explores Android apps by taking screenshots, analyzing UI elements,
tapping through screens, and building a navigation graph. Recognizes common UI
patterns (login screens, settings, nav drawers, bottom bars, onboarding) and
auto-generates playbooks from learned navigation graphs.

Data persisted to: data/app_learner/knowledge/{package}.json

Usage:
    from src.app_learner import AppLearner, get_app_learner

    learner = get_app_learner()
    graph = await learner.explore_app("com.instagram.android", max_steps=50)
    playbook = learner.generate_playbook("com.instagram.android", "post_photo")
    path = learner.find_path("com.instagram.android", "home", "settings")

CLI:
    python -m src.app_learner explore --package com.instagram.android --steps 50
    python -m src.app_learner path --package com.instagram.android --from home --to settings
    python -m src.app_learner playbook --package com.instagram.android --goal post_photo
    python -m src.app_learner knowledge --package com.instagram.android
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("app_learner")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "app_learner"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"


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


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


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

class ScreenType(str, Enum):
    """Recognized screen types in an app."""
    HOME = "home"
    LOGIN = "login"
    SIGNUP = "signup"
    SETTINGS = "settings"
    PROFILE = "profile"
    FEED = "feed"
    SEARCH = "search"
    COMPOSE = "compose"
    DETAIL = "detail"
    LIST = "list"
    MENU = "menu"
    NAV_DRAWER = "nav_drawer"
    DIALOG = "dialog"
    ONBOARDING = "onboarding"
    PERMISSION = "permission"
    LOADING = "loading"
    ERROR = "error"
    CAMERA = "camera"
    GALLERY = "gallery"
    CHECKOUT = "checkout"
    UNKNOWN = "unknown"


class ElementType(str, Enum):
    """UI element types."""
    BUTTON = "button"
    TEXT_FIELD = "text_field"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    SWITCH = "switch"
    LINK = "link"
    IMAGE = "image"
    ICON = "icon"
    TAB = "tab"
    MENU_ITEM = "menu_item"
    LIST_ITEM = "list_item"
    BACK_BUTTON = "back_button"
    NAV_ITEM = "nav_item"
    FAB = "fab"  # floating action button
    SCROLL_VIEW = "scroll_view"
    UNKNOWN = "unknown"


class ExploreStrategy(str, Enum):
    """Exploration strategies for BFS/DFS."""
    BFS = "bfs"
    DFS = "dfs"
    RANDOM = "random"
    TARGETED = "targeted"


class PlaybookStepType(str, Enum):
    """Types of steps in a playbook."""
    TAP = "tap"
    TYPE_TEXT = "type_text"
    SWIPE = "swipe"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    VERIFY = "verify"
    BACK = "back"
    HOME = "home"
    SCROLL = "scroll"
    LONG_PRESS = "long_press"
    CONDITIONAL = "conditional"


@dataclass
class UIElement:
    """A UI element detected on a screen."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    element_type: ElementType = ElementType.UNKNOWN
    text: str = ""
    description: str = ""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    tappable: bool = True
    confidence: float = 0.5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["element_type"] = self.element_type.value
        return d


@dataclass
class ScreenNode:
    """A screen (state) in the navigation graph."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    screen_type: ScreenType = ScreenType.UNKNOWN
    title: str = ""
    description: str = ""
    elements: List[UIElement] = field(default_factory=list)
    screenshot_hash: str = ""
    screenshot_path: str = ""
    package: str = ""
    activity: str = ""
    visit_count: int = 0
    first_seen: str = field(default_factory=_now_iso)
    last_seen: str = field(default_factory=_now_iso)
    confidence: float = 0.5

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "screen_type": self.screen_type.value,
            "title": self.title,
            "description": self.description,
            "elements": [e.to_dict() for e in self.elements],
            "screenshot_hash": self.screenshot_hash,
            "package": self.package,
            "activity": self.activity,
            "visit_count": self.visit_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "confidence": self.confidence,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ScreenNode":
        elements = [UIElement(**e) if isinstance(e, dict) else e for e in data.pop("elements", [])]
        for e in elements:
            if isinstance(e, UIElement) and isinstance(e.element_type, str):
                e.element_type = ElementType(e.element_type)
        st = data.pop("screen_type", "unknown")
        node = cls(screen_type=ScreenType(st), elements=elements, **data)
        return node


@dataclass
class NavigationEdge:
    """An edge (transition) between two screens."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    from_screen: str = ""
    to_screen: str = ""
    action: str = ""  # "tap", "swipe", "back", etc.
    target_element: str = ""  # description or element_id
    target_x: int = 0
    target_y: int = 0
    success_count: int = 0
    fail_count: int = 0
    created_at: str = field(default_factory=_now_iso)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success_rate"] = self.success_rate
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "NavigationEdge":
        data.pop("success_rate", None)
        return cls(**data)


@dataclass
class PlaybookStep:
    """A step in an auto-generated playbook."""
    step_type: PlaybookStepType = PlaybookStepType.TAP
    description: str = ""
    target: str = ""
    value: str = ""
    x: int = 0
    y: int = 0
    wait_seconds: float = 1.0
    expected_screen: str = ""
    fallback: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["step_type"] = self.step_type.value
        return d


@dataclass
class Playbook:
    """An auto-generated playbook for completing a task in an app."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    package: str = ""
    goal: str = ""
    description: str = ""
    steps: List[PlaybookStep] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    success_count: int = 0
    fail_count: int = 0
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "package": self.package,
            "goal": self.goal,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "confidence": self.confidence,
        }


@dataclass
class AppKnowledge:
    """Complete knowledge about an app's navigation and UI."""
    package: str = ""
    app_name: str = ""
    screens: Dict[str, ScreenNode] = field(default_factory=dict)
    edges: List[NavigationEdge] = field(default_factory=list)
    playbooks: Dict[str, Playbook] = field(default_factory=dict)
    patterns: Dict[str, Any] = field(default_factory=dict)
    explore_count: int = 0
    last_explored: str = ""
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "package": self.package,
            "app_name": self.app_name,
            "screens": {k: v.to_dict() for k, v in self.screens.items()},
            "edges": [e.to_dict() for e in self.edges],
            "playbooks": {k: v.to_dict() for k, v in self.playbooks.items()},
            "patterns": self.patterns,
            "explore_count": self.explore_count,
            "last_explored": self.last_explored,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppKnowledge":
        screens = {}
        for k, v in data.get("screens", {}).items():
            screens[k] = ScreenNode.from_dict(v) if isinstance(v, dict) else v
        edges = [
            NavigationEdge.from_dict(e) if isinstance(e, dict) else e
            for e in data.get("edges", [])
        ]
        playbooks = {}
        for k, v in data.get("playbooks", {}).items():
            if isinstance(v, dict):
                steps = [PlaybookStep(**s) if isinstance(s, dict) else s for s in v.pop("steps", [])]
                playbooks[k] = Playbook(steps=steps, **v)
            else:
                playbooks[k] = v
        return cls(
            package=data.get("package", ""),
            app_name=data.get("app_name", ""),
            screens=screens,
            edges=edges,
            playbooks=playbooks,
            patterns=data.get("patterns", {}),
            explore_count=data.get("explore_count", 0),
            last_explored=data.get("last_explored", ""),
            created_at=data.get("created_at", _now_iso()),
        )


# Screen pattern keywords for classification
SCREEN_PATTERNS = {
    ScreenType.LOGIN: ["log in", "login", "sign in", "email", "password", "username", "forgot password"],
    ScreenType.SIGNUP: ["sign up", "create account", "register", "join", "get started"],
    ScreenType.SETTINGS: ["settings", "preferences", "configuration", "account settings"],
    ScreenType.PROFILE: ["profile", "edit profile", "my account", "bio", "followers", "following"],
    ScreenType.FEED: ["feed", "for you", "timeline", "home", "discover"],
    ScreenType.SEARCH: ["search", "explore", "find", "browse"],
    ScreenType.COMPOSE: ["compose", "new post", "create", "write", "share", "upload"],
    ScreenType.ONBOARDING: ["welcome", "next", "skip", "tutorial", "get started"],
    ScreenType.PERMISSION: ["allow", "deny", "permission", "access", "location", "camera", "microphone"],
    ScreenType.LOADING: ["loading", "please wait", "connecting"],
    ScreenType.ERROR: ["error", "something went wrong", "try again", "retry", "oops"],
    ScreenType.CAMERA: ["camera", "photo", "video", "capture", "record"],
    ScreenType.GALLERY: ["gallery", "photos", "albums", "camera roll"],
    ScreenType.CHECKOUT: ["checkout", "payment", "cart", "order", "buy"],
}


# ---------------------------------------------------------------------------
# AppLearner
# ---------------------------------------------------------------------------

class AppLearner:
    """
    Self-teaching app navigator that explores Android apps via BFS,
    builds navigation graphs, and generates playbooks.

    Usage:
        learner = get_app_learner()
        knowledge = await learner.explore_app("com.instagram.android")
        path = learner.find_path("com.instagram.android", "home", "settings")
    """

    def __init__(
        self,
        controller: Any = None,
        vision: Any = None,
        memory: Any = None,
        data_dir: Optional[Path] = None,
    ):
        self._controller = controller
        self._vision = vision
        self._memory = memory
        self._data_dir = data_dir or DATA_DIR
        self._knowledge_dir = self._data_dir / "knowledge"
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)

        self._apps: Dict[str, AppKnowledge] = {}
        self._current_screen: Optional[ScreenNode] = None
        self._exploration_active: bool = False
        self._correction_log: List[Dict[str, Any]] = []

        self._load_all_knowledge()
        logger.info("AppLearner initialized (%d apps known)", len(self._apps))

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
    def memory(self):
        if self._memory is None:
            try:
                from src.agent_memory import get_memory
                self._memory = get_memory()
            except ImportError:
                logger.debug("AgentMemory not available")
        return self._memory

    # ── Persistence ──

    def _load_all_knowledge(self) -> None:
        """Load all app knowledge files from disk."""
        for path in self._knowledge_dir.glob("*.json"):
            try:
                data = _load_json(path)
                if data and data.get("package"):
                    self._apps[data["package"]] = AppKnowledge.from_dict(data)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path.name, exc)

    def _save_knowledge(self, package: str) -> None:
        """Save knowledge for a specific app."""
        if package in self._apps:
            path = self._knowledge_dir / f"{package}.json"
            _save_json(path, self._apps[package].to_dict())

    def get_knowledge(self, package: str) -> Optional[AppKnowledge]:
        """Get knowledge for a specific app."""
        return self._apps.get(package)

    def list_known_apps(self) -> List[Dict[str, Any]]:
        """List all apps we have knowledge about."""
        return [
            {
                "package": k.package,
                "app_name": k.app_name,
                "screens": len(k.screens),
                "edges": len(k.edges),
                "playbooks": len(k.playbooks),
                "explore_count": k.explore_count,
                "last_explored": k.last_explored,
            }
            for k in self._apps.values()
        ]

    # ── Screen analysis ──

    async def _capture_and_analyze(self) -> Tuple[str, dict]:
        """Take a screenshot and analyze it with vision."""
        screenshot = await self.controller.screenshot()
        if self.vision:
            analysis = await self.vision.analyze_screen(screenshot_path=screenshot)
            if not isinstance(analysis, dict):
                analysis = {"raw": str(analysis)}
        else:
            analysis = {}
        return screenshot, analysis

    def _classify_screen(self, analysis: dict) -> ScreenType:
        """Classify a screen based on its visible content."""
        text = ""
        if isinstance(analysis, dict):
            visible = analysis.get("visible_text", "")
            if isinstance(visible, list):
                text = " ".join(visible).lower()
            else:
                text = str(visible).lower()
            # Also check element descriptions
            for elem in analysis.get("tappable_elements", []):
                if isinstance(elem, dict):
                    text += " " + str(elem.get("text", "")).lower()
                    text += " " + str(elem.get("description", "")).lower()

        best_type = ScreenType.UNKNOWN
        best_score = 0

        for screen_type, keywords in SCREEN_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_type = screen_type

        return best_type

    def _extract_elements(self, analysis: dict) -> List[UIElement]:
        """Extract UI elements from vision analysis."""
        elements = []
        if not isinstance(analysis, dict):
            return elements

        for elem_data in analysis.get("tappable_elements", []):
            if not isinstance(elem_data, dict):
                continue
            text = str(elem_data.get("text", ""))
            desc = str(elem_data.get("description", ""))
            combined = (text + " " + desc).lower()

            # Classify element type
            etype = ElementType.UNKNOWN
            if any(kw in combined for kw in ["button", "btn", "submit", "ok", "cancel"]):
                etype = ElementType.BUTTON
            elif any(kw in combined for kw in ["input", "text field", "edit text", "type"]):
                etype = ElementType.TEXT_FIELD
            elif any(kw in combined for kw in ["checkbox", "check box"]):
                etype = ElementType.CHECKBOX
            elif any(kw in combined for kw in ["switch", "toggle"]):
                etype = ElementType.SWITCH
            elif any(kw in combined for kw in ["link", "href", "http"]):
                etype = ElementType.LINK
            elif any(kw in combined for kw in ["tab", "bottom nav"]):
                etype = ElementType.TAB
            elif any(kw in combined for kw in ["menu", "option"]):
                etype = ElementType.MENU_ITEM
            elif any(kw in combined for kw in ["back", "arrow left", "navigate up"]):
                etype = ElementType.BACK_BUTTON
            elif any(kw in combined for kw in ["fab", "floating", "plus", "add"]):
                etype = ElementType.FAB
            elif any(kw in combined for kw in ["image", "photo", "picture"]):
                etype = ElementType.IMAGE
            elif any(kw in combined for kw in ["icon"]):
                etype = ElementType.ICON

            ui_elem = UIElement(
                element_type=etype,
                text=text,
                description=desc,
                x=elem_data.get("x", 0),
                y=elem_data.get("y", 0),
                width=elem_data.get("width", 0),
                height=elem_data.get("height", 0),
                tappable=elem_data.get("tappable", True),
                confidence=elem_data.get("confidence", 0.5),
            )
            elements.append(ui_elem)

        return elements

    def _compute_screen_hash(self, analysis: dict) -> str:
        """Compute a hash to identify similar screens."""
        text = ""
        if isinstance(analysis, dict):
            visible = analysis.get("visible_text", "")
            if isinstance(visible, list):
                text = "|".join(sorted(visible))
            else:
                text = str(visible)
            # Include element descriptions
            elem_texts = []
            for elem in analysis.get("tappable_elements", []):
                if isinstance(elem, dict):
                    elem_texts.append(str(elem.get("text", "")))
            text += "|" + "|".join(sorted(elem_texts))
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _find_matching_screen(self, package: str, screen_hash: str) -> Optional[ScreenNode]:
        """Find a screen node that matches the given hash."""
        knowledge = self._apps.get(package)
        if not knowledge:
            return None
        for node in knowledge.screens.values():
            if node.screenshot_hash == screen_hash:
                return node
        return None

    async def _identify_current_screen(self, package: str) -> ScreenNode:
        """Capture, analyze, and identify the current screen."""
        screenshot, analysis = await self._capture_and_analyze()
        screen_hash = self._compute_screen_hash(analysis)

        # Check if we've seen this screen before
        existing = self._find_matching_screen(package, screen_hash)
        if existing:
            existing.visit_count += 1
            existing.last_seen = _now_iso()
            self._current_screen = existing
            return existing

        # New screen
        screen_type = self._classify_screen(analysis)
        elements = self._extract_elements(analysis)

        # Get current activity
        activity = ""
        try:
            output = await self.controller._adb_shell(
                "dumpsys activity activities | grep mResumedActivity"
            )
            match = re.search(r'(\S+)/(\S+)', output)
            if match:
                activity = match.group(2)
        except Exception:
            pass

        title = ""
        if isinstance(analysis, dict):
            title = analysis.get("title", "") or analysis.get("current_screen", "")

        node = ScreenNode(
            screen_type=screen_type,
            title=title or screen_type.value,
            description=str(analysis.get("description", ""))[:200] if isinstance(analysis, dict) else "",
            elements=elements,
            screenshot_hash=screen_hash,
            screenshot_path=screenshot,
            package=package,
            activity=activity,
            visit_count=1,
        )

        # Store in knowledge
        if package not in self._apps:
            self._apps[package] = AppKnowledge(package=package)
        self._apps[package].screens[node.id] = node
        self._current_screen = node

        logger.info("New screen discovered: %s (%s) in %s",
                     node.title, node.screen_type.value, package)
        return node

    # ── Exploration ──

    async def explore_app(
        self,
        package: str,
        max_steps: int = 50,
        strategy: ExploreStrategy = ExploreStrategy.BFS,
        explore_depth: int = 5,
    ) -> AppKnowledge:
        """
        Explore an app by navigating through screens.

        Uses BFS exploration: start from current screen, tap each element,
        observe the new screen, record the transition, then backtrack.

        Args:
            package: App package name.
            max_steps: Maximum number of actions to take.
            strategy: Exploration strategy (BFS, DFS, RANDOM).
            explore_depth: Maximum depth for exploration.

        Returns:
            Updated AppKnowledge for the app.
        """
        if package not in self._apps:
            self._apps[package] = AppKnowledge(package=package)

        knowledge = self._apps[package]
        self._exploration_active = True

        # Ensure app is running
        try:
            current = await self.controller.get_current_app()
            if current != package:
                await self.controller.launch_app(package)
                await asyncio.sleep(3.0)
        except Exception as exc:
            logger.error("Failed to launch %s: %s", package, exc)
            return knowledge

        steps_taken = 0
        visited_hashes: Set[str] = set()
        explore_queue: Deque[Tuple[str, int]] = deque()  # (screen_id, depth)

        # Identify starting screen
        start_screen = await self._identify_current_screen(package)
        visited_hashes.add(start_screen.screenshot_hash)
        explore_queue.append((start_screen.id, 0))

        logger.info("Starting %s exploration of %s (max %d steps)",
                     strategy.value, package, max_steps)

        while explore_queue and steps_taken < max_steps and self._exploration_active:
            screen_id, depth = explore_queue.popleft()
            if depth >= explore_depth:
                continue

            screen = knowledge.screens.get(screen_id)
            if not screen:
                continue

            # Try tapping each unexplored element
            for element in screen.elements:
                if steps_taken >= max_steps:
                    break
                if not element.tappable:
                    continue
                # Skip back buttons during exploration (could leave the app)
                if element.element_type == ElementType.BACK_BUTTON:
                    continue

                try:
                    # Navigate to this screen first (if not current)
                    if self._current_screen and self._current_screen.id != screen_id:
                        # Try to get back to the target screen
                        path = self._find_path_internal(package, self._current_screen.id, screen_id)
                        if not path:
                            continue
                        for edge in path:
                            await self._execute_edge(edge)
                            steps_taken += 1

                    # Tap the element
                    logger.debug("Step %d: Tap '%s' at (%d, %d)",
                                 steps_taken, element.text or element.description,
                                 element.x, element.y)
                    await self.controller.tap(element.x, element.y)
                    await asyncio.sleep(1.5)
                    steps_taken += 1

                    # Identify the new screen
                    new_screen = await self._identify_current_screen(package)

                    # Record the edge
                    edge = NavigationEdge(
                        from_screen=screen_id,
                        to_screen=new_screen.id,
                        action="tap",
                        target_element=element.text or element.description,
                        target_x=element.x,
                        target_y=element.y,
                        success_count=1,
                    )
                    knowledge.edges.append(edge)

                    # Add to explore queue if new
                    if new_screen.screenshot_hash not in visited_hashes:
                        visited_hashes.add(new_screen.screenshot_hash)
                        explore_queue.append((new_screen.id, depth + 1))

                    # Go back to continue exploring from the same screen
                    await self.controller.press_back()
                    await asyncio.sleep(1.0)
                    steps_taken += 1

                    # Re-identify (might not be back on the same screen)
                    back_screen = await self._identify_current_screen(package)
                    if back_screen.id != screen_id:
                        # Record back edge
                        back_edge = NavigationEdge(
                            from_screen=new_screen.id,
                            to_screen=back_screen.id,
                            action="back",
                            success_count=1,
                        )
                        knowledge.edges.append(back_edge)
                        # We ended up somewhere else — add it to the queue
                        if back_screen.screenshot_hash not in visited_hashes:
                            visited_hashes.add(back_screen.screenshot_hash)
                            explore_queue.append((back_screen.id, depth))
                        break  # Can't continue from the original screen

                except Exception as exc:
                    logger.warning("Exploration step failed: %s", exc)
                    steps_taken += 1

        knowledge.explore_count += 1
        knowledge.last_explored = _now_iso()
        self._exploration_active = False

        # Detect patterns
        self._detect_patterns(package)

        # Save
        self._save_knowledge(package)

        # Store in memory if available
        if self.memory:
            try:
                self.memory.store_sync(
                    content=f"Explored {package}: {len(knowledge.screens)} screens, "
                            f"{len(knowledge.edges)} transitions, {steps_taken} steps",
                    memory_type="task_result",
                    tags=["exploration", package],
                )
            except Exception:
                pass

        logger.info("Exploration complete: %d screens, %d edges, %d steps",
                     len(knowledge.screens), len(knowledge.edges), steps_taken)
        return knowledge

    async def _execute_edge(self, edge: NavigationEdge) -> bool:
        """Execute a navigation edge (tap, back, etc.)."""
        try:
            if edge.action == "tap":
                await self.controller.tap(edge.target_x, edge.target_y)
            elif edge.action == "back":
                await self.controller.press_back()
            elif edge.action == "swipe":
                # Default swipe down
                await self.controller.scroll_down(500)
            elif edge.action == "home":
                await self.controller.press_home()
            await asyncio.sleep(1.0)
            edge.success_count += 1
            return True
        except Exception as exc:
            edge.fail_count += 1
            logger.warning("Edge execution failed: %s", exc)
            return False

    def stop_exploration(self) -> None:
        """Stop an ongoing exploration."""
        self._exploration_active = False
        logger.info("Exploration stop requested")

    # ── Pattern detection ──

    def _detect_patterns(self, package: str) -> Dict[str, Any]:
        """Detect UI patterns in an app's screens."""
        knowledge = self._apps.get(package)
        if not knowledge:
            return {}

        patterns = {}

        # Detect bottom navigation bar
        bottom_nav_screens = []
        for screen in knowledge.screens.values():
            bottom_tabs = [
                e for e in screen.elements
                if e.element_type == ElementType.TAB and e.y > 1600
            ]
            if len(bottom_tabs) >= 3:
                bottom_nav_screens.append(screen.id)
                patterns["bottom_nav"] = {
                    "tabs": [{"text": t.text, "x": t.x, "y": t.y} for t in bottom_tabs],
                    "screen_count": len(bottom_nav_screens),
                }

        # Detect nav drawer
        for screen in knowledge.screens.values():
            if screen.screen_type == ScreenType.NAV_DRAWER:
                menu_items = [
                    e for e in screen.elements
                    if e.element_type == ElementType.MENU_ITEM
                ]
                patterns["nav_drawer"] = {
                    "items": [{"text": i.text, "x": i.x, "y": i.y} for i in menu_items],
                }

        # Detect common screen types
        screen_types = {}
        for screen in knowledge.screens.values():
            st = screen.screen_type.value
            screen_types[st] = screen_types.get(st, 0) + 1
        patterns["screen_type_distribution"] = screen_types

        # Detect login flow
        login_screens = [
            s for s in knowledge.screens.values()
            if s.screen_type == ScreenType.LOGIN
        ]
        if login_screens:
            patterns["has_login"] = True
            patterns["login_screen_id"] = login_screens[0].id

        knowledge.patterns = patterns
        return patterns

    # ── Pathfinding ──

    def find_path(
        self, package: str, from_screen: str, to_screen: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find a navigation path between two screens.

        Args:
            package: App package name.
            from_screen: Source screen ID or type name.
            to_screen: Target screen ID or type name.

        Returns:
            List of edge dicts describing the path, or None if no path found.
        """
        path = self._find_path_internal(package, from_screen, to_screen)
        if path:
            return [e.to_dict() for e in path]
        return None

    def _find_path_internal(
        self, package: str, from_id: str, to_id: str
    ) -> Optional[List[NavigationEdge]]:
        """Internal BFS pathfinding."""
        knowledge = self._apps.get(package)
        if not knowledge:
            return None

        # Resolve screen type names to IDs
        from_id = self._resolve_screen_id(package, from_id)
        to_id = self._resolve_screen_id(package, to_id)

        if from_id == to_id:
            return []

        # BFS
        visited = {from_id}
        queue: Deque[Tuple[str, List[NavigationEdge]]] = deque()
        queue.append((from_id, []))

        # Build adjacency list
        adj: Dict[str, List[NavigationEdge]] = {}
        for edge in knowledge.edges:
            if edge.from_screen not in adj:
                adj[edge.from_screen] = []
            adj[edge.from_screen].append(edge)

        while queue:
            current, path = queue.popleft()
            for edge in adj.get(current, []):
                if edge.to_screen in visited:
                    continue
                new_path = path + [edge]
                if edge.to_screen == to_id:
                    return new_path
                visited.add(edge.to_screen)
                queue.append((edge.to_screen, new_path))

        return None

    def _resolve_screen_id(self, package: str, screen_ref: str) -> str:
        """Resolve a screen reference (ID or type name) to a screen ID."""
        knowledge = self._apps.get(package)
        if not knowledge:
            return screen_ref

        # Direct ID match
        if screen_ref in knowledge.screens:
            return screen_ref

        # Try matching by screen type
        for sid, screen in knowledge.screens.items():
            if screen.screen_type.value == screen_ref:
                return sid
            if screen.title.lower() == screen_ref.lower():
                return sid

        return screen_ref

    # ── Playbook generation ──

    def generate_playbook(
        self, package: str, goal: str, description: str = ""
    ) -> Optional[Playbook]:
        """
        Generate a playbook from the navigation graph for a given goal.

        The goal should describe what screens to navigate to. For example:
        - "post_photo": Find path to compose/upload screen
        - "login": Find path to login screen and fill fields
        - "settings": Find path to settings screen

        Args:
            package: App package name.
            goal: Goal description (mapped to target screen types).
            description: Optional detailed description.

        Returns:
            Playbook with step-by-step instructions.
        """
        knowledge = self._apps.get(package)
        if not knowledge:
            logger.warning("No knowledge for %s", package)
            return None

        # Map goal to target screen type
        goal_screen_map = {
            "login": ScreenType.LOGIN,
            "signup": ScreenType.SIGNUP,
            "register": ScreenType.SIGNUP,
            "settings": ScreenType.SETTINGS,
            "profile": ScreenType.PROFILE,
            "search": ScreenType.SEARCH,
            "compose": ScreenType.COMPOSE,
            "post": ScreenType.COMPOSE,
            "post_photo": ScreenType.COMPOSE,
            "upload": ScreenType.COMPOSE,
            "camera": ScreenType.CAMERA,
            "gallery": ScreenType.GALLERY,
        }

        target_type = goal_screen_map.get(goal.lower())
        target_screen = None

        if target_type:
            for screen in knowledge.screens.values():
                if screen.screen_type == target_type:
                    target_screen = screen
                    break

        if not target_screen:
            # Try fuzzy matching by title
            for screen in knowledge.screens.values():
                if goal.lower() in screen.title.lower():
                    target_screen = screen
                    break

        if not target_screen:
            logger.warning("No screen found for goal '%s' in %s", goal, package)
            return None

        # Find path from home/first screen to target
        home_screen = None
        for screen in knowledge.screens.values():
            if screen.screen_type in (ScreenType.HOME, ScreenType.FEED):
                home_screen = screen
                break
        if not home_screen and knowledge.screens:
            home_screen = next(iter(knowledge.screens.values()))

        if not home_screen:
            return None

        path = self._find_path_internal(package, home_screen.id, target_screen.id)
        if path is None:
            logger.warning("No path found from %s to %s", home_screen.id, target_screen.id)
            return None

        # Convert path to playbook steps
        steps = []
        for edge in path:
            step = PlaybookStep(
                step_type=PlaybookStepType.TAP if edge.action == "tap" else PlaybookStepType.BACK,
                description=f"{edge.action}: {edge.target_element}" if edge.target_element else edge.action,
                target=edge.target_element,
                x=edge.target_x,
                y=edge.target_y,
                wait_seconds=1.5,
                expected_screen=edge.to_screen,
            )
            steps.append(step)

        playbook = Playbook(
            package=package,
            goal=goal,
            description=description or f"Navigate to {goal} in {knowledge.app_name or package}",
            steps=steps,
        )

        knowledge.playbooks[goal] = playbook
        self._save_knowledge(package)

        logger.info("Generated playbook '%s' for %s: %d steps", goal, package, len(steps))
        return playbook

    async def execute_playbook(
        self, package: str, goal: str
    ) -> Dict[str, Any]:
        """
        Execute a previously generated playbook.

        Args:
            package: App package name.
            goal: The playbook goal to execute.

        Returns:
            Dict with success status and step results.
        """
        knowledge = self._apps.get(package)
        if not knowledge:
            return {"success": False, "error": f"No knowledge for {package}"}

        playbook = knowledge.playbooks.get(goal)
        if not playbook:
            return {"success": False, "error": f"No playbook for '{goal}'"}

        # Ensure app is running
        try:
            current = await self.controller.get_current_app()
            if current != package:
                await self.controller.launch_app(package)
                await asyncio.sleep(3.0)
        except Exception as exc:
            return {"success": False, "error": f"Failed to launch app: {exc}"}

        step_results = []
        for i, step in enumerate(playbook.steps):
            try:
                if step.step_type == PlaybookStepType.TAP:
                    await self.controller.tap(step.x, step.y)
                elif step.step_type == PlaybookStepType.BACK:
                    await self.controller.press_back()
                elif step.step_type == PlaybookStepType.TYPE_TEXT:
                    await self.controller.type_text(step.value)
                elif step.step_type == PlaybookStepType.SWIPE:
                    await self.controller.scroll_down(500)
                elif step.step_type == PlaybookStepType.WAIT:
                    await asyncio.sleep(step.wait_seconds)
                elif step.step_type == PlaybookStepType.SCROLL:
                    await self.controller.scroll_down(500)
                elif step.step_type == PlaybookStepType.LONG_PRESS:
                    await self.controller.long_press(step.x, step.y)
                elif step.step_type == PlaybookStepType.HOME:
                    await self.controller.press_home()
                elif step.step_type == PlaybookStepType.SCREENSHOT:
                    await self.controller.screenshot()

                await asyncio.sleep(step.wait_seconds)
                step_results.append({"step": i, "success": True, "action": step.description})

            except Exception as exc:
                step_results.append({"step": i, "success": False, "error": str(exc)})
                playbook.fail_count += 1
                self._save_knowledge(package)
                return {"success": False, "step_results": step_results, "failed_at": i}

        playbook.success_count += 1
        self._save_knowledge(package)
        return {"success": True, "step_results": step_results}

    # ── User corrections ──

    def correct_screen_type(
        self, package: str, screen_id: str, correct_type: ScreenType
    ) -> Dict[str, Any]:
        """Apply a user correction to a screen's type classification."""
        knowledge = self._apps.get(package)
        if not knowledge:
            return {"success": False, "error": f"No knowledge for {package}"}
        screen = knowledge.screens.get(screen_id)
        if not screen:
            return {"success": False, "error": f"Screen {screen_id} not found"}

        old_type = screen.screen_type
        screen.screen_type = correct_type
        screen.confidence = 1.0  # User-corrected = high confidence

        self._correction_log.append({
            "package": package,
            "screen_id": screen_id,
            "old_type": old_type.value,
            "new_type": correct_type.value,
            "timestamp": _now_iso(),
        })

        self._save_knowledge(package)
        return {"success": True, "old_type": old_type.value, "new_type": correct_type.value}

    def correct_element(
        self, package: str, screen_id: str, element_id: str,
        correct_type: Optional[ElementType] = None,
        correct_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply a user correction to an element's classification."""
        knowledge = self._apps.get(package)
        if not knowledge:
            return {"success": False, "error": f"No knowledge for {package}"}
        screen = knowledge.screens.get(screen_id)
        if not screen:
            return {"success": False, "error": f"Screen {screen_id} not found"}

        for elem in screen.elements:
            if elem.id == element_id:
                if correct_type:
                    elem.element_type = correct_type
                if correct_text:
                    elem.text = correct_text
                elem.confidence = 1.0
                self._save_knowledge(package)
                return {"success": True, "element": elem.to_dict()}

        return {"success": False, "error": f"Element {element_id} not found"}

    # ── Statistics ──

    def stats(self, package: str = "") -> Dict[str, Any]:
        """Get statistics about learned apps."""
        if package:
            knowledge = self._apps.get(package)
            if not knowledge:
                return {"error": f"No knowledge for {package}"}
            return {
                "package": package,
                "app_name": knowledge.app_name,
                "screens": len(knowledge.screens),
                "edges": len(knowledge.edges),
                "playbooks": len(knowledge.playbooks),
                "patterns": list(knowledge.patterns.keys()),
                "explore_count": knowledge.explore_count,
                "last_explored": knowledge.last_explored,
                "screen_types": {
                    st.value: sum(1 for s in knowledge.screens.values() if s.screen_type == st)
                    for st in ScreenType if any(s.screen_type == st for s in knowledge.screens.values())
                },
            }
        return {
            "total_apps": len(self._apps),
            "total_screens": sum(len(k.screens) for k in self._apps.values()),
            "total_edges": sum(len(k.edges) for k in self._apps.values()),
            "total_playbooks": sum(len(k.playbooks) for k in self._apps.values()),
            "apps": self.list_known_apps(),
        }

    # ── Delete ──

    def delete_knowledge(self, package: str) -> Dict[str, Any]:
        """Delete all knowledge for an app."""
        if package in self._apps:
            del self._apps[package]
            path = self._knowledge_dir / f"{package}.json"
            if path.exists():
                path.unlink()
            return {"success": True, "deleted": package}
        return {"success": False, "error": f"No knowledge for {package}"}

    # ── Export/Import ──

    def export_knowledge(self, package: str, output_path: str = "") -> Dict[str, Any]:
        """Export app knowledge to a JSON file."""
        knowledge = self._apps.get(package)
        if not knowledge:
            return {"success": False, "error": f"No knowledge for {package}"}
        if not output_path:
            output_path = str(self._data_dir / f"export_{package}.json")
        _save_json(Path(output_path), knowledge.to_dict())
        return {"success": True, "path": output_path}

    def import_knowledge(self, input_path: str) -> Dict[str, Any]:
        """Import app knowledge from a JSON file."""
        data = _load_json(Path(input_path))
        if not data or not data.get("package"):
            return {"success": False, "error": "Invalid knowledge file"}
        knowledge = AppKnowledge.from_dict(data)
        self._apps[knowledge.package] = knowledge
        self._save_knowledge(knowledge.package)
        return {"success": True, "package": knowledge.package, "screens": len(knowledge.screens)}

    # ── Sync wrappers ──

    def explore_app_sync(self, package: str, **kwargs) -> AppKnowledge:
        return _run_sync(self.explore_app(package, **kwargs))

    def execute_playbook_sync(self, package: str, goal: str) -> Dict[str, Any]:
        return _run_sync(self.execute_playbook(package, goal))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[AppLearner] = None


def get_app_learner(
    controller: Any = None,
    vision: Any = None,
    memory: Any = None,
) -> AppLearner:
    """Get the singleton AppLearner instance."""
    global _instance
    if _instance is None:
        _instance = AppLearner(controller=controller, vision=vision, memory=memory)
    return _instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _cli_explore(args: argparse.Namespace) -> None:
    learner = get_app_learner()
    strategy = ExploreStrategy(args.strategy) if args.strategy else ExploreStrategy.BFS
    knowledge = learner.explore_app_sync(
        args.package, max_steps=args.steps, strategy=strategy
    )
    _print_json(knowledge.to_dict())


def _cli_knowledge(args: argparse.Namespace) -> None:
    learner = get_app_learner()
    if args.package:
        data = learner.stats(args.package)
    else:
        data = learner.stats()
    _print_json(data)


def _cli_path(args: argparse.Namespace) -> None:
    learner = get_app_learner()
    path = learner.find_path(args.package, getattr(args, "from"), args.to)
    if path:
        _print_json(path)
    else:
        print(f"No path found from '{getattr(args, 'from')}' to '{args.to}'")


def _cli_playbook(args: argparse.Namespace) -> None:
    learner = get_app_learner()
    action = args.action
    if action == "generate":
        pb = learner.generate_playbook(args.package, args.goal)
        if pb:
            _print_json(pb.to_dict())
        else:
            print(f"Could not generate playbook for '{args.goal}'")
    elif action == "run":
        result = learner.execute_playbook_sync(args.package, args.goal)
        _print_json(result)
    elif action == "list":
        knowledge = learner.get_knowledge(args.package)
        if knowledge:
            _print_json({k: v.to_dict() for k, v in knowledge.playbooks.items()})
        else:
            print(f"No knowledge for {args.package}")
    else:
        print(f"Unknown playbook action: {action}")


def _cli_correct(args: argparse.Namespace) -> None:
    learner = get_app_learner()
    if args.screen_type:
        result = learner.correct_screen_type(
            args.package, args.screen_id, ScreenType(args.screen_type)
        )
    else:
        result = learner.correct_element(
            args.package, args.screen_id, args.element_id or "",
            correct_type=ElementType(args.element_type) if args.element_type else None,
            correct_text=args.text,
        )
    _print_json(result)


def _cli_delete(args: argparse.Namespace) -> None:
    learner = get_app_learner()
    result = learner.delete_knowledge(args.package)
    _print_json(result)


def _cli_export(args: argparse.Namespace) -> None:
    learner = get_app_learner()
    result = learner.export_knowledge(args.package, args.output or "")
    _print_json(result)


def _cli_import(args: argparse.Namespace) -> None:
    learner = get_app_learner()
    result = learner.import_knowledge(args.input)
    _print_json(result)


def _cli_apps(args: argparse.Namespace) -> None:
    learner = get_app_learner()
    apps = learner.list_known_apps()
    _print_json(apps)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="app_learner",
        description="OpenClaw Empire — Self-Teaching App Navigator",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # explore
    exp = sub.add_parser("explore", help="Explore an app's UI")
    exp.add_argument("--package", required=True)
    exp.add_argument("--steps", type=int, default=50)
    exp.add_argument("--strategy", choices=["bfs", "dfs", "random"], default="bfs")
    exp.set_defaults(func=_cli_explore)

    # knowledge
    kn = sub.add_parser("knowledge", help="View app knowledge")
    kn.add_argument("--package", default="")
    kn.set_defaults(func=_cli_knowledge)

    # path
    pa = sub.add_parser("path", help="Find navigation path")
    pa.add_argument("--package", required=True)
    pa.add_argument("--from", required=True, dest="from")
    pa.add_argument("--to", required=True)
    pa.set_defaults(func=_cli_path)

    # playbook
    pb = sub.add_parser("playbook", help="Playbook management")
    pb.add_argument("action", choices=["generate", "run", "list"])
    pb.add_argument("--package", required=True)
    pb.add_argument("--goal", default="")
    pb.set_defaults(func=_cli_playbook)

    # correct
    cr = sub.add_parser("correct", help="Apply user corrections")
    cr.add_argument("--package", required=True)
    cr.add_argument("--screen-id", required=True)
    cr.add_argument("--screen-type", default=None)
    cr.add_argument("--element-id", default=None)
    cr.add_argument("--element-type", default=None)
    cr.add_argument("--text", default=None)
    cr.set_defaults(func=_cli_correct)

    # delete
    dl = sub.add_parser("delete", help="Delete app knowledge")
    dl.add_argument("--package", required=True)
    dl.set_defaults(func=_cli_delete)

    # export
    ex = sub.add_parser("export", help="Export app knowledge")
    ex.add_argument("--package", required=True)
    ex.add_argument("--output", default="")
    ex.set_defaults(func=_cli_export)

    # import
    im = sub.add_parser("import", help="Import app knowledge")
    im.add_argument("--input", required=True)
    im.set_defaults(func=_cli_import)

    # apps
    ap = sub.add_parser("apps", help="List all known apps")
    ap.set_defaults(func=_cli_apps)

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
