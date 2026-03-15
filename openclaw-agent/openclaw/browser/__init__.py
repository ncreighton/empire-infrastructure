"""Browser automation — browser-use wrapper, stealth, CAPTCHA handling."""

from openclaw.browser.browser_manager import BrowserManager
from openclaw.browser.gologin_manager import GoLoginBrowserManager
from openclaw.browser.proxy_manager import ProxyManager
from openclaw.browser.captcha_handler import CaptchaHandler
from openclaw.browser.session_manager import SessionManager
from openclaw.browser.step_router import StepRouter
from openclaw.browser.identity_manager import IdentityManager
from openclaw.browser.browsing_warmup import BrowsingWarmup
from openclaw.browser.cookie_seeder import CookieSeeder
from openclaw.browser.stealth import (
    get_browser_config,
    get_random_user_agent,
    get_random_viewport,
    get_stealth_args,
    add_human_delays,
    randomize_delay,
)

__all__ = [
    "BrowserManager",
    "GoLoginBrowserManager",
    "IdentityManager",
    "BrowsingWarmup",
    "CookieSeeder",
    "ProxyManager",
    "CaptchaHandler",
    "SessionManager",
    "StepRouter",
    "get_browser_config",
    "get_random_user_agent",
    "get_random_viewport",
    "get_stealth_args",
    "add_human_delays",
    "randomize_delay",
]
