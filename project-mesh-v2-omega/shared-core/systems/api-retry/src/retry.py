"""
api-retry — Shared retry logic with exponential backoff.
Extracted from common patterns across Grimoire, VideoForge, Dashboard.
"""

import time
import logging
import functools
from typing import Optional, Callable, Any, Tuple, Type

log = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_BACKOFF_FACTOR = 2.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exc = e
                    if attempt == max_retries:
                        log.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    log.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                    if on_retry:
                        on_retry(attempt, e, delay)
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


def api_request(
    method: str,
    url: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = 30,
    **kwargs
) -> "requests.Response":
    """Make an HTTP request with retry logic. Requires `requests` package."""
    import requests

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code in RETRYABLE_STATUS_CODES:
                if attempt == max_retries:
                    resp.raise_for_status()
                delay = min(DEFAULT_BASE_DELAY * (DEFAULT_BACKOFF_FACTOR ** attempt), DEFAULT_MAX_DELAY)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        delay = min(float(retry_after), DEFAULT_MAX_DELAY)
                log.warning(f"HTTP {resp.status_code} from {url}, retrying in {delay:.1f}s")
                time.sleep(delay)
                continue
            return resp
        except requests.exceptions.ConnectionError as e:
            last_exc = e
            if attempt == max_retries:
                raise
            delay = min(DEFAULT_BASE_DELAY * (DEFAULT_BACKOFF_FACTOR ** attempt), DEFAULT_MAX_DELAY)
            log.warning(f"Connection error to {url}, retrying in {delay:.1f}s: {e}")
            time.sleep(delay)
        except requests.exceptions.Timeout as e:
            last_exc = e
            if attempt == max_retries:
                raise
            delay = min(DEFAULT_BASE_DELAY * (DEFAULT_BACKOFF_FACTOR ** attempt), DEFAULT_MAX_DELAY)
            log.warning(f"Timeout from {url}, retrying in {delay:.1f}s")
            time.sleep(delay)
    raise last_exc
