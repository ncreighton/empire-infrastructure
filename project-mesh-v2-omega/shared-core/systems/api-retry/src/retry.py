"""
api-retry -- Shared retry logic with exponential backoff.
Extracted from common patterns across Grimoire, VideoForge, Dashboard.

Provides:
- @with_retry decorator for any function
- RetryConfig dataclass for configuration
- api_request() for HTTP requests with retry + rate limit handling
- async variants for FastAPI services
"""

import time
import logging
import functools
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Tuple, Type, Set

log = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    retryable_status_codes: Set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504}
    )
    timeout: int = 30
    on_retry: Optional[Callable] = None

    @classmethod
    def fast(cls) -> "RetryConfig":
        """Fast retry config for quick operations."""
        return cls(max_retries=2, base_delay=0.5, max_delay=5.0)

    @classmethod
    def patient(cls) -> "RetryConfig":
        """Patient retry config for slow APIs (ElevenLabs, Creatomate, FAL)."""
        return cls(max_retries=4, base_delay=2.0, max_delay=60.0, timeout=120)

    @classmethod
    def aggressive(cls) -> "RetryConfig":
        """Aggressive retry for critical operations."""
        return cls(max_retries=5, base_delay=1.0, max_delay=120.0, backoff_factor=3.0)


def _compute_delay(config: RetryConfig, attempt: int,
                   response=None) -> float:
    """Compute delay for next retry, respecting Retry-After header."""
    delay = min(
        config.base_delay * (config.backoff_factor ** attempt),
        config.max_delay
    )
    # Honor Retry-After header from 429 responses
    if response is not None and hasattr(response, "headers"):
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                delay = min(float(retry_after), config.max_delay)
            except (ValueError, TypeError):
                pass
    return delay


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
    config: Optional[RetryConfig] = None,
):
    """Decorator for retry with exponential backoff.

    Usage:
        @with_retry(max_retries=3)
        def call_api():
            ...

        @with_retry(config=RetryConfig.patient())
        def slow_api_call():
            ...
    """
    if config is None:
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            retryable_exceptions=retryable_exceptions,
            on_retry=on_retry,
        )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exc = e
                    if attempt == config.max_retries:
                        log.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, config.max_retries + 1, e
                        )
                        raise
                    delay = _compute_delay(config, attempt)
                    log.warning(
                        "%s attempt %d failed: %s. Retrying in %.1fs",
                        func.__name__, attempt + 1, e, delay
                    )
                    if config.on_retry:
                        config.on_retry(attempt, e, delay)
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


def api_request(
    method: str,
    url: str,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> "requests.Response":
    """Make an HTTP request with retry logic.

    Handles rate limiting (429), server errors (5xx), timeouts, and
    connection errors with exponential backoff.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        url: Request URL
        config: Optional RetryConfig (uses defaults if not provided)
        **kwargs: Passed to requests.request()

    Returns:
        requests.Response object

    Raises:
        requests.HTTPError: After all retries exhausted
        requests.ConnectionError: After all retries exhausted
        requests.Timeout: After all retries exhausted
    """
    import requests

    if config is None:
        config = RetryConfig()

    # Set timeout from config if not explicitly provided
    if "timeout" not in kwargs:
        kwargs["timeout"] = config.timeout

    last_exc = None
    for attempt in range(config.max_retries + 1):
        try:
            resp = requests.request(method, url, **kwargs)

            if resp.status_code in config.retryable_status_codes:
                if attempt == config.max_retries:
                    resp.raise_for_status()
                delay = _compute_delay(config, attempt, resp)
                log.warning(
                    "HTTP %d from %s, retrying in %.1fs",
                    resp.status_code, url, delay
                )
                time.sleep(delay)
                continue

            return resp

        except requests.exceptions.ConnectionError as e:
            last_exc = e
            if attempt == config.max_retries:
                raise
            delay = _compute_delay(config, attempt)
            log.warning(
                "Connection error to %s, retrying in %.1fs: %s",
                url, delay, e
            )
            time.sleep(delay)

        except requests.exceptions.Timeout as e:
            last_exc = e
            if attempt == config.max_retries:
                raise
            delay = _compute_delay(config, attempt)
            log.warning(
                "Timeout from %s, retrying in %.1fs", url, delay
            )
            time.sleep(delay)

    raise last_exc


def api_get(url: str, config: Optional[RetryConfig] = None,
            **kwargs) -> "requests.Response":
    """Convenience wrapper for GET requests with retry."""
    return api_request("GET", url, config=config, **kwargs)


def api_post(url: str, config: Optional[RetryConfig] = None,
             **kwargs) -> "requests.Response":
    """Convenience wrapper for POST requests with retry."""
    return api_request("POST", url, config=config, **kwargs)
