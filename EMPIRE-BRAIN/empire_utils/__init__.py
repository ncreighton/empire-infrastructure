"""Empire Utils — Shared base classes for all Empire intelligence systems.

Eliminates ~2,600 lines of duplicated boilerplate across 16+ projects.

Usage:
    from empire_utils import create_empire_app, BaseSQLiteCodex
    from empire_utils.forge_base import BaseScout, BaseSentinel, BaseOracle, BaseSmith
    from empire_utils.amplify_base import BaseAmplifyPipeline
"""

from empire_utils.fastapi_boot import create_empire_app, to_dict
from empire_utils.sqlite_codex import BaseSQLiteCodex, content_hash, get_db

__all__ = [
    "create_empire_app",
    "to_dict",
    "BaseSQLiteCodex",
    "content_hash",
    "get_db",
]
