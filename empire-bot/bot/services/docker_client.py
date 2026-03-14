"""Docker compose operations for VPS container management."""

import asyncio
import logging

logger = logging.getLogger(__name__)

COMPOSE_DIR = "/opt/empire"
COMPOSE_CMD = "docker compose"


async def _run(cmd: str, timeout: float = 30) -> tuple[int, str]:
    """Run a shell command and return (returncode, output)."""
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=COMPOSE_DIR,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, stdout.decode(errors="replace").strip()
    except asyncio.TimeoutError:
        return 1, "Command timed out"
    except Exception as e:
        return 1, str(e)


async def ps() -> tuple[int, str]:
    """List running containers."""
    return await _run(f"{COMPOSE_CMD} ps --format 'table {{{{.Name}}}}\\t{{{{.Status}}}}\\t{{{{.Ports}}}}'")


async def restart(service: str) -> tuple[int, str]:
    """Restart a specific service."""
    # Sanitize service name
    safe = "".join(c for c in service if c.isalnum() or c in "-_")
    return await _run(f"{COMPOSE_CMD} restart {safe}", timeout=60)


async def logs(service: str, lines: int = 30) -> tuple[int, str]:
    """Get recent logs for a service."""
    safe = "".join(c for c in service if c.isalnum() or c in "-_")
    return await _run(f"{COMPOSE_CMD} logs --tail={lines} --no-color {safe}")


async def up(service: str) -> tuple[int, str]:
    """Start a specific service."""
    safe = "".join(c for c in service if c.isalnum() or c in "-_")
    return await _run(f"{COMPOSE_CMD} up -d {safe}", timeout=60)


async def down(service: str) -> tuple[int, str]:
    """Stop a specific service."""
    safe = "".join(c for c in service if c.isalnum() or c in "-_")
    return await _run(f"{COMPOSE_CMD} stop {safe}", timeout=60)
