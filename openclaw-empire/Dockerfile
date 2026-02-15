# ===========================================================================
# OpenClaw Empire API Server — Multi-stage Docker Build
# ===========================================================================
# Python 3.12 slim base, non-root user, health check, minimal image size.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build-time system dependencies (for any compiled Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python packages into a prefix we can copy later
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean production image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="Nick Creighton <ncreighton@openclaw.empire>"
LABEL description="OpenClaw Empire API Server — 58+ endpoint intelligence system"

WORKDIR /app

# Install only runtime system dependencies (curl for health check)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd --gid 1000 openclaw && \
    useradd --uid 1000 --gid openclaw --shell /bin/bash --create-home openclaw

# Copy application source code
COPY src/ /app/src/
COPY skills/ /app/skills/
COPY workspace/ /app/workspace/

# Create data directories (will be overridden by volume mounts, but ensures
# the paths exist if no volume is attached)
RUN mkdir -p /app/data/forge /app/data/amplify /app/data/scheduler \
             /app/data/screenshots /app/data/vision \
             /app/configs /app/dashboard && \
    chown -R openclaw:openclaw /app

# Switch to non-root user
USER openclaw

# Expose API port
EXPOSE 8765

# Health check — hit the /health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Default environment (overridable at runtime)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENCLAW_API_PORT=8765

# Entrypoint — run the FastAPI server via uvicorn
ENTRYPOINT ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8765"]
