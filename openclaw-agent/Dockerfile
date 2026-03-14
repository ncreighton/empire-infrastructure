FROM python:3.12-slim

WORKDIR /app

# Install system deps for Playwright + cryptography compilation
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    libssl-dev \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (provides uvx, required by browser-use for extensions)
RUN pip install --no-cache-dir uv

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright system deps first (needs root)
RUN playwright install-deps chromium

# Create non-root user
RUN useradd -m -u 1000 openclaw

# Set Playwright browsers path to user-accessible location
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.playwright-browsers

# Install Playwright Chromium to the shared path
RUN mkdir -p /app/.playwright-browsers && \
    playwright install chromium && \
    chown -R openclaw:openclaw /app/.playwright-browsers

# Copy application
COPY . .

# Create data directories + GoLogin browser/profile dirs
RUN mkdir -p /app/data/screenshots /app/data/sessions /app/logs \
    /home/openclaw/.gologin/browser /home/openclaw/.gologin/profiles && \
    chown -R openclaw:openclaw /app /home/openclaw/.gologin

USER openclaw

# Expose port
EXPOSE 8100

# Health check using curl (lighter than Python import)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8100"]
