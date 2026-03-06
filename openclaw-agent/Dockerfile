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

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application
COPY . .

# Create data directories
RUN mkdir -p /app/data/screenshots /app/data/sessions /app/logs

# Create non-root user
RUN useradd -m -u 1000 openclaw && chown -R openclaw:openclaw /app
USER openclaw

# Expose port
EXPOSE 8100

# Health check using curl (lighter than Python import)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8100"]
