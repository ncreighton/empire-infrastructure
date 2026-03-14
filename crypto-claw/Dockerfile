FROM python:3.12-slim

WORKDIR /app

RUN groupadd -r moneyclaw && useradd -r -g moneyclaw -s /sbin/nologin moneyclaw

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data && chown -R moneyclaw:moneyclaw /app/data

USER moneyclaw

EXPOSE 8110

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8110/health'); assert r.status_code == 200"

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8110"]
