-- ═══════════════════════════════════════════════════════════════
-- EMPIRE AI ARSENAL — PostgreSQL Init
-- Creates databases for each service + enables extensions
-- ═══════════════════════════════════════════════════════════════

-- Enable extensions on default database
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Service databases
CREATE DATABASE litellm;
CREATE DATABASE langfuse;
CREATE DATABASE n8n;
CREATE DATABASE dify;
CREATE DATABASE authentik;

-- Enable pgvector on each database that needs it
\c litellm
CREATE EXTENSION IF NOT EXISTS vector;

\c langfuse
CREATE EXTENSION IF NOT EXISTS vector;

\c dify
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

\c n8n
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Service registry table (on main arsenal db)
\c arsenal
CREATE TABLE IF NOT EXISTS service_registry (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    tier INTEGER NOT NULL,
    internal_url VARCHAR(255) NOT NULL,
    external_port INTEGER,
    status VARCHAR(20) DEFAULT 'active',
    health_endpoint VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO service_registry (name, tier, internal_url, external_port, health_endpoint) VALUES
    ('postgres',     1, 'postgres:5432',      5432, NULL),
    ('redis',        1, 'redis:6379',         6379, NULL),
    ('qdrant',       1, 'qdrant:6333',        6333, '/healthz'),
    ('litellm',      1, 'litellm:4000',       4000, '/health'),
    ('ollama',       2, 'ollama:11434',       11434, '/api/version'),
    ('open-webui',   2, 'open-webui:8080',     3000, '/health'),
    ('n8n',          2, 'n8n:5678',            5678, '/healthz'),
    ('dify-api',     2, 'dify-api:5001',       5001, NULL),
    ('crawl4ai',     3, 'crawl4ai:11235',     11235, '/health'),
    ('searxng',      3, 'searxng:8080',        8080, '/healthz'),
    ('browserless',  3, 'browserless:3000',    3002, '/pressure'),
    ('firecrawl',    3, 'firecrawl-api:3003',  3003, NULL),
    ('langfuse',     4, 'langfuse:3000',       3004, '/api/public/health'),
    ('uptime-kuma',  4, 'uptime-kuma:3001',    3005, '/'),
    ('dozzle',       4, 'dozzle:8080',         9999, NULL),
    ('traefik',      5, 'traefik:443',         443,  NULL),
    ('authentik',    5, 'authentik:9000',       9000, NULL),
    ('speaches',     6, 'speaches:8000',       8100, '/health'),
    ('docling',      6, 'docling:5001',        5002, '/health')
ON CONFLICT (name) DO NOTHING;
