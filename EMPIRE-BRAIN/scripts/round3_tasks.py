#!/usr/bin/env python3
"""Round 3 Brain Enhancements: code solutions + pattern cross-refs."""
import sqlite3
import json
import hashlib

DB_PATH = "knowledge/brain.db"

def populate_code_solutions(conn):
    """Task 29: Populate code_solutions from known patterns."""
    solutions = [
        {
            "problem": "WordPress REST API authentication with application password",
            "solution": (
                "import requests, base64\n"
                "def wp_auth_headers(user, app_password):\n"
                '    token = base64.b64encode(f"{user}:{app_password}".encode()).decode()\n'
                '    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}'
            ),
            "language": "python",
            "project_slug": "empire-master",
            "tags": "wordpress,api,auth,rest",
        },
        {
            "problem": "FastAPI service with health endpoint and CORS",
            "solution": (
                "from fastapi import FastAPI\n"
                "from fastapi.middleware.cors import CORSMiddleware\n"
                'app = FastAPI(title="Service")\n'
                'app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])\n'
                '@app.get("/health")\n'
                'def health(): return {"status": "healthy"}'
            ),
            "language": "python",
            "project_slug": "empire-brain",
            "tags": "fastapi,api,health,cors",
        },
        {
            "problem": "SQLite connection with Row factory for dict-like access",
            "solution": (
                "import sqlite3\n"
                "def get_db(path):\n"
                "    conn = sqlite3.connect(path)\n"
                '    conn.row_factory = sqlite3.Row  # bracket access: row["col"]\n'
                '    conn.execute("PRAGMA journal_mode=WAL")\n'
                "    return conn\n"
                '# NOTE: sqlite3.Row supports row["key"] but NOT row.get("key", default)'
            ),
            "language": "python",
            "project_slug": "empire-brain",
            "tags": "sqlite,database,row-factory,gotcha",
        },
        {
            "problem": "ElevenLabs TTS with tmpfiles.org hosting for Creatomate",
            "solution": (
                "import requests\n"
                "def tts_and_host(text, voice_id, api_key):\n"
                '    resp = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",\n'
                '        headers={"xi-api-key": api_key},\n'
                '        json={"text": text, "model_id": "eleven_turbo_v2_5"})\n'
                "    # Upload to tmpfiles.org (Creatomate can fetch, unlike catbox.moe)\n"
                '    upload = requests.post("https://tmpfiles.org/api/v1/upload",\n'
                '        files={"file": ("audio.mp3", resp.content)})\n'
                '    url = upload.json()["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")\n'
                "    return url"
            ),
            "language": "python",
            "project_slug": "videoforge-engine",
            "tags": "elevenlabs,tts,audio,hosting,creatomate",
        },
        {
            "problem": "Image re-hosting to freeimage.host for Creatomate compatibility",
            "solution": (
                "import requests, base64\n"
                "def rehost_image(image_url):\n"
                "    img_data = requests.get(image_url).content\n"
                "    b64 = base64.b64encode(img_data).decode()\n"
                '    resp = requests.post("https://freeimage.host/api/1/upload",\n'
                '        data={"key": "6d207e02198a847aa98d0a2a901485a5",\n'
                '              "source": b64, "format": "json"})\n'
                '    return resp.json()["image"]["url"]  # iili.io CDN URL'
            ),
            "language": "python",
            "project_slug": "videoforge-engine",
            "tags": "image,hosting,freeimage,rehost,creatomate",
        },
        {
            "problem": "n8n webhook with PostgreSQL upsert using n8n expressions",
            "solution": (
                "-- n8n Postgres node v2.5 uses n8n expressions, NOT positional params\n"
                "-- In the Query field:\n"
                "INSERT INTO brain_projects (slug, name, category)\n"
                "VALUES ({{ $json.slug }}, {{ $json.name }}, {{ $json.category }})\n"
                "ON CONFLICT (slug) DO UPDATE SET\n"
                "  name = EXCLUDED.name, category = EXCLUDED.category;\n"
                "-- IMPORTANT: NOT $1, $2 style params"
            ),
            "language": "sql",
            "project_slug": "empire-brain",
            "tags": "n8n,postgres,upsert,webhook,gotcha",
        },
        {
            "problem": "Grimoire moon phase calculation without ephem library",
            "solution": (
                "import math, datetime\n"
                "def moon_phase(dt=None):\n"
                "    dt = dt or datetime.datetime.now()\n"
                "    diff = dt - datetime.datetime(2000, 1, 6, 18, 14)\n"
                "    days = diff.days + diff.seconds / 86400\n"
                "    lunations = days / 29.53058867\n"
                "    phase = lunations % 1\n"
                '    names = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",\n'
                '             "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]\n'
                "    return names[int(phase * 8) % 8]"
            ),
            "language": "python",
            "project_slug": "grimoire-intelligence",
            "tags": "moon,astronomy,grimoire,calculation",
        },
        {
            "problem": "Anthropic API call with prompt caching for cost optimization",
            "solution": (
                "from anthropic import Anthropic\n"
                "client = Anthropic()\n"
                "msg = client.messages.create(\n"
                '    model="claude-sonnet-4-20250514",\n'
                "    max_tokens=2048,\n"
                '    system=[{"type": "text", "text": system_prompt,\n'
                '             "cache_control": {"type": "ephemeral"}}],\n'
                '    messages=[{"role": "user", "content": user_input}])\n'
                "# Cache reads: 90% cheaper than non-cached"
            ),
            "language": "python",
            "project_slug": "empire-master",
            "tags": "anthropic,api,caching,cost-optimization",
        },
        {
            "problem": "Deduplication via content hash before INSERT",
            "solution": (
                "import hashlib\n"
                "def content_hash(text):\n"
                '    return hashlib.sha256(text.encode()).hexdigest()[:16]\n\n'
                "def insert_if_new(conn, table, content, **fields):\n"
                "    h = content_hash(content)\n"
                '    exists = conn.execute(f"SELECT 1 FROM {table} WHERE content_hash=?", (h,)).fetchone()\n'
                "    if not exists:\n"
                "        conn.execute(f\"INSERT INTO {table} (content, content_hash) VALUES (?, ?)\",\n"
                "                     (content, h))\n"
                "        return True\n"
                "    return False"
            ),
            "language": "python",
            "project_slug": "empire-brain",
            "tags": "deduplication,hash,database,pattern",
        },
        {
            "problem": "ADB reconnection with Tailscale for remote phone control",
            "solution": (
                "import subprocess\n"
                'ADB = r"C:\\Users\\ncreighton\\AppData\\Roaming\\GeeLark\\adb\\adb.exe"\n'
                'DEVICE = "100.79.124.62:5555"\n'
                "def reconnect():\n"
                "    subprocess.run([ADB, 'disconnect'], capture_output=True)\n"
                "    result = subprocess.run([ADB, 'connect', DEVICE], capture_output=True, text=True)\n"
                '    if "connected" in result.stdout:\n'
                "        subprocess.run([ADB, 'tcpip', '5555'], capture_output=True)\n"
                "        return True\n"
                "    return False"
            ),
            "language": "python",
            "project_slug": "geelark-automation",
            "tags": "adb,android,tailscale,reconnect,phone",
        },
        {
            "problem": "Revid.ai text-to-video API with correct field names",
            "solution": (
                "import requests\n"
                'headers = {"key": REVID_API_KEY, "Content-Type": "application/json"}\n'
                '# CRITICAL: slug is "faceless-video" (not "text-to-video")\n'
                '# CRITICAL: field is "inputText" (not "text")\n'
                "# CRITICAL: must include hasToSearchMedia: true\n"
                "payload = {\n"
                '    "slug": "faceless-video",\n'
                '    "inputText": narration_script,\n'
                '    "hasToSearchMedia": True,\n'
                '    "resolution": "SHORTS"\n'
                "}\n"
                'resp = requests.post("https://www.revid.ai/api/public/v2/generate",\n'
                "    json=payload, headers=headers)"
            ),
            "language": "python",
            "project_slug": "revid-forge",
            "tags": "revid,video,api,gotcha",
        },
        {
            "problem": "Windows Task Scheduler hidden service with VBS wrapper",
            "solution": (
                "# All scheduled tasks MUST use wscript.exe + VBS wrapper\n"
                "# Direct python.exe or .bat creates visible popup windows\n"
                "# Universal pattern:\n"
                "# 1. launchers/run-hidden.vbs (WshShell.Run with 0=hidden)\n"
                "# 2. launchers/run-hidden.ps1 (wraps any command)\n"
                "# 3. Task Scheduler -> wscript.exe run-hidden.vbs <cmd>\n"
                "#\n"
                "# Register via:\n"
                "# schtasks /create /tn 'ServiceName' /tr\n"
                '#   \'wscript.exe "C:\\path\\run-hidden.vbs" "python service.py"\'\n'
                "# /sc onlogon /delay 0000:15"
            ),
            "language": "powershell",
            "project_slug": "empire-master",
            "tags": "windows,scheduler,hidden,service,vbs",
        },
    ]

    inserted = 0
    for sol in solutions:
        h = hashlib.sha256(sol["solution"].encode()).hexdigest()[:16]
        exists = conn.execute(
            "SELECT 1 FROM code_solutions WHERE content_hash=?", (h,)
        ).fetchone()
        if not exists:
            conn.execute(
                """INSERT INTO code_solutions
                   (problem, solution, language, project_slug, tags, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (sol["problem"], sol["solution"], sol["language"],
                 sol["project_slug"], sol["tags"], h),
            )
            inserted += 1

    conn.commit()
    return inserted


def add_used_by_crossrefs(conn):
    """Task 30: Add used_by cross-references from patterns to projects."""
    patterns = conn.execute(
        "SELECT id, name, used_by_projects FROM patterns WHERE used_by_projects IS NOT NULL"
    ).fetchall()

    added = 0
    for pat in patterns:
        try:
            projects = json.loads(pat["used_by_projects"])
        except (json.JSONDecodeError, TypeError):
            continue

        for proj_slug in projects:
            proj = conn.execute(
                "SELECT id FROM projects WHERE slug = ?", (proj_slug,)
            ).fetchone()
            if not proj:
                continue

            exists = conn.execute(
                """SELECT 1 FROM cross_references
                   WHERE source_type='pattern' AND source_id=?
                   AND target_type='project' AND target_id=?
                   AND relationship='used_by'""",
                (pat["id"], proj["id"]),
            ).fetchone()

            if not exists:
                conn.execute(
                    """INSERT INTO cross_references
                       (source_type, source_id, target_type, target_id, relationship, strength)
                       VALUES ('pattern', ?, 'project', ?, 'used_by', 0.8)""",
                    (pat["id"], proj["id"]),
                )
                added += 1

    conn.commit()
    return added


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Task 29
    print("=== TASK 29: POPULATING CODE SOLUTIONS ===")
    count = populate_code_solutions(conn)
    print(f"  Inserted {count} code solutions")
    total = conn.execute("SELECT COUNT(*) FROM code_solutions").fetchone()[0]
    print(f"  Total code solutions: {total}")

    # Task 30
    print("\n=== TASK 30: ADDING PATTERN->PROJECT CROSS-REFS ===")
    count = add_used_by_crossrefs(conn)
    print(f"  Added {count} used_by cross-references")

    # Final stats
    print("\n=== FINAL CROSS-REF STATS ===")
    for r in conn.execute(
        "SELECT relationship, COUNT(*) FROM cross_references GROUP BY relationship ORDER BY COUNT(*) DESC"
    ):
        print(f"  {r[0]}: {r[1]}")
    total = conn.execute("SELECT COUNT(*) FROM cross_references").fetchone()[0]
    print(f"  Total: {total}")

    conn.close()
