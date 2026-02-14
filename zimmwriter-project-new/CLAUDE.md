# ZimmWriter Desktop Controller — Claude Code Project

## Overview
Full programmatic control of ZimmWriter v10.846+ desktop application on Windows.
Uses pywinauto (Windows UI Automation API) to interact with every UI element.
Exposes all functionality via a FastAPI REST server on port 8765.
Integrates with n8n workflows for fully automated content pipelines across 16 WordPress sites.

## Project Structure
```
zimmwriter-project/
├── CLAUDE.md                          # This file — project instructions
├── README.md                          # User-facing documentation
├── requirements.txt                   # Python dependencies
├── setup.bat                          # One-time Windows dependency installer
├── start-server.bat                   # Launch the API server
├── src/
│   ├── __init__.py
│   ├── controller.py                  # Core ZimmWriter controller (pywinauto)
│   ├── api.py                         # FastAPI REST server
│   ├── site_presets.py                # All 16 site preset configurations
│   ├── csv_generator.py               # SEO CSV generation utilities
│   ├── monitor.py                     # Progress monitoring & notifications
│   ├── orchestrator.py                # Multi-site job orchestration
│   └── utils.py                       # Shared utilities (clipboard, logging, etc.)
├── configs/
│   ├── site-configs.json              # Full site configuration database (all 16 sites)
│   └── outline-patterns.md            # ZimmWriter outline variable reference
├── scripts/
│   ├── discover_controls.py           # First-run UI mapping tool
│   ├── quick_test.py                  # Quick connection test
│   └── run_batch.py                   # CLI batch runner
├── workflows/
│   ├── n8n-basic-workflow.json        # Simple: connect → configure → start
│   ├── n8n-multi-site-workflow.json   # Advanced: run across multiple sites
│   └── n8n-webhook-trigger.json       # Webhook-triggered generation
├── tests/
│   └── test_controller.py             # Unit tests
├── logs/                              # Runtime logs
└── output/                            # Generated CSVs and screenshots
```

## Tech Stack
- **pywinauto** — Windows UI Automation API bindings (the core)
- **pyautogui** — Screenshot fallback + coordinate-based clicks
- **FastAPI** — REST API server
- **uvicorn** — ASGI server
- **psutil** — Process management
- **Pillow** — Screenshot capture

## How It Works
1. `pywinauto` connects to a running ZimmWriter instance via Windows UIA
2. Every button, checkbox, dropdown, and text field is addressable by name or automation ID
3. The FastAPI server wraps all controller methods as HTTP endpoints
4. n8n workflows call the API endpoints to orchestrate multi-site generation
5. The orchestrator can run sequential jobs across all 16 sites unattended

## First-Time Setup
1. Run `setup.bat` to install Python dependencies
2. Open ZimmWriter to the Bulk Writer screen
3. Run `python scripts/discover_controls.py` to map all UI control names/IDs
4. Review `output/zimmwriter_control_map.json` and update `src/controller.py` if any names differ
5. Run `start-server.bat` to launch the API on http://localhost:8765

## Key API Endpoints
- `POST /connect` — Connect to running ZimmWriter
- `POST /launch` — Launch ZimmWriter and connect
- `GET /status` — Full status dump (all controls, states)
- `POST /configure` — Set all dropdown settings
- `POST /checkboxes` — Set all checkbox states
- `POST /feature-toggle` — Toggle right-side feature buttons
- `POST /titles` — Set bulk article titles
- `POST /load-csv` — Load SEO CSV file
- `POST /start` — Start Bulk Writer
- `POST /stop` — Stop Bulk Writer
- `POST /run-job` — Complete end-to-end job
- `POST /presets/{domain}` — Apply site-specific preset
- `POST /orchestrate` — Multi-site sequential job runner
- `GET /controls/dump` — Raw UI control tree (for debugging)

## Site Preset Domains
All 16 sites are pre-configured in `src/site_presets.py`:
- witchcraftforbeginners.com, smarthomewizards.com, aiinactionhub.com
- mythicalarchives.com, family-flourish.com, smarthomegearreviews.com
- theconnectedhaven.com, clearainews.com, aidiscoverydigest.com
- wearablegearreviews.com, pulsegearreviews.com, aiinactionblueprint.com
- wealthfromai.com, celebrationseason.net, sprout-and-spruce.com
- flavors-and-forks.com, bulletjournals.net, manifestandalign.com

## Important Notes
- ZimmWriter must be RUNNING and VISIBLE for the controller to work
- Run `discover_controls.py` after any ZimmWriter update to re-map control names
- The API server must run ON THE SAME Windows machine as ZimmWriter
- File paths in CSV loading must use absolute Windows paths (C:\\...)
- The controller uses clipboard paste for fast text insertion
- Feature toggle buttons have Enabled/Disabled states in their text

## Commands for Nick
```bash
# Quick test connection
python scripts/quick_test.py

# Start API server
python -m uvicorn src.api:app --host 0.0.0.0 --port 8765

# Run a batch job from CLI
python scripts/run_batch.py --site smarthomewizards.com --csv "C:\batches\smart_home.csv"

# Discover all UI controls
python scripts/discover_controls.py

# Generate a CSV from titles
python -c "from src.csv_generator import quick_csv; quick_csv(['Title 1', 'Title 2'], 'output/batch.csv', 'smarthomewizards.com')"
```
