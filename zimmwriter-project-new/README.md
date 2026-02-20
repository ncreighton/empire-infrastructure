# ZimmWriter Desktop Controller

**100% programmatic control of ZimmWriter** via Windows UI Automation.

Control every button, dropdown, checkbox, text field, and feature toggle — from Python, n8n, Claude Code, or any HTTP client. Pre-configured for all 16 websites in the publishing empire.

---

## Features

- **Full UI Control** — Every Bulk Writer element: dropdowns, checkboxes, toggles, text fields, profiles
- **REST API** — FastAPI server on port 8765 with interactive Swagger docs
- **16 Site Presets** — All sites pre-configured with correct ZimmWriter settings
- **Multi-Site Orchestration** — Run sequential jobs across multiple sites unattended
- **Progress Monitoring** — Track completion, estimate ETAs, save job logs
- **n8n Integration** — Example workflows for automated content pipelines
- **UI Discovery** — First-run tool maps every control with exact names/IDs
- **CLI Runner** — Run batch jobs from the command line
- **Claude Code Project** — Full project structure with CLAUDE.md

---

## Quick Start

### 1. Install
```
Double-click: setup.bat
```

### 2. Map Controls (first time)
```
Open ZimmWriter → python scripts/discover_controls.py
```

### 3. Start API
```
Double-click: start-server.bat
```

### 4. Use It

**From n8n:**
```
POST http://localhost:8765/connect
POST http://localhost:8765/presets/smarthomewizards.com/apply
POST http://localhost:8765/load-csv  {"csv_path": "C:\\batches\\smart_home.csv"}
POST http://localhost:8765/start
```

**From Python:**
```python
from src.controller import quick_connect
zw = quick_connect()
zw.set_bulk_titles(["Best Smart Locks 2025", "Ring vs Nest"])
zw.configure_bulk_writer(section_length="Medium", voice="Second Person")
zw.set_checkboxes(lists=True, tables=True, enable_h3=True)
zw.enable_serp_scraping(True)
zw.start_bulk_writer()
```

**From CLI:**
```bash
python scripts/run_batch.py --site smarthomewizards.com --csv "C:\batches\smart_home.csv" --wait
```

---

## Pre-Configured Sites (18 Total)

| Domain | Niche | Voice |
|--------|-------|-------|
| witchcraftforbeginners.com | Witchcraft & Spirituality | 2nd Person |
| smarthomewizards.com | Smart Home Automation | 2nd Person |
| aiinactionhub.com | AI & Technology | 2nd Person |
| mythicalarchives.com | Mythology & Folklore | 3rd Person |
| family-flourish.com | Family & Parenting | 2nd Person |
| smarthomegearreviews.com | Smart Home Product Reviews | 2nd Person |
| theconnectedhaven.com | Smart Home Lifestyle | 2nd Person |
| clearainews.com | AI News & Journalism | 3rd Person |
| aidiscoverydigest.com | AI Research & Analysis | 3rd Person |
| wearablegearreviews.com | Wearable Tech Reviews | 2nd Person |
| pulsegearreviews.com | EDC & Tactical Gear | 2nd Person |
| aiinactionblueprint.com | AI Automation & Workflows | 2nd Person |
| wealthfromai.com | AI-Powered Income | 2nd Person |
| celebrationseason.net | Holiday & Celebrations | 2nd Person |
| sprout-and-spruce.com | Gardening & Plant Care | 2nd Person |
| flavors-and-forks.com | Recipes & Cooking | 2nd Person |
| bulletjournals.net | Bullet Journaling | 2nd Person |
| manifestandalign.com | Manifestation & Spirituality | 2nd Person |

---

## API Endpoints

Full interactive docs at: **http://localhost:8765/docs**

### Connection
| Endpoint | Description |
|----------|-------------|
| `POST /connect` | Connect to running ZimmWriter |
| `POST /launch` | Launch and connect |
| `GET /status` | Full status dump |
| `POST /screenshot` | Capture window state |

### Bulk Writer
| Endpoint | Description |
|----------|-------------|
| `POST /titles` | Set article titles |
| `POST /load-csv` | Load SEO CSV |
| `POST /configure` | Set all dropdowns |
| `POST /checkboxes` | Set all checkboxes |
| `POST /feature-toggle` | Toggle feature buttons |

### Execution
| Endpoint | Description |
|----------|-------------|
| `POST /start` | Start Bulk Writer |
| `POST /stop` | Stop generation |
| `POST /run-job` | Complete end-to-end job |
| `POST /orchestrate` | Multi-site job queue |

### Presets
| Endpoint | Description |
|----------|-------------|
| `GET /presets` | List all site presets |
| `POST /presets/{domain}/apply` | Apply site configuration |

### Discovery
| Endpoint | Description |
|----------|-------------|
| `GET /controls/dump` | Raw control tree |
| `GET /controls/buttons` | All buttons |
| `GET /controls/checkboxes` | All checkboxes + states |
| `GET /controls/dropdowns` | All dropdowns + values |

---

## Project Structure

```
zimmwriter-project/
├── CLAUDE.md                    # Claude Code project instructions
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.bat                    # One-time installer
├── start-server.bat             # API server launcher
├── auto-start-claude.bat        # Claude Code launcher
├── claude_desktop_config.json   # MCP configuration
├── test-connection.sh           # Connection test
├── src/
│   ├── __init__.py
│   ├── controller.py            # Core pywinauto controller
│   ├── api.py                   # FastAPI REST server
│   ├── site_presets.py          # All 18 site configurations
│   ├── csv_generator.py         # SEO CSV generation
│   ├── monitor.py               # Progress monitoring
│   ├── orchestrator.py          # Multi-site job runner
│   └── utils.py                 # Shared utilities
├── configs/
│   ├── site-configs.json        # Full site config database
│   └── outline-patterns.md      # ZimmWriter outline reference
├── scripts/
│   ├── discover_controls.py     # First-run UI mapping
│   ├── quick_test.py            # Quick connection test
│   └── run_batch.py             # CLI batch runner
├── workflows/
│   ├── n8n-basic-workflow.json
│   └── n8n-multi-site-workflow.json
├── tests/
│   └── test_controller.py
├── logs/                        # Runtime logs
└── output/                      # Screenshots, CSVs, control maps
```

---

## How It Works

**pywinauto** uses the Windows UI Automation API (UIA) to find and interact with every UI element in ZimmWriter. This is the same technology behind enterprise RPA tools like UiPath and Blue Prism. It can:

- Find elements by name, automation ID, control type, or class
- Click buttons (even hidden ones)
- Set dropdown selections
- Toggle checkboxes
- Paste text via clipboard (fast for large title lists)
- Read current states of all controls
- Monitor window titles for progress

The FastAPI server wraps all controller methods as HTTP endpoints, making it callable from n8n, scripts, or any HTTP client.

---

## Troubleshooting

**"Could not connect"** → ZimmWriter must be running and visible. Try running as Administrator.

**Controls not found** → Run `discover_controls.py` to get exact names. ZimmWriter updates may rename controls.

**Slow text entry** → The API uses clipboard paste by default (fast mode). Falls back to keystrokes if needed.

**File dialogs** → Use absolute Windows paths with double backslashes: `C:\\Users\\Nick\\batch.csv`
