# ZimmWriter Pipeline

AI content generation pipeline with SEO optimization, WordPress publishing, and multi-site support. Integrates with 16 WordPress sites for automated article creation, editing, and publishing.

## Trigger Phrases

- "Write an article about [topic]"
- "Generate content for [site]"
- "Publish article to [site]"
- "Run content pipeline for [keyword]"
- "SEO optimize article [ID]"
- "Check article quality score"
- "Generate bulk content for [site]"

## API Endpoints

| Method | Path | Handler | File |
|--------|------|---------|------|
| GET | `/` | `root` | `src\api.py` |
| POST | `/batch/execute` | `batch_execute` | `src\api.py` |
| POST | `/batch/prepare` | `batch_prepare` | `src\api.py` |
| POST | `/batch/resume/{batch_id}` | `batch_resume` | `src\api.py` |
| GET | `/batch/review/{batch_id}` | `batch_review` | `src\api.py` |
| GET | `/batch/status/{batch_id}` | `batch_status` | `src\api.py` |
| POST | `/bring-to-front` | `bring_to_front` | `src\api.py` |
| POST | `/campaign/batch` | `campaign_batch` | `src\api.py` |
| POST | `/campaign/classify` | `campaign_classify` | `src\api.py` |
| POST | `/campaign/plan` | `campaign_plan` | `src\api.py` |
| POST | `/campaign/run` | `campaign_run` | `src\api.py` |
| POST | `/checkboxes` | `set_checkboxes` | `src\api.py` |
| POST | `/clear` | `clear` | `src\api.py` |
| POST | `/click` | `click` | `src\api.py` |
| POST | `/config/alt-images` | `config_alt_images` | `src\api.py` |
| POST | `/config/custom-outline` | `config_custom_outline` | `src\api.py` |
| POST | `/config/custom-prompt` | `config_custom_prompt` | `src\api.py` |
| POST | `/config/deep-research` | `config_deep_research` | `src\api.py` |
| POST | `/config/link-pack` | `config_link_pack` | `src\api.py` |
| POST | `/config/seo-csv` | `config_seo_csv` | `src\api.py` |
| POST | `/config/serp-scraping` | `config_serp` | `src\api.py` |
| POST | `/config/style-mimic` | `config_style_mimic` | `src\api.py` |
| POST | `/config/webhook` | `config_webhook` | `src\api.py` |
| POST | `/config/wordpress` | `config_wordpress` | `src\api.py` |
| POST | `/config/youtube-videos` | `config_youtube_videos` | `src\api.py` |
| POST | `/configure` | `configure` | `src\api.py` |
| POST | `/connect` | `connect` | `src\api.py` |
| GET | `/controls/buttons` | `list_buttons` | `src\api.py` |
| GET | `/controls/checkboxes` | `list_checkboxes` | `src\api.py` |
| GET | `/controls/dropdowns` | `list_dropdowns` | `src\api.py` |
| GET | `/controls/dump` | `dump_controls` | `src\api.py` |
| GET | `/controls/text-fields` | `list_text_fields` | `src\api.py` |
| POST | `/feature-toggle` | `feature_toggle` | `src\api.py` |
| GET | `/health` | `health` | `src\api.py` |
| GET | `/intelligence/codex/history/{domain}` | `codex_domain_history` | `src\api.py` |
| POST | `/intelligence/detect-errors` | `detect_errors_intel` | `src\api.py` |
| POST | `/intelligence/enhanced-run` | `enhanced_run` | `src\api.py` |
| GET | `/intelligence/forge/stats` | `forge_stats` | `src\api.py` |
| GET | `/intelligence/jobs` | `active_jobs` | `src\api.py` |
| POST | `/intelligence/post-job` | `post_job_learning` | `src\api.py` |
| POST | `/intelligence/pre-job` | `pre_job_analysis` | `src\api.py` |
| GET | `/intelligence/progress/{job_id}` | `check_job_progress` | `src\api.py` |
| GET | `/intelligence/screenpipe/errors` | `screenpipe_errors` | `src\api.py` |
| GET | `/intelligence/screenpipe/state` | `screenpipe_state` | `src\api.py` |
| GET | `/intelligence/screenpipe/timeline` | `screenpipe_timeline` | `src\api.py` |
| GET | `/intelligence/stats` | `intelligence_stats` | `src\api.py` |
| POST | `/intelligence/verify-screen` | `verify_screen` | `src\api.py` |
| GET | `/is-running` | `is_running` | `src\api.py` |
| POST | `/launch` | `launch` | `src\api.py` |
| POST | `/link-packs/build` | `link_pack_build` | `src\api.py` |
| POST | `/link-packs/build-all` | `link_pack_build_all` | `src\api.py` |
| GET | `/link-packs/list` | `link_pack_list` | `src\api.py` |
| POST | `/load-csv` | `load_csv` | `src\api.py` |
| GET | `/model-stats` | `model_stats` | `src\api.py` |
| GET | `/model-stats/report` | `model_stats_report` | `src\api.py` |
| POST | `/orchestrate` | `orchestrate` | `src\api.py` |
| POST | `/orchestrate/skip` | `orchestrate_skip` | `src\api.py` |
| GET | `/orchestrate/status` | `orchestrate_status` | `src\api.py` |
| GET | `/presets` | `list_presets` | `src\api.py` |
| GET | `/presets/{domain}` | `get_preset_config` | `src\api.py` |
| POST | `/presets/{domain}/apply` | `apply_preset` | `src\api.py` |
| POST | `/profile/load` | `load_profile` | `src\api.py` |
| POST | `/profile/save` | `save_profile` | `src\api.py` |
| POST | `/profile/update` | `update_profile` | `src\api.py` |
| POST | `/run-job` | `run_job` | `src\api.py` |
| GET | `/screen/available` | `screen_available` | `src\api.py` |
| GET | `/screen/current` | `screen_current` | `src\api.py` |
| POST | `/screen/menu` | `screen_menu` | `src\api.py` |
| POST | `/screen/navigate` | `screen_navigate` | `src\api.py` |
| POST | `/screenshot` | `screenshot` | `src\api.py` |
| POST | `/set-checkbox` | `set_checkbox_single` | `src\api.py` |
| POST | `/set-dropdown` | `set_dropdown` | `src\api.py` |
| POST | `/set-text` | `set_text` | `src\api.py` |
| POST | `/start` | `start` | `src\api.py` |
| GET | `/status` | `status` | `src\api.py` |
| POST | `/stop` | `stop` | `src\api.py` |
| POST | `/titles` | `set_titles` | `src\api.py` |
| GET | `/titles/existing/{domain}` | `titles_existing` | `src\api.py` |

## Key Components

- **ZimmWriterController** (`src\controller.py`) — 100 methods: Full programmatic control of ZimmWriter desktop application. Uses Windows UI Automation API via pywi
- **BatchCampaign** (`src\batch_campaign.py`) — 19 methods: Master orchestrator for full batch campaign pipeline.
- **ScreenpipeAgent** (`src\screenpipe_agent.py`) — 17 methods: Passive monitoring layer that uses Screenpipe's OCR to observe ZimmWriter state without interfering 
- **Codex** (`src\forge_intelligence.py`) — 15 methods: Persistent memory that learns from every ZimmWriter job execution. Stores outcomes, failure patterns
- **VisionAgent** (`src\vision_agent.py`) — 15 methods: Visual verification and guidance layer for ZimmWriter automation. Uses the Empire Vision Service to 
- **IntelligenceHub** (`src\intelligence.py`) — 12 methods: Central intelligence coordinator. Initializes all subsystems and provides unified methods for enhanc
- **Orchestrator** (`src\orchestrator.py`) — 11 methods: Runs ZimmWriter jobs across multiple sites sequentially.  Usage:     orch = Orchestrator()     orch.
- **ScreenNavigator** (`src\screen_navigator.py`) — 9 methods: Navigate between ZimmWriter screens.  Requires a connected ZimmWriterController instance. All screen
- **VisionVerifiedController** (`src\vision_agent.py`) — 9 methods: Wrapper around ZimmWriterController that adds vision verification to critical operations. Uses the V
- **TestArticleTypes** (`tests\test_controller.py`) — 9 methods
- **JobMonitor** (`src\monitor.py`) — 8 methods: Monitor ZimmWriter bulk generation progress.
- **TestSitePresets** (`tests\test_controller.py`) — 8 methods
- **AmplifyPipeline** (`src\amplify_pipeline.py`) — 7 methods: Unified AMPLIFY Pipeline. Runs all six stages in sequence on configurations and actions.  Usage:    
- **TitleChecker** (`src\title_checker.py`) — 7 methods: Checks WordPress sites for existing titles to prevent duplicates.
- **HTMLContentExtractor** (`scripts\audit_articles.py`) — 6 methods: Extracts plain text, headings, and structural info from HTML content.

## Key Functions

- `download_with_api(vid_id, title, proxy_url)` — Download transcript using youtube-transcript-api. (`download_transcripts.py`)
- `main()` (`download_transcripts.py`)
- `handle_starttag(self, tag, attrs)` (`scripts\audit_articles.py`)
- `handle_endtag(self, tag)` (`scripts\audit_articles.py`)
- `handle_data(self, data)` (`scripts\audit_articles.py`)
- `get_plain_text(self)` (`scripts\audit_articles.py`)
- `get_sections(self)` (`scripts\audit_articles.py`)
- `extract_content(html)` — Parse HTML and return structured content data. (`scripts\audit_articles.py`)
- `detect_red_flags(plain_text, headings, sections)` — Detect potential quality issues in article content. (`scripts\audit_articles.py`)
- `compute_readability_score(plain_text)` — Compute a simple readability estimate (Flesch-like). (`scripts\audit_articles.py`)
- `count_syllables(word)` (`scripts\audit_articles.py`)
- `fetch_site_articles(domain, timeout)` — Fetch the 5 most recent published posts from a WordPress site. (`scripts\audit_articles.py`)
- `analyze_article(post, domain)` — Analyze a single WordPress post and return structured metrics. (`scripts\audit_articles.py`)
- `run_audit()` — Run the full audit across all 14 sites. (`scripts\audit_articles.py`)
- `read_combo_items(hwnd, max_items)` — Read all items from a ComboBox. (`scripts\discover_all_feature_windows.py`)
- `read_edit_text(hwnd, max_chars)` — Read text from an Edit control. (`scripts\discover_all_feature_windows.py`)
- `read_checkbox_state(hwnd)` — Read checkbox state (0=unchecked, 1=checked). (`scripts\discover_all_feature_windows.py`)
- `enumerate_window_controls(win)` — Enumerate all child controls in a window. (`scripts\discover_all_feature_windows.py`)
- `discover_feature_window(zw, feature_key, feature_id, window_title)` — Open a feature config window and enumerate its controls. (`scripts\discover_all_feature_windows.py`)
- `discover_image_options_window(zw, button_id, label)` — Open an Image Options (O button) window and enumerate controls. (`scripts\discover_all_feature_windows.py`)

## Stats

- **Functions**: 645
- **Classes**: 79
- **Endpoints**: 78
- **Files**: 163
- **Category**: content-tools
- **Tech Stack**: python, powershell, claude-code
