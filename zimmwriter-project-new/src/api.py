"""
ZimmWriter Controller API Server
Complete REST API wrapping all controller functionality.

Start: python -m uvicorn src.api:app --host 0.0.0.0 --port 8765
Docs:  http://localhost:8765/docs
"""

import os
import time
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .controller import ZimmWriterController
from .site_presets import SITE_PRESETS, get_preset, get_all_domains
from .monitor import JobMonitor
from .orchestrator import Orchestrator, CampaignOrchestrator
from .intelligence import IntelligenceHub
from .campaign_engine import CampaignEngine
from .article_types import classify_title, classify_titles, get_dominant_type
from .screen_navigator import ScreenNavigator, Screen, MENU_BUTTONS
from .link_pack_builder import LinkPackBuilder

app = FastAPI(
    title="ZimmWriter Controller API",
    description="Full programmatic control of ZimmWriter desktop application. "
                "Covers all Bulk Writer settings, profiles, execution, and multi-site orchestration.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global controller and intelligence hub
_ctrl: Optional[ZimmWriterController] = None
_intel: Optional[IntelligenceHub] = None

def get_ctrl() -> ZimmWriterController:
    global _ctrl
    if _ctrl is None or not _ctrl._connected:
        _ctrl = ZimmWriterController()
        if not _ctrl.connect():
            raise HTTPException(503, "ZimmWriter not running or not found")
    return _ctrl

def get_intel() -> IntelligenceHub:
    global _intel
    if _intel is None:
        _intel = IntelligenceHub()
    return _intel


# ═══════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════

class ConnectReq(BaseModel):
    exe_path: Optional[str] = None

class TitlesReq(BaseModel):
    titles: List[str]
    separator: str = "\n"

class CSVReq(BaseModel):
    csv_path: str

class ConfigureReq(BaseModel):
    h2_count: Optional[str] = None
    h2_auto_limit: Optional[int] = None
    h2_upper_limit: Optional[int] = None
    h2_lower_limit: Optional[int] = None
    ai_outline_quality: Optional[str] = None
    section_length: Optional[str] = None
    voice: Optional[str] = None
    intro: Optional[str] = None
    faq: Optional[str] = None
    audience_personality: Optional[str] = None
    ai_model: Optional[str] = None
    style_of: Optional[str] = None
    featured_image: Optional[str] = None
    subheading_image_quantity: Optional[str] = None
    subheading_images_model: Optional[str] = None
    ai_model_image_prompts: Optional[str] = None
    output_language: Optional[str] = None
    ai_model_translation: Optional[str] = None

class CheckboxReq(BaseModel):
    literary_devices: Optional[bool] = None
    lists: Optional[bool] = None
    tables: Optional[bool] = None
    blockquotes: Optional[bool] = None
    nuke_ai_words: Optional[bool] = None
    bold_readability: Optional[bool] = None
    key_takeaways: Optional[bool] = None
    disable_skinny_paragraphs: Optional[bool] = None
    enable_h3: Optional[bool] = None
    disable_active_voice: Optional[bool] = None
    disable_conclusion: Optional[bool] = None
    auto_style: Optional[bool] = None
    automatic_keywords: Optional[bool] = None
    image_prompt_per_h2: Optional[bool] = None
    progress_indicator: Optional[bool] = None
    overwrite_url_cache: Optional[bool] = None

class FeatureReq(BaseModel):
    feature: str
    enable: bool = True

class ProfileReq(BaseModel):
    name: str

class JobReq(BaseModel):
    titles: Optional[List[str]] = None
    csv_path: Optional[str] = None
    site_config: Optional[Dict[str, Any]] = None
    profile_name: Optional[str] = None
    wait: bool = False

class ClickReq(BaseModel):
    name: Optional[str] = None
    auto_id: Optional[str] = None
    title_re: Optional[str] = None

class TextReq(BaseModel):
    name: Optional[str] = None
    auto_id: Optional[str] = None
    value: str
    fast: bool = True

class DropdownReq(BaseModel):
    name: Optional[str] = None
    auto_id: Optional[str] = None
    value: str

class CheckboxSingleReq(BaseModel):
    name: Optional[str] = None
    auto_id: Optional[str] = None
    checked: bool = True

class WordPressConfigReq(BaseModel):
    site_url: Optional[str] = None
    user_name: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    author: Optional[str] = None
    article_status: str = "draft"
    disable_meta_desc: bool = False
    disable_auto_tags: bool = False

class SerpScrapingConfigReq(BaseModel):
    country: Optional[str] = None
    language: Optional[str] = None
    enable: bool = True

class DeepResearchConfigReq(BaseModel):
    ai_model: Optional[str] = None
    links_per_article: Optional[str] = None
    links_per_subheading: Optional[str] = None

class LinkPackConfigReq(BaseModel):
    pack_name: Optional[str] = None
    insertion_limit: Optional[str] = None

class StyleMimicConfigReq(BaseModel):
    style_text: Optional[str] = None

class CustomOutlineConfigReq(BaseModel):
    outline_text: Optional[str] = None
    outline_name: Optional[str] = None

class CustomPromptConfigReq(BaseModel):
    prompt_text: Optional[str] = None
    prompt_name: Optional[str] = None

class YouTubeVideosConfigReq(BaseModel):
    enable: bool = True
    max_videos: Optional[str] = None

class WebhookConfigReq(BaseModel):
    webhook_url: Optional[str] = None
    webhook_name: Optional[str] = None

class AltImagesConfigReq(BaseModel):
    featured_model: Optional[str] = None
    subheading_model: Optional[str] = None

class SeoCsvConfigReq(BaseModel):
    csv_path: str

class OrchestrateReq(BaseModel):
    jobs: List[Dict[str, Any]]  # List of {domain, titles?, csv_path?, profile_name?}
    delay_between: int = 10


# ═══════════════════════════════════════════
# CONNECTION
# ═══════════════════════════════════════════

@app.post("/connect", tags=["Connection"])
def connect(req: ConnectReq = ConnectReq()):
    global _ctrl
    _ctrl = ZimmWriterController(exe_path=req.exe_path)
    if _ctrl.connect():
        return {"status": "connected", "window": _ctrl.get_window_title()}
    raise HTTPException(503, "Could not connect")

@app.post("/launch", tags=["Connection"])
def launch(req: ConnectReq = ConnectReq()):
    global _ctrl
    _ctrl = ZimmWriterController(exe_path=req.exe_path)
    if _ctrl.launch():
        return {"status": "launched", "window": _ctrl.get_window_title()}
    raise HTTPException(503, "Could not launch")

@app.get("/status", tags=["Connection"])
def status():
    return get_ctrl().get_status()

@app.get("/is-running", tags=["Connection"])
def is_running():
    return {"running": ZimmWriterController().is_running()}

@app.post("/bring-to-front", tags=["Connection"])
def bring_to_front():
    get_ctrl().bring_to_front()
    return {"status": "ok"}

@app.post("/screenshot", tags=["Connection"])
def screenshot():
    path = get_ctrl().take_screenshot()
    return {"status": "ok", "filepath": path}


# ═══════════════════════════════════════════
# ELEMENT DISCOVERY
# ═══════════════════════════════════════════

@app.get("/controls/dump", tags=["Discovery"])
def dump_controls():
    return {"controls": get_ctrl().dump_controls()}

@app.get("/controls/buttons", tags=["Discovery"])
def list_buttons():
    return {"buttons": get_ctrl().get_all_buttons()}

@app.get("/controls/checkboxes", tags=["Discovery"])
def list_checkboxes():
    return {"checkboxes": get_ctrl().get_all_checkboxes()}

@app.get("/controls/dropdowns", tags=["Discovery"])
def list_dropdowns():
    return {"dropdowns": get_ctrl().get_all_dropdowns()}

@app.get("/controls/text-fields", tags=["Discovery"])
def list_text_fields():
    return {"text_fields": get_ctrl().get_all_text_fields()}


# ═══════════════════════════════════════════
# GENERIC INTERACTIONS
# ═══════════════════════════════════════════

@app.post("/click", tags=["Generic"])
def click(req: ClickReq):
    get_ctrl().click_button(name=req.name, auto_id=req.auto_id, title_re=req.title_re)
    return {"status": "clicked"}

@app.post("/set-text", tags=["Generic"])
def set_text(req: TextReq):
    zw = get_ctrl()
    if req.fast:
        zw.set_text_fast(name=req.name, auto_id=req.auto_id, value=req.value)
    else:
        zw.set_text_field(name=req.name, auto_id=req.auto_id, value=req.value)
    return {"status": "ok"}

@app.post("/set-dropdown", tags=["Generic"])
def set_dropdown(req: DropdownReq):
    get_ctrl().set_dropdown(name=req.name, auto_id=req.auto_id, value=req.value)
    return {"status": "ok"}

@app.post("/set-checkbox", tags=["Generic"])
def set_checkbox_single(req: CheckboxSingleReq):
    get_ctrl().set_checkbox(name=req.name, auto_id=req.auto_id, checked=req.checked)
    return {"status": "ok"}


# ═══════════════════════════════════════════
# ZIMMWRITER-SPECIFIC
# ═══════════════════════════════════════════

@app.post("/titles", tags=["Bulk Writer"])
def set_titles(req: TitlesReq):
    get_ctrl().set_bulk_titles(req.titles, req.separator)
    return {"status": "ok", "count": len(req.titles)}

@app.post("/load-csv", tags=["Bulk Writer"])
def load_csv(req: CSVReq):
    get_ctrl().load_seo_csv(req.csv_path)
    return {"status": "ok", "csv": req.csv_path}

@app.post("/configure", tags=["Bulk Writer"])
def configure(req: ConfigureReq):
    get_ctrl().configure_bulk_writer(**req.model_dump(exclude_none=True))
    return {"status": "configured"}

@app.post("/checkboxes", tags=["Bulk Writer"])
def set_checkboxes(req: CheckboxReq):
    get_ctrl().set_checkboxes(**req.model_dump(exclude_none=True))
    return {"status": "ok"}

@app.post("/feature-toggle", tags=["Bulk Writer"])
def feature_toggle(req: FeatureReq):
    get_ctrl().toggle_feature(req.feature, req.enable)
    return {"status": "ok", "feature": req.feature, "enabled": req.enable}


# ═══════════════════════════════════════════
# CONFIG WINDOWS
# ═══════════════════════════════════════════

@app.post("/config/wordpress", tags=["Config Windows"])
def config_wordpress(req: WordPressConfigReq):
    result = get_ctrl().configure_wordpress_upload(
        site_url=req.site_url, user_name=req.user_name,
        category=req.category, sub_category=req.sub_category,
        author=req.author, article_status=req.article_status,
        disable_meta_desc=req.disable_meta_desc,
        disable_auto_tags=req.disable_auto_tags,
    )
    return {"status": "configured" if result else "failed", "feature": "wordpress"}

@app.post("/config/serp-scraping", tags=["Config Windows"])
def config_serp(req: SerpScrapingConfigReq):
    result = get_ctrl().configure_serp_scraping(
        country=req.country, language=req.language, enable=req.enable,
    )
    return {"status": "configured" if result else "failed", "feature": "serp_scraping"}

@app.post("/config/deep-research", tags=["Config Windows"])
def config_deep_research(req: DeepResearchConfigReq):
    result = get_ctrl().configure_deep_research(
        ai_model=req.ai_model, links_per_article=req.links_per_article,
        links_per_subheading=req.links_per_subheading,
    )
    return {"status": "configured" if result else "failed", "feature": "deep_research"}

@app.post("/config/link-pack", tags=["Config Windows"])
def config_link_pack(req: LinkPackConfigReq):
    result = get_ctrl().configure_link_pack(
        pack_name=req.pack_name, insertion_limit=req.insertion_limit,
    )
    return {"status": "configured" if result else "failed", "feature": "link_pack"}

@app.post("/config/style-mimic", tags=["Config Windows"])
def config_style_mimic(req: StyleMimicConfigReq):
    result = get_ctrl().configure_style_mimic(style_text=req.style_text)
    return {"status": "configured" if result else "failed", "feature": "style_mimic"}

@app.post("/config/custom-outline", tags=["Config Windows"])
def config_custom_outline(req: CustomOutlineConfigReq):
    result = get_ctrl().configure_custom_outline(
        outline_text=req.outline_text, outline_name=req.outline_name,
    )
    return {"status": "configured" if result else "failed", "feature": "custom_outline"}

@app.post("/config/custom-prompt", tags=["Config Windows"])
def config_custom_prompt(req: CustomPromptConfigReq):
    result = get_ctrl().configure_custom_prompt(
        prompt_text=req.prompt_text, prompt_name=req.prompt_name,
    )
    return {"status": "configured" if result else "failed", "feature": "custom_prompt"}

@app.post("/config/youtube-videos", tags=["Config Windows"])
def config_youtube_videos(req: YouTubeVideosConfigReq):
    result = get_ctrl().configure_youtube_videos(
        enable=req.enable, max_videos=req.max_videos,
    )
    return {"status": "configured" if result else "failed", "feature": "youtube_videos"}

@app.post("/config/webhook", tags=["Config Windows"])
def config_webhook(req: WebhookConfigReq):
    result = get_ctrl().configure_webhook(
        webhook_url=req.webhook_url, webhook_name=req.webhook_name,
    )
    return {"status": "configured" if result else "failed", "feature": "webhook"}

@app.post("/config/alt-images", tags=["Config Windows"])
def config_alt_images(req: AltImagesConfigReq):
    result = get_ctrl().configure_alt_images(
        featured_model=req.featured_model, subheading_model=req.subheading_model,
    )
    return {"status": "configured" if result else "failed", "feature": "alt_images"}

@app.post("/config/seo-csv", tags=["Config Windows"])
def config_seo_csv(req: SeoCsvConfigReq):
    result = get_ctrl().configure_seo_csv(csv_path=req.csv_path)
    return {"status": "configured" if result else "failed", "feature": "seo_csv"}


# ═══════════════════════════════════════════
# PROFILES
# ═══════════════════════════════════════════

@app.post("/profile/load", tags=["Profiles"])
def load_profile(req: ProfileReq):
    get_ctrl().load_profile(req.name)
    return {"status": "loaded", "profile": req.name}

@app.post("/profile/save", tags=["Profiles"])
def save_profile(req: ProfileReq):
    get_ctrl().save_profile(req.name)
    return {"status": "saved", "profile": req.name}

@app.post("/profile/update", tags=["Profiles"])
def update_profile():
    get_ctrl().update_profile()
    return {"status": "updated"}


# ═══════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════

@app.post("/start", tags=["Execution"])
def start():
    get_ctrl().start_bulk_writer()
    return {"status": "started"}

@app.post("/stop", tags=["Execution"])
def stop():
    get_ctrl().stop_bulk_writer()
    return {"status": "stopped"}

@app.post("/clear", tags=["Execution"])
def clear():
    get_ctrl().clear_all_data()
    return {"status": "cleared"}

@app.post("/run-job", tags=["Execution"])
def run_job(req: JobReq):
    result = get_ctrl().run_job(
        titles=req.titles, csv_path=req.csv_path,
        site_config=req.site_config, profile_name=req.profile_name,
        wait=req.wait,
    )
    return {"status": "running" if result else "failed"}


# ═══════════════════════════════════════════
# SITE PRESETS
# ═══════════════════════════════════════════

@app.get("/presets", tags=["Presets"])
def list_presets():
    return {
        "presets": {
            domain: {"niche": cfg["niche"], "domain": domain}
            for domain, cfg in SITE_PRESETS.items()
        }
    }

@app.get("/presets/{domain}", tags=["Presets"])
def get_preset_config(domain: str):
    preset = get_preset(domain)
    if not preset:
        raise HTTPException(404, f"No preset for: {domain}")
    return preset

@app.post("/presets/{domain}/apply", tags=["Presets"])
def apply_preset(domain: str):
    preset = get_preset(domain)
    if not preset:
        raise HTTPException(404, f"No preset for: {domain}")
    get_ctrl().apply_site_config(preset)
    return {"status": "applied", "domain": domain}


# ═══════════════════════════════════════════
# ORCHESTRATION (Multi-site)
# ═══════════════════════════════════════════

@app.post("/orchestrate", tags=["Orchestration"])
def orchestrate(req: OrchestrateReq, background_tasks: BackgroundTasks):
    """
    Run multiple jobs across sites. Each job: {domain, titles?, csv_path?, profile_name?}
    Runs in background - check /orchestrate/status for progress.
    """
    orch = Orchestrator(get_ctrl())
    for job in req.jobs:
        orch.add_job(
            domain=job["domain"],
            titles=job.get("titles"),
            csv_path=job.get("csv_path"),
            profile_name=job.get("profile_name"),
            wait=job.get("wait", True),
        )

    # Run in background
    background_tasks.add_task(orch.run_all, req.delay_between)
    return {"status": "orchestration_started", "job_count": len(req.jobs)}


# ═══════════════════════════════════════════
# CAMPAIGN ENGINE
# ═══════════════════════════════════════════

class CampaignPlanReq(BaseModel):
    domain: str
    titles: List[str]

class CampaignRunReq(BaseModel):
    domain: str
    titles: List[str]
    profile_name: Optional[str] = None
    wait: bool = True

class CampaignBatchReq(BaseModel):
    campaigns: List[Dict[str, Any]]  # [{domain, titles, profile_name?, wait?}]
    delay_between: int = 10

@app.post("/campaign/plan", tags=["Campaign"])
def campaign_plan(req: CampaignPlanReq):
    """
    Plan a campaign: analyze titles, detect types, select settings,
    generate SEO CSV. Does NOT execute — returns plan for review.
    """
    engine = CampaignEngine()
    plan, csv_path = engine.plan_and_generate(req.domain, req.titles)
    summary = engine.get_campaign_summary(plan)
    return {
        "status": "planned",
        "csv_path": csv_path,
        "summary": summary,
        "title_types": plan.title_types,
        "per_title_config": plan.per_title_config,
    }

@app.post("/campaign/run", tags=["Campaign"])
def campaign_run(req: CampaignRunReq, background_tasks: BackgroundTasks):
    """
    Plan and execute a single-site campaign with intelligent settings.
    Uses CampaignEngine for title analysis + SEO CSV generation,
    then runs via CampaignOrchestrator.
    """
    co = CampaignOrchestrator(get_ctrl())
    if req.wait:
        result = co.run_campaign(
            domain=req.domain, titles=req.titles,
            profile_name=req.profile_name, wait=True,
        )
        return result
    else:
        background_tasks.add_task(
            co.run_campaign,
            req.domain, req.titles, req.profile_name, True,
        )
        return {"status": "started_in_background", "domain": req.domain}

@app.post("/campaign/batch", tags=["Campaign"])
def campaign_batch(req: CampaignBatchReq, background_tasks: BackgroundTasks):
    """
    Run campaigns across multiple sites. Each item: {domain, titles, profile_name?, wait?}
    Runs in background.
    """
    co = CampaignOrchestrator(get_ctrl())
    background_tasks.add_task(co.run_multi_campaign, req.campaigns, req.delay_between)
    return {"status": "batch_started", "campaign_count": len(req.campaigns)}

@app.post("/campaign/classify", tags=["Campaign"])
def campaign_classify(req: CampaignPlanReq):
    """Classify article titles by type without generating a full plan."""
    types = classify_titles(req.titles)
    dominant = get_dominant_type(req.titles)
    return {
        "title_types": types,
        "dominant_type": dominant,
        "type_counts": dict(__import__("collections").Counter(types.values())),
    }


# ═══════════════════════════════════════════
# SCREEN NAVIGATION
# ═══════════════════════════════════════════

@app.get("/screen/current", tags=["Navigation"])
def screen_current():
    """Detect which ZimmWriter screen is currently active."""
    ctrl = get_ctrl()
    nav = ScreenNavigator(ctrl)
    screen = nav.detect_screen()
    return {"screen": screen.value, "title": ctrl.get_window_title()}

@app.get("/screen/available", tags=["Navigation"])
def screen_available():
    """List all navigable screens with their menu button info."""
    return {
        "screens": [
            {"screen": s.value, "auto_id": info["auto_id"], "label": info["label"]}
            for s, info in MENU_BUTTONS.items()
        ]
    }

class NavigateReq(BaseModel):
    screen: str = Field(..., description="Target screen name (e.g., 'bulk_writer', 'seo_writer')")

@app.post("/screen/navigate", tags=["Navigation"])
def screen_navigate(req: NavigateReq):
    """Navigate to a specific screen (via Menu hub)."""
    try:
        target = Screen(req.screen)
    except ValueError:
        raise HTTPException(400, f"Unknown screen: {req.screen}")
    ctrl = get_ctrl()
    nav = ScreenNavigator(ctrl)
    success = nav.navigate_to(target)
    return {"success": success, "screen": nav.detect_screen().value}

@app.post("/screen/menu", tags=["Navigation"])
def screen_menu():
    """Navigate back to Menu from any screen."""
    ctrl = get_ctrl()
    nav = ScreenNavigator(ctrl)
    success = nav.back_to_menu()
    return {"success": success, "screen": nav.detect_screen().value}


# ═══════════════════════════════════════════
# LINK PACKS
# ═══════════════════════════════════════════

class LinkPackBuildReq(BaseModel):
    domain: str = Field(..., description="Domain to build link pack for")
    max_posts: int = Field(100, description="Maximum posts to fetch")

@app.post("/link-packs/build", tags=["Link Packs"])
def link_pack_build(req: LinkPackBuildReq):
    """Build a link pack for a site by scraping its WordPress REST API."""
    preset = get_preset(req.domain)
    if not preset:
        raise HTTPException(404, f"No preset for: {req.domain}")
    wp = preset.get("wordpress_settings", {})
    site_url = wp.get("site_url")
    if not site_url:
        raise HTTPException(400, f"No site_url in preset for: {req.domain}")

    builder = LinkPackBuilder()
    pack_text = builder.build_pack(req.domain, site_url, max_posts=req.max_posts)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "link_packs")
    os.makedirs(output_dir, exist_ok=True)
    link_count = len(pack_text.strip().split("\n")) if pack_text else 0
    path = builder.save_pack(req.domain, pack_text, output_dir)
    return {"domain": req.domain, "links": link_count, "path": path}

@app.post("/link-packs/build-all", tags=["Link Packs"])
def link_pack_build_all():
    """Build link packs for all sites that have WordPress settings."""
    builder = LinkPackBuilder()
    results = []
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "link_packs")
    os.makedirs(output_dir, exist_ok=True)
    for domain, config in SITE_PRESETS.items():
        wp = config.get("wordpress_settings", {})
        site_url = wp.get("site_url")
        if not site_url:
            results.append({"domain": domain, "status": "skipped", "reason": "no site_url"})
            continue
        try:
            pack_text = builder.build_pack(domain, site_url, max_posts=100)
            link_count = len(pack_text.strip().split("\n")) if pack_text else 0
            path = builder.save_pack(domain, pack_text, output_dir)
            results.append({"domain": domain, "status": "ok", "links": link_count, "path": path})
        except Exception as e:
            results.append({"domain": domain, "status": "error", "error": str(e)})
    return {"results": results, "built": sum(1 for r in results if r["status"] == "ok")}

@app.get("/link-packs/list", tags=["Link Packs"])
def link_pack_list():
    """List all saved link pack files."""
    pack_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "link_packs")
    if not os.path.isdir(pack_dir):
        return {"packs": []}
    packs = []
    for f in os.listdir(pack_dir):
        if f.endswith(".txt"):
            path = os.path.join(pack_dir, f)
            with open(path, encoding="utf-8") as fh:
                line_count = sum(1 for _ in fh)
            packs.append({"name": f, "path": path, "links": line_count})
    return {"packs": packs}


# ═══════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "name": "ZimmWriter Controller API",
        "version": "1.2.0",
        "docs": "http://localhost:8765/docs",
        "sites_configured": len(SITE_PRESETS),
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ═══════════════════════════════════════════
# INTELLIGENCE (FORGE + AMPLIFY + Vision + Screenpipe)
# ═══════════════════════════════════════════

class PreJobReq(BaseModel):
    domain: str
    titles: Optional[List[str]] = None
    config_overrides: Optional[Dict[str, Any]] = None

class PostJobReq(BaseModel):
    job_id: str
    success: bool
    duration_seconds: int = 0
    error: Optional[str] = None
    articles_generated: int = 0

class EnhancedJobReq(BaseModel):
    domain: str
    titles: Optional[List[str]] = None
    profile_name: Optional[str] = None
    wait: bool = True

class VisionVerifyReq(BaseModel):
    expected_screen: Optional[str] = None

@app.post("/intelligence/pre-job", tags=["Intelligence"])
def pre_job_analysis(req: PreJobReq):
    """
    Run full pre-job analysis: FORGE audit + AMPLIFY enhancement + validation.
    Returns readiness assessment, auto-fixes, and risk prediction.
    """
    intel = get_intel()
    preset = get_preset(req.domain)
    if not preset:
        raise HTTPException(404, f"No preset for: {req.domain}")

    config = {**preset}
    if req.config_overrides:
        config.update(req.config_overrides)

    result = intel.pre_job(config, req.titles)
    return result

@app.post("/intelligence/post-job", tags=["Intelligence"])
def post_job_learning(req: PostJobReq):
    """Record job outcome for FORGE Codex learning."""
    intel = get_intel()
    intel.post_job(
        req.job_id, req.success, req.duration_seconds,
        req.error, req.articles_generated,
        controller=_ctrl,
    )
    return {"status": "recorded", "job_id": req.job_id}

@app.post("/intelligence/enhanced-run", tags=["Intelligence"])
def enhanced_run(req: EnhancedJobReq, background_tasks: BackgroundTasks):
    """
    Run a complete job with full intelligence integration:
    FORGE analysis + AMPLIFY enhancement + Vision verification + Screenpipe monitoring.
    """
    intel = get_intel()
    ctrl = get_ctrl()

    preset = get_preset(req.domain)
    if not preset:
        raise HTTPException(404, f"No preset for: {req.domain}")

    if req.wait:
        result = intel.enhanced_run_job(
            ctrl, preset, req.titles, req.profile_name, wait=True,
        )
        return result
    else:
        background_tasks.add_task(
            intel.enhanced_run_job,
            ctrl, preset, req.titles, req.profile_name, True,
        )
        return {"status": "started_in_background", "domain": req.domain}

@app.get("/intelligence/progress/{job_id}", tags=["Intelligence"])
def check_job_progress(job_id: str):
    """Check job progress via Screenpipe + Vision monitoring."""
    intel = get_intel()
    return intel.check_progress(job_id, controller=_ctrl)

@app.post("/intelligence/verify-screen", tags=["Intelligence"])
def verify_screen(req: VisionVerifyReq):
    """Verify current ZimmWriter screen via Vision Service."""
    intel = get_intel()
    ctrl = get_ctrl()
    expected = req.expected_screen or "Bulk Writer"
    return intel.verify_screen(expected, controller=ctrl)

@app.post("/intelligence/detect-errors", tags=["Intelligence"])
def detect_errors_intel():
    """Check for errors using both Vision and Screenpipe."""
    intel = get_intel()
    return intel.detect_errors(controller=_ctrl)

@app.get("/intelligence/screenpipe/state", tags=["Intelligence"])
def screenpipe_state():
    """Read current ZimmWriter state via Screenpipe OCR."""
    intel = get_intel()
    return intel.screenpipe.read_current_state()

@app.get("/intelligence/screenpipe/errors", tags=["Intelligence"])
def screenpipe_errors(minutes_back: int = 10):
    """Search recent Screenpipe captures for ZimmWriter errors."""
    intel = get_intel()
    return {"errors": intel.screenpipe.search_errors(minutes_back)}

@app.get("/intelligence/screenpipe/timeline", tags=["Intelligence"])
def screenpipe_timeline(hours_back: int = 4):
    """Get ZimmWriter activity timeline from Screenpipe."""
    intel = get_intel()
    return {"timeline": intel.screenpipe.get_activity_timeline(hours_back)}

@app.get("/intelligence/stats", tags=["Intelligence"])
def intelligence_stats():
    """Get comprehensive intelligence system statistics."""
    intel = get_intel()
    return intel.get_stats()

@app.get("/intelligence/jobs", tags=["Intelligence"])
def active_jobs():
    """Get all currently tracked jobs."""
    intel = get_intel()
    return intel.get_active_jobs()

@app.get("/intelligence/forge/stats", tags=["Intelligence"])
def forge_stats():
    """Get FORGE Intelligence Engine statistics."""
    intel = get_intel()
    return intel.forge.get_stats()

@app.get("/intelligence/codex/history/{domain}", tags=["Intelligence"])
def codex_domain_history(domain: str):
    """Get FORGE Codex learning data for a domain."""
    intel = get_intel()
    history = intel.forge.codex.get_domain_history(domain)
    if not history:
        return {"domain": domain, "history": None, "message": "No data yet"}
    return {"domain": domain, "history": history}


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  ZimmWriter Controller API v1.0")
    print("  http://localhost:8765")
    print("  Docs: http://localhost:8765/docs")
    print(f"  Sites: {len(SITE_PRESETS)}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8765)
