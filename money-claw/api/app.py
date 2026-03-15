"""MoneyClaw FastAPI — Web frontend + API for readings, auth, and management.

Serves Jinja2 templates for the Mystic Luna web experience,
plus JSON API endpoints for AJAX reading generation and admin tools.
Port 8160 (configurable).
"""

import hashlib
import hmac
import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, Form
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from moneyclaw.brain import Brain
from moneyclaw.memory import Memory
from moneyclaw.scheduler import Scheduler
from moneyclaw.agents import LunaGuardian
from moneyclaw.auth import Auth
from moneyclaw.services.luna.persona import get_moon_phase, ZODIAC_SIGNS, ZODIAC_DATA, ZODIAC_LOOKUP
from moneyclaw.services.luna.companion import CompanionEngine, MOODS, RELATIONSHIP_LEVELS
from moneyclaw.services.avatar.pipeline import AvatarPipeline
from moneyclaw.services.email import EmailService
from moneyclaw.services.luna.luna_engine import LunaEngine
from moneyclaw.config import get_config, VIDEOS_DIR, AUDIO_DIR
from moneyclaw.services.telegram_bot import get_bot

logger = logging.getLogger("moneyclaw.api")

memory = Memory()
brain = Brain(memory)
scheduler = Scheduler(brain)
auth = Auth(memory)
companion = CompanionEngine(memory)
avatar = AvatarPipeline(memory)
email_service = EmailService()
luna_engine = LunaEngine(memory)
guardian = LunaGuardian()
config = get_config()

# Template and static paths
WEB_DIR = Path(__file__).parent.parent / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler
    scheduler.start()
    logger.info("MoneyClaw API started — scheduler running")

    # Start Telegram bot (non-blocking background task)
    tg_bot = get_bot(memory)
    if tg_bot:
        try:
            await tg_bot.start()
            logger.info("Luna Telegram bot started")
        except Exception as e:
            logger.warning("Telegram bot failed to start: %s", e)
    else:
        logger.info("Telegram bot not configured — skipping")

    yield

    # Stop Telegram bot
    if tg_bot:
        try:
            await tg_bot.stop()
        except Exception as e:
            logger.debug("Telegram bot stop error: %s", e)

    scheduler.stop()
    logger.info("MoneyClaw API stopped")


app = FastAPI(
    title="MoneyClaw",
    description="Autonomous revenue-generating AI agent — Mystic Luna spiritual readings",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if config.log_level == "DEBUG" else None,
    redoc_url=None,
)

# CORS — allow the site domain plus localhost for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        config.site_url,
        "http://localhost:8160",
        "http://127.0.0.1:8160",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Mount static files and generated media
app.mount("/media/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="media_videos")
app.mount("/media/audio", StaticFiles(directory=str(AUDIO_DIR)), name="media_audio")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ═══════════════════════════════════════════════════════════════════════
# Rate Limiting (in-memory, simple per-IP)
# ═══════════════════════════════════════════════════════════════════════

_rate_limits: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 30  # per window for reading generation


def _check_rate_limit(client_ip: str, limit: int = RATE_LIMIT_MAX_REQUESTS) -> bool:
    """Return True if request is allowed, False if rate limited."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    # Clean old entries
    _rate_limits[client_ip] = [t for t in _rate_limits[client_ip] if t > window_start]
    if len(_rate_limits[client_ip]) >= limit:
        return False
    _rate_limits[client_ip].append(now)
    return True


# ═══════════════════════════════════════════════════════════════════════
# Error Handlers
# ═══════════════════════════════════════════════════════════════════════

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    ctx = template_context(
        request,
        status_code=404,
        error_title="Page Not Found",
        error_message="The spirits cannot find what you seek. Perhaps it moved to another plane.",
    )
    return templates.TemplateResponse("error.html", ctx, status_code=404)


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    logger.error("Internal error on %s: %s", request.url.path, exc)
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
    ctx = template_context(
        request,
        status_code=500,
        error_title="The Ether Is Disturbed",
        error_message="Something unexpected happened. Luna is regrouping. Please try again.",
    )
    return templates.TemplateResponse("error.html", ctx, status_code=500)


# ═══════════════════════════════════════════════════════════════════════
# Auth Middleware Helper
# ═══════════════════════════════════════════════════════════════════════

def get_current_user(request: Request) -> dict | None:
    """Read session cookie and return user or None."""
    token = request.cookies.get("session_token")
    if not token:
        return None
    return auth.get_user_by_token(token)


def template_context(request: Request, **extra) -> dict:
    """Build common template context with user, moon, and request."""
    user = get_current_user(request)
    moon = get_moon_phase()
    return {"request": request, "user": user, "moon": moon, **extra}


# ═══════════════════════════════════════════════════════════════════════
# Web Routes — Server-Rendered Pages
# ═══════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    readings = brain.readings.get_available_readings()
    ctx = template_context(request, readings=readings)
    return templates.TemplateResponse("landing.html", ctx)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = get_current_user(request)
    if user:
        return RedirectResponse("/dashboard", status_code=302)
    ctx = template_context(request, error=None, success=None)
    return templates.TemplateResponse("login.html", ctx)


@app.post("/login")
async def login_submit(request: Request,
                       email: str = Form(...),
                       password: str = Form(...)):
    result = auth.login(email, password)
    if "error" in result:
        ctx = template_context(request, error=result["error"], success=None)
        return templates.TemplateResponse("login.html", ctx)

    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie(
        key="session_token",
        value=result["token"],
        httponly=True,
        max_age=86400 * 30,
        samesite="lax",
    )
    return response


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, plan: str = ""):
    """Register page — ?plan= query param is preserved so post-registration
    the user can be redirected to Stripe checkout for that plan."""
    user = get_current_user(request)
    if user:
        if plan:
            return RedirectResponse(f"/api/subscribe?plan={plan}", status_code=302)
        return RedirectResponse("/dashboard", status_code=302)
    ctx = template_context(request, error=None, intended_plan=plan)
    return templates.TemplateResponse("register.html", ctx)


@app.post("/register")
async def register_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(""),
    zodiac: str = Form(""),
    intended_plan: str = Form(""),
):
    """Register and redirect to Stripe checkout if an intended_plan was stored."""
    result = auth.register(email, password, name=name, zodiac_sign=zodiac)
    if "error" in result:
        ctx = template_context(request, error=result["error"], intended_plan=intended_plan)
        return templates.TemplateResponse("register.html", ctx)

    # Send welcome email (non-blocking)
    try:
        email_service.send_welcome_email(to_email=email, name=name or "Seeker")
    except Exception as e:
        logger.debug("Welcome email failed: %s", e)

    # After successful registration, go to plan checkout or dashboard
    if intended_plan and intended_plan in ("seeker", "mystic", "inner-circle"):
        redirect_url = f"/api/subscribe?plan={intended_plan}"
    else:
        redirect_url = "/dashboard"

    response = RedirectResponse(redirect_url, status_code=302)
    response.set_cookie(
        key="session_token",
        value=result["token"],
        httponly=True,
        max_age=86400 * 30,
        samesite="lax",
    )
    return response


@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session_token")
    if token:
        auth.logout(token)
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("session_token")
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    moon = get_moon_phase()
    readings = auth.get_user_readings(user["id"])

    ctx = template_context(request, readings=readings)
    return templates.TemplateResponse("dashboard.html", ctx)


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    user = get_current_user(request)
    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    financial = brain.ledger.get_financial_summary()
    customers = memory.get_customer_stats()
    services = memory.get_popular_services()
    learnings = memory.get_top_learnings(limit=10)
    experiments = memory.get_active_experiments()
    user_count = auth.get_user_count()
    avatar_health = avatar.status
    sched_status = scheduler.get_status()

    ctx = template_context(
        request,
        brain_state=brain.state.value,
        financial=financial,
        customers=customers,
        services=services,
        learnings=learnings,
        experiments=experiments,
        user_count=user_count,
        avatar_status=avatar_health,
        email_available=email_service.available,
        telegram_configured=bool(config.telegram.bot_token),
        scheduler_status=sched_status,
    )
    return templates.TemplateResponse("admin.html", ctx)


@app.post("/api/admin/daily-post")
async def api_admin_daily_post(request: Request):
    """Manually trigger the daily Telegram content post (admin only).

    Useful for testing the post format and confirming Telegram delivery
    without waiting for the 6 AM scheduled job.
    """
    user = get_current_user(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        from moneyclaw.services.daily_poster import run_daily_post
        result = run_daily_post()
        return result
    except Exception as e:
        logger.error("Manual daily post failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pricing", response_class=HTMLResponse)
async def pricing_page(request: Request):
    readings = brain.readings.get_available_readings()
    ctx = template_context(request, readings=readings)
    return templates.TemplateResponse("pricing.html", ctx)


@app.get("/free-reading", response_class=HTMLResponse)
async def free_reading_page(request: Request):
    ctx = template_context(request)
    return templates.TemplateResponse("free_reading.html", ctx)


@app.get("/reading/{spread_type}", response_class=HTMLResponse)
async def reading_page(request: Request, spread_type: str):
    available = brain.readings.get_available_readings()
    spread = None
    for r in available:
        if r["id"] == spread_type:
            spread = r
            break

    if not spread:
        return RedirectResponse("/pricing", status_code=302)

    ctx = template_context(request, spread=spread, spread_type=spread_type)
    return templates.TemplateResponse("reading.html", ctx)


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Live chat with Luna — the core experience."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login?next=/chat", status_code=302)
    ctx = template_context(request)
    return templates.TemplateResponse("chat.html", ctx)


@app.get("/luna-live", response_class=HTMLResponse)
async def luna_live_page(request: Request):
    """Interactive face-to-face session with Luna's AI avatar."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login?next=/luna-live", status_code=302)
    ctx = template_context(request)
    return templates.TemplateResponse("luna_live.html", ctx)


@app.get("/companion", response_class=HTMLResponse)
async def companion_page(request: Request):
    """Luna's companion page — relationship tracker, mood, chat."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    # Record visit and get greeting
    visit = companion.record_visit(user["id"], user.get("name", "Seeker"))
    profile = companion.get_profile(user["id"])
    card_patterns = companion.get_card_patterns(user["id"])
    mood_patterns = companion._detect_mood_patterns(user["id"])
    messages = companion.get_conversation_history(user["id"], limit=10)

    ctx = template_context(
        request,
        visit=visit,
        profile=profile,
        card_patterns=card_patterns,
        mood_patterns=mood_patterns,
        messages=messages,
        moods=MOODS,
        levels=RELATIONSHIP_LEVELS,
    )
    return templates.TemplateResponse("companion.html", ctx)


# ═══════════════════════════════════════════════════════════════════════
# Post-Payment Reading Delivery
# ═══════════════════════════════════════════════════════════════════════

@app.get("/reading/complete", response_class=HTMLResponse)
async def reading_complete_page(request: Request, session_id: str = ""):
    """Post-payment page — retrieves and displays the completed reading."""
    user = get_current_user(request)
    moon = get_moon_phase()

    # Check Accept header for JSON polling
    if "application/json" in request.headers.get("accept", ""):
        # Polling check from the frontend
        if session_id:
            reading_data = _find_reading_by_stripe_session(session_id)
            if reading_data:
                return JSONResponse({"ready": True})
        return JSONResponse({"ready": False})

    reading_text = None
    spread_name = None
    cards = None
    video_path = None
    audio_path = None
    email_sent = False
    error = None

    if not session_id:
        error = "No session ID provided. If you completed a payment, check your email for your reading."
    else:
        # Look up reading by Stripe session metadata
        reading_data = _find_reading_by_stripe_session(session_id)

        if reading_data:
            reading_text = reading_data.get("response", "")
            spread_name = reading_data.get("spread_type", "Your Reading")
            cards = reading_data.get("cards", [])
            video_path = reading_data.get("video_path")
            audio_path = reading_data.get("audio_path")
            email_sent = reading_data.get("email_sent", False)
        else:
            # Reading may still be generating — show processing state
            reading_text = None

    ctx = template_context(
        request,
        reading=reading_text,
        spread_name=spread_name,
        cards=cards or [],
        video_path=video_path,
        audio_path=audio_path,
        email_sent=email_sent,
        error=error,
    )
    return templates.TemplateResponse("reading_complete.html", ctx)


def _find_reading_by_stripe_session(session_id: str) -> dict | None:
    """Look up a reading by its Stripe session ID."""
    try:
        with memory._conn() as conn:
            row = conn.execute(
                """SELECT * FROM interactions
                   WHERE channel = 'stripe'
                   AND interaction_type = 'reading'
                   ORDER BY created_at DESC LIMIT 1"""
            ).fetchone()
            if row:
                return dict(row)
    except Exception as e:
        logger.error("Failed to find reading for session %s: %s", session_id, e)
    return None


# ═══════════════════════════════════════════════════════════════════════
# Legal Pages
# ═══════════════════════════════════════════════════════════════════════

TERMS_CONTENT = """
<h2>1. Nature of Services</h2>
<p>Mystic Luna provides AI-powered spiritual guidance, tarot readings, astrology insights, and related content. All readings are generated by artificial intelligence and are presented for entertainment and personal reflection purposes.</p>
<p><strong>Important:</strong> Luna's readings are not a substitute for professional medical, psychological, legal, or financial advice. If you are in crisis, please contact a qualified professional.</p>

<h2>2. AI Transparency</h2>
<p>Luna Moonshadow is an AI-powered spiritual guide. We are transparent about this: all readings, horoscopes, and guidance are generated by artificial intelligence trained on centuries of mystical and spiritual tradition. No human psychic or reader is involved in generating your reading.</p>

<h2>3. Payment & Delivery</h2>
<p>Paid readings are delivered instantly upon payment confirmation. All payments are processed securely through Stripe. Prices are in USD and include all applicable fees.</p>

<h2>4. Refund Policy</h2>
<p>We want you to be satisfied with your reading. If you are not satisfied, contact us within 48 hours of your reading for a full refund. Monthly subscriptions can be canceled at any time; you'll retain access through the end of your billing period.</p>

<h2>5. Intellectual Property</h2>
<p>Your readings are yours. You may save, print, and share your readings freely. The Mystic Luna brand, website design, and underlying technology remain our property.</p>

<h2>6. Acceptable Use</h2>
<p>You agree not to use Mystic Luna's services for any harmful, fraudulent, or illegal purpose. You must be 18 years or older to purchase readings.</p>

<h2>7. Limitation of Liability</h2>
<p>Mystic Luna provides spiritual guidance for personal reflection. We make no guarantees about the accuracy of predictions or outcomes. You assume full responsibility for decisions made based on readings.</p>

<h2>8. Contact</h2>
<p>Questions about these terms? Reach out through our Telegram channel or email support.</p>
"""

PRIVACY_CONTENT = """
<h2>1. Information We Collect</h2>
<p><strong>Account Information:</strong> Email address, name (optional), zodiac sign (optional), birth date (optional).</p>
<p><strong>Reading Data:</strong> Your questions, card draws, and reading history to personalize your experience.</p>
<p><strong>Payment Data:</strong> Processed securely by Stripe. We do not store your credit card information.</p>
<p><strong>Usage Data:</strong> Pages visited, features used, and interaction patterns to improve our service.</p>

<h2>2. How We Use Your Data</h2>
<ul>
<li>Generate personalized readings and horoscopes</li>
<li>Track your spiritual journey across sessions</li>
<li>Send reading deliveries and account notifications</li>
<li>Improve Luna's reading quality and accuracy</li>
</ul>

<h2>3. AI Processing</h2>
<p>Your questions and context are processed by AI models (Anthropic Claude) to generate readings. Reading text (first 500 characters) is stored for quality improvement. Full questions and responses are processed in memory and not permanently stored in raw form.</p>

<h2>4. Data Sharing</h2>
<p>We do not sell your personal data. Data is shared only with:</p>
<ul>
<li><strong>Stripe</strong> — payment processing</li>
<li><strong>Anthropic</strong> — AI reading generation (anonymized)</li>
<li><strong>ElevenLabs / Hedra</strong> — voice and video generation for avatar readings (anonymized)</li>
</ul>

<h2>5. Data Retention</h2>
<p>Account data is retained while your account is active. Reading history is stored indefinitely to enable pattern tracking. You may request deletion of your data at any time.</p>

<h2>6. Your Rights</h2>
<p>You have the right to access, correct, or delete your personal data. Contact us to exercise these rights.</p>

<h2>7. Security</h2>
<p>We use industry-standard encryption, secure authentication, and password hashing (PBKDF2-SHA256) to protect your data.</p>
"""

REFUND_CONTENT = """
<h2>Our Promise</h2>
<p>We want every reading to resonate with your spirit. If it doesn't, we'll make it right.</p>

<h2>Single Readings</h2>
<p>If you're not satisfied with a paid reading, contact us within <strong>48 hours</strong> of delivery for a full refund. No questions asked.</p>

<h2>Monthly Subscriptions</h2>
<p>Cancel anytime. You'll retain access through the end of your current billing period. No partial-month refunds, but we will refund the full current month if you cancel within the first 7 days.</p>

<h2>How to Request a Refund</h2>
<p>Contact us via Telegram or email with your order details. Refunds are processed within 5-7 business days back to your original payment method.</p>

<h2>Exceptions</h2>
<p>Free readings are, well, free — no refund needed. Custom ritual/spell designs are non-refundable once delivered, as they require significant personalization work.</p>
"""


@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    ctx = template_context(request, page_title="Terms of Service", content=TERMS_CONTENT)
    return templates.TemplateResponse("legal.html", ctx)


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    ctx = template_context(request, page_title="Privacy Policy", content=PRIVACY_CONTENT)
    return templates.TemplateResponse("legal.html", ctx)


@app.get("/refund-policy", response_class=HTMLResponse)
async def refund_page(request: Request):
    ctx = template_context(request, page_title="Refund Policy", content=REFUND_CONTENT)
    return templates.TemplateResponse("legal.html", ctx)


# ═══════════════════════════════════════════════════════════════════════
# AJAX API — Called by frontend JavaScript
# ═══════════════════════════════════════════════════════════════════════

class FreeReadingRequest(BaseModel):
    spread_type: str = "daily_pull"
    question: str | None = None
    email: str | None = None


class CheckoutRequest(BaseModel):
    spread_type: str
    question: str | None = None
    email: str | None = None


@app.post("/api/free-reading")
async def api_free_reading(request: Request, req: FreeReadingRequest):
    """Generate a free reading via AJAX."""
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip, limit=10):
        raise HTTPException(status_code=429, detail="Too many readings requested. Please wait a moment.")

    customer_id = None
    if req.email:
        customer_id = hashlib.sha256(f"web:{req.email}".encode()).hexdigest()[:16]

    try:
        result = await brain.readings.generate_reading(
            spread_type=req.spread_type,
            question=req.question,
            customer_id=customer_id,
            channel="web",
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Free reading generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Reading generation failed")


@app.post("/api/checkout")
async def api_checkout(req: CheckoutRequest):
    """Create a Stripe Checkout Session for a paid reading."""
    available = brain.readings.get_available_readings()
    spread = None
    for r in available:
        if r["id"] == req.spread_type:
            spread = r
            break

    if not spread:
        raise HTTPException(status_code=404, detail="Spread type not found")

    if spread["price"] in ("free", "Free"):
        raise HTTPException(status_code=400, detail="This reading is free")

    # Try Stripe
    stripe_key = config.stripe.api_key
    if stripe_key:
        try:
            import stripe
            stripe.api_key = stripe_key

            price_cents = round(float(spread["price"]) * 100)

            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": f"Mystic Luna — {spread['name']}",
                            "description": f"{spread['cards']}-card personalized tarot reading",
                        },
                        "unit_amount": price_cents,
                    },
                    "quantity": 1,
                }],
                mode="payment",
                success_url=config.site_url + "/reading/complete?session_id={CHECKOUT_SESSION_ID}",
                cancel_url=config.site_url + "/reading/" + req.spread_type,
                metadata={
                    "spread_type": req.spread_type,
                    "question": req.question or "",
                    "email": req.email or "",
                },
                customer_email=req.email if req.email else None,
            )

            return {"checkout_url": session.url}

        except ImportError:
            logger.warning("stripe package not installed, falling back to direct generation")
        except Exception as e:
            logger.error("Stripe checkout failed: %s", e)

    # Fallback: no Stripe configured — generate reading directly
    return {"checkout_url": None, "message": "Stripe not configured — use free reading endpoint"}


# ═══════════════════════════════════════════════════════════════════════
# Companion API — Mood, Chat, Relationship
# ═══════════════════════════════════════════════════════════════════════

class MoodRequest(BaseModel):
    mood: str
    energy_level: int = 3
    notes: str = ""


@app.post("/api/companion/mood")
async def api_companion_mood(request: Request, req: MoodRequest):
    """Log mood and get Luna's empathetic response."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to log your mood")
    result = companion.log_mood(
        user["id"], req.mood, req.energy_level, req.notes,
        user_name=user.get("name", "Seeker")
    )
    return result


@app.get("/api/companion/profile")
async def api_companion_profile(request: Request):
    """Get companion profile for current user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    return companion.get_profile(user["id"])


@app.get("/api/companion/moods")
async def api_companion_mood_history(request: Request):
    """Get mood history."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    return companion.get_mood_history(user["id"])


@app.get("/api/companion/patterns")
async def api_companion_patterns(request: Request):
    """Get reading patterns for current user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    return companion.get_card_patterns(user["id"])


@app.get("/api/companion/messages")
async def api_companion_messages(request: Request):
    """Get conversation history."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    return companion.get_conversation_history(user["id"])


# ═══════════════════════════════════════════════════════════════════════
# Luna Intelligence Engine — Chat, Context-Aware Responses
# ═══════════════════════════════════════════════════════════════════════

class LunaChatRequest(BaseModel):
    message: str
    channel: str = "web"


@app.post("/api/luna/chat")
async def api_luna_chat(request: Request, req: LunaChatRequest):
    """Chat with Luna via the full intelligence engine."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to chat with Luna")

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip, limit=20):
        raise HTTPException(status_code=429, detail="Too many messages. Please wait a moment.")

    try:
        result = luna_engine.respond(
            user_id=str(user["id"]),
            message=req.message,
            channel=req.channel,
        )
        return {
            "response": result.text,
            "mode": result.mode,
            "quality_score": result.quality_score,
            "cost_cents": result.cost_cents,
            "handler": result.handler,
            "context": result.context_summary,
        }
    except Exception as e:
        logger.error("Luna chat failed: %s", e)
        raise HTTPException(status_code=500, detail="Luna is gathering her thoughts... please try again.")


# ═══════════════════════════════════════════════════════════════════════
# Avatar API — Voice, Video, Real-Time Sessions
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/avatar/status")
async def api_avatar_status():
    """Check which avatar services are available."""
    return avatar.status


@app.get("/api/avatar/health")
async def api_avatar_health():
    """Full health check with usage stats for all avatar services."""
    return avatar.get_service_health()


class VideoGenerationRequest(BaseModel):
    text: str
    content_type: str = "reading"
    aspect_ratio: str = "1:1"


@app.post("/api/avatar/generate-video")
async def api_avatar_generate_video(req: VideoGenerationRequest):
    """Generate a Luna talking-head video from text."""
    if not avatar.status["full_pipeline"]:
        missing = []
        if not avatar.voice.available:
            missing.append("ELEVENLABS_API_KEY")
        if not avatar.video.available:
            missing.append("HEDRA_API_KEY")
        if not avatar.video.portrait_path:
            missing.append("Luna portrait at assets/luna_portrait.png")
        raise HTTPException(
            status_code=503,
            detail=f"Avatar pipeline not ready. Missing: {', '.join(missing)}"
        )

    result = avatar.generate_reading_video(
        reading_text=req.text,
        spread_type=req.content_type,
        aspect_ratio=req.aspect_ratio,
    )
    return result.to_dict()


@app.post("/api/avatar/generate-audio")
async def api_avatar_generate_audio(req: VideoGenerationRequest):
    """Generate Luna's voice audio from text (no video)."""
    if not avatar.voice.available:
        raise HTTPException(status_code=503, detail="ElevenLabs not configured")
    return avatar.generate_voice_only(req.text)


@app.post("/api/avatar/live-session")
async def api_avatar_live_session(request: Request):
    """Create a real-time interactive Luna session.

    Returns WebRTC session credentials for the frontend to connect
    to the live avatar stream.
    """
    if not avatar.realtime.available:
        raise HTTPException(status_code=503, detail="Simli not configured")

    user = get_current_user(request)
    customer_id = str(user["id"]) if user else "anonymous"
    return avatar.create_live_session(customer_id=customer_id)


@app.get("/api/avatar/voices")
async def api_avatar_voices():
    """List available ElevenLabs voices for Luna."""
    return avatar.voice.list_voices()


# ═══════════════════════════════════════════════════════════════════════
# JSON API — Readings, Feedback, System Status
# ═══════════════════════════════════════════════════════════════════════

class ReadingRequest(BaseModel):
    spread_type: str
    question: str | None = None
    customer_id: str | None = None
    channel: str = "web"
    context: str | None = None


class FeedbackRequest(BaseModel):
    customer_id: str
    interaction_id: int
    score: float = Field(ge=1.0, le=5.0)
    comment: str | None = None


class YesNoRequest(BaseModel):
    question: str
    customer_id: str | None = None


# --- Health ---

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "moneyclaw",
        "state": brain.state.value,
        "scheduler": scheduler.get_status(),
    }


@app.get("/api/status")
async def system_status():
    return brain.status()


# --- Readings (Revenue Generator) ---

@app.get("/api/readings/available")
async def available_readings():
    return brain.readings.get_available_readings()


@app.post("/api/readings/generate")
async def generate_reading(req: ReadingRequest):
    try:
        result = await brain.readings.generate_reading(
            spread_type=req.spread_type,
            question=req.question,
            customer_id=req.customer_id,
            channel=req.channel,
            context=req.context,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Reading generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Reading generation failed")


@app.post("/api/readings/yes-no")
async def yes_no_reading(req: YesNoRequest):
    return brain.readings.quick_yes_no(
        question=req.question,
        customer_id=req.customer_id,
    )


@app.get("/api/readings/daily")
async def daily_reading():
    card = brain.readings.tarot.daily_card()
    moon = get_moon_phase()
    return {"card": card.to_dict(), "moon_phase": moon}


# --- Moon & Astrology ---

@app.get("/api/moon")
async def moon_phase():
    return get_moon_phase()


@app.get("/api/horoscopes")
async def horoscope_signs():
    return {"signs": ZODIAC_SIGNS}


@app.get("/api/horoscope/{sign}")
async def api_horoscope_sign(request: Request, sign: str):
    """Generate a daily horoscope for a specific zodiac sign."""
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip, limit=30):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait.")

    sign_data = ZODIAC_LOOKUP.get(sign.lower())
    if not sign_data:
        raise HTTPException(status_code=404, detail=f"Unknown sign: {sign}")

    moon = get_moon_phase()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Deterministic seed for same horoscope all day
    seed = hashlib.sha256(f"horoscope:{sign.lower()}:{today}".encode()).hexdigest()
    rng = __import__("random").Random(seed)

    # Generate horoscope elements
    themes = {
        "Fire": ["passion", "courage", "action", "inspiration", "creativity", "ambition"],
        "Earth": ["stability", "growth", "abundance", "patience", "practicality", "nurturing"],
        "Air": ["communication", "ideas", "connection", "freedom", "clarity", "learning"],
        "Water": ["intuition", "emotion", "healing", "depth", "compassion", "transformation"],
    }
    element = sign_data["element"]
    sign_themes = themes.get(element, themes["Fire"])
    day_theme = rng.choice(sign_themes)
    secondary_theme = rng.choice([t for t in sign_themes if t != day_theme])

    crystals = {
        "Fire": ["Carnelian", "Citrine", "Red Jasper", "Sunstone", "Fire Agate", "Garnet"],
        "Earth": ["Moss Agate", "Green Aventurine", "Tiger's Eye", "Smoky Quartz", "Jade", "Malachite"],
        "Air": ["Amethyst", "Clear Quartz", "Lapis Lazuli", "Fluorite", "Selenite", "Aquamarine"],
        "Water": ["Moonstone", "Rose Quartz", "Labradorite", "Pearl", "Black Tourmaline", "Opal"],
    }
    colors = {
        "Fire": ["Red", "Orange", "Gold", "Amber", "Crimson", "Scarlet"],
        "Earth": ["Green", "Brown", "Forest Green", "Olive", "Emerald", "Copper"],
        "Air": ["Blue", "Lavender", "Sky Blue", "Silver", "White", "Periwinkle"],
        "Water": ["Teal", "Deep Blue", "Violet", "Indigo", "Pearl White", "Sea Green"],
    }
    lucky_numbers = rng.sample(range(1, 50), 3)

    # Build horoscope text based on moon phase + element + sign
    moon_influence = {
        "new": "The New Moon invites you to set powerful intentions today.",
        "waxing_crescent": "Under the Waxing Crescent, your seeds of intention begin to sprout.",
        "first_quarter": "The First Quarter Moon brings a call to action and decisive energy.",
        "waxing_gibbous": "The Waxing Gibbous refines your path — adjust and persist.",
        "full": "The Full Moon illuminates truths and brings emotions to the surface.",
        "waning_gibbous": "The Waning Gibbous encourages sharing your wisdom with others.",
        "last_quarter": "The Last Quarter asks you to release what no longer serves you.",
        "waning_crescent": "The Waning Crescent whispers of rest, reflection, and surrender.",
    }

    ruler_influence = {
        "Mars": "Mars fuels your fire today — channel it into purposeful action, not conflict.",
        "Venus": "Venus blesses your connections — beauty, love, and harmony are amplified.",
        "Mercury": "Mercury sharpens your mind — communication flows easily, ideas sparkle.",
        "Moon": "The Moon deepens your emotional awareness — trust your gut feelings.",
        "Sun": "The Sun illuminates your path — your natural radiance draws others in.",
        "Pluto": "Pluto stirs transformation beneath the surface — embrace the becoming.",
        "Jupiter": "Jupiter expands your horizons — think bigger, reach further.",
        "Saturn": "Saturn rewards discipline — structure and patience yield lasting results.",
        "Uranus": "Uranus sparks unexpected insights — be open to sudden shifts and breakthroughs.",
        "Neptune": "Neptune dissolves boundaries — your imagination and empathy are heightened.",
    }

    moon_msg = moon_influence.get(moon.get("key", ""), "The cosmic energies are shifting.")
    ruler_msg = ruler_influence.get(sign_data["ruler"], "Your ruling planet supports your journey.")

    horoscope_parts = [
        f"{moon_msg}",
        f"\nAs a {element} sign ruled by {sign_data['ruler']}, your energy today centers on {day_theme} and {secondary_theme}. {ruler_msg}",
        f"\nThe {moon.get('phase', 'current moon')} at {moon.get('illumination', 0)}% illumination creates a {['receptive', 'dynamic', 'contemplative', 'expansive'][rng.randint(0, 3)]} atmosphere for your sign. Pay attention to themes of {day_theme} arising in unexpected places — a conversation, a song, a passing thought may carry exactly the message you need.",
        f"\n{moon.get('guidance', 'Follow your intuition')}. This is especially potent for {sign_data['name']} right now.",
    ]

    horoscope_text = "\n".join(horoscope_parts)

    return {
        "sign": sign_data["name"],
        "symbol": sign_data["symbol"],
        "dates": sign_data["dates"],
        "element": f"{sign_data['element']} · {sign_data['modality']}",
        "horoscope": horoscope_text,
        "lucky": {
            "crystal": rng.choice(crystals.get(element, ["Amethyst"])),
            "color": rng.choice(colors.get(element, ["Blue"])),
            "number": ", ".join(str(n) for n in sorted(lucky_numbers)),
        },
        "moon_phase": moon.get("phase", ""),
        "date": today,
    }


@app.get("/api/tarot/deck")
async def tarot_deck():
    """Return the full 78-card tarot deck for the gallery."""
    cards = brain.readings.tarot.get_full_deck()
    return [c.to_dict() for c in cards]


# --- Web Pages: Tarot Gallery & Horoscope ---

@app.get("/tarot", response_class=HTMLResponse)
async def tarot_gallery_page(request: Request):
    ctx = template_context(request)
    return templates.TemplateResponse("tarot.html", ctx)


@app.get("/horoscope", response_class=HTMLResponse)
async def horoscope_page(request: Request):
    ctx = template_context(request, signs=ZODIAC_DATA)
    return templates.TemplateResponse("horoscope.html", ctx)


# --- Feedback ---

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    return brain.handle_feedback(
        customer_id=req.customer_id,
        interaction_id=req.interaction_id,
        score=req.score,
        comment=req.comment,
    )


# --- Financial & Customer Stats ---

@app.get("/api/status/financial")
async def financial_status():
    return brain.ledger.get_financial_summary()


@app.get("/api/status/financial/daily")
async def daily_financial():
    return brain.ledger.daily_report()


@app.get("/api/status/financial/roi")
async def reading_roi():
    return brain.ledger.calculate_reading_roi()


@app.get("/api/status/customers")
async def customer_stats():
    return memory.get_customer_stats()


@app.get("/api/status/services")
async def popular_services():
    return memory.get_popular_services()


# --- Learning & Self-Improvement ---

@app.get("/api/learning/insights")
async def learnings():
    return memory.get_top_learnings(limit=20)


@app.get("/api/learning/experiments")
async def experiments():
    return memory.get_active_experiments()


@app.get("/api/learning/patterns")
async def patterns():
    return brain.patterns.detect_all()


@app.get("/api/learning/satisfaction")
async def satisfaction():
    return brain.feedback.analyze_satisfaction_trends()


@app.get("/api/learning/suggestions")
async def improvement_suggestions():
    return brain.feedback.get_improvement_suggestions()


@app.get("/api/learning/upgrades")
async def upgrade_history():
    return brain.upgrader.get_upgrade_history()


@app.get("/api/learning/decisions")
async def decision_stats():
    return memory.get_decision_success_rate()


# --- Heartbeat & Scheduler ---

@app.post("/api/heartbeat")
async def trigger_heartbeat():
    result = brain.heartbeat()
    return {
        "state": result.state,
        "health": result.health_score,
        "actions": result.actions_taken,
        "patterns": result.patterns_found,
        "upgrades": result.upgrades_applied,
        "duration_ms": result.duration_ms,
    }


@app.get("/api/scheduler")
async def scheduler_status():
    return scheduler.get_status()


@app.post("/api/scheduler/run/{job_name}")
async def run_job(job_name: str):
    return scheduler.run_once(job_name)


# --- Stripe Webhook ---

# Map Stripe product names (lower-case) to subscription tiers
_STRIPE_PRODUCT_TIER: dict[str, str] = {
    "mystic luna — seeker": "seeker",
    "mystic luna — mystic": "mystic",
    "mystic luna — inner circle": "inner_circle",
}


def _resolve_subscription_tier(stripe_module, subscription_obj: dict) -> str | None:
    """Return the tier key for a Stripe subscription object, or None."""
    try:
        items = subscription_obj.get("items", {}).get("data", [])
        for item in items:
            price_id = item.get("price", {}).get("id", "")
            if price_id:
                price = stripe_module.Price.retrieve(price_id, expand=["product"])
                product_name = price.product.name.lower()
                for key, tier in _STRIPE_PRODUCT_TIER.items():
                    if key in product_name:
                        return tier
    except Exception as e:
        logger.warning("Could not resolve subscription tier: %s", e)
    return None


def _find_user_for_stripe_event(customer_email: str, stripe_customer_id: str) -> dict | None:
    """Find a local user by Stripe customer ID, falling back to email."""
    user = None
    if stripe_customer_id:
        user = auth.get_user_by_stripe_customer(stripe_customer_id)
    if not user and customer_email:
        user = auth.get_user_by_email(customer_email)
    return user


@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    body = await request.body()
    sig = request.headers.get("stripe-signature", "")

    # Verify Stripe signature if webhook secret is configured
    webhook_secret = config.stripe.webhook_secret
    stripe_module = None
    if webhook_secret:
        try:
            import stripe as _stripe
            stripe_module = _stripe
            event = _stripe.Webhook.construct_event(body, sig, webhook_secret)
        except ImportError:
            # No stripe package — fall back to raw JSON
            try:
                event = json.loads(body)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid payload")
        except Exception as e:
            logger.warning("Stripe signature verification failed: %s", e)
            raise HTTPException(status_code=400, detail="Invalid signature")
    else:
        try:
            event = json.loads(body)
            try:
                import stripe as _stripe
                stripe_module = _stripe
            except ImportError:
                pass
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload")

    if stripe_module:
        stripe_module.api_key = config.stripe.api_key

    event_type = event.get("type", "")

    # ── checkout.session.completed ────────────────────────────────────────────
    if event_type == "checkout.session.completed":
        session = event.get("data", {}).get("object", {})
        amount = session.get("amount_total", 0)
        customer_email = session.get("customer_email") or ""
        stripe_customer_id = session.get("customer") or ""
        metadata = session.get("metadata", {})
        spread_type = metadata.get("spread_type", "unknown")
        mode = session.get("mode", "payment")  # "payment" or "subscription"

        # ── Bundle purchase: add credits to user account ──────────────────────
        if metadata.get("type") == "bundle":
            bundle_id = metadata.get("bundle_id", "")
            user_id = metadata.get("user_id", "")
            readings = int(metadata.get("readings", 0))
            stripe_payment_id = session.get("id", "")
            bundle_name = bundle_id  # human-readable name resolved below
            # Find the bundle definition for its display name
            for _b in READING_BUNDLES:
                if _b["id"] == bundle_id:
                    bundle_name = _b["name"]
                    break
            if user_id and readings > 0:
                memory.add_credits(
                    user_id=user_id,
                    credits=readings,
                    bundle_name=bundle_name,
                    stripe_payment_id=stripe_payment_id,
                )
                brain.ledger.record_sale(
                    source="bundles",
                    amount_cents=amount,
                    product=bundle_name,
                    customer_id=customer_email,
                    stripe_id=stripe_payment_id,
                )
                logger.info(
                    "Bundle fulfilled: user=%s bundle=%s credits=%d",
                    user_id, bundle_name, readings,
                )
            else:
                logger.warning(
                    "Bundle webhook missing user_id or readings: metadata=%s", metadata
                )
            return {"status": "bundle_fulfilled", "credits_added": readings}

        # ── Subscription purchase: upgrade user tier ──────────────────────────
        if mode == "subscription":
            subscription_id = session.get("subscription")
            tier = None

            if subscription_id and stripe_module:
                try:
                    sub = stripe_module.Subscription.retrieve(subscription_id)
                    tier = _resolve_subscription_tier(stripe_module, sub)
                except Exception as e:
                    logger.error("Could not retrieve subscription %s: %s", subscription_id, e)

            if tier:
                user = _find_user_for_stripe_event(customer_email, stripe_customer_id)
                if user:
                    # Also store the stripe_customer_id for future webhook lookups
                    auth.update_user(
                        user["id"],
                        stripe_customer_id=stripe_customer_id or user.get("stripe_customer_id", ""),
                    )
                    auth.update_subscription(user["id"], tier)
                    logger.info(
                        "Subscription activated: user=%s tier=%s", user["email"], tier
                    )
                else:
                    logger.warning(
                        "Subscription purchased by unknown user: email=%s customer=%s tier=%s",
                        customer_email, stripe_customer_id, tier,
                    )

            brain.ledger.record_sale(
                source="subscriptions",
                amount_cents=amount,
                product=tier or "subscription",
                customer_id=customer_email,
                stripe_id=session.get("id"),
            )
            return {"status": "subscription_activated", "tier": tier}

        # ── One-time payment: generate and deliver reading ────────────────────
        brain.ledger.record_sale(
            source="readings",
            amount_cents=amount,
            product=spread_type,
            customer_id=customer_email,
            stripe_id=session.get("id"),
        )

        question = metadata.get("question")
        if question and spread_type != "unknown":
            try:
                reading_result = await brain.readings.generate_reading(
                    spread_type=spread_type,
                    question=question,
                    customer_id=customer_email,
                    channel="stripe",
                )

                # Send reading via email
                if customer_email and reading_result.get("reading"):
                    cards = reading_result.get("spread", {}).get("cards", [])
                    email_service.send_reading_email(
                        to_email=customer_email,
                        reading_text=reading_result["reading"],
                        spread_name=reading_result.get("spread", {}).get("spread_name", spread_type),
                        cards=cards,
                        moon=reading_result.get("moon_phase"),
                    )

                # Deliver via Telegram if the buyer came from Telegram
                tg_chat_id = metadata.get("telegram_chat_id")
                if tg_chat_id:
                    tg_bot = get_bot(memory)
                    if tg_bot:
                        try:
                            await tg_bot.send_paid_reading(
                                chat_id=tg_chat_id,
                                spread_type=spread_type,
                                question=question,
                                customer_id=customer_email or f"tg:{tg_chat_id}",
                            )
                        except Exception as tg_err:
                            logger.error("Telegram reading delivery failed: %s", tg_err)

                    # Notify Nick of the sale
                    if tg_bot and amount > 0:
                        try:
                            await tg_bot.notify_nick(
                                f"💰 *New sale!*\n"
                                f"Spread: {spread_type}\n"
                                f"Amount: ${amount / 100:.2f}\n"
                                f"From: {customer_email or 'Telegram user'}"
                            )
                        except Exception:
                            pass

            except Exception as e:
                logger.error("Post-payment reading generation failed: %s", e)

        logger.info(
            "Payment received: $%.2f for %s from %s",
            amount / 100, spread_type, customer_email
        )
        return {"status": "recorded"}

    # ── customer.subscription.deleted — downgrade to free ────────────────────
    if event_type == "customer.subscription.deleted":
        sub_obj = event.get("data", {}).get("object", {})
        stripe_customer_id = sub_obj.get("customer") or ""
        customer_email = ""

        # Try to find the customer email from Stripe
        if stripe_customer_id and stripe_module:
            try:
                customer = stripe_module.Customer.retrieve(stripe_customer_id)
                customer_email = customer.get("email") or ""
            except Exception as e:
                logger.warning("Could not retrieve Stripe customer %s: %s", stripe_customer_id, e)

        user = _find_user_for_stripe_event(customer_email, stripe_customer_id)
        if user:
            auth.update_subscription(user["id"], "free")
            logger.info(
                "Subscription canceled — downgraded to free: user=%s", user.get("email")
            )
        else:
            logger.warning(
                "Subscription deleted for unknown user: customer=%s email=%s",
                stripe_customer_id, customer_email,
            )
        return {"status": "subscription_canceled"}

    # ── customer.subscription.updated — handle plan changes ──────────────────
    if event_type == "customer.subscription.updated":
        sub_obj = event.get("data", {}).get("object", {})
        stripe_customer_id = sub_obj.get("customer") or ""
        status = sub_obj.get("status", "")

        customer_email = ""
        if stripe_customer_id and stripe_module:
            try:
                customer = stripe_module.Customer.retrieve(stripe_customer_id)
                customer_email = customer.get("email") or ""
            except Exception:
                pass

        user = _find_user_for_stripe_event(customer_email, stripe_customer_id)
        if user and status in ("active", "trialing"):
            tier = _resolve_subscription_tier(stripe_module, sub_obj) if stripe_module else None
            if tier:
                auth.update_subscription(user["id"], tier)
                logger.info(
                    "Subscription updated: user=%s tier=%s", user.get("email"), tier
                )
        elif user and status in ("canceled", "unpaid", "past_due"):
            auth.update_subscription(user["id"], "free")
            logger.info(
                "Subscription degraded to free (status=%s): user=%s", status, user.get("email")
            )
        return {"status": "subscription_updated"}

    return {"status": "ignored", "event_type": event_type}


# ═══════════════════════════════════════════════════════════════════════
# Knowledge Base — Public Endpoints (no auth required)
# ═══════════════════════════════════════════════════════════════════════

from moneyclaw.knowledge import (
    # correspondences
    get_herb, get_crystal, get_correspondences_for_intention,
    search_herbs, search_crystals, get_herb_of_day, get_crystal_of_day,
    HERBS, CRYSTALS, COLORS, ELEMENTS, PLANETS, INTENTION_MAP,
    # moon
    calculate_moon_phase_precise, get_phase_data, get_sign_data, get_next_30_days_moon,
    MOON_PHASES, MOON_IN_SIGNS,
    # wheel_of_year
    get_current_sabbat, get_next_sabbat, get_seasonal_context, get_all_sabbats,
    # spells
    get_all_spell_types, get_spell_template, get_spell_types_for_intention,
    generate_spell_suggestion, score_spell,
    # meditations
    get_all_meditations_summary, get_meditation, get_meditations_for_intention,
    # planetary hours
    get_today_planetary_overview, get_all_hours_for_day,
    get_best_hours_for_intention, get_current_planetary_hour, get_day_ruler,
    # journal prompts
    get_daily_prompt, get_moon_prompts, get_sabbat_prompts,
    get_shadow_prompt, get_prompts_for_theme, get_all_themes,
)
from moneyclaw.grimoire_db import GrimoireDB

# Module-level singletons (created lazily on first use)
_grimoire_db: GrimoireDB | None = None


def _get_grimoire_db() -> GrimoireDB:
    global _grimoire_db
    if _grimoire_db is None:
        _grimoire_db = GrimoireDB()
    return _grimoire_db


@app.get("/api/knowledge/herbs")
async def api_knowledge_herbs(request: Request, q: str = "", intention: str = ""):
    """Return all herbs or filter by keyword/intention."""
    if intention:
        corr = get_correspondences_for_intention(intention)
        return {"herbs": corr.get("herbs", []), "source": "intention"}
    if q:
        results = search_herbs(q)
        return {"herbs": results, "query": q, "count": len(results)}
    # Return full list (name + icon + brief description)
    herbs_list = [
        {
            "name": h["name"],
            "display_name": h.get("display_name", h["name"]),
            "icon": h.get("icon", "🌿"),
            "element": h.get("element", ""),
            "planet": h.get("planet", ""),
            "magical_properties": h.get("magical_properties", [])[:3],
            "beginner_tip": h.get("beginner_tip", ""),
        }
        for h in HERBS.values()
    ]
    return {"herbs": herbs_list, "count": len(herbs_list)}


@app.get("/api/knowledge/crystals")
async def api_knowledge_crystals(request: Request, q: str = "", intention: str = ""):
    """Return all crystals or filter by keyword/intention."""
    if intention:
        corr = get_correspondences_for_intention(intention)
        return {"crystals": corr.get("crystals", []), "source": "intention"}
    if q:
        results = search_crystals(q)
        return {"crystals": results, "query": q, "count": len(results)}
    crystals_list = [
        {
            "name": c["name"],
            "display_name": c.get("display_name", c["name"]),
            "icon": c.get("icon", "💎"),
            "element": c.get("element", ""),
            "planet": c.get("planet", ""),
            "magical_properties": c.get("magical_properties", [])[:3],
            "chakra": c.get("chakra", ""),
            "beginner_tip": c.get("beginner_tip", ""),
        }
        for c in CRYSTALS.values()
    ]
    return {"crystals": crystals_list, "count": len(crystals_list)}


@app.get("/api/knowledge/moon")
async def api_knowledge_moon(request: Request):
    """Return all moon phase data plus current moon info."""
    now = datetime.now(timezone.utc)
    phase_key, illumination, zodiac_sign = calculate_moon_phase_precise(
        now.year, now.month, now.day, now.hour
    )
    current_phase = get_phase_data(phase_key) or {}
    current_sign = get_sign_data(zodiac_sign) or {}

    return {
        "phases": list(MOON_PHASES.values()),
        "signs": list(MOON_IN_SIGNS.values()),
        "current": {
            "phase_key": phase_key,
            "phase_name": current_phase.get("name", phase_key),
            "phase_icon": current_phase.get("icon", "🌙"),
            "illumination": illumination,
            "zodiac_sign": zodiac_sign,
            "sign_name": current_sign.get("name", zodiac_sign),
            "sign_symbol": current_sign.get("symbol", ""),
            "affirmation": current_phase.get("affirmation", ""),
            "ritual_suggestion": current_phase.get("ritual_suggestion", ""),
        },
    }


@app.get("/api/knowledge/moon/calendar")
async def api_knowledge_moon_calendar(request: Request, days: int = 30):
    """Return moon phase calendar for the next N days (max 90)."""
    days = min(days, 90)
    calendar = get_next_30_days_moon()
    return {"calendar": calendar[:days], "days": days}


@app.get("/api/knowledge/sabbats")
async def api_knowledge_sabbats(request: Request):
    """Return all sabbats plus current and next upcoming sabbat."""
    _ts = datetime.now()
    current = get_current_sabbat(_ts.month, _ts.day)
    next_sabbat = get_next_sabbat(_ts.month, _ts.day)
    seasonal = get_seasonal_context(_ts.month)
    all_sabbats = get_all_sabbats()
    return {
        "sabbats": all_sabbats,
        "current": current,
        "next": next_sabbat,
        "seasonal_context": seasonal,  # prose string
    }


@app.get("/api/knowledge/sabbats/current")
async def api_knowledge_sabbats_current(request: Request):
    """Return current and next sabbat only."""
    _ts = datetime.now()
    current = get_current_sabbat(_ts.month, _ts.day)
    next_sabbat = get_next_sabbat(_ts.month, _ts.day)
    return {"current": current, "next": next_sabbat, "seasonal_context": get_seasonal_context(_ts.month)}


@app.get("/api/knowledge/spells/types")
async def api_knowledge_spell_types(request: Request, intention: str = ""):
    """Return all spell types or filtered by intention."""
    if intention:
        types = get_spell_types_for_intention(intention)
        return {"spell_types": types, "intention": intention}
    return {"spell_types": get_all_spell_types()}


@app.get("/api/knowledge/correspondences/{intention}")
async def api_knowledge_correspondences(request: Request, intention: str):
    """Return full correspondences for a spiritual intention."""
    corr = get_correspondences_for_intention(intention)
    if not corr:
        raise HTTPException(
            status_code=404,
            detail=f"No correspondences found for intention '{intention}'. "
                   f"Try: love, protection, abundance, healing, intuition, peace, "
                   f"courage, banishing, divination, creativity, communication, "
                   f"shadow_work, grounding, transformation, prosperity",
        )
    return corr


@app.get("/api/knowledge/planetary-hours")
async def api_knowledge_planetary_hours(request: Request, weekday: int | None = None):
    """Return planetary hours overview and today's schedule."""
    overview = get_today_planetary_overview()
    now = datetime.now()
    wd = weekday if weekday is not None else now.weekday()
    all_hours = get_all_hours_for_day(wd, now.year, now.month, now.day)
    return {
        "overview": overview,
        "hours": all_hours,
    }


@app.get("/api/knowledge/meditations")
async def api_knowledge_meditations(request: Request, intention: str = "", key: str = ""):
    """Return meditation frameworks, optionally filtered."""
    if key:
        med = get_meditation(key)
        if not med:
            raise HTTPException(status_code=404, detail=f"Meditation '{key}' not found")
        return med
    if intention:
        results = get_meditations_for_intention(intention)
        return {"meditations": results, "intention": intention}
    return {"meditations": get_all_meditations_summary()}


@app.get("/api/knowledge/journal-prompts")
async def api_knowledge_journal_prompts(
    request: Request,
    theme: str = "daily",
    index: int = 0,
):
    """Return journal prompts for a given theme."""
    available = get_all_themes()
    prompts = get_prompts_for_theme(theme)
    single = prompts[index % len(prompts)] if prompts else ""
    return {
        "theme": theme,
        "prompts": prompts,
        "featured": single,
        "available_themes": available,
    }


# ═══════════════════════════════════════════════════════════════════════
# Workshop — Spell / Ritual Crafting (auth required)
# ═══════════════════════════════════════════════════════════════════════

class CraftSpellRequest(BaseModel):
    intention: str
    spell_type: str = ""
    moon_phase: str = ""
    notes: str = ""
    difficulty: str = ""
    save_to_grimoire: bool = False


class ScoreSpellRequest(BaseModel):
    title: str
    intention: str = ""
    spell_type: str = ""
    ingredients: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    moon_phase: str = ""
    day_of_week: str = ""
    planet: str = ""
    notes: str = ""


@app.post("/api/workshop/craft")
async def api_workshop_craft(request: Request, req: CraftSpellRequest):
    """Generate a spell suggestion for a given intention."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to access the workshop")

    suggestion = generate_spell_suggestion(req.intention, req.spell_type or None)
    if not suggestion:
        raise HTTPException(
            status_code=422,
            detail=f"Could not generate a spell for intention '{req.intention}'",
        )

    # Optionally save to the user's grimoire
    if req.save_to_grimoire:
        db = _get_grimoire_db()
        spell_id = db.add_spell(
            user_id=str(user["id"]),
            title=suggestion.get("title", f"{req.intention.title()} Spell"),
            intention=req.intention,
            spell_type=suggestion.get("spell_type", ""),
            ingredients=suggestion.get("materials", {}).get("required", []),
            steps=suggestion.get("steps", []),
            moon_phase=req.moon_phase or suggestion.get("timing", {}).get("moon_phase", ""),
            notes=req.notes,
        )
        db.check_milestones(str(user["id"]))
        suggestion["saved_spell_id"] = spell_id

    return suggestion


@app.post("/api/workshop/score")
async def api_workshop_score(request: Request, req: ScoreSpellRequest):
    """Score a spell based on correspondence completeness."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to score your spell")

    result = score_spell(req.model_dump())
    return result


# ═══════════════════════════════════════════════════════════════════════
# Grimoire — Personal Book of Shadows (auth required)
# ═══════════════════════════════════════════════════════════════════════

class AddSpellRequest(BaseModel):
    title: str
    intention: str = ""
    spell_type: str = ""
    ingredients: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    moon_phase: str = ""
    day_of_week: str = ""
    planet: str = ""
    notes: str = ""
    rating: int | None = None
    cast_on: str | None = None


class AddHerbRequest(BaseModel):
    herb_name: str
    use_case: str = ""
    experience: str = ""
    rating: int | None = None
    notes: str = ""


class AddCrystalRequest(BaseModel):
    crystal_name: str
    cleansing_method: str = ""
    intentions: list[str] = Field(default_factory=list)
    experiences: str = ""
    notes: str = ""


class AddMoonEntryRequest(BaseModel):
    moon_phase: str
    moon_date: str | None = None
    intention: str = ""
    reflection: str = ""
    what_released: str = ""
    what_manifested: str = ""
    ritual_performed: str = ""
    energy_level: int | None = None


class LogPracticeRequest(BaseModel):
    practice_type: str
    duration_minutes: int | None = None
    notes: str = ""
    mood_before: int | None = None
    mood_after: int | None = None
    log_date: str | None = None


@app.get("/api/grimoire/overview")
async def api_grimoire_overview(request: Request):
    """Return the user's grimoire overview: stats, recent entries, milestones."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to access your grimoire")
    db = _get_grimoire_db()
    return db.get_grimoire_overview(str(user["id"]))


@app.get("/api/grimoire/spells")
async def api_grimoire_spells(request: Request, limit: int = 50):
    """Return the user's saved spells."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    return {"spells": db.get_spells(str(user["id"]), limit=min(limit, 200))}


@app.post("/api/grimoire/spells")
async def api_grimoire_add_spell(request: Request, req: AddSpellRequest):
    """Add a spell to the Book of Shadows."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    spell_id = db.add_spell(
        user_id=str(user["id"]),
        title=req.title,
        intention=req.intention,
        spell_type=req.spell_type,
        ingredients=req.ingredients,
        steps=req.steps,
        moon_phase=req.moon_phase,
        day_of_week=req.day_of_week,
        planet=req.planet,
        notes=req.notes,
        rating=req.rating,
        cast_on=req.cast_on,
    )
    new_milestones = db.check_milestones(str(user["id"]))
    return {"spell_id": spell_id, "new_milestones": new_milestones}


@app.get("/api/grimoire/herbs")
async def api_grimoire_herbs(request: Request):
    """Return the user's herb journal."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    return {"herbs": db.get_herbs(str(user["id"]))}


@app.post("/api/grimoire/herbs")
async def api_grimoire_add_herb(request: Request, req: AddHerbRequest):
    """Add an herb experience to the journal."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    entry_id = db.add_herb_entry(
        user_id=str(user["id"]),
        herb_name=req.herb_name,
        use_case=req.use_case,
        experience=req.experience,
        rating=req.rating,
        notes=req.notes,
    )
    new_milestones = db.check_milestones(str(user["id"]))
    return {"entry_id": entry_id, "new_milestones": new_milestones}


@app.get("/api/grimoire/crystals")
async def api_grimoire_crystals(request: Request):
    """Return the user's crystal collection."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    return {"crystals": db.get_crystals(str(user["id"]))}


@app.post("/api/grimoire/crystals")
async def api_grimoire_add_crystal(request: Request, req: AddCrystalRequest):
    """Add a crystal to the collection."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    entry_id = db.add_crystal(
        user_id=str(user["id"]),
        crystal_name=req.crystal_name,
        cleansing_method=req.cleansing_method,
        intentions=req.intentions,
        experiences=req.experiences,
        notes=req.notes,
    )
    new_milestones = db.check_milestones(str(user["id"]))
    return {"entry_id": entry_id, "new_milestones": new_milestones}


@app.get("/api/grimoire/moon-journal")
async def api_grimoire_moon_journal(request: Request, limit: int = 30):
    """Return the user's moon phase journal."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    return {"entries": db.get_moon_journal(str(user["id"]), limit=min(limit, 100))}


@app.post("/api/grimoire/moon-journal")
async def api_grimoire_add_moon_entry(request: Request, req: AddMoonEntryRequest):
    """Log a moon phase journal entry."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    entry_id = db.add_moon_entry(
        user_id=str(user["id"]),
        moon_phase=req.moon_phase,
        moon_date=req.moon_date,
        intention=req.intention,
        reflection=req.reflection,
        what_released=req.what_released,
        what_manifested=req.what_manifested,
        ritual_performed=req.ritual_performed,
        energy_level=req.energy_level,
    )
    new_milestones = db.check_milestones(str(user["id"]))
    return {"entry_id": entry_id, "new_milestones": new_milestones}


@app.get("/api/grimoire/practice")
async def api_grimoire_practice(request: Request, limit: int = 50):
    """Return the user's practice log."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    return {"practice_log": db.get_practice_log(str(user["id"]), limit=min(limit, 200))}


@app.post("/api/grimoire/practice")
async def api_grimoire_log_practice(request: Request, req: LogPracticeRequest):
    """Log a practice session."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    entry_id = db.log_practice(
        user_id=str(user["id"]),
        practice_type=req.practice_type,
        duration_minutes=req.duration_minutes,
        notes=req.notes,
        mood_before=req.mood_before,
        mood_after=req.mood_after,
        log_date=req.log_date,
    )
    new_milestones = db.check_milestones(str(user["id"]))
    return {"entry_id": entry_id, "new_milestones": new_milestones}


@app.get("/api/grimoire/milestones")
async def api_grimoire_milestones(request: Request):
    """Return achieved milestones for the user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    return {"milestones": db.get_milestones(str(user["id"]))}


# ═══════════════════════════════════════════════════════════════════════
# Daily Practice — Full Daily Guidance Package
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/daily")
async def api_daily(request: Request):
    """Return a complete daily spiritual guidance package.

    Public endpoint — works for anonymous users too, though it won't include
    personalised grimoire data.
    """
    now = datetime.now(timezone.utc)
    local_now = datetime.now()

    # Moon
    phase_key, illumination, zodiac_sign = calculate_moon_phase_precise(
        now.year, now.month, now.day, now.hour
    )
    phase_data = get_phase_data(phase_key) or {}
    sign_data = get_sign_data(zodiac_sign) or {}

    # Planetary hours
    planetary_overview = get_today_planetary_overview()
    weekday = local_now.weekday()
    current_hour_float = local_now.hour + local_now.minute / 60.0
    current_planet_hour = get_current_planetary_hour(weekday, current_hour_float)

    # Sabbat context
    seasonal = get_seasonal_context(local_now.month)
    next_sabbat = get_next_sabbat(local_now.month, local_now.day)

    # Daily journal prompt (cycles by day-of-year)
    day_of_year = local_now.timetuple().tm_yday
    daily_prompt = get_daily_prompt(day_of_year)

    # Moon journal prompt
    moon_prompt = get_moon_prompts(phase_key)
    featured_moon_prompt = moon_prompt[0] if moon_prompt else daily_prompt

    # Herb and crystal of the day
    herb_of_day = get_herb_of_day(day_of_year)
    crystal_of_day = get_crystal_of_day(day_of_year)

    # Best planetary hours for common intentions
    best_love_hours = get_best_hours_for_intention("love", weekday)[:2]
    best_abundance_hours = get_best_hours_for_intention("abundance", weekday)[:2]

    return {
        "date": local_now.strftime("%Y-%m-%d"),
        "day_of_week": local_now.strftime("%A"),
        "moon": {
            "phase_key": phase_key,
            "phase_name": phase_data.get("name", phase_key),
            "phase_icon": phase_data.get("icon", "🌙"),
            "illumination": illumination,
            "zodiac_sign": zodiac_sign,
            "sign_name": sign_data.get("name", zodiac_sign),
            "sign_symbol": sign_data.get("symbol", ""),
            "keyword": phase_data.get("keyword", ""),
            "affirmation": phase_data.get("affirmation", ""),
            "ritual_suggestion": phase_data.get("ritual_suggestion", ""),
            "journal_prompts": moon_prompt[:3],
        },
        "planetary": {
            "day_ruler": planetary_overview.get("day_ruler", ""),
            "current_planet": current_planet_hour.get("planet", ""),
            "current_hour_end": current_planet_hour.get("hour_end_display", ""),
            "current_best_for": current_planet_hour.get("best_for", ""),
            "is_daytime": current_planet_hour.get("is_daytime", True),
            "best_love_hours": best_love_hours,
            "best_abundance_hours": best_abundance_hours,
        },
        "seasonal": {
            "context": seasonal,  # prose string
            "next_sabbat": next_sabbat,
        },
        "daily_prompt": daily_prompt,
        "featured_moon_prompt": featured_moon_prompt,
        "herb_of_day": {
            "name": herb_of_day.get("name", ""),
            "icon": herb_of_day.get("icon", "🌿"),
            "beginner_tip": herb_of_day.get("beginner_tip", ""),
            "magical_properties": herb_of_day.get("magical_properties", [])[:3],
        } if herb_of_day else None,
        "crystal_of_day": {
            "name": crystal_of_day.get("name", ""),
            "icon": crystal_of_day.get("icon", "💎"),
            "beginner_tip": crystal_of_day.get("beginner_tip", ""),
            "magical_properties": crystal_of_day.get("magical_properties", [])[:3],
        } if crystal_of_day else None,
    }


# ═══════════════════════════════════════════════════════════════════════
# Moon Timing — Intent-based timing guidance
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/moon/timing")
async def api_moon_timing(request: Request, intention: str = "general"):
    """Return moon-phase timing advice for a given spiritual intention.

    Combines current moon data with the correspondence knowledge base to
    produce human-readable guidance. No auth required.
    """
    now = datetime.now(timezone.utc)
    phase_key, illumination, zodiac_sign = calculate_moon_phase_precise(
        now.year, now.month, now.day, now.hour
    )
    phase_data = get_phase_data(phase_key) or {}
    sign_data = get_sign_data(zodiac_sign) or {}
    corr = get_correspondences_for_intention(intention) or {}

    ideal_phases = corr.get("best_moon_phases", [])
    ideal_elements = corr.get("elements", [])

    is_ideal_phase = bool(ideal_phases) and any(
        p.lower() in phase_key.lower() for p in ideal_phases
    )
    is_compatible_sign = bool(ideal_elements) and sign_data.get("element", "") in ideal_elements

    score = 50
    if is_ideal_phase:
        score += 30
    if is_compatible_sign:
        score += 20

    action_map = {
        "new": "cast new-beginning spells and set intentions under",
        "waxing_crescent": "plant seeds of intention under",
        "first_quarter": "push forward and take action under",
        "waxing_gibbous": "refine and persist under",
        "full": "release, celebrate, and manifest under",
        "waning_gibbous": "share wisdom and give thanks under",
        "last_quarter": "let go and banish under",
        "waning_crescent": "rest, reflect, and surrender under",
    }
    action_phrase = action_map.get(phase_key, "work with")
    phase_name = phase_data.get("name", phase_key)
    sign_name = sign_data.get("name", zodiac_sign)

    if is_ideal_phase:
        timing_verdict = "Excellent timing"
        guidance = (
            f"The {phase_name} is one of the best phases for {intention} work. "
            f"The Moon in {sign_name} amplifies this energy further. "
            f"This is a powerful moment to {action_phrase} the {phase_name}."
        )
    elif score >= 50:
        timing_verdict = "Good timing"
        guidance = (
            f"The {phase_name} supports your {intention} intention. "
            f"You can {action_phrase} this moon phase with confidence."
        )
    else:
        timing_verdict = "Workable timing"
        guidance = (
            f"While the {phase_name} is not the traditional peak for {intention} work, "
            f"intention and focus can overcome timing. Work with what you have — "
            f"the Moon in {sign_name} brings its own gifts."
        )

    next_ideal = None
    if not is_ideal_phase and ideal_phases:
        phases_order = [
            "new", "waxing_crescent", "first_quarter", "waxing_gibbous",
            "full", "waning_gibbous", "last_quarter", "waning_crescent",
        ]
        try:
            cur_idx = phases_order.index(phase_key)
        except ValueError:
            cur_idx = 0
        for offset in range(1, 9):
            candidate = phases_order[(cur_idx + offset) % 8]
            if any(p.lower() in candidate.lower() for p in ideal_phases):
                days_away = offset * 3
                candidate_data = get_phase_data(candidate) or {}
                next_ideal = {
                    "phase_key": candidate,
                    "phase_name": candidate_data.get("name", candidate),
                    "approx_days": days_away,
                }
                break

    return {
        "intention": intention,
        "timing_score": score,
        "timing_verdict": timing_verdict,
        "guidance": guidance,
        "current_moon": {
            "phase_key": phase_key,
            "phase_name": phase_name,
            "phase_icon": phase_data.get("icon", "🌙"),
            "illumination": illumination,
            "zodiac_sign": zodiac_sign,
            "sign_name": sign_name,
        },
        "is_ideal_phase": is_ideal_phase,
        "ideal_phases": ideal_phases,
        "next_ideal_phase": next_ideal,
        "ritual_suggestion": phase_data.get("ritual_suggestion", ""),
        "correspondences": {
            "herbs": corr.get("herbs", [])[:3],
            "crystals": corr.get("crystals", [])[:3],
            "colors": corr.get("colors", [])[:3],
            "candles": corr.get("candles", []),
        },
    }


@app.post("/api/moon/timing")
async def api_moon_timing_post(request: Request):
    """POST version of moon timing — accepts JSON body with 'intention' field.

    Identical logic to the GET version; added so Guardian and frontend
    can POST to this endpoint without query-string encoding.
    """
    body = await request.json()
    intention = body.get("intention", "general")

    now = datetime.now(timezone.utc)
    phase_key, illumination, zodiac_sign = calculate_moon_phase_precise(
        now.year, now.month, now.day, now.hour
    )
    phase_data = get_phase_data(phase_key) or {}
    sign_data = get_sign_data(zodiac_sign) or {}
    corr = get_correspondences_for_intention(intention) or {}

    ideal_phases = corr.get("best_moon_phases", [])
    ideal_elements = corr.get("elements", [])

    is_ideal_phase = bool(ideal_phases) and any(
        p.lower() in phase_key.lower() for p in ideal_phases
    )
    is_compatible_sign = bool(ideal_elements) and sign_data.get("element", "") in ideal_elements

    score = 50
    if is_ideal_phase:
        score += 30
    if is_compatible_sign:
        score += 20

    action_map = {
        "new": "cast new-beginning spells and set intentions under",
        "waxing_crescent": "plant seeds of intention under",
        "first_quarter": "push forward and take action under",
        "waxing_gibbous": "refine and persist under",
        "full": "release, celebrate, and manifest under",
        "waning_gibbous": "share wisdom and give thanks under",
        "last_quarter": "let go and banish under",
        "waning_crescent": "rest, reflect, and surrender under",
    }
    action_phrase = action_map.get(phase_key, "work with")
    phase_name = phase_data.get("name", phase_key)
    sign_name = sign_data.get("name", zodiac_sign)

    if is_ideal_phase:
        timing_verdict = "Excellent timing"
        guidance = (
            f"The {phase_name} is one of the best phases for {intention} work. "
            f"The Moon in {sign_name} amplifies this energy further. "
            f"This is a powerful moment to {action_phrase} the {phase_name}."
        )
    elif score >= 50:
        timing_verdict = "Good timing"
        guidance = (
            f"The {phase_name} supports your {intention} intention. "
            f"You can {action_phrase} this moon phase with confidence."
        )
    else:
        timing_verdict = "Workable timing"
        guidance = (
            f"While the {phase_name} is not the traditional peak for {intention} work, "
            f"intention and focus can overcome timing. Work with what you have — "
            f"the Moon in {sign_name} brings its own gifts."
        )

    next_ideal = None
    if not is_ideal_phase and ideal_phases:
        phases_order = [
            "new", "waxing_crescent", "first_quarter", "waxing_gibbous",
            "full", "waning_gibbous", "last_quarter", "waning_crescent",
        ]
        try:
            cur_idx = phases_order.index(phase_key)
        except ValueError:
            cur_idx = 0
        for offset in range(1, 9):
            candidate = phases_order[(cur_idx + offset) % 8]
            if any(p.lower() in candidate.lower() for p in ideal_phases):
                days_away = offset * 3
                candidate_data = get_phase_data(candidate) or {}
                next_ideal = {
                    "phase_key": candidate,
                    "phase_name": candidate_data.get("name", candidate),
                    "approx_days": days_away,
                }
                break

    return {
        "intention": intention,
        "timing_score": score,
        "timing_verdict": timing_verdict,
        "guidance": guidance,
        "current_moon": {
            "phase_key": phase_key,
            "phase_name": phase_name,
            "phase_icon": phase_data.get("icon", "🌙"),
            "illumination": illumination,
            "zodiac_sign": zodiac_sign,
            "sign_name": sign_name,
        },
        "is_ideal_phase": is_ideal_phase,
        "ideal_phases": ideal_phases,
        "next_ideal_phase": next_ideal,
        "ritual_suggestion": phase_data.get("ritual_suggestion", ""),
        "correspondences": {
            "herbs": corr.get("herbs", [])[:3],
            "crystals": corr.get("crystals", [])[:3],
            "colors": corr.get("colors", [])[:3],
            "candles": corr.get("candles", []),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# New Feature Pages — Grimoire, Moon Oracle, Workshop, Encyclopedia, etc.
# ═══════════════════════════════════════════════════════════════════════

@app.get("/grimoire", response_class=HTMLResponse)
async def grimoire_page(request: Request):
    """Personal Book of Shadows — API endpoints are auth-gated, page is public."""
    ctx = template_context(request)
    return templates.TemplateResponse("grimoire.html", ctx)


@app.get("/moon-oracle", response_class=HTMLResponse)
async def moon_oracle_page(request: Request):
    """Live moon phase tracker, 30-day calendar, and timing calculator."""
    ctx = template_context(request)
    return templates.TemplateResponse("moon_oracle.html", ctx)


@app.get("/spell-workshop", response_class=HTMLResponse)
async def spell_workshop_page(request: Request):
    """4-step spell crafting wizard — auth required to save spells."""
    ctx = template_context(request)
    return templates.TemplateResponse("spell_workshop.html", ctx)


@app.get("/encyclopedia", response_class=HTMLResponse)
async def encyclopedia_page(request: Request):
    """Searchable herb and crystal reference library."""
    ctx = template_context(request)
    return templates.TemplateResponse("herb_crystal_encyclopedia.html", ctx)


@app.get("/herb-crystal-encyclopedia", response_class=HTMLResponse)
async def herb_crystal_encyclopedia_page(request: Request):
    """Searchable herb and crystal reference library (canonical URL)."""
    ctx = template_context(request)
    return templates.TemplateResponse("herb_crystal_encyclopedia.html", ctx)


@app.get("/wheel-of-year", response_class=HTMLResponse)
async def wheel_of_year_page(request: Request):
    """Interactive sabbat wheel and seasonal ritual guide."""
    ctx = template_context(request)
    return templates.TemplateResponse("wheel_of_year.html", ctx)


@app.get("/meditation", response_class=HTMLResponse)
async def meditation_page(request: Request):
    """Guided meditation frameworks with ambient timer."""
    ctx = template_context(request)
    return templates.TemplateResponse("meditation.html", ctx)


@app.get("/daily", response_class=HTMLResponse)
async def daily_practice_page(request: Request):
    """Daily practice hub — personalized bento-grid spiritual dashboard."""
    ctx = template_context(request)
    return templates.TemplateResponse("daily_practice.html", ctx)


# ═══════════════════════════════════════════════════════════════════════
# Missing Route Fixes — redirect aliases and API gaps
# ═══════════════════════════════════════════════════════════════════════

@app.get("/moon", response_class=HTMLResponse)
async def moon_redirect():
    """Redirect /moon → /moon-oracle (dashboard links here)."""
    return RedirectResponse("/moon-oracle", status_code=301)


@app.get("/readings", response_class=HTMLResponse)
async def readings_redirect():
    """Redirect /readings → /pricing (dashboard 'see all' links here)."""
    return RedirectResponse("/pricing", status_code=301)


@app.get("/guardian", response_class=HTMLResponse)
async def guardian_page(request: Request):
    ctx = template_context(request)
    return templates.TemplateResponse("guardian.html", ctx)


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """About Luna Moonshadow — AI-powered spiritual guide."""
    moon = get_moon_phase()
    ctx = template_context(request, moon=moon)
    return templates.TemplateResponse("about.html", ctx)


# ── Grimoire: sabbat celebration endpoint ───────────────────────────────────

class AddSabbatRequest(BaseModel):
    sabbat_name: str
    sabbat_date: str | None = None
    celebration_notes: str = ""
    rituals_performed: str = ""
    offerings: str = ""
    intentions: str = ""
    reflections: str = ""
    energy_level: int | None = None


@app.post("/api/grimoire/sabbats")
async def api_grimoire_add_sabbat(request: Request, req: AddSabbatRequest):
    """Save a sabbat celebration entry to the grimoire."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    db = _get_grimoire_db()
    try:
        entry_id = db.add_sabbat_entry(
            user_id=str(user["id"]),
            sabbat_name=req.sabbat_name,
            sabbat_date=req.sabbat_date,
            celebration_notes=req.celebration_notes,
            rituals_performed=req.rituals_performed,
            offerings=req.offerings,
            intentions=req.intentions,
            reflections=req.reflections,
            energy_level=req.energy_level,
        )
        new_milestones = db.check_milestones(str(user["id"]))
        return {"entry_id": entry_id, "new_milestones": new_milestones}
    except AttributeError:
        # add_sabbat_entry may not exist in all DB versions — log to practice instead
        logger.warning("add_sabbat_entry not available, logging as practice session")
        entry_id = db.log_practice(
            user_id=str(user["id"]),
            practice_type=f"sabbat:{req.sabbat_name}",
            notes=req.celebration_notes or req.reflections,
        )
        new_milestones = db.check_milestones(str(user["id"]))
        return {"entry_id": entry_id, "new_milestones": new_milestones}


# ── Grimoire: practice-log alias ─────────────────────────────────────────────

@app.post("/api/grimoire/practice-log")
async def api_grimoire_practice_log_alias(request: Request, req: LogPracticeRequest):
    """Alias for POST /api/grimoire/practice — same behaviour, different path."""
    return await api_grimoire_log_practice(request, req)


# ── Meditation: start endpoint ───────────────────────────────────────────────

class MeditationStartRequest(BaseModel):
    meditation_key: str = "moon_meditation"
    intention: str = ""
    duration_minutes: int = 10


@app.post("/api/meditation/start")
async def api_meditation_start(request: Request, req: MeditationStartRequest):
    """Begin a meditation session.

    If ElevenLabs is configured, generate a short TTS opening for the meditation.
    Otherwise return text-only mode.
    """
    # Fetch the meditation script
    med = get_meditation(req.meditation_key)
    if not med:
        # Fallback: produce a generic opening based on intention
        med = {
            "name": req.intention or "Spiritual Meditation",
            "opening": (
                f"Close your eyes and take a deep breath. "
                f"You are entering a sacred space of {req.intention or 'stillness and peace'}. "
                f"Allow yourself to release the tension of the day."
            ),
        }

    opening_text = med.get("opening") or med.get("description") or (
        "Close your eyes. Breathe deeply. You are safe and held by the universe."
    )

    # Attempt ElevenLabs TTS for the opening paragraph
    audio_url = None
    if avatar.voice.available:
        try:
            result = avatar.generate_voice_only(opening_text[:500])
            audio_url = result.get("audio_url") or result.get("url")
        except Exception as e:
            logger.debug("Meditation TTS generation failed: %s", e)

    return {
        "meditation": med,
        "opening_text": opening_text,
        "audio_url": audio_url,
        "duration_minutes": req.duration_minutes,
        "message": "audio ready" if audio_url else "text mode only",
    }


# ── Guardian Agent API ───────────────────────────────────────────────────────

@app.get("/api/guardian/status")
async def guardian_status():
    """Get Luna Guardian agent status and dashboard."""
    return guardian.get_status()


@app.post("/api/guardian/scan")
async def guardian_scan(request: Request):
    """Trigger a guardian scan manually. Body: {"tier": "pulse|scan|intel|daily"}"""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    tier = body.get("tier", "scan")
    if tier not in ("pulse", "scan", "intel", "daily"):
        raise HTTPException(400, f"Invalid tier: {tier}. Use pulse, scan, intel, or daily")
    result = await guardian.run_once_async(tier)
    return result


@app.get("/api/guardian/issues")
async def guardian_issues(severity: str | None = None):
    """Get open issues found by the guardian."""
    return {"issues": guardian.memory.get_open_issues(severity)}


@app.post("/api/guardian/issues/{issue_id}/resolve")
async def guardian_resolve_issue(issue_id: int, request: Request):
    """Mark an issue as resolved."""
    body = await request.json()
    guardian.memory.resolve_issue(issue_id, body.get("fix", "Manual resolution"))
    return {"status": "resolved", "issue_id": issue_id}


@app.get("/api/guardian/ideas")
async def guardian_ideas(category: str | None = None, limit: int = 20):
    """Get top enhancement ideas ranked by priority."""
    return {"ideas": guardian.memory.get_top_ideas(limit, category)}


@app.post("/api/guardian/ideas/{idea_id}/implement")
async def guardian_implement_idea(idea_id: int):
    """Mark an idea as implemented."""
    guardian.memory.implement_idea(idea_id)
    return {"status": "implemented", "idea_id": idea_id}


@app.get("/api/guardian/learnings")
async def guardian_learnings(domain: str | None = None):
    """Get agent learnings sorted by confidence."""
    return {"learnings": guardian.memory.get_learnings(domain)}


@app.get("/api/guardian/heartbeats")
async def guardian_heartbeats(tier: str | None = None, limit: int = 20):
    """Get heartbeat history."""
    return {"heartbeats": guardian.memory.get_heartbeat_history(tier, limit)}


@app.get("/api/guardian/scans")
async def guardian_scans(target: str | None = None, limit: int = 50):
    """Get recent scan results."""
    return {"scans": guardian.memory.get_recent_scans(target, limit)}


# ═══════════════════════════════════════════════════════════════════════
# Reading Bundle / Credits System
# ═══════════════════════════════════════════════════════════════════════

READING_BUNDLES = [
    {
        "id": "pack_3",
        "name": "3 Reading Pack",
        "readings": 3,
        "price_cents": 1799,  # $17.99 (save 15%)
        "savings": "Save 15%",
        "description": "3 premium readings — use on any spread, any time",
    },
    {
        "id": "pack_5",
        "name": "5 Reading Pack",
        "readings": 5,
        "price_cents": 2499,  # $24.99 (save 25%)
        "savings": "Save 25%",
        "description": "5 premium readings — best value for regular seekers",
    },
    {
        "id": "pack_10",
        "name": "10 Reading Pack",
        "readings": 10,
        "price_cents": 3999,  # $39.99 (save 40%)
        "savings": "Save 40%",
        "description": "10 premium readings — for dedicated practitioners",
    },
]


@app.get("/api/bundles")
async def get_bundles():
    """Return all available reading bundle options."""
    return {"bundles": READING_BUNDLES}


@app.get("/api/user/credits")
async def get_user_credits(request: Request):
    """Return the current user's remaining reading credits."""
    user = get_current_user(request)
    if not user:
        return {"credits": 0, "logged_in": False}
    credits = memory.get_credits(user["id"])
    return {"credits": credits, "logged_in": True, "user_id": user["id"]}


@app.get("/api/user/subscription")
async def get_user_subscription(request: Request):
    """Return the current user's subscription tier and status."""
    user = get_current_user(request)
    if not user:
        return {"tier": "free", "logged_in": False}
    return {
        "tier": user.get("subscription_tier", "free"),
        "status": user.get("subscription_status", "none"),
        "logged_in": True,
        "user_id": user["id"],
    }


# ═══════════════════════════════════════════════════════════════════════
# Subscription Plans — Stripe Checkout for recurring plans
# ═══════════════════════════════════════════════════════════════════════

SUBSCRIPTION_PLANS = {
    "seeker": {
        "name": "Seeker",
        "price_cents": 999,   # $9.99/month
        "description": "Weekly personalized card pull, moon phase alerts, priority chat",
        "features": [
            "Weekly personalized tarot pull",
            "Moon phase ritual reminders",
            "Priority Luna chat",
            "Exclusive journal prompts",
        ],
    },
    "mystic": {
        "name": "Mystic",
        "price_cents": 1999,  # $19.99/month
        "description": "Everything in Seeker + monthly birth chart transit, custom rituals",
        "features": [
            "Everything in Seeker",
            "Monthly birth chart transit reading",
            "Custom ritual recommendations each moon phase",
            "2 premium readings/month included",
            "Exclusive Substack content",
        ],
    },
    "inner-circle": {
        "name": "Inner Circle",
        "price_cents": 2999,  # $29.99/month
        "description": "Full spiritual mentorship — unlimited readings, voice sessions",
        "features": [
            "Everything in Mystic",
            "Unlimited premium readings",
            "Monthly 1-on-1 voice session with Luna",
            "Personalized spell/ritual creation",
            "Early access to new features",
            "Private Discord channel",
        ],
    },
}


@app.get("/api/plans")
async def get_plans():
    """Return all subscription plan details."""
    return {"plans": SUBSCRIPTION_PLANS}


@app.get("/api/subscribe")
async def subscribe_redirect(request: Request, plan: str = ""):
    """Create a Stripe Checkout Session for a subscription plan.

    This is the endpoint that /register redirects to after signup when
    the user selected a plan on the pricing page.
    """
    user = get_current_user(request)
    if not user:
        return RedirectResponse(f"/register?plan={plan}", status_code=302)

    plan_data = SUBSCRIPTION_PLANS.get(plan)
    if not plan_data:
        return RedirectResponse("/pricing", status_code=302)

    try:
        import stripe
        stripe.api_key = config.stripe.api_key
        if not stripe.api_key:
            raise ValueError("Stripe API key not configured")

        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "unit_amount": plan_data["price_cents"],
                    "recurring": {"interval": "month"},
                    "product_data": {
                        "name": f"Mystic Luna — {plan_data['name']}",
                        "description": plan_data["description"],
                    },
                },
                "quantity": 1,
            }],
            customer_email=user.get("email"),
            metadata={
                "plan": plan,
                "user_id": str(user["id"]),
            },
            success_url=f"{config.site_url}/dashboard?subscribed={plan}",
            cancel_url=f"{config.site_url}/pricing?cancelled=1",
        )
        return RedirectResponse(session.url, status_code=303)

    except ImportError:
        logger.warning("Stripe not installed — cannot process subscription")
        return RedirectResponse("/pricing?error=stripe_not_configured", status_code=302)
    except Exception as e:
        logger.error("Subscription checkout failed: %s", e)
        return RedirectResponse("/pricing?error=checkout_failed", status_code=302)


@app.post("/api/subscription/cancel")
async def cancel_subscription(request: Request):
    """Cancel the current user's subscription (downgrades to free at period end)."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Sign in required")

    tier = user.get("subscription_tier", "free")
    if tier == "free":
        return {"status": "already_free"}

    stripe_cid = user.get("stripe_customer_id")
    if stripe_cid:
        try:
            import stripe
            stripe.api_key = config.stripe.api_key
            # Cancel all active subscriptions for this customer
            subs = stripe.Subscription.list(customer=stripe_cid, status="active")
            for sub in subs.auto_paging_iter():
                stripe.Subscription.modify(sub.id, cancel_at_period_end=True)
                logger.info("Subscription %s set to cancel at period end", sub.id)
            return {"status": "canceling_at_period_end"}
        except ImportError:
            pass
        except Exception as e:
            logger.error("Stripe cancel failed: %s", e)

    # Fallback: downgrade immediately if Stripe is unavailable
    auth.update_subscription(user["id"], "free")
    return {"status": "canceled"}


# ═══════════════════════════════════════════════════════════════════════
# Credit-Based Reading Redemption
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/readings/redeem")
async def redeem_reading(request: Request):
    """Use a reading credit to generate a reading without payment.

    Body: {"spread_type": "celtic_cross", "question": "..."}
    """
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Sign in to use reading credits")

    body = await request.json()
    spread_type = body.get("spread_type", "past_present_future")
    question = body.get("question", "")

    # Verify credits
    credits = memory.get_credits(user["id"])
    if credits <= 0:
        raise HTTPException(402, "No reading credits remaining. Purchase a bundle to continue.")

    # Validate spread type
    available = brain.readings.get_available_readings()
    spread = next((r for r in available if r["id"] == spread_type), None)
    if not spread:
        raise HTTPException(400, f"Unknown spread type: {spread_type}")

    # Redeem credit first
    if not memory.redeem_credit(user["id"], spread_type):
        raise HTTPException(402, "Credit redemption failed")

    # Generate the reading
    try:
        result = await brain.readings.generate_reading(
            spread_type=spread_type,
            question=question,
            customer_id=str(user["id"]),
            channel="credit",
        )
        remaining = memory.get_credits(user["id"])
        result["credits_remaining"] = remaining
        return result
    except Exception as e:
        # Refund the credit on failure
        memory.add_credits(user["id"], 1, "refund_failed_reading")
        logger.error("Credit reading generation failed (credit refunded): %s", e)
        raise HTTPException(500, "Reading generation failed — your credit has been refunded")


# ═══════════════════════════════════════════════════════════════════════
# Account Management
# ═══════════════════════════════════════════════════════════════════════

@app.get("/account", response_class=HTMLResponse)
async def account_page(request: Request):
    """User account management page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login?next=/account", status_code=302)

    credits = memory.get_credits(user["id"])
    profile = companion.get_profile(user["id"])

    ctx = template_context(
        request,
        credits=credits,
        profile=profile,
        plans=SUBSCRIPTION_PLANS,
        zodiac_signs=ZODIAC_SIGNS,
    )
    return templates.TemplateResponse("account.html", ctx)


@app.post("/api/account/update")
async def update_account(request: Request):
    """Update user profile (name, zodiac sign)."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Sign in required")

    body = await request.json()
    updates = {}
    if "name" in body:
        updates["name"] = body["name"].strip()[:100]
    if "zodiac" in body and body["zodiac"] in ZODIAC_SIGNS + [""]:
        updates["zodiac_sign"] = body["zodiac"]
    if "birth_date" in body:
        updates["birth_date"] = body["birth_date"]

    if not updates:
        return {"status": "no_changes"}

    result = auth.update_user(user["id"], **updates)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"status": "updated", "user": result.get("user")}


@app.post("/api/account/change-password")
async def change_password(request: Request):
    """Change the current user's password."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Sign in required")

    body = await request.json()
    current = body.get("current_password", "")
    new_pw = body.get("new_password", "")

    if not current or not new_pw:
        raise HTTPException(400, "Both current_password and new_password required")
    if len(new_pw) < 8:
        raise HTTPException(400, "New password must be at least 8 characters")

    # Verify current password
    login_result = auth.login(user["email"], current)
    if "error" in login_result:
        raise HTTPException(403, "Current password is incorrect")

    result = auth.update_user(user["id"], password=new_pw)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"status": "password_changed"}


@app.post("/api/bundles/purchase")
async def purchase_bundle(request: Request):
    """Create a Stripe Checkout Session for a reading bundle."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Sign in to purchase reading bundles")

    body = await request.json()
    bundle_id = body.get("bundle_id")
    bundle = next((b for b in READING_BUNDLES if b["id"] == bundle_id), None)
    if not bundle:
        raise HTTPException(400, f"Unknown bundle: {bundle_id}")

    try:
        import stripe
        stripe.api_key = config.stripe.api_key

        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "unit_amount": bundle["price_cents"],
                    "product_data": {
                        "name": f"Mystic Luna \u2014 {bundle['name']}",
                        "description": bundle["description"],
                    },
                },
                "quantity": 1,
            }],
            metadata={
                "type": "bundle",
                "bundle_id": bundle_id,
                "user_id": str(user["id"]),
                "readings": str(bundle["readings"]),
            },
            success_url=f"{config.site_url}/pricing?bundle=success",
            cancel_url=f"{config.site_url}/pricing?bundle=cancelled",
        )
        return {"checkout_url": session.url}
    except ImportError:
        raise HTTPException(503, "Stripe not configured")
    except Exception as e:
        logger.error("Bundle checkout failed: %s", e)
        raise HTTPException(500, f"Checkout failed: {e}")
