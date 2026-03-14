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
from moneyclaw.auth import Auth
from moneyclaw.services.luna.persona import get_moon_phase, ZODIAC_SIGNS
from moneyclaw.services.luna.companion import CompanionEngine, MOODS, RELATIONSHIP_LEVELS
from moneyclaw.services.avatar.pipeline import AvatarPipeline
from moneyclaw.services.email import EmailService
from moneyclaw.services.luna.luna_engine import LunaEngine
from moneyclaw.config import get_config, VIDEOS_DIR, AUDIO_DIR

logger = logging.getLogger("moneyclaw.api")

memory = Memory()
brain = Brain(memory)
scheduler = Scheduler(brain)
auth = Auth(memory)
companion = CompanionEngine(memory)
avatar = AvatarPipeline(memory)
email_service = EmailService()
luna_engine = LunaEngine(memory)
config = get_config()

# Template and static paths
WEB_DIR = Path(__file__).parent.parent / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    logger.info("MoneyClaw API started — scheduler running")
    yield
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
async def register_page(request: Request):
    user = get_current_user(request)
    if user:
        return RedirectResponse("/dashboard", status_code=302)
    ctx = template_context(request, error=None)
    return templates.TemplateResponse("register.html", ctx)


@app.post("/register")
async def register_submit(request: Request,
                          email: str = Form(...),
                          password: str = Form(...),
                          name: str = Form(""),
                          zodiac: str = Form("")):
    result = auth.register(email, password, name=name, zodiac_sign=zodiac)
    if "error" in result:
        ctx = template_context(request, error=result["error"])
        return templates.TemplateResponse("register.html", ctx)

    # Send welcome email (non-blocking)
    try:
        email_service.send_welcome_email(to_email=email, name=name or "Seeker")
    except Exception as e:
        logger.debug("Welcome email failed: %s", e)

    response = RedirectResponse("/dashboard", status_code=302)
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

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    body = await request.body()
    sig = request.headers.get("stripe-signature", "")

    # Verify Stripe signature if webhook secret is configured
    webhook_secret = config.stripe.webhook_secret
    if webhook_secret:
        try:
            import stripe
            event = stripe.Webhook.construct_event(body, sig, webhook_secret)
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
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload")

    event_type = event.get("type", "")

    if event_type == "checkout.session.completed":
        session = event.get("data", {}).get("object", {})
        amount = session.get("amount_total", 0)
        customer_email = session.get("customer_email", "")
        metadata = session.get("metadata", {})
        spread_type = metadata.get("spread_type", "unknown")

        brain.ledger.record_sale(
            source="readings",
            amount_cents=amount,
            product=spread_type,
            customer_id=customer_email,
            stripe_id=session.get("id"),
        )

        # Auto-generate and deliver the reading
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
            except Exception as e:
                logger.error("Post-payment reading generation failed: %s", e)

        logger.info(
            "Payment received: $%.2f for %s from %s",
            amount / 100, spread_type, customer_email
        )
        return {"status": "recorded"}

    return {"status": "ignored", "event_type": event_type}
