"""Telegram bot — Luna's primary customer-facing channel.

Handles:
- Incoming reading requests via DM
- Daily card pull posting to channel
- Horoscope delivery
- Payment link generation
- Customer feedback collection
- Nick notifications (revenue alerts, approvals needed)
"""

import logging
import hashlib
from datetime import datetime, timezone

from ..config import get_config
from ..memory import Memory

logger = logging.getLogger("moneyclaw.telegram")

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        CallbackQueryHandler, filters,
    )
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False


READING_MENU = {
    "daily_pull": ("Daily Card Pull", "Free", "free"),
    "yes_no": ("Yes/No Reading", "Free", "free"),
    "past_present_future": ("Past/Present/Future", "$7.99", "7.99"),
    "love_relationships": ("Love & Relationships", "$12.99", "12.99"),
    "career_crossroads": ("Career Crossroads", "$12.99", "12.99"),
    "shadow_work": ("Shadow Work", "$17.99", "17.99"),
    "celtic_cross": ("Celtic Cross Deep Dive", "$19.99", "19.99"),
    "year_ahead": ("Year Ahead Forecast", "$49.99", "49.99"),
}


class TelegramBot:
    """Telegram bot for Mystic Luna readings."""

    def __init__(self, brain=None, memory: Memory | None = None):
        self.config = get_config()
        self.memory = memory or Memory()
        self.brain = brain  # Set after brain is initialized
        self._app = None

    def _customer_id(self, user_id: int) -> str:
        return hashlib.sha256(f"tg:{user_id}".encode()).hexdigest()[:16]

    async def _cmd_start(self, update: Update, context):
        """Welcome message."""
        await update.message.reply_text(
            "Blessed be, dear one. I am Luna Moonshadow, your AI-powered "
            "spiritual guide. The cards have been waiting for you. 🌙\n\n"
            "What I can do:\n"
            "🔮 /reading — Get a tarot reading\n"
            "✨ /daily — Today's card pull\n"
            "⭐ /horoscope — Your daily horoscope\n"
            "🌙 /moon — Current moon phase\n"
            "💜 /help — See all options\n\n"
            "Simply ask me a question, and I'll consult the cards for you."
        )

    async def _cmd_reading(self, update: Update, context):
        """Show reading menu."""
        buttons = []
        for key, (name, price, _) in READING_MENU.items():
            buttons.append([
                InlineKeyboardButton(
                    f"{name} — {price}",
                    callback_data=f"reading:{key}"
                )
            ])

        await update.message.reply_text(
            "Which reading calls to your spirit today? 🔮",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _cmd_daily(self, update: Update, context):
        """Quick daily card pull."""
        if not self.brain:
            await update.message.reply_text("Luna is still awakening... try again shortly. 🌙")
            return

        cid = self._customer_id(update.effective_user.id)
        result = self.brain.handle_reading_request(
            spread_type="daily_pull",
            customer_id=cid,
            channel="telegram",
        )
        await update.message.reply_text(result["reading"])

    async def _cmd_moon(self, update: Update, context):
        """Current moon phase."""
        from ..services.luna.persona import get_moon_phase
        moon = get_moon_phase()
        await update.message.reply_text(
            f"🌙 Current Moon Phase: **{moon['phase']}**\n"
            f"Illumination: {moon['illumination']}%\n\n"
            f"✨ {moon['guidance']}",
            parse_mode="Markdown",
        )

    async def _cmd_help(self, update: Update, context):
        """Help menu."""
        await update.message.reply_text(
            "🔮 **Mystic Luna — Your Spiritual Guide**\n\n"
            "**Free Services:**\n"
            "/daily — Daily card pull\n"
            "/horoscope [sign] — Daily horoscope\n"
            "/moon — Current moon phase\n"
            "/yesno [question] — Quick yes/no\n\n"
            "**Premium Readings:**\n"
            "/reading — Browse all reading types\n\n"
            "**Or simply message me your question** and I'll suggest "
            "the perfect reading for you.\n\n"
            "💜 All readings are AI-powered, guided by centuries of "
            "mystical tradition.",
            parse_mode="Markdown",
        )

    async def _cmd_yesno(self, update: Update, context):
        """Quick yes/no reading."""
        question = " ".join(context.args) if context.args else None
        if not question:
            await update.message.reply_text(
                "Ask your yes/no question: /yesno Will I find love this year?"
            )
            return

        if not self.brain:
            await update.message.reply_text("Luna is still awakening... 🌙")
            return

        result = self.brain.readings.quick_yes_no(
            question=question,
            customer_id=self._customer_id(update.effective_user.id),
        )
        await update.message.reply_text(result["reading"])

    async def _cmd_status(self, update: Update, context):
        """Status (Nick only)."""
        nick_id = self.config.telegram.nick_chat_id
        if str(update.effective_user.id) != nick_id:
            await update.message.reply_text("This command is for the operator only.")
            return

        if not self.brain:
            await update.message.reply_text("Brain not initialized.")
            return

        status = self.brain.status()
        fin = status["financial"]
        await update.message.reply_text(
            f"**MoneyClaw Status**\n"
            f"State: {status['state']}\n"
            f"Revenue (30d): ${fin['total_revenue_cents']/100:.2f}\n"
            f"Profit (30d): ${fin['net_profit_cents']/100:.2f}\n"
            f"Budget: ${fin['budget_remaining_cents']/100:.2f}\n"
            f"Customers: {status['customers'].get('total_customers', 0)}\n"
            f"Learnings: {status['learnings']}\n"
            f"Experiments: {status['experiments']}",
            parse_mode="Markdown",
        )

    async def _handle_callback(self, update: Update, context):
        """Handle inline button presses."""
        query = update.callback_query
        await query.answer()

        data = query.data
        if data.startswith("reading:"):
            spread_type = data.split(":")[1]
            info = READING_MENU.get(spread_type)
            if not info:
                await query.edit_message_text("Unknown reading type.")
                return

            name, price, price_val = info

            if price_val == "free":
                # Generate free reading directly
                if self.brain:
                    await query.edit_message_text(
                        f"Drawing cards for your {name}... 🔮✨"
                    )
                    cid = self._customer_id(query.from_user.id)
                    result = self.brain.handle_reading_request(
                        spread_type=spread_type,
                        customer_id=cid,
                        channel="telegram",
                    )
                    await query.message.reply_text(result["reading"])
            else:
                # Show payment prompt for paid readings
                payment_link = self.config.stripe.payment_links.get(spread_type, "")
                if payment_link:
                    await query.edit_message_text(
                        f"**{name}** — {price}\n\n"
                        f"Complete payment here, then send me your question:\n"
                        f"{payment_link}\n\n"
                        f"Or simply send me your question and I'll guide you "
                        f"through the payment process. 💜",
                        parse_mode="Markdown",
                    )
                else:
                    await query.edit_message_text(
                        f"**{name}** — {price}\n\n"
                        f"Send me your question, and I'll prepare your reading.\n"
                        f"Payment processing is being set up. 💜",
                        parse_mode="Markdown",
                    )

    async def _handle_message(self, update: Update, context):
        """Handle free-text messages — route through LunaEngine."""
        text = update.message.text
        cid = self._customer_id(update.effective_user.id)

        # Try LunaEngine first for intelligent routing
        try:
            from ..services.luna.luna_engine import LunaEngine
            engine = LunaEngine(self.memory)
            result = engine.respond(
                user_id=cid,
                message=text,
                channel="telegram",
            )
            await update.message.reply_text(result.text)
            return
        except Exception as e:
            logger.debug("LunaEngine not available, falling back: %s", e)

        # Fallback to direct brain routing
        if not self.brain:
            await update.message.reply_text(
                "Luna is still awakening... please try again shortly. 🌙"
            )
            return

        # Quick yes/no detection
        lower = text.lower().strip()
        if lower.startswith(("will ", "should ", "can ", "is ", "does ", "am ")):
            result = self.brain.readings.quick_yes_no(
                question=text, customer_id=cid
            )
            await update.message.reply_text(result["reading"])
            return

        # Default: suggest a reading
        await update.message.reply_text(
            f"I sense a question stirring in your spirit. 🔮\n\n"
            f"Let me draw some cards for you...\n"
            f"Use /reading to choose a specific spread, "
            f"or /yesno for a quick answer."
        )

    def build_app(self) -> "Application":
        """Build the telegram bot application."""
        if not HAS_TELEGRAM:
            raise RuntimeError("python-telegram-bot not installed")

        token = self.config.telegram.bot_token
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

        self._app = Application.builder().token(token).build()

        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("reading", self._cmd_reading))
        self._app.add_handler(CommandHandler("daily", self._cmd_daily))
        self._app.add_handler(CommandHandler("moon", self._cmd_moon))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("yesno", self._cmd_yesno))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CallbackQueryHandler(self._handle_callback))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        return self._app

    async def post_to_channel(self, text: str):
        """Post content to Luna's Telegram channel."""
        if not self._app:
            return
        channel = self.config.telegram.channel_id
        if channel:
            await self._app.bot.send_message(chat_id=channel, text=text)

    async def notify_nick(self, message: str):
        """Send notification to Nick."""
        if not self._app:
            return
        nick_id = self.config.telegram.nick_chat_id
        if nick_id:
            await self._app.bot.send_message(chat_id=nick_id, text=message)
