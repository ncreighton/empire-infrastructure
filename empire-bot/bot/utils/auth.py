"""Authentication decorator for admin-only commands."""

import functools
import logging
from telegram import Update
from telegram.ext import ContextTypes

from bot.config import ADMIN_IDS

logger = logging.getLogger(__name__)


def admin_only(func):
    """Decorator that restricts handler to whitelisted admin user IDs."""
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user = update.effective_user
        if not user or user.id not in ADMIN_IDS:
            uid = user.id if user else "unknown"
            logger.warning("Unauthorized access attempt from user %s", uid)
            if update.message:
                await update.message.reply_text("Access denied.")
            elif update.callback_query:
                await update.callback_query.answer("Access denied.", show_alert=True)
            return
        return await func(update, context, *args, **kwargs)
    return wrapper
