"""Async Telegram Bot using python-telegram-bot v20+.

Commands:
    /start  -- Welcome message
    /help   -- Usage guide
    /ask    -- RAG pipeline query
    /image  -- Vision pipeline (reply/send after uploading)
    /summarize -- Summarize last 3 interactions
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import Optional

import redis.asyncio as aioredis
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

from history.session import SessionHistory  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram_bot")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
RESULT_POLL_TIMEOUT = 30  # seconds
HELP_TEXT = """
*AvivaBot — Hybrid AI Assistant* 

*Commands:*

`/ask <your question>` — Ask anything from the knowledge base
  _Example:_ `/ask What is the WFH policy?`

`/image` — Send this command, then upload a photo to get an AI description

`/summarize` — View a summary of your last 3 interactions

`/help` — Show this help message

*Powered by:* phi3:mini (RAG) + LLaVA (Vision) + ChromaDB + Redis Streams
"""


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

_redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """Lazily initialise and return the Redis client."""
    global _redis_client
    if _redis_client is None:
        pool = aioredis.ConnectionPool.from_url(
            REDIS_URL, max_connections=20, decode_responses=True
        )
        _redis_client = aioredis.Redis(connection_pool=pool)
    return _redis_client


async def push_to_stream(stream: str, payload: dict) -> str:
    """Add *payload* to *stream* and return the message ID."""
    r = await get_redis()
    msg_id = await r.xadd(stream, {k: str(v) for k, v in payload.items()})
    return msg_id


async def poll_result(result_key: str, timeout: int = RESULT_POLL_TIMEOUT) -> Optional[dict]:
    """
    Poll Redis for the worker result using BLPOP.

    Args:
        result_key: The Redis key where the worker will push the result.
        timeout: Maximum seconds to wait.

    Returns:
        Parsed result dict, or None on timeout.
    """
    r = await get_redis()
    result = await r.blpop(result_key, timeout=timeout)
    if result is None:
        return None
    _, raw = result
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    await update.message.reply_text(
        f"Hi {user.first_name}! I'm *AvivaBot*, your AI assistant.\n\n"
        "Type `/help` to see what I can do!",
        parse_mode=ParseMode.MARKDOWN,
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)


async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /ask <query> command.
    Pushes query to text_stream and polls for the RAG worker result.
    """
    user_id = str(update.effective_user.id)
    query = " ".join(context.args).strip() if context.args else ""

    if not query:
        await update.message.reply_text(
            "Please provide a question. Example: `/ask What is the leave policy?`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    result_key = f"result:rag:{uuid.uuid4().hex}"
    payload = {
        "user_id": user_id,
        "query": query,
        "result_key": result_key,
        "timestamp": int(time.time()),
    }

    thinking_msg = await update.message.reply_text("Thinking... Please wait.")

    await push_to_stream("text_stream", payload)
    logger.info("Pushed RAG query from user %s to text_stream.", user_id)

    result = await poll_result(result_key, timeout=RESULT_POLL_TIMEOUT)

    if result is None:
        await thinking_msg.edit_text(
            "Still processing, please wait or try again.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    answer = result.get("answer", "No answer returned.")
    await thinking_msg.edit_text(answer, parse_mode=ParseMode.MARKDOWN)


async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /image command — sets a flag for the user to send a photo next."""
    user_id = str(update.effective_user.id)
    context.chat_data["awaiting_image"] = user_id
    await update.message.reply_text(
        "Please upload a photo now and I'll describe it for you!",
        parse_mode=ParseMode.MARKDOWN,
    )


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle incoming photos.
    Downloads the image, encodes it as base64, pushes to image_stream,
    and polls for the vision worker result.
    """
    user_id = str(update.effective_user.id)

    # Get the highest-resolution photo
    photo = update.message.photo[-1]
    photo_file = await photo.get_file()
    image_bytes = await photo_file.download_as_bytearray()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    result_key = f"result:vision:{uuid.uuid4().hex}"
    payload = {
        "user_id": user_id,
        "image_b64": image_b64,
        "result_key": result_key,
        "timestamp": int(time.time()),
    }

    processing_msg = await update.message.reply_text("Analysing your image...")

    await push_to_stream("image_stream", payload)
    logger.info("Pushed image from user %s to image_stream.", user_id)

    result = await poll_result(result_key, timeout=RESULT_POLL_TIMEOUT)

    if result is None:
        await processing_msg.edit_text("Still processing your image, please try again.")
        return

    caption = result.get("caption", "Could not generate caption.")
    tags = result.get("tags", [])
    tags_str = " | ".join(f"`{t}`" for t in tags)

    response = f"*Caption:*\n{caption}\n\n*Tags:* {tags_str}"
    await processing_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)

    # Clear the awaiting flag
    context.chat_data.pop("awaiting_image", None)


async def summarize_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /summarize — returns last 3 conversation turns from session history."""
    user_id = str(update.effective_user.id)
    history = SessionHistory(redis_url=REDIS_URL)
    summary = await history.format_for_summary(user_id)
    await update.message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)
    await history.close()


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Build and start the Telegram bot application."""
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set!")

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("ask", ask_handler))
    application.add_handler(CommandHandler("image", image_handler))
    application.add_handler(CommandHandler("summarize", summarize_handler))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))

    logger.info("AvivaBot starting polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
