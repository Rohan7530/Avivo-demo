"""Vision Worker: consumes image_stream via Redis Streams consumer group.

Lifecycle:
1. XREADGROUP loop on "image_stream" / "vision_group".
2. For each message:
   a. Decode base64 image from message data.
   b. Run BLIP locally (vision/captioner.py) — no Ollama, fast CPU inference.
   c. Store result in Redis result key.
   d. XACK the message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, List

import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

from vision.captioner import describe_image  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("vision_worker")

STREAM_NAME = "image_stream"
GROUP_NAME = "vision_group"
CONSUMER_NAME = "vision-worker-1"
RESULT_TTL = 120  # seconds


async def process_vision_message(
    message_data: dict,
    redis: aioredis.Redis,
) -> None:
    """
    Process a single vision message from the stream.

    Args:
        message_data: Decoded stream message fields (image_b64, result_key, user_id).
        redis: Async Redis client.
    """
    t_start = time.perf_counter()

    image_b64: str = message_data.get("image_b64", "")
    result_key: str = message_data.get("result_key", "")
    user_id: str = message_data.get("user_id", "unknown")

    if not image_b64 or not result_key:
        logger.warning("Invalid vision message — missing image_b64 or result_key.")
        return

    logger.info("Processing vision request from user %s.", user_id)

    try:
        result = await describe_image(image_b64=image_b64)
    except Exception as exc:
        logger.error("Vision processing failed: %s", exc, exc_info=True)
        result = {
            "caption": "Could not process image. Please try again.",
            "tags": ["error", "retry", "unavailable"],
        }

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Vision done | user=%s total=%.0fms | caption=%.60s | tags=%s",
        user_id, total_ms, result["caption"], result["tags"],
    )

    result["total_ms"] = total_ms

    await redis.lpush(result_key, json.dumps(result))
    await redis.expire(result_key, RESULT_TTL)


async def run() -> None:
    """Start the Vision worker consumer loop."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    logger.info("Vision Worker starting up (BLIP local model — no Ollama needed).")
    logger.info("Redis: %s", redis_url)

    # Pre-load the BLIP model so first request is instant
    logger.info("Pre-loading BLIP captioning model...")
    from vision.captioner import _get_blip
    await asyncio.get_event_loop().run_in_executor(None, _get_blip)
    logger.info("BLIP model ready.")

    pool = aioredis.ConnectionPool.from_url(redis_url, max_connections=10, decode_responses=True)
    redis = aioredis.Redis(connection_pool=pool)

    # Create consumer group
    try:
        await redis.xgroup_create(STREAM_NAME, GROUP_NAME, id="0", mkstream=True)
        logger.info("Consumer group '%s' created on stream '%s'.", GROUP_NAME, STREAM_NAME)
    except Exception:
        logger.info("Consumer group '%s' already exists.", GROUP_NAME)

    logger.info("Vision Worker ready. Listening on '%s'...", STREAM_NAME)

    while True:
        try:
            entries: List[Any] = await redis.xreadgroup(
                groupname=GROUP_NAME,
                consumername=CONSUMER_NAME,
                streams={STREAM_NAME: ">"},
                count=1,
                block=5000,
            )

            if not entries:
                continue

            for stream, messages in entries:
                for msg_id, msg_data in messages:
                    try:
                        await process_vision_message(
                            message_data=msg_data,
                            redis=redis,
                        )
                        await redis.xack(STREAM_NAME, GROUP_NAME, msg_id)
                        logger.debug("XACK message %s.", msg_id)
                    except Exception as exc:
                        logger.error(
                            "Error processing vision message %s: %s", msg_id, exc, exc_info=True
                        )

        except asyncio.CancelledError:
            logger.info("Vision Worker shutting down.")
            break
        except Exception as exc:
            logger.error("Unexpected error in vision loop: %s", exc, exc_info=True)
            await asyncio.sleep(2)

    await redis.aclose()


if __name__ == "__main__":
    asyncio.run(run())
