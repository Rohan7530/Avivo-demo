"""RAG Worker: consumes text_stream via Redis Streams consumer group.

Lifecycle:
1. Load sentence-transformer model + ChromaDB client at startup (once).
2. XREADGROUP loop on "text_stream" / "rag_group".
3. For each message:
   a. Check L1/L2 cache by query hash.
   b. On miss: embed query → retrieve top-3 chunks → build prompt → call Ollama.
   c. Store answer in Redis result key (LPUSH result:{result_key}).
   d. Update session history.
   e. XACK the message.
4. Log embed_ms, retrieval_ms, llm_ms, total_ms per query.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List

import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

# Ensure ChromaDB and embedder are loaded once at module/startup time
from rag import embedder as emb_module  # noqa: E402
from rag import retriever as ret_module  # noqa: E402
from rag.prompt_builder import build_prompt, extract_source_attribution  # noqa: E402
from cache.redis_cache import RedisCache  # noqa: E402
from history.session import SessionHistory  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_worker")

STREAM_NAME = "text_stream"
GROUP_NAME = "rag_group"
CONSUMER_NAME = "rag-worker-1"
RESULT_TTL = 120  # seconds to keep result key

_MAX_RETRIES = 3
_BASE_DELAY = 1.0


# ---------------------------------------------------------------------------
# Ollama LLM call with exponential backoff
# ---------------------------------------------------------------------------

async def _call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    timeout: float = 120.0,
) -> str:
    """Call Ollama /api/generate with exponential backoff."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{ollama_url}/api/generate", json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "")
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            delay = _BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Ollama attempt %d/%d failed: %s — retrying in %.1fs.",
                attempt, _MAX_RETRIES, exc, delay,
            )
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(delay)
            else:
                raise
    return ""


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

async def process_rag_message(
    message_data: Dict[str, str],
    cache: RedisCache,
    history: SessionHistory,
    redis: aioredis.Redis,
    ollama_url: str,
    llm_model: str,
) -> None:
    """
    Process a single RAG message from the stream.

    Args:
        message_data: Decoded stream message fields.
        cache: Two-layer cache instance.
        history: Session history instance.
        redis: Async Redis client.
        ollama_url: Ollama base URL.
        llm_model: LLM model name.
    """
    t_total_start = time.perf_counter()

    user_id = message_data.get("user_id", "unknown")
    query = message_data.get("query", "")
    result_key = message_data.get("result_key", "")

    if not query or not result_key:
        logger.warning("Invalid message — missing query or result_key.")
        return

    logger.info("Processing RAG query from user %s: %.80s", user_id, query)

    # ── Cache check ──────────────────────────────────────────────────────────
    cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    cached = await cache.get(cache_key)
    if cached:
        logger.info("Cache HIT for query hash %s.", cache_key)
        await redis.lpush(result_key, json.dumps(cached))
        await redis.expire(result_key, RESULT_TTL)
        return

    # ── Embed ────────────────────────────────────────────────────────────────
    t_embed = time.perf_counter()
    query_embedding = emb_module.embed_single(query)
    embed_ms = (time.perf_counter() - t_embed) * 1000

    # ── Retrieve ─────────────────────────────────────────────────────────────
    t_retrieval = time.perf_counter()
    chunks = ret_module.retrieve(query_embedding, top_k=3)
    retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

    # ── Session history ───────────────────────────────────────────────────────
    turn_history = await history.get_last_n_turns(user_id, n=3)

    # ── Build prompt ─────────────────────────────────────────────────────────
    prompt = build_prompt(query=query, context_chunks=chunks, history=turn_history)
    sources = extract_source_attribution(chunks)

    # ── LLM call ─────────────────────────────────────────────────────────────
    t_llm = time.perf_counter()
    try:
        answer = await _call_ollama(prompt, model=llm_model, ollama_url=ollama_url)
    except Exception as exc:
        logger.error("Ollama LLM call failed: %s", exc)
        answer = "Sorry, I encountered an error generating your answer. Please try again."
        sources = ""
    llm_ms = (time.perf_counter() - t_llm) * 1000

    total_ms = (time.perf_counter() - t_total_start) * 1000
    logger.info(
        "Latency — embed=%.0fms retrieval=%.0fms llm=%.0fms total=%.0fms",
        embed_ms, retrieval_ms, llm_ms, total_ms,
    )

    # ── Format response ──────────────────────────────────────────────────────
    if sources:
        formatted_answer = f"{answer}\n\n*Sources:* {sources}"
    else:
        formatted_answer = answer

    result_payload = {
        "answer": formatted_answer,
        "sources": sources,
        "embed_ms": embed_ms,
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "total_ms": total_ms,
    }

    # ── Store result ─────────────────────────────────────────────────────────
    await redis.lpush(result_key, json.dumps(result_payload))
    await redis.expire(result_key, RESULT_TTL)

    # ── Cache the result ──────────────────────────────────────────────────────
    ttl = int(os.getenv("CACHE_TTL", "3600"))
    await cache.set(cache_key, result_payload, ttl=ttl)

    # ── Update session history ────────────────────────────────────────────────
    await history.add(user_id, role="user", content=query)
    await history.add(user_id, role="assistant", content=answer[:500])

    # ── Increment total query counter ─────────────────────────────────────────
    await redis.incr("stats:total_queries")
    latency_list_key = "stats:latencies"
    await redis.rpush(latency_list_key, int(total_ms))
    await redis.ltrim(latency_list_key, -1000, -1)  # keep last 1000


# ---------------------------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------------------------

async def run() -> None:
    """Start the RAG worker consumer loop."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm_model = os.getenv("LLM_MODEL", "phi3:mini")

    logger.info("RAG Worker starting up.")
    logger.info("Redis: %s | Ollama: %s | Model: %s", redis_url, ollama_url, llm_model)

    # Pre-load embedding model and ChromaDB (singleton pattern)
    logger.info("Pre-loading embedding model...")
    emb_module._get_model()
    logger.info("Pre-loading ChromaDB collection...")
    ret_module._get_collection()

    # Async Redis client
    pool = aioredis.ConnectionPool.from_url(redis_url, max_connections=10, decode_responses=True)
    redis = aioredis.Redis(connection_pool=pool)

    cache = RedisCache(redis_url=redis_url)
    history = SessionHistory(redis_url=redis_url)

    # Create consumer group (ignore error if already exists)
    try:
        await redis.xgroup_create(STREAM_NAME, GROUP_NAME, id="0", mkstream=True)
        logger.info("Consumer group '%s' created on stream '%s'.", GROUP_NAME, STREAM_NAME)
    except Exception:
        logger.info("Consumer group '%s' already exists.", GROUP_NAME)

    logger.info("RAG Worker ready. Listening on '%s'...", STREAM_NAME)

    while True:
        try:
            entries: List[Any] = await redis.xreadgroup(
                groupname=GROUP_NAME,
                consumername=CONSUMER_NAME,
                streams={STREAM_NAME: ">"},
                count=1,
                block=5000,  # block up to 5s
            )

            if not entries:
                continue

            for stream, messages in entries:
                for msg_id, msg_data in messages:
                    try:
                        await process_rag_message(
                            message_data=msg_data,
                            cache=cache,
                            history=history,
                            redis=redis,
                            ollama_url=ollama_url,
                            llm_model=llm_model,
                        )
                        await redis.xack(STREAM_NAME, GROUP_NAME, msg_id)
                        logger.debug("XACK message %s.", msg_id)
                    except Exception as exc:
                        logger.error("Error processing message %s: %s", msg_id, exc, exc_info=True)

        except asyncio.CancelledError:
            logger.info("RAG Worker shutting down.")
            break
        except Exception as exc:
            logger.error("Unexpected error in main loop: %s", exc, exc_info=True)
            await asyncio.sleep(2)

    await cache.close()
    await history.close()
    await redis.aclose()


if __name__ == "__main__":
    asyncio.run(run())
