"""FastAPI sidecar for AvivaBot.

Endpoints:
    GET  /health   -- Service liveness check
    POST /ask      -- RAG query (used by Gradio frontend)
    POST /vision   -- Image description (used by Gradio frontend)
    POST /ingest   -- Runtime document ingestion into ChromaDB
    GET  /stats    -- Cache hits, total queries, avg latency
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from rag import chunker, embedder, retriever  # noqa: E402
from rag.prompt_builder import build_prompt, extract_source_attribution  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
RESULT_POLL_TIMEOUT = 180  # seconds

app = FastAPI(
    title="AvivaBot API",
    version="1.0.0",
    description="FastAPI sidecar for the Aviva Hybrid AI Bot",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_redis: Optional[aioredis.Redis] = None
_start_time = time.time()


async def get_redis() -> aioredis.Redis:
    """Lazily initialise the async Redis client."""
    global _redis
    if _redis is None:
        pool = aioredis.ConnectionPool.from_url(
            REDIS_URL, max_connections=20, decode_responses=True
        )
        _redis = aioredis.Redis(connection_pool=pool)
    return _redis


async def push_to_stream(stream: str, payload: Dict[str, Any]) -> str:
    r = await get_redis()
    return await r.xadd(stream, {k: str(v) for k, v in payload.items()})


async def poll_result(result_key: str, timeout: int = RESULT_POLL_TIMEOUT) -> Optional[dict]:
    r = await get_redis()
    result = await r.blpop(result_key, timeout=timeout)
    if result is None:
        return None
    _, raw = result
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str
    user_id: str = "api-user"


class AskResponse(BaseModel):
    answer: str
    sources: str
    total_ms: float


class VisionRequest(BaseModel):
    image_base64: str
    user_id: str = "api-user"


class VisionResponse(BaseModel):
    caption: str
    tags: List[str]
    total_ms: float


class IngestRequest(BaseModel):
    doc_text: str
    doc_name: str


class IngestResponse(BaseModel):
    chunks_added: int
    doc_name: str


class StatsResponse(BaseModel):
    cache_hits: int
    total_queries: int
    avg_latency_ms: float
    uptime_seconds: float
    chroma_doc_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Operations"])
async def health() -> Dict[str, Any]:
    """Service liveness check."""
    r = await get_redis()
    try:
        await r.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "error"

    return {
        "status": "ok",
        "redis": redis_status,
        "workers": ["rag-worker-1", "vision-worker-1"],
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(request: AskRequest) -> AskResponse:
    """
    Submit a RAG query and wait for the answer.

    Pushes the query to Redis ``text_stream``, polls the result key,
    and returns the formatted answer with source attribution.
    """
    result_key = f"result:rag:{uuid.uuid4().hex}"
    payload = {
        "user_id": request.user_id,
        "query": request.query,
        "result_key": result_key,
        "timestamp": int(time.time()),
    }

    await push_to_stream("text_stream", payload)
    logger.info("POST /ask | user=%s | query=%.80s", request.user_id, request.query)

    result = await poll_result(result_key, timeout=RESULT_POLL_TIMEOUT)
    if result is None:
        raise HTTPException(status_code=504, detail="Worker timed out. Please try again.")

    return AskResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", ""),
        total_ms=result.get("total_ms", 0.0),
    )


@app.post("/vision", response_model=VisionResponse, tags=["Vision"])
async def vision(request: VisionRequest) -> VisionResponse:
    """
    Submit an image for vision analysis.

    Pushes the base64 image to Redis ``image_stream`` and returns
    the caption and tags from the LLaVA model.
    """
    result_key = f"result:vision:{uuid.uuid4().hex}"
    payload = {
        "user_id": request.user_id,
        "image_b64": request.image_base64,
        "result_key": result_key,
        "timestamp": int(time.time()),
    }

    await push_to_stream("image_stream", payload)
    logger.info("POST /vision | user=%s", request.user_id)

    result = await poll_result(result_key, timeout=RESULT_POLL_TIMEOUT)
    if result is None:
        raise HTTPException(status_code=504, detail="Vision worker timed out. Please try again.")

    return VisionResponse(
        caption=result.get("caption", ""),
        tags=result.get("tags", []),
        total_ms=result.get("total_ms", 0.0),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Knowledge Base"])
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    Ingest a new document into ChromaDB at runtime.

    Chunks the provided text, embeds it, and upserts it into the
    persistent ChromaDB collection.
    """
    if not request.doc_text.strip():
        raise HTTPException(status_code=400, detail="doc_text cannot be empty.")

    chunks = chunker.chunk_text(
        text=request.doc_text,
        source=request.doc_name,
        chunk_size=300,
        overlap=50,
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from provided text.")

    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    metadatas = [{"source": c.source} for c in chunks]
    ids = [f"{request.doc_name}:chunk:{c.chunk_index}" for c in chunks]

    retriever.add_chunks(texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    logger.info("Ingested %d chunks from doc '%s'.", len(chunks), request.doc_name)

    return IngestResponse(chunks_added=len(chunks), doc_name=request.doc_name)


@app.get("/stats", response_model=StatsResponse, tags=["Operations"])
async def stats() -> StatsResponse:
    """Return system usage statistics."""
    r = await get_redis()

    cache_hits = int((await r.get("cache:hits")) or 0)
    total_queries = int((await r.get("stats:total_queries")) or 0)

    # Average latency from the last 1000 queries
    latencies_raw = await r.lrange("stats:latencies", 0, -1)
    if latencies_raw:
        latencies = [float(v) for v in latencies_raw]
        avg_latency_ms = sum(latencies) / len(latencies)
    else:
        avg_latency_ms = 0.0

    try:
        chroma_count = retriever.collection_count()
    except Exception:
        chroma_count = 0

    return StatsResponse(
        cache_hits=cache_hits,
        total_queries=total_queries,
        avg_latency_ms=round(avg_latency_ms, 1),
        uptime_seconds=round(time.time() - _start_time, 1),
        chroma_doc_count=chroma_count,
    )
