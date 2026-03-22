# AvivaBot — Hybrid AI Telegram Bot

> A production-grade, fully async AI assistant combining RAG + Computer Vision,
> event-driven via Redis Streams, with a beautiful Gradio web frontend.

---

## 1. Project Overview & Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USERS                                    │
│         Telegram App          Gradio Web UI (public URL)        │
└────────────┬──────────────────────────┬────────────────────────┘
             │                          │ HTTP
             ▼                          ▼
  ┌─────────────────┐        ┌──────────────────────┐
  │ telegram_bot.py │        │  FastAPI /ask         │
  │  (python-tg v20)│        │  FastAPI /vision      │
  └────────┬────────┘        │  FastAPI /ingest      │
           │ XADD            │  FastAPI /stats       │
           ▼                 └──────────┬────────────┘
  ┌──────────────────────────────────────┘
  │             REDIS STREAMS
  │   text_stream ────► rag-worker (consumer group)
  │   image_stream ───► vision-worker (consumer group)
  └──────────────────────────────────────────────────
           │                            │
           ▼                            ▼
  ┌────────────────┐          ┌─────────────────────┐
  │  rag_worker.py │          │ vision_worker.py     │
  │                │          │                      │
  │ 1. L1/L2 cache │          │ 1. Decode base64     │
  │ 2. Embed query │          │ 2. Call LLaVA/Ollama │
  │ 3. ChromaDB    │          │ 3. Parse caption+tags│
  │ 4. Build prompt│          │ 4. LPUSH result key  │
  │ 5. phi3:mini   │          │ 5. XACK message      │
  │ 6. XACK        │          └─────────────────────-┘
  └────────┬───────┘
           │
           ▼
  ┌────────────────────────────────────────┐
  │              OLLAMA                    │
  │  phi3:mini (3.8B) — RAG answers        │
  │  llava — Image captioning              │
  └────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────┐
  │           ChromaDB (persistent)        │
  │  all-MiniLM-L6-v2 embeddings           │
  │  cosine similarity, top-k retrieval    │
  └────────────────────────────────────────┘
```

---

## 2. Tech Stack

| Component          | Technology                     | Justification                                      |
|--------------------|--------------------------------|----------------------------------------------------|
| Telegram Bot       | python-telegram-bot v20+       | Fully async, official SDK, handler-based           |
| Message Queue      | Redis Streams                  | At-least-once delivery, consumer groups, XACK      |
| Vector Store       | ChromaDB                       | Lightweight, persistent, cosine search, no server  |
| Embeddings         | all-MiniLM-L6-v2               | 384-dim, 80MB, fast CPU inference, high quality    |
| RAG LLM            | phi3:mini (Ollama)             | 3.8B params, fast, instruction-tuned, CPU-capable  |
| Vision LLM         | LLaVA (Ollama)                 | Best open-source multimodal model, base64 input    |
| FastAPI Sidecar    | FastAPI + uvicorn              | Async, typed, auto-docs, CORS, minimal overhead    |
| Web Frontend       | Gradio 4.x                     | share=True gives instant public URL, rich widgets  |
| Cache              | Redis + Python OrderedDict     | L1 LRU (in-process) + L2 Redis (distributed)       |
| Session History    | Redis Lists                    | LPUSH+LTRIM for rolling window, 24hr TTL           |
| Container Orchestration | Docker Compose            | Single-command deployment, named volumes, networks |

---

## 3. Quick Start

### Prerequisites
- Docker & Docker Compose installed
- A Telegram Bot token from [@BotFather](https://t.me/BotFather)

### Setup

```bash
# 1. Clone / enter the project directory
cd avivo-bot

# 2. Copy and configure environment
cp .env.example .env
# Edit .env and set your TELEGRAM_BOT_TOKEN

# 3. Build and start all services
docker-compose up --build

# 4. In a separate terminal, ingest knowledge base documents
docker-compose exec api python scripts/ingest_docs.py

# 5. Check all services are running
docker-compose ps
```

Services will start in order:
1. `redis` → `ollama` → `ollama-init` (pulls phi3:mini + llava)
2. `rag-worker`, `vision-worker`, `api`, `bot`, `frontend`

> **Note:** The first startup may take 5–10 minutes for Ollama to pull models (~4GB).

---

## 4. Telegram Bot Setup

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts to get your token
3. Copy the token into your `.env` file:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   ```
4. Start a conversation with your bot and send `/help`

**Available Commands:**

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/help` | Show usage guide |
| `/ask <question>` | Ask anything from the knowledge base |
| `/image` | Upload an image for AI description |
| `/summarize` | Show your last 3 conversation turns |

---

## 5. Gradio Frontend — Public Share URL

The Gradio frontend automatically generates a public `gradio.live` URL at startup.

To find it, check the frontend container logs:
```bash
docker-compose logs frontend
```

Look for a line like:
```
Running on public URL: https://abc123.gradio.live
```

Share this URL with anyone — no Telegram required!

The web UI offers three tabs:
- **Ask Anything** — RAG query with source accordion
- **Describe Image** — Image upload with caption + tag badges
- **System Stats** — Live metrics refreshed every 10 seconds

---

## 6. Adding New Documents to the Knowledge Base

### Option A: File-based (recommended for bulk ingestion)
```bash
# Copy your .md file into docs/
cp my_new_doc.md avivo-bot/docs/

# Re-run ingestion
docker-compose exec api python scripts/ingest_docs.py
```

### Option B: Runtime API (for single documents)
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "doc_name": "my_new_policy.md",
    "doc_text": "Your full document content here..."
  }'
```

Response: `{"chunks_added": 12, "doc_name": "my_new_policy.md"}`

---

## 7. System Design Decisions

### Why Redis Streams (not pub/sub)?

| Feature | Redis Pub/Sub | Redis Streams |
|---------|--------------|---------------|
| Persistence | No — messages lost if consumer offline | Yes — messages survive restarts |
| Consumer groups | No | Yes — multiple workers, load balanced |
| Acknowledgment | No | Yes — XACK ensures at-least-once delivery |
| Replay | No | Yes — re-read from any offset |

Streams give us **exactly the guarantees needed** for a production message queue.

### Why ChromaDB?

- **Zero operational overhead:** Runs embedded within the worker process (no separate server)
- **Persistent:** Data survives restarts via a Docker volume mount
- **Native cosine similarity:** Perfect for semantic search
- **Fast:** Sub-10ms retrieval for small to medium collections

### Why phi3:mini?

- **3.8B parameters** — fits comfortably in 8GB RAM, runs on CPU
- **Instruction-tuned** — follows prompts reliably
- **Fast:** ~3–8s per response on modern CPU hardware
- **Fallback:** Replace `LLM_MODEL=mistral` in `.env` to use Mistral 7B

---

## 8. Caching Strategy

```
Query arrives
     │
     ▼
L1 Cache (in-memory OrderedDict, maxsize=100, LRU eviction)
     │ miss
     ▼
L2 Cache (Redis key: cache:{md5(query)}, TTL=3600s)
     │ miss
     ▼
Full RAG Pipeline (embed → retrieve → LLM → answer)
     │
     ├─► Store in L2 (Redis, TTL=1hr)
     └─► Promote to L1 (in-memory, instant future hits)
```

**Cache key:** `md5(query.lower().strip())` — normalized for robustness

**Benefits:**
- L1 latency: ~0ms (dict lookup)
- L2 latency: ~1ms (Redis GET)
- Full pipeline: 5–15 seconds
- Metric: `cache:hits` counter in Redis, exposed at `/stats`

---

## 9. Demo Screenshots

_Screenshots and recordings can be added here after deployment._

| Component | Description |
|-----------|-------------|
| Telegram Bot | `/ask` response with source attribution |
| Gradio Ask Tab | Question answered with context accordion |
| Gradio Vision Tab | Image upload → caption + keyword badges |
| Gradio Stats Tab | Live cache hit rate and query count |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health + Redis status |
| `POST` | `/ask` | Submit RAG query |
| `POST` | `/vision` | Submit image for description |
| `POST` | `/ingest` | Add document to knowledge base |
| `GET` | `/stats` | System metrics |

Interactive API docs available at: `http://localhost:8000/docs`

---

## License

MIT License — see [LICENSE](LICENSE) for details.
