"""Gradio frontend for Aviv0Bot.

Three tabs:
  1. Ask Anything  -- RAG pipeline via FastAPI /ask
  2. Describe Image -- Vision pipeline via FastAPI /vision
  3. System Stats   -- Auto-refreshing stats from FastAPI /stats

Launched with share=True to generate a public gradio.live URL.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, Tuple

import httpx
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend")

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180.0


# ---------------------------------------------------------------------------
# API helper functions
# ---------------------------------------------------------------------------

def _post(endpoint: str, payload: Dict[str, Any], timeout: float = REQUEST_TIMEOUT) -> Dict:
    """Synchronous HTTP POST to the FastAPI sidecar."""
    url = f"{FASTAPI_URL}{endpoint}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        return {"error": f"API error {exc.response.status_code}: {exc.response.text}"}
    except Exception as exc:
        return {"error": str(exc)}


def _get(endpoint: str, timeout: float = 10.0) -> Dict:
    """Synchronous HTTP GET to the FastAPI sidecar."""
    url = f"{FASTAPI_URL}{endpoint}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tab 1: Ask Anything (RAG)
# ---------------------------------------------------------------------------

def ask_question(query: str, user_id: str = "gradio-user") -> Tuple[str, str]:
    """
    Send a RAG query to the FastAPI /ask endpoint.

    Returns:
        (answer_markdown, sources_string)
    """
    if not query.strip():
        return "Please enter a question.", ""

    result = _post("/ask", {"query": query, "user_id": user_id})

    if "error" in result:
        return f"Error: {result['error']}", ""

    answer = result.get("answer", "No answer returned.")
    sources = result.get("sources", "")
    total_ms = result.get("total_ms", 0)

    sources_display = f"Source: {sources}" if sources else "No source available."
    answer_display = f"{answer}\n\n*({total_ms:.0f} ms)*"

    return answer_display, sources_display


# ---------------------------------------------------------------------------
# Tab 2: Describe Image (Vision)
# ---------------------------------------------------------------------------

def describe_image(image_path: str, user_id: str = "gradio-user") -> Tuple[str, Dict]:
    """
    Send an image to the FastAPI /vision endpoint.

    Returns:
        (caption_string, tags_label_dict)
    """
    if image_path is None:
        return "Please upload an image.", {}

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as exc:
        return f"Could not read image: {exc}", {}

    result = _post("/vision", {"image_base64": image_b64, "user_id": user_id})

    if "error" in result:
        return f"Error: {result['error']}", {}

    caption = result.get("caption", "No caption generated.")
    tags = result.get("tags", [])
    total_ms = result.get("total_ms", 0)

    # Build label dict for gr.Label (tag → confidence)
    tag_labels = {tag: 1.0 / len(tags) for tag in tags} if tags else {}

    caption_display = f"{caption}\n\n*({total_ms:.0f} ms)*"
    return caption_display, tag_labels


# ---------------------------------------------------------------------------
# Tab 3: System Stats
# ---------------------------------------------------------------------------

def fetch_stats() -> str:
    """Fetch and format stats from the FastAPI /stats endpoint."""
    data = _get("/stats")

    if "error" in data:
        return f"Could not fetch stats: {data['error']}"

    cache_hits = data.get("cache_hits", 0)
    total_queries = data.get("total_queries", 0)
    avg_latency = data.get("avg_latency_ms", 0.0)
    uptime = data.get("uptime_seconds", 0.0)
    chroma_count = data.get("chroma_doc_count", 0)

    cache_rate = f"{(cache_hits / max(total_queries, 1)) * 100:.1f}%"
    uptime_fmt = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"

    return (
        f"**Cache Hit Rate:** {cache_rate} ({cache_hits} hits / {total_queries} queries)\n\n"
        f"**Avg Response Latency:** {avg_latency:.0f} ms\n\n"
        f"**ChromaDB Documents:** {chroma_count} chunks\n\n"
        f"**API Uptime:** {uptime_fmt}"
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    ["What is the work from home policy?"],
    ["How many days of leave do I get per year?"],
    ["How does the RAG pipeline work?"],
    ["What are the ingredients for Chicken Tikka Masala?"],
    ["What should I do on my first day at Aviv0 Tech?"],
]

CSS = """
.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.header-banner h1 {
    color: #e94560;
    font-size: 2.4em;
    margin: 0;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.header-banner p {
    color: #a0aec0;
    margin: 8px 0 0;
    font-size: 1.05em;
}
footer { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.gray,
    ),
    css=CSS,
    title="Aviv0Bot — Hybrid AI Assistant",
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-banner">
        <h1>Aviv0Bot</h1>
        <p>Hybrid AI Assistant &mdash; RAG + Vision, powered by phi3:mini &amp; LLaVA</p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Ask Anything ─────────────────────────────────────────────
        with gr.TabItem("Ask Anything"):
            gr.Markdown(
                "### Ask a question from the knowledge base\n"
                "I'll retrieve the most relevant context and generate an accurate answer."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    ask_query = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g. What is the WFH policy?",
                        lines=3,
                    )
                    ask_btn = gr.Button("Ask", variant="primary", size="lg")
                with gr.Column(scale=2):
                    ask_answer = gr.Markdown(label="Answer")
                    with gr.Accordion("Source Documents", open=False):
                        ask_sources = gr.Markdown(label="Sources")

            gr.Examples(
                examples=EXAMPLE_QUERIES,
                inputs=ask_query,
                label="Example Questions",
            )
            ask_btn.click(
                fn=ask_question,
                inputs=[ask_query],
                outputs=[ask_answer, ask_sources],
            )

        # ── Tab 2: Describe Image ───────────────────────────────────────────
        with gr.TabItem("Describe Image"):
            gr.Markdown(
                "### Upload an image for AI Vision analysis\n"
                "Aviv0Bot will generate a caption and 3 keyword tags using LLaVA."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    image_input = gr.Image(
                        type="filepath",
                        label="Upload Image",
                        height=300,
                    )
                    vision_btn = gr.Button("Describe", variant="primary", size="lg")
                with gr.Column(scale=3):
                    vision_caption = gr.Markdown(label="Caption")
                    vision_tags = gr.Label(label="Keyword Tags", num_top_classes=3)

            vision_btn.click(
                fn=describe_image,
                inputs=[image_input],
                outputs=[vision_caption, vision_tags],
            )

        # ── Tab 3: System Stats ─────────────────────────────────────────────
        with gr.TabItem("System Stats"):
            gr.Markdown(
                "### Real-time System Statistics\n"
                "Auto-refreshes every 10 seconds."
            )
            stats_display = gr.Markdown(value=fetch_stats())
            refresh_btn = gr.Button("Refresh Now", variant="secondary")

            # Auto-refresh via timer
            demo.load(
                fn=fetch_stats,
                inputs=None,
                outputs=stats_display,
                every=10,
            )
            refresh_btn.click(fn=fetch_stats, inputs=None, outputs=stats_display)


if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
    )
