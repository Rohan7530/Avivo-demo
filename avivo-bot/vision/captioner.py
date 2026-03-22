"""Vision captioner using Salesforce BLIP — runs locally on CPU, no Ollama needed.

Replaces the LLaVA/Ollama-based pipeline with a fast local inference model:
  - Model: Salesforce/blip-image-captioning-base (~950 MB, one-time download)
  - CPU inference: ~1–3 seconds per image
  - Tags: extracted from caption using simple keyword filtering

Usage:
    result = await describe_image(image_bytes=<bytes>)
    # or
    result = await describe_image(image_b64=<base64_string>)

Returns:
    {"caption": str, "tags": [str, str, str]}
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton model loader (loaded once per process)
# ---------------------------------------------------------------------------

_blip_processor = None
_blip_model = None

def _get_blip() -> Tuple[Any, Any]:
    """Load BLIP processor and model (once per process)."""
    global _blip_processor, _blip_model
    if _blip_processor is None:
        logger.info("Loading BLIP captioning model (first call only)...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch

        model_name = "Salesforce/blip-image-captioning-base"
        _blip_processor = BlipProcessor.from_pretrained(model_name)
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU-safe
        )
        _blip_model.eval()
        logger.info("BLIP model loaded successfully.")
    return _blip_processor, _blip_model


# ---------------------------------------------------------------------------
# Tag extraction from caption
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "on", "at", "by",
    "for", "with", "about", "against", "between", "through", "during",
    "this", "that", "these", "those", "it", "its", "and", "or", "but",
    "very", "so", "just", "as", "into", "from", "there", "their", "they",
    "image", "photo", "picture", "shows", "shown", "display", "displays",
    "features", "depicts", "set", "showing",
}


def _extract_tags(caption: str, n: int = 3) -> List[str]:
    """Extract top-N meaningful keywords from a caption string."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", caption.lower())
    seen = set()
    tags = []
    for w in words:
        if w not in _STOPWORDS and w not in seen:
            seen.add(w)
            tags.append(w)
        if len(tags) == n:
            break
    while len(tags) < n:
        tags.append("untagged")
    return tags


# ---------------------------------------------------------------------------
# Core caption function (sync, run in thread pool)
# ---------------------------------------------------------------------------

def _caption_sync(image_bytes: bytes) -> Dict[str, Any]:
    """Run BLIP captioning synchronously (called via asyncio executor)."""
    from PIL import Image
    import torch

    processor, model = _get_blip()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=4,
        )
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    tags = _extract_tags(caption)

    return {"caption": caption, "tags": tags}


# ---------------------------------------------------------------------------
# Public async API (same interface as the old Ollama captioner)
# ---------------------------------------------------------------------------

def _parse_vision_response(raw: str) -> Dict[str, Any]:
    """Kept for test compatibility — parses a caption string into result dict."""
    tags = _extract_tags(raw)
    return {"caption": raw.strip(), "tags": tags}


async def describe_image(
    image_bytes: Optional[bytes] = None,
    image_b64: Optional[str] = None,
    ollama_url: Optional[str] = None,   # ignored — kept for API compatibility
    model: Optional[str] = None,        # ignored — kept for API compatibility
) -> Dict[str, Any]:
    """
    Generate a caption and 3 keyword tags for the given image using BLIP.

    Runs BLIP inference in a thread pool so the async event loop is not blocked.

    Args:
        image_bytes: Raw image bytes.
        image_b64:   Pre-encoded base64 image string.
        ollama_url:  Ignored (kept for backwards compatibility).
        model:       Ignored (kept for backwards compatibility).

    Returns:
        Dict with ``caption`` (str) and ``tags`` (list[str]).
    """
    if image_bytes is None and image_b64 is None:
        raise ValueError("Either image_bytes or image_b64 must be provided.")

    if image_bytes is None:
        image_bytes = base64.b64decode(image_b64)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        functools.partial(_caption_sync, image_bytes),
    )

    logger.info(
        "BLIP vision result — caption: %.80s | tags: %s",
        result["caption"],
        result["tags"],
    )
    return result
