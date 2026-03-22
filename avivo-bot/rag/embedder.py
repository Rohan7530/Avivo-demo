"""Embedder: wraps sentence-transformers for text embedding.

The model is loaded once at module level to avoid repeated cold-starts.
"""

from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load the model once per process
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    """Return the singleton SentenceTransformer model, loading it if needed."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        logger.info("Loading embedding model '%s' — first call only.", model_name)
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(texts: List[str]) -> List[List[float]]:
    """
    Encode a list of strings into dense float vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_single(text: str) -> List[float]:
    """
    Embed a single string.

    Args:
        text: Input string.

    Returns:
        Embedding vector as a list of floats.
    """
    results = embed([text])
    return results[0] if results else []
