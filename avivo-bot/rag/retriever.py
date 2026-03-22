"""Retriever: ChromaDB-based top-k semantic search.

The ChromaDB client and collection are initialised once at module level.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton ChromaDB client & collection
# ---------------------------------------------------------------------------
_chroma_client = None
_collection = None
COLLECTION_NAME = "aviva_knowledge_base"


def _get_collection():
    """Return the ChromaDB collection, initialising it if needed."""
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore

    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    logger.info("Initialising ChromaDB at '%s'.", db_path)

    _chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "ChromaDB collection '%s' ready. Documents: %d",
        COLLECTION_NAME,
        _collection.count(),
    )
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_chunks(
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
) -> None:
    """
    Upsert chunks into the ChromaDB collection.

    Args:
        texts: Raw text of each chunk.
        embeddings: Pre-computed embedding vectors.
        metadatas: Metadata dicts (e.g. {'source': 'company_policies.md'}).
        ids: Unique string IDs for each chunk.
    """
    collection = _get_collection()
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    logger.debug("Upserted %d chunks into ChromaDB.", len(texts))


def retrieve(
    query_embedding: List[float],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Return the top-k most similar chunks for *query_embedding*.

    Args:
        query_embedding: Query vector from the embedder.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys ``text``, ``source``, ``distance``.
    """
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty -- no results returned.")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, distances):
        output.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "distance": dist,
            }
        )

    logger.debug("Retrieved %d chunks for query.", len(output))
    return output


def collection_count() -> int:
    """Return the total number of documents in the collection."""
    return _get_collection().count()
