"""One-time script: chunk, embed, and store all docs/ Markdown files in ChromaDB.

Run with:
    python scripts/ingest_docs.py

Or from the Docker Compose setup:
    docker-compose run --rm api python scripts/ingest_docs.py
"""

from __future__ import annotations

import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from rag import chunker, embedder, retriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("ingest")

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")


def ingest_all() -> None:
    """Read all .md files from docs/, chunk, embed, and upsert into ChromaDB."""
    if not os.path.isdir(DOCS_DIR):
        logger.error("docs/ directory not found at '%s'. Run from project root.", DOCS_DIR)
        sys.exit(1)

    md_files = [
        os.path.join(DOCS_DIR, f)
        for f in sorted(os.listdir(DOCS_DIR))
        if f.endswith(".md")
    ]

    if not md_files:
        logger.warning("No .md files found in '%s'.", DOCS_DIR)
        return

    logger.info("Found %d document(s): %s", len(md_files), [os.path.basename(f) for f in md_files])

    total_chunks = 0

    for filepath in md_files:
        logger.info("Processing '%s'...", os.path.basename(filepath))

        chunks = chunker.chunk_document(filepath, chunk_size=300, overlap=50)
        if not chunks:
            logger.warning("No chunks generated from '%s'. Skipping.", filepath)
            continue

        texts = [c.text for c in chunks]
        metadatas = [{"source": c.source} for c in chunks]
        ids = [f"{c.source}:chunk:{c.chunk_index}" for c in chunks]

        logger.info("Embedding %d chunks from '%s'...", len(chunks), os.path.basename(filepath))
        embeddings = embedder.embed(texts)

        retriever.add_chunks(texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        logger.info("  Upserted %d chunks from '%s'.", len(chunks), os.path.basename(filepath))
        total_chunks += len(chunks)

    logger.info(
        "Ingestion complete. %d total chunks across %d documents stored in ChromaDB.",
        total_chunks,
        len(md_files),
    )
    logger.info("ChromaDB collection now has %d documents.", retriever.collection_count())


if __name__ == "__main__":
    ingest_all()
