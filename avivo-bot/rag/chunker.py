"""Document chunker: splits text into overlapping fixed-size chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """Represents a single text chunk with metadata."""

    text: str
    source: str
    chunk_index: int


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Split *text* into overlapping chunks of *chunk_size* characters.

    Args:
        text: Raw document text to split.
        source: Document name / filename for metadata.
        chunk_size: Maximum characters per chunk (default 300).
        overlap: Number of characters shared between consecutive chunks (default 50).

    Returns:
        List of :class:`Chunk` objects.
    """
    if not text or not text.strip():
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    text = text.strip()

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunks.append(
                Chunk(
                    text=chunk_text_content,
                    source=source,
                    chunk_index=idx,
                )
            )
            idx += 1

        if end == len(text):
            break

        # Slide forward by (chunk_size - overlap) to create the overlap
        start += chunk_size - overlap

    return chunks


def chunk_document(filepath: str, chunk_size: int = 300, overlap: int = 50) -> List[Chunk]:
    """
    Read a file from *filepath* and return its chunks.

    Args:
        filepath: Absolute or relative path to the document.
        chunk_size: Characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of :class:`Chunk` objects.
    """
    import os

    source = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as fh:
        text = fh.read()

    return chunk_text(text, source=source, chunk_size=chunk_size, overlap=overlap)
