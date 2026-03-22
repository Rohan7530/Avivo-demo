"""Unit tests for the RAG pipeline (chunker, embedder, retriever, prompt_builder)."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch

from rag.chunker import chunk_text, chunk_document, Chunk
from rag.prompt_builder import build_prompt, extract_source_attribution


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    def test_basic_chunking(self):
        text = "A" * 600
        chunks = chunk_text(text, source="test.md", chunk_size=300, overlap=50)
        assert len(chunks) >= 2
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.source == "test.md"

    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = chunk_text(text, source="short.md", chunk_size=300, overlap=50)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("", source="empty.md")
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        chunks = chunk_text("   \n\t  ", source="ws.md")
        assert chunks == []

    def test_chunk_overlap(self):
        text = "A" * 400
        chunks = chunk_text(text, source="test.md", chunk_size=300, overlap=100)
        # Second chunk should start at offset 200 (300-100)
        assert len(chunks) == 2
        # Each chunk should share 100 chars of overlap
        assert len(chunks[0].text) == 300
        assert len(chunks[1].text) == 200  # remaining

    def test_chunk_indices_are_sequential(self):
        text = "X" * 1000
        chunks = chunk_text(text, source="big.md", chunk_size=200, overlap=20)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_chunk_document_reads_file(self, tmp_path):
        doc = tmp_path / "test_doc.md"
        doc.write_text("Hello from file! " * 30)
        chunks = chunk_document(str(doc))
        assert len(chunks) >= 1
        assert chunks[0].source == "test_doc.md"


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_build_prompt_contains_query(self):
        prompt = build_prompt(
            query="What is the leave policy?",
            context_chunks=[],
        )
        assert "What is the leave policy?" in prompt

    def test_build_prompt_contains_system(self):
        prompt = build_prompt(query="test", context_chunks=[])
        assert "AvivaBot" in prompt
        assert "SYSTEM" in prompt

    def test_build_prompt_with_context(self):
        chunks = [{"text": "You get 18 days of leave.", "source": "policies.md"}]
        prompt = build_prompt(query="leave?", context_chunks=chunks)
        assert "You get 18 days of leave." in prompt
        assert "policies.md" in prompt

    def test_build_prompt_with_history(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        prompt = build_prompt(query="How are you?", context_chunks=[], history=history)
        assert "Hello" in prompt
        assert "Hi there!" in prompt

    def test_history_limited_to_6_entries(self):
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        prompt = build_prompt(query="q", context_chunks=[], history=history)
        # Should only contain last 6 messages
        assert "msg 14" in prompt
        assert "msg 0" not in prompt

    def test_extract_source_attribution_deduplicates(self):
        chunks = [
            {"source": "policy.md"},
            {"source": "policy.md"},
            {"source": "faq.md"},
        ]
        result = extract_source_attribution(chunks)
        assert result == "policy.md, faq.md"

    def test_extract_source_attribution_empty(self):
        result = extract_source_attribution([])
        assert result == ""


# ---------------------------------------------------------------------------
# Embedder tests (mocked)
# ---------------------------------------------------------------------------

class TestEmbedder:
    @patch("rag.embedder._get_model")
    def test_embed_returns_vectors(self, mock_get_model):
        mock_model = MagicMock()
        fake_array = MagicMock()
        fake_array.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_model.encode.return_value = fake_array
        mock_get_model.return_value = mock_model

        from rag.embedder import embed
        result = embed(["test text"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)

    @patch("rag.embedder._get_model")
    def test_embed_empty_list_returns_empty(self, mock_get_model):
        from rag.embedder import embed
        result = embed([])
        assert result == []


# ---------------------------------------------------------------------------
# Retriever tests (mocked ChromaDB)
# ---------------------------------------------------------------------------

class TestRetriever:
    @patch("rag.retriever._get_collection")
    def test_retrieve_returns_structured_results(self, mock_get_collection):
        mock_col = MagicMock()
        mock_col.count.return_value = 5
        mock_col.query.return_value = {
            "documents": [["chunk text 1", "chunk text 2"]],
            "metadatas": [[{"source": "doc1.md"}, {"source": "doc2.md"}]],
            "distances": [[0.1, 0.2]],
        }
        mock_get_collection.return_value = mock_col

        from rag.retriever import retrieve
        results = retrieve(query_embedding=[0.1] * 384, top_k=2)

        assert len(results) == 2
        assert results[0]["text"] == "chunk text 1"
        assert results[0]["source"] == "doc1.md"
        assert results[0]["distance"] == 0.1

    @patch("rag.retriever._get_collection")
    def test_retrieve_empty_collection(self, mock_get_collection):
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_get_collection.return_value = mock_col

        from rag.retriever import retrieve
        results = retrieve(query_embedding=[0.1] * 384)
        assert results == []
