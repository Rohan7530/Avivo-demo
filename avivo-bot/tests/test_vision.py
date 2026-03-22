"""Unit tests for the vision captioner (vision/captioner.py)."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, patch

from vision.captioner import _parse_vision_response, describe_image


# ---------------------------------------------------------------------------
# Parser tests (synchronous, no mocking needed)
# ---------------------------------------------------------------------------

class TestVisionParser:
    def test_parses_caption_and_tags(self):
        raw = (
            "A golden retriever sits on grass next to a red ball. "
            "The dog looks happy and playful.\n"
            "Tags: dog, outdoor, happy"
        )
        result = _parse_vision_response(raw)
        assert "golden retriever" in result["caption"]
        assert len(result["tags"]) == 3
        assert "dog" in result["tags"]

    def test_tags_lowercased(self):
        raw = "An image of a cat.\nTags: Fluffy, Indoor, White"
        result = _parse_vision_response(raw)
        assert all(t == t.lower() for t in result["tags"])

    def test_missing_tags_filled_with_untagged(self):
        raw = "A plain white background."
        result = _parse_vision_response(raw)
        assert len(result["tags"]) == 3
        assert "untagged" in result["tags"]

    def test_tags_truncated_to_3(self):
        raw = "A busy market scene.\nTags: market, people, fruits, vegetables, colorful, busy"
        result = _parse_vision_response(raw)
        assert len(result["tags"]) == 3

    def test_semicolon_separated_tags(self):
        raw = "Dark sky at night.\nTags: night; stars; moon"
        result = _parse_vision_response(raw)
        assert len(result["tags"]) == 3


# ---------------------------------------------------------------------------
# describe_image tests (async, mocked Ollama call)
# ---------------------------------------------------------------------------

class TestDescribeImage:
    @pytest.mark.asyncio
    @patch("vision.captioner._call_ollama_vision", new_callable=AsyncMock)
    async def test_describe_image_with_b64(self, mock_ollama):
        mock_ollama.return_value = (
            "A serene mountain landscape with snow-capped peaks. "
            "The sky is clear and blue.\nTags: mountain, snow, landscape"
        )

        result = await describe_image(
            image_b64="dGVzdA==",  # "test" base64
            ollama_url="http://localhost:11434",
            model="llava",
        )

        assert "caption" in result
        assert "tags" in result
        assert len(result["tags"]) == 3
        mock_ollama.assert_called_once()

    @pytest.mark.asyncio
    @patch("vision.captioner._call_ollama_vision", new_callable=AsyncMock)
    async def test_describe_image_with_bytes(self, mock_ollama):
        mock_ollama.return_value = "A red car on a road.\nTags: car, road, red"

        result = await describe_image(
            image_bytes=b"fake_image_bytes",
            ollama_url="http://localhost:11434",
            model="llava",
        )

        assert result["caption"] != ""
        assert len(result["tags"]) == 3

    @pytest.mark.asyncio
    async def test_describe_image_no_input_raises(self):
        with pytest.raises(ValueError):
            await describe_image()  # no image_bytes or image_b64
