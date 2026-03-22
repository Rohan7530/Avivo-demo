"""Unit tests for the two-layer cache (cache/redis_cache.py)."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cache.redis_cache import RedisCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cache(redis_mock) -> RedisCache:
    """Create a RedisCache instance with a mocked Redis client."""
    cache = RedisCache(redis_url="redis://localhost:6379")
    cache._redis = redis_mock
    return cache


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

class TestRedisCache:
    @pytest.mark.asyncio
    async def test_l1_cache_hit(self):
        """L1 cache should return value without hitting Redis."""
        redis_mock = AsyncMock()
        cache = make_cache(redis_mock)

        # Pre-populate L1
        cache._l1["my_key"] = "l1_value"
        cache._l1.move_to_end("my_key")

        result = await cache.get("my_key")
        assert result == "l1_value"
        redis_mock.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_l2_cache_hit(self):
        """L2 Redis cache hit should populate L1 and return value."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=json.dumps("l2_value"))
        redis_mock.incr = AsyncMock()
        cache = make_cache(redis_mock)

        result = await cache.get("my_key")
        assert result == "l2_value"
        assert cache._l1.get("my_key") == "l2_value"

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self):
        """Cache miss should return None."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)
        cache = make_cache(redis_mock)

        result = await cache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_stores_in_l1_and_l2(self):
        """set() should populate L1 and call Redis setex."""
        redis_mock = AsyncMock()
        redis_mock.setex = AsyncMock()
        cache = make_cache(redis_mock)

        await cache.set("test_key", {"data": 42}, ttl=600)

        assert cache._l1.get("test_key") == {"data": 42}
        redis_mock.setex.assert_called_once()
        call_args = redis_mock.setex.call_args
        assert call_args[0][0] == "cache:test_key"
        assert call_args[0][1] == 600

    @pytest.mark.asyncio
    async def test_invalidate_removes_from_l1_and_l2(self):
        """invalidate() should remove from both L1 and Redis."""
        redis_mock = AsyncMock()
        redis_mock.delete = AsyncMock()
        cache = make_cache(redis_mock)

        cache._l1["del_key"] = "some_value"
        await cache.invalidate("del_key")

        assert "del_key" not in cache._l1
        redis_mock.delete.assert_called_once_with("cache:del_key")

    def test_l1_lru_eviction(self):
        """L1 should evict the LRU entry when maxsize exceeded."""
        from cache.redis_cache import _L1_MAXSIZE
        cache = RedisCache.__new__(RedisCache)
        from collections import OrderedDict
        cache._l1 = OrderedDict()
        cache._redis = None

        # Fill L1 to capacity
        for i in range(_L1_MAXSIZE):
            cache._l1_set(f"key_{i}", i)

        assert len(cache._l1) == _L1_MAXSIZE

        # Add one more — should evict LRU (key_0)
        cache._l1_set("new_key", "new_value")
        assert "key_0" not in cache._l1
        assert "new_key" in cache._l1
        assert len(cache._l1) == _L1_MAXSIZE

    @pytest.mark.asyncio
    async def test_redis_error_is_handled_gracefully(self):
        """Redis errors should not raise exceptions to the caller."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        cache = make_cache(redis_mock)

        result = await cache.get("some_key")
        assert result is None  # should return None gracefully
