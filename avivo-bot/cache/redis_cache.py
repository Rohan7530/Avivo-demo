"""Two-layer cache: L1 in-memory LRU dict + L2 Redis hash with TTL.

Usage::

    cache = RedisCache(redis_url="redis://localhost:6379")
    await cache.set("my_key", "my_value", ttl=3600)
    value = await cache.get("my_key")   # hits L1 first, then L2
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from typing import Any, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_L1_MAXSIZE = 100


class RedisCache:
    """
    Two-layer cache.

    - L1: Python OrderedDict (in-process LRU, maxsize=100).
    - L2: Redis HSET/GET with TTL via EXPIRE.

    Metrics: cache hits increment the ``cache:hits`` Redis counter.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis: Optional[aioredis.Redis] = None
        self._l1: OrderedDict[str, Any] = OrderedDict()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _get_redis(self) -> aioredis.Redis:
        """Return the async Redis client, creating it lazily."""
        if self._redis is None:
            pool = aioredis.ConnectionPool.from_url(
                self._url,
                max_connections=20,
                decode_responses=True,
            )
            self._redis = aioredis.Redis(connection_pool=pool)
        return self._redis

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    # ------------------------------------------------------------------
    # L1 helpers
    # ------------------------------------------------------------------

    def _l1_get(self, key: str) -> Any:
        """Return value from L1, None if missing. Moves hit to end (LRU)."""
        if key in self._l1:
            self._l1.move_to_end(key)
            return self._l1[key]
        return None

    def _l1_set(self, key: str, value: Any) -> None:
        """Insert value into L1, evicting LRU entry if over capacity."""
        if key in self._l1:
            self._l1.move_to_end(key)
        self._l1[key] = value
        if len(self._l1) > _L1_MAXSIZE:
            self._l1.popitem(last=False)  # evict least-recently-used

    def _l1_delete(self, key: str) -> None:
        self._l1.pop(key, None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value, checking L1 then L2.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None on miss.
        """
        # L1 check
        value = self._l1_get(key)
        if value is not None:
            logger.debug("L1 cache hit for key '%s'.", key)
            try:
                r = await self._get_redis()
                await r.incr("cache:hits")
            except Exception:
                pass
            return value

        # L2 check
        try:
            r = await self._get_redis()
            raw = await r.get(f"cache:{key}")
            if raw is not None:
                logger.debug("L2 cache hit for key '%s'.", key)
                value = json.loads(raw)
                self._l1_set(key, value)
                await r.incr("cache:hits")
                return value
        except Exception as exc:
            logger.warning("Redis cache GET error: %s", exc)

        return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Store a value in both L1 and L2.

        Args:
            key: Cache key.
            value: JSON-serialisable value.
            ttl: Time-to-live in seconds for L2 (default 3600).
        """
        self._l1_set(key, value)
        try:
            r = await self._get_redis()
            serialised = json.dumps(value)
            await r.setex(f"cache:{key}", ttl, serialised)
            logger.debug("Cached key '%s' with TTL=%ds.", key, ttl)
        except Exception as exc:
            logger.warning("Redis cache SET error: %s", exc)

    async def invalidate(self, key: str) -> None:
        """
        Remove a key from both L1 and L2.

        Args:
            key: Cache key to delete.
        """
        self._l1_delete(key)
        try:
            r = await self._get_redis()
            await r.delete(f"cache:{key}")
            logger.debug("Invalidated cache key '%s'.", key)
        except Exception as exc:
            logger.warning("Redis cache DELETE error: %s", exc)

    async def get_hit_count(self) -> int:
        """Return the total number of cache hits recorded in Redis."""
        try:
            r = await self._get_redis()
            val = await r.get("cache:hits")
            return int(val) if val else 0
        except Exception:
            return 0
