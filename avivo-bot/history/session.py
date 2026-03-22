"""Per-user session history stored in Redis Lists.

Structure:
    key   = "history:{user_id}"
    value = JSON-encoded list of {"role": "user|assistant", "content": "..."}
    max   = 6 entries (3 turns: 3 user + 3 assistant)
    TTL   = 86400 seconds (24 hours)

Usage::

    session = SessionHistory(redis_url="redis://localhost:6379")
    await session.add(user_id="123", role="user", content="Hello!")
    history = await session.get(user_id="123")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 6   # 3 turns = 3 user + 3 assistant messages
_TTL_SECONDS = 86400  # 24 hours


class SessionHistory:
    """Manages per-user conversation history in a Redis List."""

    def __init__(self, redis_url: str | None = None) -> None:
        self._url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        """Lazy-init the async Redis client."""
        if self._redis is None:
            pool = aioredis.ConnectionPool.from_url(
                self._url,
                max_connections=20,
                decode_responses=True,
            )
            self._redis = aioredis.Redis(connection_pool=pool)
        return self._redis

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    @staticmethod
    def _key(user_id: str) -> str:
        return f"history:{user_id}"

    async def add(self, user_id: str, role: str, content: str) -> None:
        """
        Append a message to the user's history.

        RPUSH the new entry, then LTRIM to keep only the last _MAX_ENTRIES.
        Resets the 24-hour TTL on every write.

        Args:
            user_id: Unique identifier for the user.
            role: "user" or "assistant".
            content: Message text.
        """
        key = self._key(user_id)
        entry = json.dumps({"role": role, "content": content})
        try:
            r = await self._get_redis()
            pipe = r.pipeline()
            pipe.rpush(key, entry)
            pipe.ltrim(key, -_MAX_ENTRIES, -1)  # keep last N entries
            pipe.expire(key, _TTL_SECONDS)
            await pipe.execute()
            logger.debug("Added %s message to history for user %s.", role, user_id)
        except Exception as exc:
            logger.warning("SessionHistory.add error: %s", exc)

    async def get(self, user_id: str) -> List[Dict[str, str]]:
        """
        Retrieve all history entries for the user.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            List of dicts with 'role' and 'content'.
        """
        key = self._key(user_id)
        try:
            r = await self._get_redis()
            raw_entries = await r.lrange(key, 0, -1)
            history = [json.loads(e) for e in raw_entries]
            logger.debug("Fetched %d history entries for user %s.", len(history), user_id)
            return history
        except Exception as exc:
            logger.warning("SessionHistory.get error: %s", exc)
            return []

    async def get_last_n_turns(self, user_id: str, n: int = 3) -> List[Dict[str, str]]:
        """
        Retrieve the last *n* conversation turns (2*n messages).

        Args:
            user_id: Unique identifier for the user.
            n: Number of turns to retrieve.

        Returns:
            List of dicts with 'role' and 'content'.
        """
        all_history = await self.get(user_id)
        return all_history[-(2 * n):]

    async def clear(self, user_id: str) -> None:
        """
        Delete all history for the user.

        Args:
            user_id: Unique identifier for the user.
        """
        key = self._key(user_id)
        try:
            r = await self._get_redis()
            await r.delete(key)
            logger.debug("Cleared history for user %s.", user_id)
        except Exception as exc:
            logger.warning("SessionHistory.clear error: %s", exc)

    async def format_for_summary(self, user_id: str) -> str:
        """
        Return a human-readable summary of the last 3 turns.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            A formatted string of the conversation.
        """
        history = await self.get_last_n_turns(user_id, n=3)
        if not history:
            return "No conversation history found."

        lines = ["*Your last 3 interactions:*\n"]
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            emoji = "You" if role == "user" else "Bot"
            lines.append(f"*{emoji}:* {content}")

        return "\n".join(lines)
