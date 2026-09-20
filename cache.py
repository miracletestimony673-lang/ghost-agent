"""Cache: Valkey/Redis cache class and singleton instance."""

from typing import Optional
import redis.asyncio as redis_async

from config import VALKEY_URL

class Cache:
    """Valkey/Redis cache wrapper."""

    def __init__(self, url: str):
        self._url = url
        self._client: Optional[redis_async.Redis] = None
        self._enabled = bool(url)

    async def _get_client(self) -> Optional[redis_async.Redis]:
        """Get or create the Redis client."""
        if not self._enabled:
            return None
        if self._client is None:
            try:
                self._client = redis_async.from_url(
                    self._url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0,
                )
            except Exception:
                self._enabled = False
                return None
        return self._client

    async def get(self, key: str) -> Optional[str]:
        """Get a value from cache."""
        client = await self._get_client()
        if client is None:
            return None
        try:
            return await client.get(key)
        except Exception:
            return None

    async def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        """Set a value with expiration."""
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.setex(key, ttl_seconds, value)
        except Exception:
            pass

    async def delete(self, key: str) -> None:
        """Delete a key from cache."""
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.delete(key)
        except Exception:
            pass

# Singleton cache instance
cache = Cache(VALKEY_URL)