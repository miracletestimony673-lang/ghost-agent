"""
Valkey cache wrapper.

The Cache class is defensive: it never raises. If Valkey is unreachable
or a command fails, callers get None (on get) or a no-op (on setex), and
a warning is logged. The backend must keep working without the cache.
"""

import logging
from typing import Optional

import redis.asyncio as redis_async

logger = logging.getLogger("ghost.cache")


class Cache:
    def __init__(self, url: str):
        self._url = url
        self._client: Optional[redis_async.Redis] = None
        self._enabled = bool(url)
        self._reported_disabled = False
        self._reported_connect_error = False

    async def _get_client(self) -> Optional[redis_async.Redis]:
        if not self._enabled:
            if not self._reported_disabled:
                logger.warning("Cache disabled: VALKEY_URL is not set")
                self._reported_disabled = True
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
            except Exception as e:
                if not self._reported_connect_error:
                    logger.warning("Cache client construction failed: %s", e)
                    self._reported_connect_error = True
                self._enabled = False
                return None
        return self._client

    async def get(self, key: str) -> Optional[str]:
        client = await self._get_client()
        if client is None:
            return None
        try:
            return await client.get(key)
        except Exception as e:
            logger.warning("Cache get failed for key %s: %s", key, e)
            return None

    async def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.setex(key, ttl_seconds, value)
        except Exception as e:
            logger.warning("Cache setex failed for key %s: %s", key, e)

    async def delete(self, key: str) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.delete(key)
        except Exception as e:
            logger.warning("Cache delete failed for key %s: %s", key, e)

    async def ping(self) -> bool:
        """
        Returns True if the cache responds to PING. Used by /debug/cache.
        """
        client = await self._get_client()
        if client is None:
            return False
        try:
            return bool(await client.ping())
        except Exception as e:
            logger.warning("Cache ping failed: %s", e)
            return False

    async def probe(self) -> dict:
        """
        Runs a round trip write + read. Returns a report.
        Used by /debug/cache.
        """
        report: dict = {
            "url_set": bool(self._url),
            "url_scheme": (self._url.split("://", 1)[0] if self._url else None),
            "client_built": False,
            "ping_ok": False,
            "set_ok": False,
            "get_ok": False,
            "value_matches": False,
            "error": None,
        }
        if not self._url:
            report["error"] = "VALKEY_URL is not set"
            return report

        client = await self._get_client()
        if client is None:
            report["error"] = "client could not be constructed"
            return report
        report["client_built"] = True

        try:
            report["ping_ok"] = bool(await client.ping())
        except Exception as e:
            report["error"] = f"ping failed: {e}"
            return report

        probe_key = "debug:probe"
        probe_value = "ok"
        try:
            await client.setex(probe_key, 10, probe_value)
            report["set_ok"] = True
        except Exception as e:
            report["error"] = f"setex failed: {e}"
            return report

        try:
            read = await client.get(probe_key)
            report["get_ok"] = True
            report["value_matches"] = (read == probe_value)
        except Exception as e:
            report["error"] = f"get failed: {e}"
            return report

        return report


# The singleton. Instantiated in main.py from config.VALKEY_URL.
cache: Cache = Cache("")