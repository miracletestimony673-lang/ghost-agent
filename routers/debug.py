"""Debug endpoints. Not for production use but harmless."""

from fastapi import APIRouter

from cache import cache

router = APIRouter(tags=["debug"])


@router.get("/debug/cache")
async def debug_cache() -> dict:
    """
    Reports the state of the Valkey cache.

    Returns a JSON object describing whether the URL is set, whether the
    client was constructed, whether ping succeeded, whether a write and
    read round trip succeeded, and the error message if anything failed.

    No auth required. Safe to leave in place since it exposes no secrets:
    the URL scheme is reported but not the URL or the password.
    """
    return await cache.probe()