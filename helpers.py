"""Shared helper functions."""

import time
import uuid
import json
from typing import Any

import bcrypt
import jwt
from fastapi import HTTPException, Depends, Header
import asyncpg

from config import JWT_SECRET, JWT_ALGORITHM, TOKEN_EXPIRE_DAYS, BCRYPT_ROUNDS
from db import get_pool
from cache import cache

def now_ms() -> int:
    """Return current timestamp in milliseconds."""
    return int(time.time() * 1000)

def make_account_id() -> str:
    """Generate a new account ID."""
    return f"acc_{uuid.uuid4().hex[:16]}"

def hash_password(plain: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(
        plain.encode("utf-8"),
        bcrypt.gensalt(rounds=BCRYPT_ROUNDS),
    ).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def make_token(account_id: str) -> tuple[str, int]:
    """Create a JWT token for an account."""
    now = now_ms()
    expires_ms = now + TOKEN_EXPIRE_DAYS * 24 * 60 * 60 * 1000
    payload = {
        "sub": account_id,
        "iat": now // 1000,
        "exp": expires_ms // 1000,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, expires_ms

def decode_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT token."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="session expired")
    except Exception:
        raise HTTPException(status_code=401, detail="invalid session")

async def get_current_account(
    authorization: str | None = Header(default=None),
    pool: asyncpg.Pool = Depends(get_pool),
) -> str:
    """Get the current account ID from the authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="missing session token")

    # Try Valkey cache first
    cache_key = f"session:{token}"
    cached = await cache.get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            account_id = data.get("account_id")
            expires_at = data.get("expires_at")
            if account_id and expires_at and expires_at > now_ms():
                return account_id
        except Exception:
            pass

    # Fall back to Postgres
    row = await pool.fetchrow(
        "SELECT account_id, expires_at FROM sessions WHERE token = $1",
        token,
    )
    if row is None:
        raise HTTPException(status_code=401, detail="invalid session")
    if row["expires_at"] < now_ms():
        await pool.execute("DELETE FROM sessions WHERE token = $1", token)
        await cache.delete(cache_key)
        raise HTTPException(status_code=401, detail="session expired")

    decode_token(token)

    # Populate cache with 5-minute TTL
    await cache.setex(
        cache_key,
        300,
        json.dumps({
            "account_id": row["account_id"],
            "expires_at": row["expires_at"],
        }),
    )
    return row["account_id"]

# Google Auth helper
def verify_google_id_token(token: str) -> dict[str, Any]:
    """Verify a Google ID token."""
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
    from config import GOOGLE_CLIENT_ID

    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_CLIENT_ID is not configured",
        )
    try:
        info = google_id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"invalid google token: {e}",
        )
    if not info.get("email_verified"):
        raise HTTPException(
            status_code=401,
            detail="google email is not verified",
        )
    return info