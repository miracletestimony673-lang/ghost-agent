"""Auth router: /health, /auth/* endpoints."""

from typing import Any
from fastapi import APIRouter, Depends, Header, HTTPException
import asyncpg

from db import get_pool
from helpers import get_current_account, verify_google_id_token, make_token, hash_password, verify_password, now_ms, make_account_id
from models import RegisterRequest, LoginRequest, GoogleAuthRequest

router = APIRouter()

@router.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}

@router.post("/auth/google")
async def auth_google(
    req: GoogleAuthRequest,
    pool: asyncpg.Pool = Depends(get_pool),
) -> dict[str, Any]:
    if not req.idToken:
        raise HTTPException(status_code=400, detail="missing idToken")

    # Verify Google token
    google_info = verify_google_id_token(req.idToken)
    email = google_info.get("email")
    display_name = google_info.get("name") or email.split("@")[0]

    if not email:
        raise HTTPException(status_code=401, detail="google token missing email")

    email = email.lower()

    # Check existing account
    row = await pool.fetchrow(
        "SELECT account_id, auth_provider FROM accounts WHERE email = $1",
        email,
    )

    if row is not None:
        # Account exists - check auth provider
        if row["auth_provider"] == "password":
            raise HTTPException(
                status_code=409,
                detail="this email is registered with a password, sign in with your password",
            )
        # Google account exists - use it
        account_id = row["account_id"]
    else:
        # Create new Google account
        account_id = make_account_id()
        await pool.execute(
            """
            INSERT INTO accounts (account_id, email, display_name, password_hash, auth_provider, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            account_id,
            email,
            display_name,
            None,  # password_hash is NULL for Google accounts
            "google",
            now_ms(),
        )

    # Issue Ghost JWT
    token, expires_ms = make_token(account_id)

    # Store session
    await pool.execute(
        """
        INSERT INTO sessions (token, account_id, expires_at, created_at)
        VALUES ($1, $2, $3, $4)
        """,
        token,
        account_id,
        expires_ms,
        now_ms(),
    )

    return {
        "session": token,
        "expiresAt": expires_ms,
        "accountId": account_id,
        "displayName": display_name,
    }

@router.post("/auth/register", status_code=201)
async def register(
    req: RegisterRequest,
    pool: asyncpg.Pool = Depends(get_pool),
) -> dict[str, Any]:
    email = req.email.lower()
    existing = await pool.fetchrow("SELECT account_id FROM accounts WHERE email = $1", email)
    if existing is not None:
        raise HTTPException(status_code=409, detail="email already registered")

    account_id = make_account_id()
    display_name = req.displayName or email.split("@")[0]
    await pool.execute(
        """
        INSERT INTO accounts (account_id, email, display_name, password_hash, created_at)
        VALUES ($1, $2, $3, $4, $5)
        """,
        account_id,
        email,
        display_name,
        hash_password(req.password),
        now_ms(),
    )
    return {
        "accountId": account_id,
        "email": email,
        "displayName": display_name,
    }

@router.post("/auth/login")
async def login(
    req: LoginRequest,
    pool: asyncpg.Pool = Depends(get_pool),
) -> dict[str, Any]:
    email = req.email.lower()
    row = await pool.fetchrow(
        "SELECT account_id, display_name, password_hash FROM accounts WHERE email = $1",
        email,
    )
    if row is None:
        raise HTTPException(status_code=401, detail="invalid email or password")
    if not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid email or password")

    token, expires_ms = make_token(row["account_id"])
    await pool.execute(
        """
        INSERT INTO sessions (token, account_id, expires_at, created_at)
        VALUES ($1, $2, $3, $4)
        """,
        token,
        row["account_id"],
        expires_ms,
        now_ms(),
    )
    return {
        "session": token,
        "expiresAt": expires_ms,
        "accountId": row["account_id"],
        "displayName": row["display_name"],
    }

@router.post("/auth/logout")
async def logout(
    authorization: str | None = Header(default=None),
    pool: asyncpg.Pool = Depends(get_pool),
) -> dict[str, bool]:
    from cache import cache
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        await pool.execute("DELETE FROM sessions WHERE token = $1", token)
        await cache.delete(f"session:{token}")
    return {"ok": True}

@router.get("/auth/session")
async def session_check(
    account_id: str = Depends(get_current_account),
    pool: asyncpg.Pool = Depends(get_pool),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    token = (authorization or "").removeprefix("Bearer ").strip()
    row = await pool.fetchrow(
        "SELECT expires_at FROM sessions WHERE token = $1",
        token,
    )
    if row is None:
        raise HTTPException(status_code=401, detail="session not found")
    return {
        "valid": True,
        "accountId": account_id,
        "expiresAt": row["expires_at"],
    }