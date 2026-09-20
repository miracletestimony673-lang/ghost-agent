"""
Ghost Agent backend with Postgres persistence.

Endpoints:
    GET  /health
    POST /auth/register
    POST /auth/login
    POST /auth/logout
    GET  /auth/session
    POST /chat
    GET  /capabilities
    GET  /models

Storage: Postgres. Accounts and sessions persist across redeploys.

Provider: Groq. The Groq API key is read from GROQ_API_KEY.
"""

import os
import time
import uuid
import json
from typing import Any, Optional

import asyncpg
import bcrypt
import httpx
import jwt
import redis.asyncio as redis_async
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, EmailStr, Field

# Google Auth imports
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
DATABASE_URL = os.getenv("DATABASE_URL", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
VALKEY_URL = os.getenv("VALKEY_URL", "")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7

GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_CHAT_URL = f"{GROQ_BASE}/chat/completions"

BCRYPT_ROUNDS = 12

MODEL_BY_TASK = {
    "text":           "openai/gpt-oss-120b",
    "text-fast":      "openai/gpt-oss-20b",
    "text-tiny":      "llama-3.1-8b-instant",
    "reasoning":      "openai/gpt-oss-120b",
    "reasoning-deep": "openai/gpt-oss-120b",
    "vision":         "qwen/qwen3.8-27b",
    "moderation":     "openai/gpt-oss-safeguard-20b",
    "tts":            "canopylabs/orpheus-v1-english",
    "tts-arabic":     "canopylabs/orpheus-arabic-saudi",
    "stt":            "whisper-large-v3",
    "stt-fast":       "whisper-large-v3-turbo",
}

TTS_STT_ENABLED = False

app = FastAPI(title="Ghost Agent Backend", version="phase4-pg")

# ----------------------------------------------------------------------------
# Valkey Cache
# ----------------------------------------------------------------------------

class Cache:
    def __init__(self, url: str):
        self._url = url
        self._client: Optional[redis_async.Redis] = None
        self._enabled = bool(url)

    async def _get_client(self) -> Optional[redis_async.Redis]:
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
        client = await self._get_client()
        if client is None:
            return None
        try:
            return await client.get(key)
        except Exception:
            return None

    async def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.setex(key, ttl_seconds, value)
        except Exception:
            pass

    async def delete(self, key: str) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.delete(key)
        except Exception:
            pass

cache = Cache(VALKEY_URL)

# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------

_db_pool: Optional[asyncpg.Pool] = None

async def get_pool() -> asyncpg.Pool:
    if _db_pool is None:
        raise HTTPException(status_code=500, detail="database not initialized")
    return _db_pool

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id   TEXT PRIMARY KEY,
    email        TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    password_hash TEXT NULL,
    auth_provider TEXT NOT NULL DEFAULT 'password',
    created_at   BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    account_id  TEXT NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
    expires_at  BIGINT NOT NULL,
    created_at  BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_account_id ON sessions(account_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
"""

MIGRATION_SQL = """
ALTER TABLE accounts ALTER COLUMN password_hash DROP NOT NULL;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS auth_provider TEXT NOT NULL DEFAULT 'password';
"""

@app.on_event("startup")
async def on_startup():
    global _db_pool
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Configure it in Render's environment variables."
        )
    _db_pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=1,
        max_size=5,
        command_timeout=30,
    )
    async with _db_pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
        try:
            await conn.execute(MIGRATION_SQL)
        except Exception:
            pass

    # Warm up Valkey cache connection
    try:
        await cache.get("__probe__")
    except Exception:
        pass

@app.on_event("shutdown")
async def on_shutdown():
    global _db_pool
    if _db_pool is not None:
        await _db_pool.close()
        _db_pool = None

# ----------------------------------------------------------------------------
# DTOs
# ----------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    displayName: str | None = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ChatMessage(BaseModel):
    role: str
    content: Any = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    task: str = "text"
    stream: bool = False
    tools: list[dict] | None = None

class WebSearchRequest(BaseModel):
    query: str
    maxResults: int = 5

class WebFetchRequest(BaseModel):
    url: str

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def now_ms() -> int:
    return int(time.time() * 1000)

def make_account_id() -> str:
    return f"acc_{uuid.uuid4().hex[:16]}"

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(
        plain.encode("utf-8"),
        bcrypt.gensalt(rounds=BCRYPT_ROUNDS),
    ).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def make_token(account_id: str) -> tuple[str, int]:
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

# ----------------------------------------------------------------------------
# Routes — health
# ----------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}

# ----------------------------------------------------------------------------
# Routes — auth
# ----------------------------------------------------------------------------

# Google Sign-In helper
def verify_google_id_token(token: str) -> dict[str, Any]:
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

class GoogleAuthRequest(BaseModel):
    idToken: str

@app.post("/auth/google")
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

@app.post("/auth/register", status_code=201)
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

@app.post("/auth/login")
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

@app.post("/auth/logout")
async def logout(
    authorization: str | None = Header(default=None),
    pool: asyncpg.Pool = Depends(get_pool),
) -> dict[str, bool]:
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        await pool.execute("DELETE FROM sessions WHERE token = $1", token)
        await cache.delete(f"session:{token}")
    return {"ok": True}

@app.get("/auth/session")
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

# ----------------------------------------------------------------------------
# Routes — capabilities
# ----------------------------------------------------------------------------

@app.get("/capabilities")
async def capabilities(account_id: str = Depends(get_current_account)) -> dict[str, bool]:
    cached = await cache.get("capabilities:v1")
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    payload = {
        "text": True,
        "vision": True,
        "reasoning": True,
        "tts": TTS_STT_ENABLED,
        "stt": TTS_STT_ENABLED,
    }
    await cache.setex("capabilities:v1", 3600, json.dumps(payload))
    return payload

# ----------------------------------------------------------------------------
# Routes — models
# ----------------------------------------------------------------------------

@app.get("/models")
async def models() -> dict[str, str]:
    cached = await cache.get("models:v1")
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    payload = dict(MODEL_BY_TASK)
    await cache.setex("models:v1", 3600, json.dumps(payload))
    return payload

# ----------------------------------------------------------------------------
# Routes — web (Tavily proxy)
# ----------------------------------------------------------------------------

@app.post("/web/search")
async def web_search(
    req: WebSearchRequest,
    account_id: str = Depends(get_current_account),
) -> dict[str, Any]:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not TAVILY_API_KEY:
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY is not configured")

    max_results = max(1, min(req.maxResults, 10))

    payload = {
        "query": req.query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=500, detail="search provider timeout")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"search provider error: {type(e).__name__}")

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"search provider rejected request: {resp.status_code}")

    data = resp.json()
    results = []
    for r in (data.get("results") or [])[:max_results]:
        results.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "content": (r.get("content") or "")[:4000],
        })

    return {"query": req.query, "results": results}

@app.post("/web/fetch")
async def web_fetch(
    req: WebFetchRequest,
    account_id: str = Depends(get_current_account),
) -> dict[str, Any]:
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="url must not be empty")
    if not TAVILY_API_KEY:
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY is not configured")

    payload = {"urls": [req.url]}

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                "https://api.tavily.com/extract",
                headers={
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=500, detail="fetch provider timeout")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"fetch provider error: {type(e).__name__}")

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"fetch provider rejected request: {resp.status_code}")

    data = resp.json()
    results = data.get("results") or []
    failed = data.get("failed_results") or []

    if not results:
        reason = "could not extract content"
        if failed:
            reason = failed[0].get("error") or reason
        raise HTTPException(status_code=500, detail=f"fetch failed: {reason}")

    first = results[0]
    return {
        "url": first.get("url") or req.url,
        "title": first.get("title") or "",
        "content": (first.get("raw_content") or "")[:8000],
    }

# ----------------------------------------------------------------------------
# Routes — chat
# ----------------------------------------------------------------------------

async def stream_groq(groq_messages, model, tools):
    payload = {
        "model": model,
        "messages": groq_messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                GROQ_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    error_msg = body[:200].decode("utf-8", "ignore") if body else "Unknown error"
                    yield f"data: {{\"error\":\"provider error {response.status_code}: {error_msg}\"}}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        yield f"{line}\n\n"
    except httpx.TimeoutException:
        yield 'data: {"error":"upstream timeout"}\n\n'
        yield "data: [DONE]\n\n"
    except httpx.HTTPError as e:
        yield f'data: {{\"error\":\"upstream error: {type(e).__name__}\"}}\n\n'
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_msg = str(e)[:200]
        yield f'data: {{\"error\":\"stream error: {error_msg}\"}}\n\n'
        yield "data: [DONE]\n\n"

@app.post("/chat")
async def chat(req: ChatRequest, account_id: str = Depends(get_current_account)):
    if req.task in ("tts", "tts-arabic", "stt", "stt-fast"):
        if not TTS_STT_ENABLED:
            raise HTTPException(
                status_code=501,
                detail=f"task '{req.task}' is not enabled yet",
            )

    if req.task == "vision":
        has_image = False
        for m in req.messages:
            content = m.content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        has_image = True
                        break
            if has_image:
                break
        if not has_image:
            raise HTTPException(
                status_code=400,
                detail="task 'vision' requires an image",
            )

    model = MODEL_BY_TASK.get(req.task)
    if not model:
        raise HTTPException(status_code=400, detail=f"unsupported task: {req.task}")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server")

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    for m in req.messages:
        if m.role not in ("system", "user", "assistant", "tool"):
            raise HTTPException(status_code=400, detail=f"invalid role: {m.role}")
        if m.role not in ("tool",) and m.content is None and m.tool_calls is None:
            raise HTTPException(status_code=400, detail="message content must not be null")

    # Build messages for Groq, preserving all fields
    groq_messages = []
    for m in req.messages:
        msg = {"role": m.role}
        if m.content is not None:
            msg["content"] = m.content
        if m.tool_calls is not None:
            msg["tool_calls"] = m.tool_calls
        if m.tool_call_id is not None:
            msg["tool_call_id"] = m.tool_call_id
        groq_messages.append(msg)

    if req.stream:
        return StreamingResponse(
            stream_groq(groq_messages, model, req.tools),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    payload = {
        "model": model,
        "messages": groq_messages,
        "stream": False,
    }

    # Forward tools if provided
    if req.tools:
        payload["tools"] = req.tools

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                GROQ_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                content=json.dumps(payload),
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=500, detail="upstream timeout from provider")
    except httpx.HTTPException as e:
        raise HTTPException(status_code=500, detail=f"upstream error: {type(e).__name__}")

    if resp.status_code == 429:
        retry_after_ms = 3000
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = body.get("error") or body.get("detail")
                if isinstance(detail, dict) and "retry_after" in detail:
                    retry_after_ms = int(detail["retry_after"]) * 1000
        except Exception:
            pass
        raise HTTPException(status_code=429, detail={"retryAfterMs": retry_after_ms})

    if resp.status_code >= 500:
        raise HTTPException(status_code=500, detail=f"provider error {resp.status_code}")

    if resp.status_code >= 400:
        try:
            body = resp.json()
            message = None
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    message = err.get("message")
                elif isinstance(err, str):
                    message = err
        except Exception:
            message = None
        raise HTTPException(
            status_code=500,
            detail=f"provider rejected request: {message or resp.status_code}",
        )

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(status_code=500, detail="provider returned non-JSON response")

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    # Preserve the full message, including tool_calls
    response_message = {
        "role": message.get("role", "assistant"),
        "content": message.get("content"),
    }

    # Preserve tool_calls if present
    if message.get("tool_calls"):
        response_message["tool_calls"] = message["tool_calls"]

    usage = data.get("usage") or {}
    return {
        "message": response_message,
        "model": data.get("model", model),
        "usage": {
            "promptTokens": int(usage.get("prompt_tokens", 0)),
            "completionTokens": int(usage.get("completion_tokens", 0)),
        },
    }

# ----------------------------------------------------------------------------
# Error handler
# ----------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, (dict, list)):
        body = detail
    else:
        body = {"detail": detail}
    return JSONResponse(status_code=exc.status_code, content=body)