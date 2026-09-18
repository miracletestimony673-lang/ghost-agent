"""
Ghost Agent backend.

Frozen contract, Phase 4 (extended).

Endpoints:
    GET  /health
    POST /auth/register
    POST /auth/login
    POST /auth/logout
    GET  /auth/session
    POST /chat
    GET  /capabilities
    GET  /models

Storage: in-memory dict. Development only.

Provider: Groq. The Groq API key is read from GROQ_API_KEY and is never
returned in any response.
"""

import os
import time
import uuid
import json
from typing import Any

import bcrypt
import httpx
import jwt
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7

GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_CHAT_URL = f"{GROQ_BASE}/chat/completions"
GROQ_AUDIO_TRANSCRIPTION_URL = f"{GROQ_BASE}/audio/transcriptions"
GROQ_AUDIO_SPEECH_URL = f"{GROQ_BASE}/audio/speech"

BCRYPT_ROUNDS = 12

# Model mapping. Server-side only. The app sends `task`, the backend picks
# the model. This lets us swap Groq models without an app update.
#
# Groq's catalogue changes often. If a model returns "does not exist or you
# do not have access", it has either been retired or gated to a paid tier.
# Replace the id here and redeploy.
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

# Phase 4 app surface is text only. These tasks are accepted by the backend
# but the app will not send them until later phases. We do NOT return 501
# for them anymore because the models exist.
TTS_STT_ENABLED = False  # flip to True when the app has UI for them


app = FastAPI(title="Ghost Agent Backend", version="phase4")


# ----------------------------------------------------------------------------
# In-memory stores (development only)
# ----------------------------------------------------------------------------

accounts_by_email: dict[str, dict[str, Any]] = {}
sessions: dict[str, dict[str, Any]] = {}


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
    content: Any


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    task: str = "text"
    stream: bool = False


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

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


def issue_token(account_id: str) -> tuple[str, int]:
    now_ms = int(time.time() * 1000)
    expires_ms = now_ms + TOKEN_EXPIRE_DAYS * 24 * 60 * 60 * 1000
    payload = {
        "sub": account_id,
        "iat": now_ms // 1000,
        "exp": expires_ms // 1000,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    sessions[token] = {"accountId": account_id, "expiresAt": expires_ms}
    return token, expires_ms


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="session expired")
    except Exception:
        raise HTTPException(status_code=401, detail="invalid session")


def get_current_account(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="missing session token")
    if token not in sessions:
        raise HTTPException(status_code=401, detail="invalid session")
    info = sessions[token]
    if info["expiresAt"] < int(time.time() * 1000):
        sessions.pop(token, None)
        raise HTTPException(status_code=401, detail="session expired")
    decode_token(token)
    return info["accountId"]


# ----------------------------------------------------------------------------
# Routes — health
# ----------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


# ----------------------------------------------------------------------------
# Routes — auth
# ----------------------------------------------------------------------------

@app.post("/auth/register", status_code=201)
async def register(req: RegisterRequest) -> dict[str, Any]:
    email = req.email.lower()
    if email in accounts_by_email:
        raise HTTPException(status_code=409, detail="email already registered")

    account_id = make_account_id()
    display_name = req.displayName or email.split("@")[0]
    accounts_by_email[email] = {
        "accountId": account_id,
        "email": email,
        "displayName": display_name,
        "passwordHash": hash_password(req.password),
    }
    return {
        "accountId": account_id,
        "email": email,
        "displayName": display_name,
    }


@app.post("/auth/login")
async def login(req: LoginRequest) -> dict[str, Any]:
    email = req.email.lower()
    account = accounts_by_email.get(email)
    if not account:
        raise HTTPException(status_code=401, detail="invalid email or password")
    if not verify_password(req.password, account["passwordHash"]):
        raise HTTPException(status_code=401, detail="invalid email or password")

    token, expires_ms = issue_token(account["accountId"])
    return {
        "session": token,
        "expiresAt": expires_ms,
        "accountId": account["accountId"],
        "displayName": account["displayName"],
    }


@app.post("/auth/logout")
async def logout(authorization: str | None = Header(default=None)) -> dict[str, bool]:
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        sessions.pop(token, None)
    return {"ok": True}


@app.get("/auth/session")
async def session_check(account_id: str = Depends(get_current_account)) -> dict[str, Any]:
    expires_at = None
    for info in sessions.values():
        if info["accountId"] == account_id:
            expires_at = info["expiresAt"]
            break
    if expires_at is None:
        raise HTTPException(status_code=401, detail="session not found")
    return {
        "valid": True,
        "accountId": account_id,
        "expiresAt": expires_at,
    }


# ----------------------------------------------------------------------------
# Routes — capabilities
# ----------------------------------------------------------------------------

@app.get("/capabilities")
async def capabilities(account_id: str = Depends(get_current_account)) -> dict[str, bool]:
    return {
        "text": True,
        "vision": True,
        "reasoning": True,
        "tts": TTS_STT_ENABLED,
        "stt": TTS_STT_ENABLED,
    }


# ----------------------------------------------------------------------------
# Routes — models (informational, no auth, safe to expose)
# ----------------------------------------------------------------------------

@app.get("/models")
async def models() -> dict[str, str]:
    # Return the current server-side model mapping. Useful for debugging and
    # for a future in-app model picker. Contains no secrets.
    return dict(MODEL_BY_TASK)


# ----------------------------------------------------------------------------
# Routes — chat
# ----------------------------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest, account_id: str = Depends(get_current_account)) -> dict[str, Any]:
    # TTS / STT are gated. When the app has UI for them, flip TTS_STT_ENABLED.
    if req.task in ("tts", "tts-arabic", "stt", "stt-fast"):
        if not TTS_STT_ENABLED:
            raise HTTPException(
                status_code=501,
                detail=f"task '{req.task}' is not enabled yet",
            )

    # Vision guardrail: reject if no image is present.
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
        raise HTTPException(
            status_code=400,
            detail=f"unsupported task: {req.task}",
        )

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured on the server",
        )

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    for m in req.messages:
        if m.role not in ("system", "user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail=f"invalid role: {m.role}",
            )
        if m.content is None:
            raise HTTPException(
                status_code=400,
                detail="message content must not be null",
            )

    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "stream": False,
    }

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
    except httpx.HTTPError as e:
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
        raise HTTPException(
            status_code=429,
            detail={"retryAfterMs": retry_after_ms},
        )

    if resp.status_code >= 500:
        raise HTTPException(
            status_code=500,
            detail=f"provider error {resp.status_code}",
        )

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
    content = message.get("content") or ""

    usage = data.get("usage") or {}
    return {
        "message": {"role": "assistant", "content": content},
        "model": data.get("model", model),
        "usage": {
            "promptTokens": int(usage.get("prompt_tokens", 0)),
            "completionTokens": int(usage.get("completion_tokens", 0)),
        },
    }


# ----------------------------------------------------------------------------
# Error handler — every error body must include a `detail` field as JSON.
# ----------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, (dict, list)):
        body = detail
    else:
        body = {"detail": detail}
    return JSONResponse(status_code=exc.status_code, content=body)