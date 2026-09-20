"""
Ghost Agent backend.

FastAPI application. Wires configuration, database, cache, and routers.

Endpoints:
    GET  /health
    POST /auth/register
    POST /auth/login
    POST /auth/google
    POST /auth/logout
    GET  /auth/session
    POST /chat
    POST /chats
    GET  /chats
    GET  /chats/{chat_id}
    PATCH /chats/{chat_id}
    DELETE /chats/{chat_id}
    POST /web/search
    POST /web/fetch
    POST /audio/transcribe
    POST /audio/speak
    GET  /models
    GET  /capabilities
    GET  /debug/cache

Storage: Postgres. Sessions and chats persist across redeploys.

Cache: Valkey. Sessions and TTS audio are cached. Falls back to Postgres
if the cache is unavailable.

Provider: Groq. The Groq API key is read from GROQ_API_KEY.
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from cache import init_cache
from config import VALKEY_URL
from db import close_db, init_db
from routers import audio as audio_router
from routers import auth as auth_router
from routers import chat as chat_router
from routers import chats as chats_router
from routers import debug as debug_router
from routers import misc as misc_router
from routers import web as web_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ghost.main")


app = FastAPI(title="Ghost Agent Backend", version="phase9")


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router.router)
app.include_router(chat_router.router)
app.include_router(chats_router.router)
app.include_router(web_router.router)
app.include_router(audio_router.router)
app.include_router(misc_router.router)
app.include_router(debug_router.router)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    # Cache first. It is best-effort; a failure here does not stop the app.
    try:
        init_cache(VALKEY_URL)
        logger.info("Cache initialized (url_set=%s)", bool(VALKEY_URL))
    except Exception as e:
        logger.warning("Cache initialization raised: %s", e)

    # Database. Required.
    await init_db()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await close_db()


# ---------------------------------------------------------------------------
# Error handler. Every error body carries a `detail` field as JSON.
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, (dict, list)):
        body = detail
    else:
        body = {"detail": detail}
    return JSONResponse(status_code=exc.status_code, content=body)