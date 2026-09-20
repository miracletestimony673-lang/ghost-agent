"""
Ghost Agent backend with Postgres persistence.

Endpoints:
    GET  /health
    POST /auth/register
    POST /auth/login
    POST /auth/google
    POST /auth/logout
    GET  /auth/session
    POST /chat
    GET  /chats
    POST /chats
    GET  /chats/{chat_id}
    PATCH /chats/{chat_id}
    DELETE /chats/{chat_id}
    GET  /capabilities
    GET  /models
    POST /web/search
    POST /web/fetch

Storage: Postgres. Accounts and sessions persist across redeploys.

Provider: Groq. The Groq API key is read from GROQ_API_KEY.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from db import init_db, close_db
from cache import cache
from routers import auth as auth_router
from routers import chat as chat_router
from routers import chats as chats_router
from routers import web as web_router
from routers import misc as misc_router

app = FastAPI(title="Ghost Agent Backend", version="phase4-pg")

# Include all routers
app.include_router(auth_router.router)
app.include_router(chat_router.router)
app.include_router(chats_router.router)
app.include_router(web_router.router)
app.include_router(misc_router.router)

# Startup and shutdown events
@app.on_event("startup")
async def on_startup():
    await init_db()
    # Warm up Valkey cache connection
    try:
        await cache.get("__probe__")
    except Exception:
        pass

@app.on_event("shutdown")
async def on_shutdown():
    await close_db()

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, (dict, list)):
        body = detail
    else:
        body = {"detail": detail}
    return JSONResponse(status_code=exc.status_code, content=body)