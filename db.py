"""Database: Postgres pool, schema, migrations."""

import asyncpg
from typing import Optional
from fastapi import HTTPException

from config import DATABASE_URL

# Global pool
_db_pool: Optional[asyncpg.Pool] = None

async def get_pool() -> asyncpg.Pool:
    """FastAPI dependency to get the database pool."""
    if _db_pool is None:
        raise HTTPException(status_code=500, detail="database not initialized")
    return _db_pool

# Schema SQL
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

CREATE TABLE IF NOT EXISTS chats (
    chat_id       TEXT PRIMARY KEY,
    account_id    TEXT NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
    title         TEXT,
    created_at    BIGINT NOT NULL,
    updated_at    BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id    TEXT PRIMARY KEY,
    chat_id       TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    role          TEXT NOT NULL,
    content       TEXT,
    tool_calls    JSONB,
    tool_call_id  TEXT,
    created_at    BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chats_account_id ON chats(account_id);
CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
"""

# Migration SQL
MIGRATION_SQL = """
ALTER TABLE accounts ALTER COLUMN password_hash DROP NOT NULL;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS auth_provider TEXT NOT NULL DEFAULT 'password';

-- no migration needed; both tables are new
"""

async def init_db() -> None:
    """Initialize database: create pool, run schema, run migrations."""
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

async def close_db() -> None:
    """Close the database pool."""
    global _db_pool
    if _db_pool is not None:
        await _db_pool.close()
        _db_pool = None