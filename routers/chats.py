"""Chats router: /chats/* endpoints for chat history management."""

from typing import Any
from fastapi import APIRouter, Depends, Header, HTTPException
import asyncpg
import uuid

from db import get_pool
from helpers import get_current_account, now_ms
from models import (
    CreateChatResponse,
    ChatListResponse,
    ChatListItem,
    ChatDetailResponse,
    ChatMessageResponse,
    RenameChatRequest,
    RenameChatResponse,
    DeleteChatResponse,
)

router = APIRouter()

@router.post("/chats", status_code=201)
async def create_chat(
    account_id: str = Depends(get_current_account),
    pool: asyncpg.Pool = Depends(get_pool),
) -> CreateChatResponse:
    chat_id = f"chat_{uuid.uuid4().hex[:16]}"
    now = now_ms()
    await pool.execute(
        """
        INSERT INTO chats (chat_id, account_id, title, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5)
        """,
        chat_id,
        account_id,
        None,
        now,
        now,
    )
    return CreateChatResponse(
        chatId=chat_id,
        title=None,
        createdAt=now,
        updatedAt=now,
    )

@router.get("/chats")
async def list_chats(
    account_id: str = Depends(get_current_account),
    pool: asyncpg.Pool = Depends(get_pool),
) -> ChatListResponse:
    rows = await pool.fetch(
        """
        SELECT chat_id, title, created_at, updated_at
        FROM chats
        WHERE account_id = $1
        ORDER BY updated_at DESC
        LIMIT 100
        """,
        account_id,
    )
    chats = []
    for row in rows:
        message_count = await pool.fetchval(
            "SELECT COUNT(*) FROM messages WHERE chat_id = $1",
            row["chat_id"],
        )
        chats.append(ChatListItem(
            chatId=row["chat_id"],
            title=row["title"],
            createdAt=row["created_at"],
            updatedAt=row["updated_at"],
            messageCount=message_count,
        ))
    return ChatListResponse(chats=chats)

@router.get("/chats/{chat_id}")
async def get_chat(
    chat_id: str,
    account_id: str = Depends(get_current_account),
    pool: asyncpg.Pool = Depends(get_pool),
) -> ChatDetailResponse:
    row = await pool.fetchrow(
        "SELECT chat_id, account_id, title, created_at, updated_at FROM chats WHERE chat_id = $1",
        chat_id,
    )
    if row is None or row["account_id"] != account_id:
        raise HTTPException(status_code=404, detail="chat not found")

    messages = await pool.fetch(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE chat_id = $1
        ORDER BY created_at ASC
        """,
        chat_id,
    )
    message_responses = [
        ChatMessageResponse(
            role=m["role"],
            content=m["content"],
            createdAt=m["created_at"],
        )
        for m in messages
    ]
    return ChatDetailResponse(
        chatId=row["chat_id"],
        title=row["title"],
        createdAt=row["created_at"],
        updatedAt=row["updated_at"],
        messages=message_responses,
    )

@router.patch("/chats/{chat_id}")
async def rename_chat(
    chat_id: str,
    req: RenameChatRequest,
    account_id: str = Depends(get_current_account),
    pool: asyncpg.Pool = Depends(get_pool),
) -> RenameChatResponse:
    if not req.title or len(req.title) > 100:
        raise HTTPException(status_code=400, detail="title must be non-empty and at most 100 characters")

    # Verify ownership
    chat_account_id = await pool.fetchval(
        "SELECT account_id FROM chats WHERE chat_id = $1",
        chat_id,
    )
    if chat_account_id is None or chat_account_id != account_id:
        raise HTTPException(status_code=404, detail="chat not found")

    updated_at = now_ms()
    await pool.execute(
        "UPDATE chats SET title = $1, updated_at = $2 WHERE chat_id = $3",
        req.title,
        updated_at,
        chat_id,
    )
    return RenameChatResponse(
        chatId=chat_id,
        title=req.title,
        updatedAt=updated_at,
    )

@router.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    account_id: str = Depends(get_current_account),
    pool: asyncpg.Pool = Depends(get_pool),
) -> DeleteChatResponse:
    # Verify ownership
    chat_account_id = await pool.fetchval(
        "SELECT account_id FROM chats WHERE chat_id = $1",
        chat_id,
    )
    if chat_account_id is None or chat_account_id != account_id:
        raise HTTPException(status_code=404, detail="chat not found")

    await pool.execute("DELETE FROM chats WHERE chat_id = $1", chat_id)
    return DeleteChatResponse(ok=True)