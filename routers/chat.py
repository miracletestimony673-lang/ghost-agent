"""Chat router: /chat endpoint with buffered and streaming handlers."""

from typing import Any
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
import asyncpg
import httpx
import json
import uuid

from config import GROQ_API_KEY, GROQ_CHAT_URL, MODEL_BY_TASK, TTS_STT_ENABLED
from db import get_pool
from cache import cache
from helpers import get_current_account, now_ms
from models import ChatRequest, ChatMessage

router = APIRouter()

def make_message_id() -> str:
    """Generate a new message ID."""
    return f"msg_{uuid.uuid4().hex[:16]}"

async def store_user_message(pool: asyncpg.Pool, chat_id: str, content: str) -> str:
    """Store a user message and return the message_id."""
    message_id = make_message_id()
    await pool.execute(
        """
        INSERT INTO messages (message_id, chat_id, role, content, created_at)
        VALUES ($1, $2, 'user', $3, $4)
        """,
        message_id,
        chat_id,
        content,
        now_ms(),
    )
    return message_id

async def store_assistant_message(
    pool: asyncpg.Pool,
    chat_id: str,
    content: str | None = None,
    tool_calls: list[dict] | None = None,
) -> str:
    """Store an assistant message and return the message_id."""
    message_id = make_message_id()
    await pool.execute(
        """
        INSERT INTO messages (message_id, chat_id, role, content, tool_calls, created_at)
        VALUES ($1, $2, 'assistant', $3, $4, $5)
        """,
        message_id,
        chat_id,
        content,
        tool_calls,
        now_ms(),
    )
    return message_id

async def store_tool_message(
    pool: asyncpg.Pool,
    chat_id: str,
    content: str | None = None,
    tool_call_id: str | None = None,
) -> str:
    """Store a tool message and return the message_id."""
    message_id = make_message_id()
    await pool.execute(
        """
        INSERT INTO messages (message_id, chat_id, role, content, tool_call_id, created_at)
        VALUES ($1, $2, 'tool', $3, $4, $5)
        """,
        message_id,
        chat_id,
        content,
        tool_call_id,
        now_ms(),
    )
    return message_id

async def verify_chat_ownership(pool: asyncpg.Pool, chat_id: str, account_id: str) -> bool:
    """Verify that a chat exists and belongs to the account."""
    chat_account_id = await pool.fetchval(
        "SELECT account_id FROM chats WHERE chat_id = $1",
        chat_id,
    )
    return chat_account_id is not None and chat_account_id == account_id

async def update_chat_updated_at(pool: asyncpg.Pool, chat_id: str) -> None:
    """Update the chat updated_at timestamp."""
    await pool.execute(
        "UPDATE chats SET updated_at = $1 WHERE chat_id = $2",
        now_ms(),
        chat_id,
    )

async def generate_title(user_text: str, assistant_text: str) -> str:
    """Generate a chat title using the text-tiny model."""
    title_model = "llama-3.1-8b-instant"
    title_prompt = [
        {"role": "system", "content": "Generate a 3 to 6 word title for this conversation. Reply with the title only. No quotes. No punctuation at the end."},
        {"role": "user", "content": f"User: {user_text}\n\nAssistant: {assistant_text}"},
    ]

    try:
        payload = {
            "model": title_model,
            "messages": title_prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                GROQ_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if resp.status_code == 200:
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            title = message.get("content") or ""
            # Clean up: strip quotes and trim
            title = title.strip().strip('"').strip("'")
            # Trim to 100 characters
            title = title[:100]
            if title:
                return title
    except Exception:
        pass

    # Fallback: first 40 characters of user message
    return user_text[:40].strip()

async def stream_groq(groq_messages, model, tools):
    """Stream Groq response without storage."""
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

async def stream_groq_with_storage(groq_messages, model, tools, chat_id, account_id, pool):
    """Stream Groq response and store the full text at the end."""
    import re

    payload = {
        "model": model,
        "messages": groq_messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    buffer = []

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
                        # Parse and extract content for buffering
                        try:
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                yield f"{line}\n\n"
                                continue
                            data_json = json.loads(data_str)
                            if "choices" in data_json:
                                for choice in data_json["choices"]:
                                    if "delta" in choice:
                                        delta = choice["delta"]
                                        if "content" in delta:
                                            buffer.append(delta["content"])
                                        elif "tool_calls" in delta:
                                            # For tool calls, we store the full tool_calls
                                            pass
                        except Exception:
                            pass
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
    else:
        # Stream completed successfully, store the accumulated text
        try:
            full_text = "".join(buffer)
            if full_text:
                await store_assistant_message(
                    pool, chat_id, full_text, None
                )

            # Store tool messages from the request
            for m in groq_messages:
                if m.get("role") == "tool":
                    await store_tool_message(
                        pool, chat_id, m.get("content"), m.get("tool_call_id")
                    )

            # Generate title if needed
            # Get the user content from the request messages
            user_content = None
            for m in groq_messages:
                if m.get("role") == "user" and m.get("content"):
                    user_content = m["content"]
                    break

            if user_content and full_text:
                chat_title = await pool.fetchval(
                    "SELECT title FROM chats WHERE chat_id = $1",
                    chat_id,
                )
                if chat_title is None:
                    msg_count = await pool.fetchval(
                        "SELECT COUNT(*) FROM messages WHERE chat_id = $1 AND role = 'assistant'",
                        chat_id,
                    )
                    if msg_count is not None and msg_count == 1:
                        title = await generate_title(user_content, full_text)
                        await pool.execute(
                            "UPDATE chats SET title = $1, updated_at = $2 WHERE chat_id = $3",
                            title,
                            now_ms(),
                            chat_id,
                        )

            # Update chat updated_at
            await update_chat_updated_at(pool, chat_id)
        except Exception:
            pass

        return

@router.post("/chat")
async def chat(
    req: ChatRequest,
    account_id: str = Depends(get_current_account),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """Handle chat requests - both buffered and streaming."""
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

    # Handle chat_id if present
    chat_id = req.chat_id
    if chat_id:
        # Verify chat exists and belongs to account
        if not await verify_chat_ownership(pool, chat_id, account_id):
            raise HTTPException(status_code=404, detail="chat not found")

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
        if chat_id:
            # Store user message before streaming
            user_content = None
            for m in req.messages:
                if m.role == "user" and m.content is not None:
                    user_content = m.content
                    break
            if user_content:
                await store_user_message(pool, chat_id, user_content)

            # Store tool messages if present
            for m in req.messages:
                if m.role == "tool":
                    await store_tool_message(
                        pool, chat_id, m.content, m.tool_call_id
                    )

            return StreamingResponse(
                stream_groq_with_storage(
                    groq_messages, model, req.tools, chat_id, account_id, pool
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
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

    # Store messages if chat_id is present
    if chat_id:
        # Store user message
        user_content = None
        for m in req.messages:
            if m.role == "user" and m.content is not None:
                user_content = m.content
                break
        if user_content:
            await store_user_message(pool, chat_id, user_content)

        # Store tool messages if present
        for m in req.messages:
            if m.role == "tool":
                await store_tool_message(
                    pool, chat_id, m.content, m.tool_call_id
                )

        # Store assistant response
        assistant_content = message.get("content")
        assistant_tool_calls = message.get("tool_calls")
        await store_assistant_message(
            pool, chat_id, assistant_content, assistant_tool_calls
        )

        # Generate title if needed (first assistant response and title is NULL)
        if user_content and assistant_content:
            chat_title = await pool.fetchval(
                "SELECT title FROM chats WHERE chat_id = $1",
                chat_id,
            )
            if chat_title is None:
                # Check if this is the first assistant message
                msg_count = await pool.fetchval(
                    "SELECT COUNT(*) FROM messages WHERE chat_id = $1 AND role = 'assistant'",
                    chat_id,
                )
                if msg_count is not None and msg_count == 1:
                    title = await generate_title(user_content, assistant_content)
                    await pool.execute(
                        "UPDATE chats SET title = $1, updated_at = $2 WHERE chat_id = $3",
                        title,
                        now_ms(),
                        chat_id,
                    )

        # Update chat updated_at
        await update_chat_updated_at(pool, chat_id)

    usage = data.get("usage") or {}
    return {
        "message": response_message,
        "model": data.get("model", model),
        "usage": {
            "promptTokens": int(usage.get("prompt_tokens", 0)),
            "completionTokens": int(usage.get("completion_tokens", 0)),
        },
    }