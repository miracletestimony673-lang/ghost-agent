"""Pydantic DTOs for request/response models."""

from typing import Any
from pydantic import BaseModel, EmailStr, Field

# Auth models
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    displayName: str | None = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class GoogleAuthRequest(BaseModel):
    idToken: str

# Chat models
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
    chat_id: str | None = None

# Web models
class WebSearchRequest(BaseModel):
    query: str
    maxResults: int = 5

class WebFetchRequest(BaseModel):
    url: str

# Chat history DTOs
class CreateChatResponse(BaseModel):
    chatId: str
    title: str | None = None
    createdAt: int
    updatedAt: int

class ChatListItem(BaseModel):
    chatId: str
    title: str | None = None
    createdAt: int
    updatedAt: int
    messageCount: int

class ChatListResponse(BaseModel):
    chats: list[ChatListItem]

class ChatMessageResponse(BaseModel):
    role: str
    content: str | None = None
    createdAt: int

class ChatDetailResponse(BaseModel):
    chatId: str
    title: str | None = None
    createdAt: int
    updatedAt: int
    messages: list[ChatMessageResponse]

class RenameChatRequest(BaseModel):
    title: str

class RenameChatResponse(BaseModel):
    chatId: str
    title: str
    updatedAt: int

class DeleteChatResponse(BaseModel):
    ok: bool = True