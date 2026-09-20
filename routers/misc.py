"""Misc router: /models, /capabilities endpoints."""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException
import json

from config import MODEL_BY_TASK, TTS_STT_ENABLED
from cache import cache
from helpers import get_current_account

router = APIRouter()

@router.get("/capabilities")
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

@router.get("/models")
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