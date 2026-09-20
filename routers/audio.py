"""Audio router: /audio/* endpoints for speech-to-text and text-to-speech."""

import base64
import hashlib
import json

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from cache import cache
from config import GROQ_API_KEY, GROQ_BASE, MODEL_BY_TASK, STT_MODEL
from helpers import get_current_account

router = APIRouter(tags=["audio"])

GROQ_TRANSCRIBE_URL = f"{GROQ_BASE}/audio/transcriptions"
GROQ_SPEECH_URL = f"{GROQ_BASE}/audio/speech"
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_TTS_CHARS = 2500
TTS_CACHE_TTL_SECONDS = 3600
TTS_DEFAULT_VOICE = "austin"
TTS_RESPONSE_FORMAT = "wav"


# ---------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------

@router.post("/audio/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    account_id: str = Depends(get_current_account),
) -> dict[str, str]:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="audio file is empty")
    if len(contents) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail="audio file exceeds 25 MB")

    filename = file.filename or "audio.m4a"
    content_type = file.content_type or "audio/m4a"

    files = {
        "file": (filename, contents, content_type),
    }
    data = {
        "model": STT_MODEL,
        "response_format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                GROQ_TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files=files,
                data=data,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=500, detail="transcription provider timeout")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"transcription provider error: {type(e).__name__}")

    if resp.status_code != 200:
        # Include the provider's own message when available.
        detail = _extract_provider_error(resp)
        raise HTTPException(
            status_code=500,
            detail=f"transcription provider rejected request: {resp.status_code} {detail}".strip(),
        )

    payload = resp.json()
    text = payload.get("text") or ""
    return {"text": text.strip()}


# ---------------------------------------------------------------------------
# Text to speech
# ---------------------------------------------------------------------------

class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TTS_CHARS)


@router.post("/audio/speak")
async def speak(
    req: SpeakRequest,
    account_id: str = Depends(get_current_account),
) -> Response:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    model = MODEL_BY_TASK.get("tts")
    if not model:
        raise HTTPException(status_code=500, detail="tts model is not configured")

    # Cache key: sha256 over model + voice + format + text.
    # Any change to the voice or format invalidates the cache for that text.
    key_input = f"{model}\n{TTS_DEFAULT_VOICE}\n{TTS_RESPONSE_FORMAT}\n{text}".encode("utf-8")
    cache_key = "tts:" + hashlib.sha256(key_input).hexdigest()

    cached = await cache.get(cache_key)
    if cached:
        try:
            payload = json.loads(cached)
            content_type = payload.get("content_type") or "audio/wav"
            audio_b64 = payload.get("audio_b64") or ""
            audio_bytes = base64.b64decode(audio_b64)
            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={"X-Cache": "HIT"},
            )
        except Exception:
            # Corrupt or unexpected entry. Fall through and regenerate.
            pass

    request_body = {
        "model": model,
        "voice": TTS_DEFAULT_VOICE,
        "response_format": TTS_RESPONSE_FORMAT,
        "input": text,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                GROQ_SPEECH_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=request_body,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=500, detail="speech provider timeout")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"speech provider error: {type(e).__name__}")

    if resp.status_code != 200:
        detail = _extract_provider_error(resp)
        raise HTTPException(
            status_code=500,
            detail=f"speech provider rejected request: {resp.status_code} {detail}".strip(),
        )

    audio_bytes = resp.content
    content_type = resp.headers.get("content-type") or "audio/wav"

    try:
        cache_payload = json.dumps({
            "content_type": content_type,
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
        })
        await cache.setex(cache_key, TTS_CACHE_TTL_SECONDS, cache_payload)
    except Exception:
        pass

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={"X-Cache": "MISS"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_provider_error(resp: httpx.Response) -> str:
    """
    Pull a short, human-readable error out of the provider's response body.
    Falls back to an empty string if the body cannot be parsed.
    """
    try:
        body = resp.json()
    except Exception:
        try:
            text = resp.text or ""
            return text[:300]
        except Exception:
            return ""

    if not isinstance(body, dict):
        return ""

    err = body.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        if isinstance(msg, str) and msg:
            return msg[:300]
    if isinstance(err, str) and err:
        return err[:300]

    detail = body.get("detail")
    if isinstance(detail, str) and detail:
        return detail[:300]

    return ""