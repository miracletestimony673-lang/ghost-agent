"""Audio router: /audio/* endpoints for speech-to-text."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from config import GROQ_API_KEY, GROQ_BASE, STT_MODEL
from helpers import get_current_account

import httpx

router = APIRouter(tags=["audio"])

GROQ_TRANSCRIBE_URL = f"{GROQ_BASE}/audio/transcriptions"
MAX_AUDIO_BYTES = 25 * 1024 * 1024

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
        raise HTTPException(
            status_code=500,
            detail=f"transcription provider rejected request: {resp.status_code}",
        )

    payload = resp.json()
    text = payload.get("text") or ""
    return {"text": text.strip()}