"""Web router: /web/* endpoints (Tavily proxy)."""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException
import httpx

from config import TAVILY_API_KEY
from helpers import get_current_account
from models import WebSearchRequest, WebFetchRequest

router = APIRouter()

@router.post("/web/search")
async def web_search(
    req: WebSearchRequest,
    account_id: str = Depends(get_current_account),
) -> dict[str, Any]:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not TAVILY_API_KEY:
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY is not configured")

    max_results = max(1, min(req.maxResults, 10))

    payload = {
        "query": req.query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=500, detail="search provider timeout")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"search provider error: {type(e).__name__}")

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"search provider rejected request: {resp.status_code}")

    data = resp.json()
    results = []
    for r in (data.get("results") or [])[:max_results]:
        results.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "content": (r.get("content") or "")[:4000],
        })

    return {"query": req.query, "results": results}

@router.post("/web/fetch")
async def web_fetch(
    req: WebFetchRequest,
    account_id: str = Depends(get_current_account),
) -> dict[str, Any]:
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="url must not be empty")
    if not TAVILY_API_KEY:
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY is not configured")

    payload = {"urls": [req.url]}

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                "https://api.tavily.com/extract",
                headers={
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=500, detail="fetch provider timeout")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"fetch provider error: {type(e).__name__}")

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"fetch provider rejected request: {resp.status_code}")

    data = resp.json()
    results = data.get("results") or []
    failed = data.get("failed_results") or []

    if not results:
        reason = "could not extract content"
        if failed:
            reason = failed[0].get("error") or reason
        raise HTTPException(status_code=500, detail=f"fetch failed: {reason}")

    first = results[0]
    return {
        "url": first.get("url") or req.url,
        "title": first.get("title") or "",
        "content": (first.get("raw_content") or "")[:8000],
    }