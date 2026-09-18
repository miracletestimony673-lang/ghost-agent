# Ghost Agent Backend

FastAPI backend for the Ghost Agent Android app. Holds the Groq API key
server-side. The app never sees the key.

## Endpoints

- `GET  /health` — returns `{"ok": true}`
- `POST /auth/register` — create account
- `POST /auth/login` — get session token
- `POST /auth/logout` — invalidate session
- `GET  /auth/session` — validate session
- `POST /chat` — proxy to Groq
- `GET  /capabilities` — which task types are available

## Setup

1. Set environment variables:
   - `GROQ_API_KEY` (required)
   - `JWT_SECRET` (recommended; defaults to a dev value)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt