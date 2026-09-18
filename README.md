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

## Environment variables

- `GROQ_API_KEY` — required for `/chat` to work. Get one from https://console.groq.com
- `JWT_SECRET` — any random string. Change in production.

## Deploy to Render (free tier)

1. Push `main.py` and `requirements.txt` to a GitHub repo.
2. Go to https://render.com and sign in with GitHub.
3. Click **New** → **Web Service**.
4. Select your repo.
5. Render detects Python. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Under **Environment**, add:
   - `GROQ_API_KEY` = your key
   - `JWT_SECRET` = any long random string
7. Click **Create Web Service**.
8. Wait for the deploy to finish. You get a URL like `https://ghost-agent-backend.onrender.com`.

## Test

From any terminal with internet access:

    curl https://your-render-url.onrender.com/health

Expected:

    {"ok":true}

## Notes

- Storage is in-memory. Restarting the server loses all accounts. Fine for Phase 4. Replace with a real database before production.
- Free tier on Render spins down after 15 minutes of inactivity. First request after idle takes ~30 seconds.