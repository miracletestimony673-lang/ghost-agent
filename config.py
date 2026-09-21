"""Configuration: environment variables and constants."""

import os

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
DATABASE_URL = os.getenv("DATABASE_URL", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
VALKEY_URL = os.getenv("VALKEY_URL", "")

# JWT constants
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7

# Bcrypt rounds
BCRYPT_ROUNDS = 12

# Groq API endpoints
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_CHAT_URL = f"{GROQ_BASE}/chat/completions"

# STT model
STT_MODEL = "whisper-large-v3-turbo"

# Feature flags
TTS_STT_ENABLED = True

# Model mapping by task
MODEL_BY_TASK = {
    "text": "qwen/qwen3.8-27b",
    "text-fast": "openai/gpt-oss-20b",
    "reasoning": "openai/gpt-oss-120b",
    "reasoning-deep": "openai/gpt-oss-120b",
    "vision": "qwen/qwen3.8-27b",
    "moderation": "openai/gpt-oss-safeguard-20b",
    "tts": "canopylabs/orpheus-v1-english",
    "tts-arabic": "canopylabs/orpheus-arabic-saudi",
    "stt": "whisper-large-v3",
    "stt-fast": "whisper-large-v3-turbo",
}