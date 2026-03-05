"""Companion Lite — re-exports for tests and Go CLI backend."""
from __future__ import annotations

from config import (  # noqa: F401
    BASE_DIR, CFG, MODELS, ALIASES, DEFAULT_MODEL,
    SYSTEM_PROMPT, EXTRACTION_MODEL, OLLAMA_URL,
    _gemini_client, _gemini_clients,
    load_config,
)
from llm import (  # noqa: F401
    DebugLogger, debug_log, LOG_FILE,
    call_gemini_api, stream_gemini_api, call_ollama,
)
from memory import Memory  # noqa: F401
from chat import Chat  # noqa: F401
