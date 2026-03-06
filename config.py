"""Bootstrap & configuration — loads settings, initializes API clients."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from google import genai

# ── Bootstrap ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


def load_config(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)

    def resolve(obj: Any) -> Any:
        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                return os.environ.get(obj[2:-1], "")
            if obj.startswith("~"):
                return os.path.expanduser(obj)
        elif isinstance(obj, dict):
            return {k: resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve(v) for v in obj]
        return obj

    return resolve(raw)


CFG = load_config(BASE_DIR / "config" / "settings.yaml")

# ── Models ─────────────────────────────────────────────────────────────
MODELS: dict[str, dict] = {}
for key, val in CFG.get("models", {}).items():
    if val.get("provider") == "gemini":
        MODELS[key] = val

if not MODELS:
    print("Error: No gemini API models found in config/settings.yaml")
    sys.exit(1)

# ── Gemini API clients ────────────────────────────────────────────────
_api_keys = CFG.get("api_keys", {})
_gemini_clients: dict[str, genai.Client] = {}
for _key_name, _key_val in _api_keys.items():
    if _key_name.startswith("gemini") and _key_val:
        _gemini_clients[_key_name] = genai.Client(api_key=_key_val)

_gemini_client = next(iter(_gemini_clients.values()), None)


def get_client(model_key: str) -> genai.Client | None:
    """Get the Gemini API client for a model."""
    model_cfg = MODELS.get(model_key, {})
    key_name = model_cfg.get("api_key", "gemini")
    return _gemini_clients.get(key_name, _gemini_client)


# ── Aliases & defaults ────────────────────────────────────────────────
ALIASES: dict[str, str] = {}
for key in MODELS:
    if "flash" in key and "flash" not in ALIASES:
        ALIASES["flash"] = key
    elif "pro" in key and "pro" not in ALIASES:
        ALIASES["pro"] = key

DEFAULT_MODEL = ALIASES.get("flash") or next(iter(MODELS))

# ── Memory config ─────────────────────────────────────────────────────
MEM_CFG = CFG.get("memory", {})
RECALL_LIMIT = MEM_CFG.get("recall_limit", 5)
CHROMA_PATH = MEM_CFG.get("chromadb", {}).get("path", "./data/chromadb")
if CHROMA_PATH.startswith("~"):
    CHROMA_PATH = os.path.expanduser(CHROMA_PATH)
elif not os.path.isabs(CHROMA_PATH):
    CHROMA_PATH = str(BASE_DIR / CHROMA_PATH)
COLLECTION_NAME = MEM_CFG.get("chromadb", {}).get("collection", "companion")

# ── Ollama config ─────────────────────────────────────────────────────
OLLAMA_CFG = MEM_CFG.get("ollama", {})
OLLAMA_URL = OLLAMA_CFG.get("base_url", "http://localhost:11434")
EXTRACTION_MODEL = OLLAMA_CFG.get("extraction_model", "qwen3:1.7b")

# ── Fact extraction fallback (when Ollama not available) ──────────────
EXTRACTION_FALLBACK_CFG = MEM_CFG.get("extraction_fallback", {})
EXTRACTION_FALLBACK_MODEL = EXTRACTION_FALLBACK_CFG.get("model", "gemini-2.5-flash-lite")
EXTRACTION_FALLBACK_API_KEY = EXTRACTION_FALLBACK_CFG.get("api_key", "gemini")

# ── Declutter config ─────────────────────────────────────────────────
DECLUTTER_CFG = MEM_CFG.get("declutter", {})
DECLUTTER_MODEL = DECLUTTER_CFG.get("model", "gemini-2.5-flash-lite")
DECLUTTER_API_KEY = DECLUTTER_CFG.get("api_key", "gemini")

# ── Debug / logging paths ─────────────────────────────────────────────
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "debug.log"

# ── Personality / system prompt ────────────────────────────────────────
with open(BASE_DIR / "config" / "personality.yaml") as _f:
    _personality = yaml.safe_load(_f)
SYSTEM_PROMPT = _personality.get("base", "")

# ── Live API config (optional voice + text) ───────────────────────────
LIVE_CFG = CFG.get("live", {})
LIVE_ENABLED = bool(LIVE_CFG.get("enabled", False))
LIVE_MODEL = LIVE_CFG.get("model", "gemini-2.0-flash-live-001")
LIVE_API_KEY = LIVE_CFG.get("api_key", "gemini")


def get_live_client() -> genai.Client | None:
    """Return the Gemini client for Live API, or None if disabled/unconfigured."""
    if not LIVE_ENABLED:
        return None
    return _gemini_clients.get(LIVE_API_KEY, _gemini_client)


# ── TTS config ───────────────────────────────────────────────────────
TTS_CFG = CFG.get("tts", {})
TTS_BACKEND = TTS_CFG.get("backend", "kokoro")
TTS_VOICE = TTS_CFG.get("voice", "af_heart")
TTS_SPEED = float(TTS_CFG.get("speed", 1.0))
TTS_LANG = TTS_CFG.get("lang", "a")
ELEVENLABS_CFG = TTS_CFG.get("elevenlabs", {})
GOOGLE_TTS_CFG = TTS_CFG.get("google", {})
GEMINI_TTS_CFG = TTS_CFG.get("gemini_tts", {})
