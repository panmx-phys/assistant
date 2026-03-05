"""LLM callers (Gemini API, Ollama) and debug logger."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from google import genai

from config import (
    CFG, LOG_FILE, OLLAMA_URL,
    _gemini_client,
)


# ── Debug logger ───────────────────────────────────────────────────────

class DebugLogger:
    """Logs every LLM call to a JSON-lines file with full request/response and timing."""

    def __init__(self, path: Path, enabled: bool = False):
        self._path = path
        self.enabled = enabled

    def log(
        self,
        *,
        call_type: str,
        model: str,
        prompt: str,
        response: str,
        elapsed_ms: int,
        error: str | None = None,
    ):
        if not self.enabled:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "call_type": call_type,
            "model": model,
            "elapsed_ms": elapsed_ms,
            "prompt": prompt,
        }
        if error:
            entry["error"] = error
        entry["response"] = response

        with open(self._path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        status = "ERROR" if error else "OK"
        print(f"  [debug] {call_type} | {model} | {elapsed_ms}ms | {status}")


debug_log = DebugLogger(LOG_FILE, enabled=CFG.get("debug", False))


# ── Gemini API caller ─────────────────────────────────────────────────

def call_gemini_api(model: str, prompt: str, system_prompt: str = "",
                    temperature: float = 0.7, max_tokens: int = 2048,
                    client: genai.Client | None = None) -> str:
    """Call the Gemini API via google-genai SDK and return the response text."""
    cli = client or _gemini_client
    if not cli:
        raise RuntimeError("GEMINI_API_KEY not set — cannot use Gemini API")

    config = genai.types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if system_prompt:
        config.system_instruction = system_prompt

    response = cli.models.generate_content(
        model=model, contents=prompt, config=config,
    )
    return response.text.strip() if response.text else ""


def stream_gemini_api(model: str, prompt: str, system_prompt: str = "",
                      temperature: float = 0.7, max_tokens: int = 2048,
                      client: genai.Client | None = None):
    """Stream the Gemini API response, yielding text chunks as they arrive."""
    cli = client or _gemini_client
    if not cli:
        raise RuntimeError("GEMINI_API_KEY not set — cannot use Gemini API")

    config = genai.types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if system_prompt:
        config.system_instruction = system_prompt

    for chunk in cli.models.generate_content_stream(
        model=model, contents=prompt, config=config,
    ):
        if chunk.text:
            yield chunk.text


# ── Ollama caller ──────────────────────────────────────────────────────

def call_ollama(model: str, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
    """Call Ollama's generate endpoint and return the response text."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()
