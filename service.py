#!/usr/bin/env python3
"""FastAPI service wrapping the Chat engine — exposes SSE streaming + commands."""
from __future__ import annotations

import base64
import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from chat import Chat
import config
from live import run_live_turn

# ── Lifespan ──────────────────────────────────────────────────────────

chat = Chat()
_chat_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    chat.tts.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Request / Response models ─────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    use_live: bool = False  # use Gemini Live API for voice + text (optional)


class CommandRequest(BaseModel):
    command: str


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def models():
    model_info = {}
    for key, m in config.MODELS.items():
        model_info[key] = {
            "description": m.get("description", ""),
            "provider": m.get("provider", ""),
            "model": m.get("model", ""),
        }
    return {
        "models": model_info,
        "aliases": config.ALIASES,
        "current": chat._model_key,
        "current_description": chat.model.get("description", chat.model["model"]),
        "live": {
            "enabled": config.LIVE_ENABLED,
            "model": config.LIVE_MODEL if config.LIVE_ENABLED else None,
        },
    }


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    use_live = req.use_live and config.LIVE_ENABLED
    live_client = config.get_live_client() if use_live else None

    async def generate():
        if use_live and live_client:
            # Gemini Live path: stream text + audio from Live API (no TTS)
            async with _chat_lock:
                system, recent = chat._prepare(req.message)
                full_prompt = chat._build_user_prompt(recent)
                text_parts: list[str] = []
                try:
                    async for kind, data in run_live_turn(
                        live_client,
                        config.LIVE_MODEL,
                        system,
                        full_prompt,
                    ):
                        if kind == "text":
                            text_parts.append(data)
                            yield {"data": json.dumps({"chunk": data})}
                        elif kind == "audio":
                            yield {"data": json.dumps({"audio": base64.b64encode(data).decode()})}
                        elif kind == "error":
                            yield {"data": json.dumps({"error": data})}
                        elif kind == "done":
                            break
                    response_text = "".join(text_parts)
                    chat._history.append({"role": "assistant", "content": response_text})
                    chat._store_safe(req.message, response_text)
                    yield {"data": json.dumps({"done": True, "model": config.LIVE_MODEL, "live": True})}
                except Exception as e:
                    yield {"data": json.dumps({"error": str(e)})}
                    yield {"data": json.dumps({"done": True, "model": config.LIVE_MODEL, "live": True})}
            return

        # Default: stream text + TTS
        async with _chat_lock:
            label = chat.model.get("description", chat.model["model"])
            q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _stream():
                try:
                    chat.tts.start()
                    first_chunk = True
                    for chunk in chat.send_stream(req.message):
                        if first_chunk:
                            chat.tts.set_emotion(chat.last_emotion)
                            first_chunk = False
                        chat.tts.feed(chunk)
                        asyncio.run_coroutine_threadsafe(q.put(("chunk", chunk)), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(q.put(("error", str(e))), loop)
                finally:
                    chat.tts.finish()
                    asyncio.run_coroutine_threadsafe(q.put(("done", label)), loop)

            t = threading.Thread(target=_stream, daemon=True)
            t.start()

            while True:
                kind, data = await q.get()
                if kind == "chunk":
                    yield {"data": json.dumps({"chunk": data})}
                elif kind == "error":
                    yield {"data": json.dumps({"error": data})}
                elif kind == "done":
                    yield {"data": json.dumps({"done": True, "model": data, "emotion": chat.last_emotion})}
                    break

    return EventSourceResponse(generate())


@app.post("/command")
async def command_endpoint(req: CommandRequest):
    cmd_str = req.command.strip()

    # Model shortcuts
    if cmd_str in ("\\p", "\\f"):
        alias = "pro" if cmd_str == "\\p" else "flash"
        result = chat.handle_command(f"/model {alias}")
        return {"result": result or ""}

    # Intercept quit before handle_command (which calls sys.exit)
    parts = cmd_str.split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    if cmd in ("/quit", "/exit", "/q"):
        return {"result": "Bye!", "action": "quit"}

    result = chat.handle_command(cmd_str)
    if result is None:
        return {"result": "", "not_command": True}
    return {"result": result}


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("CORTANA_PORT", "8391"))
    uvicorn.run("service:app", host="127.0.0.1", port=port, log_level="warning")
