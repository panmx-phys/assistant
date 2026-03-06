"""Gemini Live API — optional real-time voice + text alongside TTS.

When config live.enabled is true, clients can request use_live on /chat to get
streamed text and native audio from the Live API instead of the TTS pipeline.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Literal

from google import genai
from google.genai import types

# Event type: "text" | "audio" | "done" | "error"
LiveEventKind = Literal["text", "audio", "done", "error"]


async def run_live_turn(
    client: genai.Client,
    model: str,
    system_instruction: str,
    user_message: str,
) -> AsyncIterator[tuple[LiveEventKind, str | bytes]]:
    """Run one user turn through the Live API; yield (kind, data) events.

    Yields:
        ("text", str) — incremental text from the model.
        ("audio", bytes) — PCM audio chunk (24 kHz, format per API).
        ("done", b"") — turn finished.
        ("error", str) — error message then done.
    """
    config = types.LiveConnectConfig(
        system_instruction=system_instruction,
        response_modalities=[types.Modality.TEXT, types.Modality.AUDIO],
    )
    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            await session.send(
                input=types.Content(
                    role="user",
                    parts=[types.Part(text=user_message)],
                ),
                end_of_turn=True,
            )
            async for message in session.receive():
                if message.server_content:
                    sc = message.server_content
                    if sc.model_turn and sc.model_turn.parts:
                        for part in sc.model_turn.parts:
                            if part.text:
                                yield ("text", part.text)
                            if part.inline_data and part.inline_data.data:
                                yield ("audio", bytes(part.inline_data.data))
                    if getattr(sc, "turn_complete", False) or getattr(
                        sc, "generation_complete", False
                    ):
                        yield ("done", b"")
                        return
            yield ("done", b"")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield ("error", str(e))
        yield ("done", b"")
