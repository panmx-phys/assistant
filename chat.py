"""Chat engine — orchestrates LLM calls, memory, and slash commands."""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime

import config
import llm
from memory import Memory
from tts import TTS

_EMOTION_RE = re.compile(r'^\s*\[(\w+)\]\s*')
VALID_EMOTIONS = frozenset({
    "neutral", "happy", "sad", "excited", "calm",
    "serious", "playful", "empathetic", "curious", "annoyed",
})


def _strip_emotion(text: str) -> tuple[str, str]:
    """Extract and strip an emotion tag from the start of text.

    Returns (cleaned_text, emotion). Emotion defaults to "neutral".
    """
    m = _EMOTION_RE.match(text)
    if m and m.group(1).lower() in VALID_EMOTIONS:
        return text[m.end():], m.group(1).lower()
    return text, "neutral"


class Chat:
    def __init__(self):
        self._memory = Memory()
        self._history: list[dict[str, str]] = []
        self._model_key = config.DEFAULT_MODEL
        self.tts = TTS()
        self.last_emotion: str = "neutral"

    @property
    def model(self) -> dict:
        return config.MODELS[self._model_key]

    def _resolve_key(self, arg: str) -> str | None:
        """Resolve a model argument — accepts full key or alias (flash/pro)."""
        if arg in config.MODELS:
            return arg
        if arg in config.ALIASES:
            return config.ALIASES[arg]
        return None

    def _build_user_prompt(self, messages: list[dict[str, str]]) -> str:
        """Build conversation text (system_instruction is passed separately)."""
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")
        return "\n\n".join(parts)

    def _prepare(self, user_msg: str) -> tuple[str, list[dict[str, str]]]:
        """Build system prompt with memories/time, append user msg, return (system, recent)."""
        memories = self._memory.recall(user_msg)
        mem_block = ""
        if memories:
            lines = []
            for fact, when in memories:
                lines.append(f"- {fact} ({when})" if when else f"- {fact}")
            mem_block = "\n\nWhat you remember about this user:\n" + "\n".join(lines)

        now = datetime.now()
        tz_name = time.tzname[time.daylight] if time.daylight else time.tzname[0]
        time_str = f"\n\nCurrent date and time: {now.strftime('%A, %B %d, %Y %I:%M %p')} ({tz_name})"

        system = config.SYSTEM_PROMPT + time_str + mem_block
        self._history.append({"role": "user", "content": user_msg})
        return system, self._history[-20:]

    def _store_safe(self, user_msg: str, response: str):
        try:
            self._memory.store(user_msg, response)
        except Exception:
            pass

    def send(self, user_msg: str) -> str:
        system, recent = self._prepare(user_msg)

        model_cfg = self.model
        model_id = model_cfg["model"]
        temperature = model_cfg.get("temperature", 0.7)
        max_tokens = model_cfg.get("max_tokens", 2048)
        client = config.get_client(self._model_key)
        full_prompt = self._build_user_prompt(recent)

        t0 = time.monotonic()
        error = None
        response_text = ""
        try:
            response_text = llm.call_gemini_api(
                model_id, full_prompt, system_prompt=system,
                temperature=temperature, max_tokens=max_tokens,
                client=client,
            )
        except Exception as e:
            error = str(e)
            response_text = f"Error: {e}"
        elapsed = int((time.monotonic() - t0) * 1000)

        response_text, self.last_emotion = _strip_emotion(response_text)

        llm.debug_log.log(
            call_type="chat", model=model_id, prompt=full_prompt,
            response=response_text, elapsed_ms=elapsed, error=error,
        )

        self._history.append({"role": "assistant", "content": response_text})
        self._store_safe(user_msg, response_text)
        return response_text

    def send_stream(self, user_msg: str):
        """Stream a response, yielding text chunks.

        The first chunk(s) are buffered until the emotion tag is parsed out.
        Yields (chunk, emotion) tuples — emotion is set once the tag is found.
        """
        system, recent = self._prepare(user_msg)

        model_cfg = self.model
        model_id = model_cfg["model"]
        temperature = model_cfg.get("temperature", 0.7)
        max_tokens = model_cfg.get("max_tokens", 2048)
        client = config.get_client(self._model_key)
        full_prompt = self._build_user_prompt(recent)

        t0 = time.monotonic()
        chunks: list[str] = []
        error = None
        emotion_parsed = False
        buffer = ""

        try:
            for chunk in llm.stream_gemini_api(
                model_id, full_prompt, system_prompt=system,
                temperature=temperature, max_tokens=max_tokens,
                client=client,
            ):
                chunks.append(chunk)
                if not emotion_parsed:
                    buffer += chunk
                    # Wait until we have enough to check for tag (e.g. "[happy] ")
                    if "]" in buffer or len(buffer) > 30:
                        cleaned, self.last_emotion = _strip_emotion(buffer)
                        emotion_parsed = True
                        if cleaned:
                            yield cleaned
                else:
                    yield chunk
        except Exception as e:
            error = str(e)
            err_msg = f"Error: {e}"
            chunks.append(err_msg)
            yield err_msg

        # If stream was too short to trigger parsing
        if not emotion_parsed and buffer:
            cleaned, self.last_emotion = _strip_emotion(buffer)
            if cleaned:
                yield cleaned

        elapsed = int((time.monotonic() - t0) * 1000)
        response_text = "".join(chunks)
        # Strip emotion from full text for history/memory
        response_text, _ = _strip_emotion(response_text)

        llm.debug_log.log(call_type="chat_stream", model=model_id, prompt=full_prompt,
                          response=response_text, elapsed_ms=elapsed, error=error)

        self._history.append({"role": "assistant", "content": response_text})
        self._store_safe(user_msg, response_text)

    def handle_command(self, msg: str) -> str | None:
        """Handle /commands. Returns response string, or None if not a command."""
        parts = msg.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/model":
            if not arg:
                lines = []
                for key, m in config.MODELS.items():
                    marker = " <--" if key == self._model_key else ""
                    alias = next((a for a, k in config.ALIASES.items() if k == key), "")
                    alias_str = f" ({alias})" if alias else ""
                    lines.append(f"  {key}{alias_str} — {m['description']}{marker}")
                return "Models:\n" + "\n".join(lines) + "\n\nUsage: /model <name|flash|pro>"
            resolved = self._resolve_key(arg)
            if resolved:
                self._model_key = resolved
                return f"Switched to {config.MODELS[resolved]['description']}"
            return f"Unknown model: {arg}. Available: {', '.join(list(config.MODELS) + list(config.ALIASES))}"

        elif cmd == "/remember":
            if not arg:
                return "Usage: /remember <fact>"
            self._memory.store_fact(arg)
            return f"Remembered: {arg}"

        elif cmd == "/memories":
            mems = self._memory.get_all()
            if not mems:
                return "No memories stored yet."
            return "Memories:\n" + "\n".join(f"  - {m}" for m in mems)

        elif cmd == "/clearmemories":
            self._memory.delete_all()
            return "All memories cleared."

        elif cmd == "/clear":
            self._history.clear()
            return "Conversation history cleared (memories preserved)."

        elif cmd == "/tts":
            self.tts.enabled = not self.tts.enabled
            state = "ON" if self.tts.enabled else "OFF"
            if not self.tts.enabled:
                self.tts.stop()
            return f"Text-to-speech {state}"

        elif cmd == "/voice":
            if not arg:
                return (
                    "Voice presets:\n"
                    "  /voice en             Switch to English (af_heart)\n"
                    "  /voice zh             Switch to Chinese (zf_xiaobei)\n"
                    "  /voice <name> [lang]  Custom voice + optional lang code\n"
                    "\n"
                    "Lang codes: a=American, b=British, z=Chinese, j=Japanese,\n"
                    "            e=Spanish, f=French, h=Hindi, i=Italian, p=Portuguese\n"
                    "\n"
                    "English voices: af_heart, af_bella, af_nicole, am_adam, am_michael\n"
                    "Chinese voices: zf_xiaobei, zf_xiaoni, zm_yunjian, zm_yunxi"
                )
            parts_v = arg.split()
            presets = {
                "en": ("af_heart", "a"),
                "english": ("af_heart", "a"),
                "zh": ("zf_xiaobei", "z"),
                "chinese": ("zf_xiaobei", "z"),
                "cn": ("zf_xiaobei", "z"),
                "jp": ("zf_xiaobei", "j"),  # placeholder
                "japanese": ("zf_xiaobei", "j"),
            }
            if parts_v[0].lower() in presets:
                voice, lang_code = presets[parts_v[0].lower()]
                self.tts.set_voice(voice, lang_code)
                return f"Voice: {voice} (lang={lang_code})"
            voice = parts_v[0]
            lang_code = parts_v[1] if len(parts_v) > 1 else None
            self.tts.set_voice(voice, lang_code)
            return f"Voice: {voice}" + (f" (lang={lang_code})" if lang_code else "")

        elif cmd == "/declutter":
            try:
                before, after = self._memory.declutter()
                removed = before - after
                return f"Decluttered: {before} → {after} memories ({removed} removed/merged)."
            except Exception as e:
                return f"Declutter failed: {e}"

        elif cmd == "/backfill":
            scored, total = self._memory.backfill_significance()
            if scored == 0:
                return f"All {total} memories already have significance scores."
            return f"Scored {scored}/{total} memories with significance ratings."

        elif cmd == "/ttsbackend":
            if not arg:
                from tts import BACKENDS
                return (
                    f"Current backend: {self.tts._backend_name if hasattr(self.tts, '_backend_name') else 'kokoro'}\n"
                    f"Available: {', '.join(BACKENDS)}\n"
                    "Usage: /ttsbackend <kokoro|elevenlabs|google>"
                )
            from tts import BACKENDS
            if arg not in BACKENDS:
                return f"Unknown backend: {arg}. Choose from: {', '.join(BACKENDS)}"
            self.tts.set_backend(arg)
            return f"TTS backend switched to: {arg}"

        elif cmd == "/debug":
            llm.debug_log.enabled = not llm.debug_log.enabled
            state = "ON" if llm.debug_log.enabled else "OFF"
            extra = f"\n  Log file: {llm.LOG_FILE}" if llm.debug_log.enabled else ""
            return f"Debug mode {state}{extra}"

        elif cmd == "/help":
            aliases_hint = ", ".join(f"{a}={k}" for a, k in config.ALIASES.items())
            return (
                "Commands:\n"
                "  /model [name]       Switch model or list models\n"
                "  /remember <fact>    Store a fact in memory\n"
                "  /memories           View all stored memories\n"
                "  /clearmemories      Delete all stored memories\n"
                "  /clear              Clear conversation history\n"
                "  /tts                Toggle text-to-speech\n"
                "  /ttsbackend [name]  Switch TTS backend (kokoro, elevenlabs, google)\n"
                "  /voice [preset]     Switch TTS voice/language (en, zh, ...)\n"
                "  /declutter          Merge duplicates & prune low-value memories\n"
                "  /backfill           Score all memories by significance\n"
                "  /debug              Toggle debug logging\n"
                "  /help               Show this help\n"
                "  /quit               Exit\n"
                "\nShortcuts: \\p = pro, \\f = flash\n"
                f"Aliases: {aliases_hint}"
            )

        elif cmd in ("/quit", "/exit", "/q"):
            print("Bye!")
            sys.exit(0)

        return None
