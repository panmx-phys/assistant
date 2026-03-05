"""Text-to-speech voice server — runs as a separate process, controlled via Unix socket.

Protocol (newline-delimited JSON over Unix socket):
  {"cmd": "start"}                          — begin a new streaming utterance
  {"cmd": "feed", "text": …}               — push a text chunk (sentences are flushed automatically)
  {"cmd": "finish"}                         — flush remaining text buffer
  {"cmd": "speak", "text": …}              — one-shot: stop current + speak full text
  {"cmd": "stop"}                           — interrupt current playback
  {"cmd": "voice", "voice": …, "lang": …}  — switch voice/language at runtime
  {"cmd": "backend", "backend": …}         — switch TTS backend at runtime
  {"cmd": "emotion", "emotion": …}         — set emotion hint for synthesis
  {"cmd": "quit"}                           — shut down the server

The server sends back: {"status": "ok"} for each command received.

Usage:
  # Start server:   python tts.py [--socket /tmp/tts.sock] [--voice af_heart] [--lang a] [--speed 1.0]
  # From client:    echo '{"cmd":"speak","text":"Hello"}' | socat - UNIX-CONNECT:/tmp/tts.sock
"""
from __future__ import annotations

import abc
import argparse
import io
import json
import os
import queue
import re
import socket
import struct
import sys
import threading

import numpy as np

_SENTENCE_END = re.compile(r'[.!?;:\n]')
DEFAULT_SOCKET = "/tmp/tts.sock"

BACKENDS = ("kokoro", "elevenlabs", "google", "gemini")


def _clean(text: str) -> str:
    return text.replace("**", "").replace("*", "").replace("`", "").replace("#", "")


# ── Backend interface ─────────────────────────────────────────────────

class TTSBackend(abc.ABC):
    """Base class for TTS synthesis backends.

    Each backend converts text → numpy float32 audio array at 24 kHz.
    """

    @abc.abstractmethod
    def synthesize(self, text: str, voice: str, speed: float,
                   emotion: str = "neutral") -> np.ndarray | None:
        """Return a float32 numpy array of audio samples at 24 kHz, or None on failure."""

    def warmup(self):
        """Optional pre-loading hook (called in a background thread)."""


# ── Kokoro backend (local) ────────────────────────────────────────────

class KokoroBackend(TTSBackend):
    def __init__(self, lang: str = "a"):
        self.lang = lang
        self._pipelines: dict[str, object] = {}

    def _ensure_env(self):
        import warnings
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    def _get_pipeline(self, lang: str | None = None):
        lang = lang or self.lang
        if lang not in self._pipelines:
            self._ensure_env()
            import torch
            from huggingface_hub import utils
            utils.disable_progress_bars()
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            from kokoro import KPipeline
            self._pipelines[lang] = KPipeline(lang_code=lang, device=device)
        return self._pipelines[lang]

    def warmup(self):
        self._get_pipeline()

    def synthesize(self, text: str, voice: str, speed: float,
                   emotion: str = "neutral") -> np.ndarray | None:
        # Kokoro doesn't support emotion control — ignored
        pipeline = self._get_pipeline()
        chunks = []
        for _gs, _ps, audio in pipeline(text, voice=voice, speed=speed):
            if audio is not None and len(audio) > 0:
                chunks.append(np.asarray(audio, dtype=np.float32))
        if chunks:
            return np.concatenate(chunks)
        return None


# ── ElevenLabs backend (online API) ───────────────────────────────────

class ElevenLabsBackend(TTSBackend):
    """Uses the ElevenLabs text-to-speech REST API.

    Requires `httpx` for HTTP requests. Audio is returned as MP3 and
    decoded to PCM float32 via the `miniaudio` library.
    """

    API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

    def __init__(self, api_key: str = "", model: str = "eleven_multilingual_v2",
                 voice_id: str = "Rachel", speed: float = 0):
        self.api_key = api_key
        self.model = model
        self.voice_id = voice_id
        self.speed = speed
        if not self.api_key:
            print("Warning: ELEVENLABS_API_KEY not set — ElevenLabs backend will fail.")

    # Map emotion → ElevenLabs voice_settings tweaks (stability, similarity_boost, style)
    _EMOTION_SETTINGS = {
        "neutral":    {"stability": 0.50, "similarity_boost": 0.75, "style": 0.0},
        "happy":      {"stability": 0.35, "similarity_boost": 0.80, "style": 0.6},
        "excited":    {"stability": 0.25, "similarity_boost": 0.85, "style": 0.8},
        "sad":        {"stability": 0.70, "similarity_boost": 0.60, "style": 0.4},
        "calm":       {"stability": 0.75, "similarity_boost": 0.70, "style": 0.1},
        "serious":    {"stability": 0.65, "similarity_boost": 0.70, "style": 0.1},
        "playful":    {"stability": 0.30, "similarity_boost": 0.80, "style": 0.7},
        "empathetic": {"stability": 0.60, "similarity_boost": 0.75, "style": 0.5},
        "curious":    {"stability": 0.40, "similarity_boost": 0.75, "style": 0.4},
        "annoyed":    {"stability": 0.55, "similarity_boost": 0.65, "style": 0.5},
    }

    def synthesize(self, text: str, voice: str, speed: float,
                   emotion: str = "neutral") -> np.ndarray | None:
        import httpx

        vid = voice or self.voice_id
        spd = self.speed or speed
        url = f"{self.API_URL}/{vid}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        settings = self._EMOTION_SETTINGS.get(emotion, self._EMOTION_SETTINGS["neutral"])
        body = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": settings["stability"],
                "similarity_boost": settings["similarity_boost"],
                "style": settings["style"],
                "speed": spd,
            },
        }
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=30.0)
            resp.raise_for_status()
            return self._decode_mp3(resp.content)
        except Exception as e:
            print(f"ElevenLabs error: {e}")
            return None

    @staticmethod
    def _decode_mp3(data: bytes) -> np.ndarray | None:
        """Decode MP3 bytes to float32 numpy array at 24 kHz."""
        try:
            import miniaudio
            decoded = miniaudio.decode(data, output_format=miniaudio.SampleFormat.FLOAT32,
                                       nchannels=1, sample_rate=24000)
            return np.frombuffer(decoded.samples, dtype=np.float32)
        except ImportError:
            # Fallback: try pydub + ffmpeg
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_mp3(io.BytesIO(data))
                seg = seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
                samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32)
                return samples / 32768.0
            except Exception:
                print("Error: Install 'miniaudio' or 'pydub' to decode ElevenLabs audio.")
                return None


# ── Google Cloud TTS backend (online API) ─────────────────────────────

class GoogleCloudBackend(TTSBackend):
    """Uses the Google Cloud Text-to-Speech REST API (v1).

    Uses API key auth (simpler than service account) via the
    texttospeech.googleapis.com/v1 REST endpoint.
    """

    API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

    # Voices that don't support SSML (use plain text fallback)
    _NO_SSML_PREFIXES = ("Journey", "Chirp")

    # Map emotion → SSML prosody adjustments (pitch, rate, volume offsets)
    _EMOTION_PROSODY = {
        "neutral":    {},
        "happy":      {"pitch": "+2st", "rate": "105%", "volume": "+1dB"},
        "excited":    {"pitch": "+4st", "rate": "115%", "volume": "+2dB"},
        "sad":        {"pitch": "-2st", "rate": "85%",  "volume": "-2dB"},
        "calm":       {"pitch": "-1st", "rate": "90%",  "volume": "-1dB"},
        "serious":    {"pitch": "-1st", "rate": "95%",  "volume": "+0dB"},
        "playful":    {"pitch": "+3st", "rate": "110%", "volume": "+1dB"},
        "empathetic": {"pitch": "-1st", "rate": "92%",  "volume": "-1dB"},
        "curious":    {"pitch": "+2st", "rate": "102%", "volume": "+0dB"},
        "annoyed":    {"pitch": "+1st", "rate": "105%", "volume": "+2dB"},
    }

    def __init__(self, api_key: str = "", language_code: str = "en-US",
                 voice_name: str = "en-US-Journey-F", speed: float = 0):
        self.api_key = api_key
        self.language_code = language_code
        self.voice_name = voice_name
        self.speed = speed
        if not self.api_key:
            print("Warning: GOOGLE_TTS_API_KEY not set — Google TTS backend will fail.")

    def _supports_ssml(self, voice_name: str) -> bool:
        """Check if the voice supports SSML (Journey/Chirp voices don't)."""
        for prefix in self._NO_SSML_PREFIXES:
            if prefix in voice_name:
                return False
        return True

    def _wrap_ssml(self, text: str, emotion: str) -> str:
        """Wrap text in SSML with prosody adjustments for the given emotion."""
        prosody = self._EMOTION_PROSODY.get(emotion, {})
        if not prosody:
            return f"<speak>{text}</speak>"
        attrs = " ".join(f'{k}="{v}"' for k, v in prosody.items())
        return f"<speak><prosody {attrs}>{text}</prosody></speak>"

    def synthesize(self, text: str, voice: str, speed: float,
                   emotion: str = "neutral") -> np.ndarray | None:
        import httpx

        vname = voice or self.voice_name
        spd = self.speed or speed
        url = f"{self.API_URL}?key={self.api_key}"

        use_ssml = self._supports_ssml(vname) and emotion != "neutral"
        if use_ssml:
            input_field = {"ssml": self._wrap_ssml(text, emotion)}
        else:
            input_field = {"text": text}

        body = {
            "input": input_field,
            "voice": {
                "languageCode": self.language_code,
                "name": vname,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": 24000,
                "speakingRate": spd,
            },
        }
        try:
            resp = httpx.post(url, json=body, timeout=30.0)
            resp.raise_for_status()
            import base64
            audio_b64 = resp.json().get("audioContent", "")
            raw = base64.b64decode(audio_b64)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            return samples / 32768.0
        except Exception as e:
            print(f"Google TTS error: {e}")
            return None


# ── Gemini 2.5 Flash TTS backend (online API) ────────────────────────

class GeminiTTSBackend(TTSBackend):
    """Uses the Gemini 2.5 Flash Preview TTS model via the google-genai SDK.

    Returns PCM int16 at 24 kHz which is converted to float32.
    Emotion is conveyed via a prompt prefix (the model follows style cues).
    """

    _EMOTION_PREFIX = {
        "neutral":    "",
        "happy":      "Say this cheerfully and warmly: ",
        "excited":    "Say this with great excitement and energy: ",
        "sad":        "Say this in a somber, melancholic tone: ",
        "calm":       "Say this in a calm, soothing voice: ",
        "serious":    "Say this in a serious, measured tone: ",
        "playful":    "Say this in a playful, lighthearted way: ",
        "empathetic": "Say this with empathy and compassion: ",
        "curious":    "Say this with curiosity and wonder: ",
        "annoyed":    "Say this with mild irritation: ",
    }

    def __init__(self, api_key: str = "", model: str = "gemini-2.5-flash-preview-tts",
                 voice_name: str = "Kore", speed: float = 0):
        self.model = model
        self.voice_name = voice_name
        self.speed = speed
        self._client = None
        self._api_key = api_key
        if not self._api_key:
            print("Warning: No API key for Gemini TTS — backend will fail.")

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def synthesize(self, text: str, voice: str, speed: float,
                   emotion: str = "neutral") -> np.ndarray | None:
        from google.genai import types

        client = self._ensure_client()
        vname = voice if voice and not voice.startswith("af_") else self.voice_name

        prefix = self._EMOTION_PREFIX.get(emotion, "")
        prompt = prefix + text

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=vname,
                            )
                        )
                    ),
                ),
            )
            data = response.candidates[0].content.parts[0].inline_data.data
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio = samples / 32768.0
            # Speed via resampling (no native support)
            spd = self.speed or speed
            if spd and abs(spd - 1.0) > 0.05:
                n = int(len(audio) / spd)
                audio = np.interp(np.linspace(0, len(audio) - 1, n),
                                  np.arange(len(audio)), audio)
            return audio
        except Exception as e:
            print(f"Gemini TTS error: {e}")
            return None


# ── Backend factory ───────────────────────────────────────────────────

def create_backend(name: str, lang: str = "a") -> TTSBackend:
    """Instantiate a TTS backend by name."""
    if name == "kokoro":
        return KokoroBackend(lang=lang)
    elif name == "elevenlabs":
        from config import ELEVENLABS_CFG
        return ElevenLabsBackend(
            api_key=ELEVENLABS_CFG.get("api_key", ""),
            model=ELEVENLABS_CFG.get("model", "eleven_multilingual_v2"),
            voice_id=ELEVENLABS_CFG.get("voice_id", "Rachel"),
            speed=float(ELEVENLABS_CFG.get("speed", 0)),
        )
    elif name == "google":
        from config import GOOGLE_TTS_CFG
        return GoogleCloudBackend(
            api_key=GOOGLE_TTS_CFG.get("api_key", ""),
            language_code=GOOGLE_TTS_CFG.get("language_code", "en-US"),
            voice_name=GOOGLE_TTS_CFG.get("voice_name", "en-US-Journey-F"),
            speed=float(GOOGLE_TTS_CFG.get("speed", 0)),
        )
    elif name == "gemini":
        from config import GEMINI_TTS_CFG
        return GeminiTTSBackend(
            api_key=GEMINI_TTS_CFG.get("api_key", ""),
            model=GEMINI_TTS_CFG.get("model", "gemini-2.5-flash-preview-tts"),
            voice_name=GEMINI_TTS_CFG.get("voice_name", "Kore"),
            speed=float(GEMINI_TTS_CFG.get("speed", 0)),
        )
    else:
        raise ValueError(f"Unknown TTS backend: {name!r}. Choose from: {BACKENDS}")


# ── Engine (unchanged public API, now delegates to backend) ───────────

class TTSEngine:
    """Synthesis + playback engine (runs in-process threads)."""

    def __init__(self, voice: str = "af_heart", lang: str = "a", speed: float = 1.0,
                 backend_name: str = "kokoro"):
        self.voice = voice
        self.lang = lang
        self.speed = max(0.5, min(2.0, speed))
        self.emotion = "neutral"
        self.backend_name = backend_name
        self._backend: TTSBackend = create_backend(backend_name, lang=lang)
        self._text_buf = ""
        self._state_lock = threading.Lock()
        self._synth_q: queue.Queue[str | None] = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue()
        self._stop_ev = threading.Event()
        self._synth_t: threading.Thread | None = None
        self._play_t: threading.Thread | None = None

    def set_voice(self, voice: str, lang: str | None = None):
        with self._state_lock:
            if lang:
                self.lang = lang
                if isinstance(self._backend, KokoroBackend):
                    self._backend.lang = lang
            self.voice = voice

    def set_emotion(self, emotion: str):
        with self._state_lock:
            self.emotion = emotion

    def set_backend(self, name: str):
        """Switch to a different TTS backend at runtime."""
        if name not in BACKENDS:
            print(f"Unknown backend: {name!r}. Choose from: {BACKENDS}")
            return
        with self._state_lock:
            self.backend_name = name
            self._backend = create_backend(name, lang=self.lang)

    def _kill_playback(self):
        import sounddevice as sd
        self._stop_ev.set()
        self._synth_q.put(None)
        self._audio_q.put(None)
        try:
            sd.stop()
        except Exception:
            pass
        if self._synth_t and self._synth_t.is_alive():
            self._synth_t.join(timeout=1)
        if self._play_t and self._play_t.is_alive():
            self._play_t.join(timeout=1)
        self._synth_t = self._play_t = None
        for q in (self._synth_q, self._audio_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _synth_loop(self):
        while not self._stop_ev.is_set():
            try:
                txt = self._synth_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if txt is None:
                self._audio_q.put(None)
                break
            try:
                with self._state_lock:
                    voice, speed, emotion = self.voice, self.speed, self.emotion
                    backend = self._backend
                audio = backend.synthesize(txt, voice, speed, emotion=emotion)
                if self._stop_ev.is_set():
                    self._audio_q.put(None)
                    return
                if audio is not None and len(audio) > 0:
                    self._audio_q.put(audio)
            except Exception:
                pass

    def _play_loop(self):
        import sounddevice as sd
        while not self._stop_ev.is_set():
            try:
                audio = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if audio is None:
                break
            try:
                sd.play(audio, samplerate=24000, blocking=True)
                if self._stop_ev.is_set():
                    sd.stop()
                    return
            except Exception:
                pass

    def _start_threads(self):
        self._stop_ev.clear()
        self._synth_t = threading.Thread(target=self._synth_loop, daemon=True)
        self._play_t = threading.Thread(target=self._play_loop, daemon=True)
        self._synth_t.start()
        self._play_t.start()

    def _flush_sentences(self):
        with self._state_lock:
            while _SENTENCE_END.search(self._text_buf):
                m = _SENTENCE_END.search(self._text_buf)
                sentence = self._text_buf[:m.end()]
                self._text_buf = self._text_buf[m.end():]
                cleaned = _clean(sentence).strip()
                if cleaned:
                    self._synth_q.put(cleaned)

    # ── Commands ──────────────────────────────────────────────────────

    def start(self):
        self._kill_playback()
        with self._state_lock:
            self._text_buf = ""
        self._start_threads()

    def feed(self, text: str):
        with self._state_lock:
            self._text_buf += text
        self._flush_sentences()

    def finish(self):
        with self._state_lock:
            leftover = _clean(self._text_buf).strip()
            self._text_buf = ""
        if leftover:
            self._synth_q.put(leftover)
        self._synth_q.put(None)

    def speak(self, text: str):
        self._kill_playback()
        with self._state_lock:
            self._text_buf = ""
        cleaned = _clean(text).strip()
        if cleaned:
            self._start_threads()
            self._synth_q.put(cleaned)
            self._synth_q.put(None)

    def stop(self):
        self._kill_playback()
        with self._state_lock:
            self._text_buf = ""


# ── Socket server ─────────────────────────────────────────────────────

def _handle_client(conn: socket.socket, engine: TTSEngine, shutdown_ev: threading.Event):
    """Handle one client connection (reads newline-delimited JSON commands)."""
    buf = b""
    with conn:
        while not shutdown_ev.is_set():
            try:
                data = conn.recv(4096)
            except OSError:
                break
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cmd = msg.get("cmd", "")
                if cmd == "start":
                    engine.start()
                elif cmd == "feed":
                    engine.feed(msg.get("text", ""))
                elif cmd == "finish":
                    engine.finish()
                elif cmd == "speak":
                    engine.speak(msg.get("text", ""))
                elif cmd == "stop":
                    engine.stop()
                elif cmd == "voice":
                    engine.set_voice(msg.get("voice", engine.voice),
                                     msg.get("lang"))
                elif cmd == "backend":
                    engine.set_backend(msg.get("backend", engine.backend_name))
                elif cmd == "emotion":
                    engine.set_emotion(msg.get("emotion", "neutral"))
                elif cmd == "quit":
                    shutdown_ev.set()
                try:
                    conn.sendall(json.dumps({"status": "ok"}).encode() + b"\n")
                except OSError:
                    break


def serve(sock_path: str, voice: str, lang: str, speed: float, backend: str = "kokoro"):
    """Run the TTS voice server."""
    engine = TTSEngine(voice=voice, lang=lang, speed=speed, backend_name=backend)
    shutdown_ev = threading.Event()

    # Pre-load the backend in background so first request is faster
    threading.Thread(target=engine._backend.warmup, daemon=True).start()

    if os.path.exists(sock_path):
        os.unlink(sock_path)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(4)
    srv.settimeout(1.0)

    print(f"TTS voice server listening on {sock_path}")
    print(f"  backend={backend}  voice={voice}  lang={lang}  speed={speed}")

    try:
        while not shutdown_ev.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(
                target=_handle_client,
                args=(conn, engine, shutdown_ev),
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        pass
    finally:
        # Silence PortAudio/sounddevice errors during teardown
        import contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            engine.stop()
        srv.close()
        try:
            os.unlink(sock_path)
        except OSError:
            pass
        print("\nTTS server stopped.")


# ── Client (used by chat.py) ─────────────────────────────────────────

class TTS:
    """Lightweight client that sends commands to the TTS voice server.

    Matches the old TTS public API so callers don't need to change.
    If the server isn't running, all calls are silent no-ops.
    """

    def __init__(self, socket_path: str = DEFAULT_SOCKET, **_kw):
        self._sock_path = socket_path
        self._enabled = True
        self._conn: socket.socket | None = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        if not value:
            self._disconnect()

    def _connect(self) -> socket.socket | None:
        if self._conn is not None:
            return self._conn
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(self._sock_path)
            self._conn = s
            return s
        except OSError:
            return None

    def _disconnect(self):
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                except OSError:
                    pass
                self._conn = None

    def _send(self, msg: dict):
        if not self._enabled:
            return
        with self._lock:
            conn = self._connect()
            if conn is None:
                return
            try:
                conn.sendall(json.dumps(msg).encode() + b"\n")
                # read ack (non-critical)
                conn.recv(256)
            except OSError:
                self._conn = None

    def start(self):
        self._send({"cmd": "start"})

    def feed(self, text: str):
        self._send({"cmd": "feed", "text": text})

    def finish(self):
        self._send({"cmd": "finish"})

    def speak(self, text: str):
        self._send({"cmd": "speak", "text": text})

    def stop(self):
        self._send({"cmd": "stop"})

    def set_voice(self, voice: str, lang: str | None = None):
        msg: dict = {"cmd": "voice", "voice": voice}
        if lang:
            msg["lang"] = lang
        self._send(msg)

    def set_emotion(self, emotion: str):
        self._send({"cmd": "emotion", "emotion": emotion})

    def set_backend(self, backend: str):
        self._send({"cmd": "backend", "backend": backend})


# ── CLI entry point ───────────────────────────────────────────────────

def main():
    from config import TTS_VOICE, TTS_SPEED, TTS_LANG, TTS_BACKEND

    parser = argparse.ArgumentParser(description="TTS voice server")
    parser.add_argument("--socket", default=DEFAULT_SOCKET, help="Unix socket path")
    parser.add_argument("--backend", default=TTS_BACKEND, choices=BACKENDS,
                        help="TTS backend (kokoro, elevenlabs, google)")
    parser.add_argument("--voice", default=TTS_VOICE, help="Voice name")
    parser.add_argument("--lang", default=TTS_LANG, help="Language code")
    parser.add_argument("--speed", type=float, default=TTS_SPEED, help="Speech speed (0.5–2.0)")
    args = parser.parse_args()
    serve(args.socket, args.voice, args.lang, args.speed, backend=args.backend)


if __name__ == "__main__":
    main()
