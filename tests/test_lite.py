"""Tests for lite.py — Memory, Chat commands, debug logging, and model switching."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import chromadb
import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ──────────────────────────────────────────────────────────


class FakeEmbeddingFunction(chromadb.EmbeddingFunction):
    """Deterministic fake embeddings — avoids downloading ONNX model."""

    def __call__(self, input):
        return [[float(hash(t) % 1000) / 1000.0] * 384 for t in input]


_fake_embed = FakeEmbeddingFunction()

# Patch get_or_create_collection globally so ChromaDB never downloads ONNX
from chromadb.api.client import Client as _ChromaClient
_original_get_or_create = _ChromaClient.get_or_create_collection


def _patched_get_or_create(self, name, **kwargs):
    kwargs.setdefault("embedding_function", _fake_embed)
    return _original_get_or_create(self, name, **kwargs)


_real_gemini_client = None
_real_gemini_clients = None

@pytest.fixture(autouse=True)
def _patch_lite_globals(tmp_path, monkeypatch):
    """Patch module globals so tests use temp dirs and don't hit real services."""
    import config
    import llm

    global _real_gemini_client, _real_gemini_clients
    _real_gemini_client = config._gemini_client
    _real_gemini_clients = config._gemini_clients.copy()

    _fake_client = object()
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chromadb"))
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_lite")
    monkeypatch.setattr(config, "LOG_FILE", tmp_path / "debug.log")
    monkeypatch.setattr(llm, "debug_log", llm.DebugLogger(tmp_path / "debug.log"))
    # Ensure API path is used in unit tests (real clients for integration tests)
    monkeypatch.setattr(config, "_gemini_client", _fake_client)
    monkeypatch.setattr(config, "_gemini_clients", {"gemini": _fake_client})
    monkeypatch.setattr(
        _ChromaClient, "get_or_create_collection", _patched_get_or_create
    )


@pytest.fixture
def memory():
    import lite
    return lite.Memory()


@pytest.fixture
def chat():
    import lite
    return lite.Chat()


# ── Memory: store_fact + get_all ─────────────────────────────────────


def test_store_fact_and_get_all(memory):
    memory.store_fact("User's name is Alice")
    memory.store_fact("User likes hiking")

    all_mems = memory.get_all()
    assert len(all_mems) == 2
    assert "User's name is Alice" in all_mems
    assert "User likes hiking" in all_mems


def test_get_all_empty(memory):
    assert memory.get_all() == []


# ── Memory: recall ───────────────────────────────────────────────────


def test_recall_empty(memory):
    assert memory.recall("anything") == []


def test_recall_returns_results(memory):
    memory.store_fact("User's name is Alice")
    memory.store_fact("User likes hiking")
    memory.store_fact("User prefers dark mode")

    results = memory.recall("what is the user's name", limit=2)
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_recall_limit_respected(memory):
    for i in range(10):
        memory.store_fact(f"Fact number {i}")

    results = memory.recall("facts", limit=3)
    assert len(results) == 3


# ── Memory: store (Ollama fact extraction) ───────────────────────────


@patch("llm.call_ollama")
def test_store_extracts_facts(mock_ollama, memory):
    mock_ollama.return_value = "User's name is Bob\nUser likes cats"

    # Use _store_sync directly to avoid thread timing issues
    memory._store_sync("My name is Bob and I love cats", "Nice to meet you Bob!")

    all_mems = memory.get_all()
    assert len(all_mems) == 2
    texts = set(all_mems)
    assert "User's name is Bob" in texts
    assert "User likes cats" in texts


@patch("llm.call_ollama")
def test_store_no_facts(mock_ollama, memory):
    mock_ollama.return_value = "NONE"

    memory.store("hello", "hi there")
    assert memory.get_all() == []


@patch("llm.call_ollama")
def test_store_handles_extraction_failure(mock_ollama, memory):
    mock_ollama.side_effect = Exception("connection refused")

    memory.store("test", "test")
    assert memory.get_all() == []


@patch("llm.call_ollama")
def test_extract_facts_strips_bullets(mock_ollama, memory):
    mock_ollama.return_value = "- Likes Python\n• Prefers VSCode\n* Uses Mac"

    facts = memory._extract_facts("I use Python in VSCode on Mac", "Cool setup!")
    assert len(facts) == 3
    fact_texts = [f["fact"] for f in facts]
    assert "Likes Python" in fact_texts
    assert "Prefers VSCode" in fact_texts
    assert "Uses Mac" in fact_texts


# ── Memory: delete_all ───────────────────────────────────────────────


def test_delete_all(memory):
    memory.store_fact("fact 1")
    memory.store_fact("fact 2")
    assert len(memory.get_all()) == 2

    memory.delete_all()
    assert memory.get_all() == []


def test_delete_all_when_empty(memory):
    memory.delete_all()
    assert memory.get_all() == []


# ── Memory: persistence across instances ─────────────────────────────


def test_memory_persists_across_instances():
    import lite

    mem1 = lite.Memory()
    mem1.store_fact("User's name is Alice")
    assert len(mem1.get_all()) == 1
    del mem1

    mem2 = lite.Memory()
    all_mems = mem2.get_all()
    assert len(all_mems) == 1
    assert "Alice" in all_mems[0]


# ── Chat: /model command ────────────────────────────────────────────


def test_model_list(chat):
    result = chat.handle_command("/model")
    assert result is not None
    assert "gemini_flash_cli" in result or "flash" in result


def test_model_switch_alias(chat):
    result = chat.handle_command("/model pro")
    assert result is not None
    assert "Pro" in result or "pro" in result


def test_model_switch_full_key(chat):
    import lite
    first_key = next(iter(lite.MODELS))
    result = chat.handle_command(f"/model {first_key}")
    assert result is not None
    assert first_key in result or "Switched" in result


def test_model_unknown(chat):
    result = chat.handle_command("/model nonexistent")
    assert "Unknown model" in result


# ── Chat: /remember + /memories ──────────────────────────────────────


def test_remember_and_memories(chat):
    result = chat.handle_command("/remember User likes Python")
    assert "Remembered" in result

    result = chat.handle_command("/memories")
    assert "User likes Python" in result


def test_remember_empty(chat):
    result = chat.handle_command("/remember")
    assert "Usage" in result


def test_memories_empty(chat):
    result = chat.handle_command("/memories")
    assert "No memories" in result


# ── Chat: /clearmemories ─────────────────────────────────────────────


def test_clearmemories(chat):
    chat.handle_command("/remember test fact")
    result = chat.handle_command("/clearmemories")
    assert "cleared" in result.lower()

    result = chat.handle_command("/memories")
    assert "No memories" in result


# ── Chat: /clear ─────────────────────────────────────────────────────


@patch("llm.call_gemini_api", return_value="Hello!")
@patch("llm.call_ollama", return_value="NONE")
def test_clear_history(mock_ollama, mock_gemini, chat):
    chat.send("hi")
    assert len(chat._history) == 2  # user + assistant

    result = chat.handle_command("/clear")
    assert "cleared" in result.lower()
    assert len(chat._history) == 0


# ── Chat: /debug toggle ─────────────────────────────────────────────


def test_debug_toggle(chat):
    import llm
    assert not llm.debug_log.enabled

    result = chat.handle_command("/debug")
    assert "ON" in result
    assert llm.debug_log.enabled

    result = chat.handle_command("/debug")
    assert "OFF" in result
    assert not llm.debug_log.enabled


# ── Chat: /help ──────────────────────────────────────────────────────


def test_help(chat):
    result = chat.handle_command("/help")
    assert "/model" in result
    assert "/remember" in result
    assert "/debug" in result
    assert "/quit" in result


# ── Chat: unknown command passes through ─────────────────────────────


def test_unknown_command(chat):
    result = chat.handle_command("/notacommand")
    assert result is None


# ── Chat: send (end-to-end with mocks) ──────────────────────────────


@patch("llm.call_gemini_api", return_value="Hi there!")
@patch("llm.call_ollama", return_value="NONE")
def test_send_basic(mock_ollama, mock_gemini, chat):
    response = chat.send("hello")
    assert response == "Hi there!"
    assert len(chat._history) == 2
    assert chat._history[0] == {"role": "user", "content": "hello"}
    assert chat._history[1] == {"role": "assistant", "content": "Hi there!"}


@patch("llm.call_gemini_api", return_value="I remember you like Python!")
@patch("llm.call_ollama", return_value="NONE")
def test_send_with_memories(mock_ollama, mock_gemini, chat):
    chat._memory.store_fact("User likes Python")

    response = chat.send("what do you know about me?")
    assert response == "I remember you like Python!"

    # Verify the system prompt included memory context
    system_sent = mock_gemini.call_args.kwargs.get("system_prompt", "")
    assert "User likes Python" in system_sent


@patch("llm.call_gemini_api", side_effect=RuntimeError("CLI failed"))
@patch("llm.call_ollama", return_value="NONE")
def test_send_handles_error(mock_ollama, mock_gemini, chat):
    response = chat.send("hello")
    assert "Error" in response


# ── Debug logging ────────────────────────────────────────────────────


@patch("llm.call_gemini_api", return_value="response text")
@patch("llm.call_ollama", return_value="NONE")
def test_debug_log_writes_to_file(mock_ollama, mock_gemini, chat):
    import config
    import llm

    llm.debug_log.enabled = True
    chat.send("test message")

    log_content = llm.debug_log._path.read_text()
    lines = [json.loads(l) for l in log_content.strip().splitlines()]

    # Should have at least a chat entry (+ possibly fact_extraction)
    chat_entries = [e for e in lines if e["call_type"] == "chat"]
    assert len(chat_entries) == 1
    assert chat_entries[0]["model"] in [m["model"] for m in config.MODELS.values()]
    assert "elapsed_ms" in chat_entries[0]
    assert chat_entries[0]["response"] == "response text"
    assert "test message" in chat_entries[0]["prompt"]


@patch("llm.call_gemini_api", return_value="ok")
@patch("llm.call_ollama", return_value="User likes tea")
def test_debug_log_includes_extraction(mock_ollama, mock_gemini, chat):
    import config
    import llm

    llm.debug_log.enabled = True
    chat.send("I like tea")

    log_content = llm.debug_log._path.read_text()
    lines = [json.loads(l) for l in log_content.strip().splitlines()]

    extraction_entries = [e for e in lines if e["call_type"] == "fact_extraction"]
    assert len(extraction_entries) == 1
    assert extraction_entries[0]["model"] == config.EXTRACTION_MODEL
    assert "elapsed_ms" in extraction_entries[0]


def test_debug_log_silent_when_disabled(chat):
    import llm

    llm.debug_log.enabled = False
    llm.debug_log.log(
        call_type="test", model="test", prompt="p", response="r", elapsed_ms=0
    )
    assert not llm.debug_log._path.exists()


# ── Config loading ───────────────────────────────────────────────────


def test_models_loaded_from_settings():
    import config

    assert len(config.MODELS) > 0
    for key, val in config.MODELS.items():
        assert val["provider"] == "gemini"
        assert "model" in val


def test_aliases_exist():
    import config

    assert "flash" in config.ALIASES or "pro" in config.ALIASES


def test_system_prompt_loaded():
    import config

    assert len(config.SYSTEM_PROMPT) > 0
    assert "companion" in config.SYSTEM_PROMPT.lower() or "Cortana" in config.SYSTEM_PROMPT


# ── Cross-session user interaction ───────────────────────────────────


@patch("llm.call_ollama")
@patch("llm.call_gemini_api")
def test_remember_name_across_sessions(mock_gemini, mock_ollama):
    """Simulate a real user: tell the assistant your name in session 1,
    then ask for it in session 2 — verify memory carries over."""
    import lite

    # -- Session 1: user introduces themselves --
    mock_gemini.return_value = "Nice to meet you, Alice!"
    mock_ollama.return_value = "User's name is Alice"

    session1 = lite.Chat()
    response1 = session1.send("Hi, my name is Alice")
    assert response1 == "Nice to meet you, Alice!"

    # Wait for background store thread to complete
    time.sleep(0.5)

    # Verify the fact was extracted and stored
    stored = session1._memory.get_all()
    assert any("Alice" in m for m in stored)

    # -- Session 2: new Chat instance (simulates app restart) --
    mock_gemini.return_value = "Your name is Alice!"
    mock_ollama.return_value = "NONE"

    session2 = lite.Chat()
    response2 = session2.send("What is my name?")
    assert response2 == "Your name is Alice!"

    # Verify the prompt sent to the LLM included the remembered fact
    system_sent = mock_gemini.call_args.kwargs.get("system_prompt", mock_gemini.call_args[0][1])
    assert "Alice" in system_sent
    assert "remember" in system_sent.lower() or "What you remember" in system_sent


@patch("llm.call_ollama")
@patch("llm.call_gemini_api")
def test_multi_fact_conversation_across_sessions(mock_gemini, mock_ollama):
    """Session 1: share multiple facts. Session 2: ask about them —
    verify all facts appear in the prompt context."""
    import lite

    # -- Session 1: user shares several facts --
    mock_gemini.return_value = "Got it, I'll remember all of that!"
    mock_ollama.return_value = (
        "User's name is Bob\n"
        "User's favorite color is blue\n"
        "User is a software engineer"
    )

    session1 = lite.Chat()
    session1.send("I'm Bob, I love the color blue, and I'm a software engineer")

    time.sleep(0.5)
    stored = session1._memory.get_all()
    assert len(stored) == 3

    # -- Session 2: fresh session asks about the user --
    mock_gemini.return_value = "You're Bob, a software engineer who loves blue!"
    mock_ollama.return_value = "NONE"

    session2 = lite.Chat()
    session2.send("Tell me what you know about me")

    system_sent = mock_gemini.call_args.kwargs.get("system_prompt", mock_gemini.call_args[0][1])
    assert "Bob" in system_sent
    # At least one of the other facts should be recalled too
    assert "blue" in system_sent or "software engineer" in system_sent


@patch("llm.call_ollama")
@patch("llm.call_gemini_api")
def test_cleared_memories_not_recalled_in_new_session(mock_gemini, mock_ollama):
    """Session 1: store a fact then clear memories.
    Session 2: the cleared fact should NOT appear in the prompt."""
    import lite

    # -- Session 1: remember then forget --
    mock_gemini.return_value = "Hi!"
    mock_ollama.return_value = "User's name is Charlie"

    session1 = lite.Chat()
    session1.send("My name is Charlie")
    time.sleep(0.5)
    assert any("Charlie" in m for m in session1._memory.get_all())

    session1.handle_command("/clearmemories")
    assert session1._memory.get_all() == []

    # -- Session 2: Charlie should be gone --
    mock_gemini.return_value = "I don't know your name yet."
    mock_ollama.return_value = "NONE"

    session2 = lite.Chat()
    session2.send("What is my name?")

    system_sent = mock_gemini.call_args.kwargs.get("system_prompt", mock_gemini.call_args[0][1])
    assert "Charlie" not in system_sent


# ── Integration: implicit facts via real Ollama ──────────────────────

def _ollama_available() -> bool:
    """Check if Ollama is running and the extraction model is pulled."""
    import config

    try:
        resp = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        # model tag may or may not include `:latest`
        base = config.EXTRACTION_MODEL.split(":")[0]
        return any(base in m for m in models)
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(), reason="Ollama not running or extraction model not available"
)


@requires_ollama
@patch("llm.call_gemini_api")
def test_implicit_name_extraction_via_ollama(mock_gemini):
    """Real Ollama extracts 'name is Alice' from casual conversation
    and a new session can recall it."""
    import lite

    mock_gemini.return_value = "Nice to meet you, Alice!"

    session1 = lite.Chat()
    session1.send("Hey there! I'm Alice and I just moved to Tokyo.")

    time.sleep(1)
    stored = session1._memory.get_all()
    assert len(stored) >= 1, f"Expected at least 1 extracted fact, got: {stored}"
    combined = " ".join(stored).lower()
    assert "alice" in combined, f"Expected 'alice' in extracted facts: {stored}"

    # -- Session 2: verify recall injects facts into prompt --
    mock_gemini.return_value = "You're Alice, living in Tokyo!"

    session2 = lite.Chat()
    session2.send("What do you remember about me?")

    system_sent = mock_gemini.call_args.kwargs.get("system_prompt", mock_gemini.call_args[0][1])
    # The recalled memories should mention Alice
    assert "alice" in system_sent.lower(), (
        f"Expected 'alice' in session 2 system prompt, got:\n{system_sent}"
    )


@requires_ollama
@patch("llm.call_gemini_api")
def test_implicit_preferences_extraction_via_ollama(mock_gemini):
    """Real Ollama picks up implicit preferences — not just explicit /remember."""
    import lite

    mock_gemini.return_value = "Python and hiking, nice combo!"

    session1 = lite.Chat()
    session1.send(
        "I've been coding in Python for about 5 years now. "
        "On weekends I usually go hiking in the mountains."
    )

    time.sleep(1)
    stored = session1._memory.get_all()
    assert len(stored) >= 1, f"Expected extracted facts, got: {stored}"
    combined = " ".join(stored).lower()
    assert "python" in combined or "hiking" in combined, (
        f"Expected 'python' or 'hiking' in facts: {stored}"
    )


@requires_ollama
@patch("llm.call_gemini_api")
def test_no_facts_from_trivial_message_via_ollama(mock_gemini):
    """Real Ollama should extract nothing (or very little) from a generic greeting."""
    import lite

    mock_gemini.return_value = "Hello! How can I help?"

    session = lite.Chat()
    session.send("hello")

    time.sleep(1)
    stored = session._memory.get_all()
    # A trivial "hello" shouldn't produce meaningful facts
    assert len(stored) <= 1, (
        f"Expected 0-1 facts from a bare greeting, got {len(stored)}: {stored}"
    )


# ── Full pipeline: all LLMs live, with timing + visualization ────────

def _gemini_api_available() -> bool:
    """Check if google-genai SDK is available and API key is set."""
    import config
    return config._gemini_client is not None


requires_all_llms = pytest.mark.skipif(
    not (_ollama_available() and _gemini_api_available()),
    reason="Requires both Ollama (with extraction model) and Gemini API key",
)


class PipelineTimer:
    """Records timing spans for each pipeline step."""

    def __init__(self):
        self.spans: list[dict] = []
        self._origin: float = time.monotonic()

    def record(self, label: str, model: str, start: float, end: float,
               output_preview: str = ""):
        self.spans.append({
            "label": label,
            "model": model,
            "start_ms": int((start - self._origin) * 1000),
            "end_ms": int((end - self._origin) * 1000),
            "duration_ms": int((end - start) * 1000),
            "output_preview": output_preview[:80],
        })

    def visualize(self, title: str, total_ms: int) -> str:
        """Render an ASCII pipeline timeline."""
        if not self.spans:
            return "(no spans recorded)"

        max_end = max(s["end_ms"] for s in self.spans)
        bar_width = 50
        label_width = max(len(s["label"]) for s in self.spans)
        model_width = max(len(s["model"]) for s in self.spans)

        lines = []
        lines.append("")
        lines.append(f"  {'=' * 70}")
        lines.append(f"  PIPELINE: {title}")
        lines.append(f"  {'=' * 70}")
        lines.append("")

        # Header
        header = f"  {'Step':<{label_width}}  {'Model':<{model_width}}  {'Timeline':<{bar_width}}  Duration"
        lines.append(header)
        lines.append(f"  {'-' * (label_width + model_width + bar_width + 12)}")

        for span in self.spans:
            if max_end == 0:
                left, width = 0, 1
            else:
                left = int(span["start_ms"] / max_end * bar_width)
                width = max(1, int(span["duration_ms"] / max_end * bar_width))
            bar = "." * left + "\u2588" * width + "." * (bar_width - left - width)
            dur = f"{span['duration_ms']:>5}ms"
            lines.append(
                f"  {span['label']:<{label_width}}  {span['model']:<{model_width}}  {bar}  {dur}"
            )

        lines.append("")

        # Summary table
        lines.append(f"  {'Step':<{label_width}}  {'Model':<{model_width}}  Duration  Output")
        lines.append(f"  {'-' * (label_width + model_width + 50)}")
        for span in self.spans:
            preview = span["output_preview"].replace("\n", " ") if span["output_preview"] else "-"
            lines.append(
                f"  {span['label']:<{label_width}}  {span['model']:<{model_width}}"
                f"  {span['duration_ms']:>5}ms   {preview}"
            )
        lines.append(f"  {'-' * (label_width + model_width + 50)}")
        lines.append(f"  {'TOTAL':<{label_width}}  {'':<{model_width}}  {total_ms:>5}ms")
        lines.append(f"  {'=' * 70}")
        lines.append("")

        return "\n".join(lines)

    def save_to_debug_log(self, title: str, total_ms: int):
        """Append the ASCII visualization + raw spans as JSON to the debug log."""
        log_dir = Path(__file__).parent.parent / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "debug.log"

        viz = self.visualize(title, total_ms)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "call_type": "pipeline_test",
            "title": title,
            "total_ms": total_ms,
            "spans": self.spans,
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.write(viz + "\n")


@requires_all_llms
def test_full_pipeline_remember_and_recall():
    """End-to-end with ALL real LLMs: Gemini API for chat, Ollama for fact extraction.

    Session 1 — user shares their name  →  Gemini responds  →  Ollama extracts facts
    Session 2 — user asks for their name →  ChromaDB recalls →  Gemini responds with context

    Every LLM call is timed and an ASCII pipeline diagram is printed.
    """
    import config
    import llm
    from chat import Chat

    # Restore real Gemini clients for integration test
    config._gemini_client = _real_gemini_client
    config._gemini_clients = _real_gemini_clients

    timer = PipelineTimer()
    pipeline_start = time.monotonic()

    # Wrap the real LLM functions with timing
    real_gemini_api = llm.call_gemini_api
    real_ollama = llm.call_ollama

    gemini_call_count = [0]
    ollama_call_count = [0]

    def timed_gemini(model, prompt, system_prompt="", temperature=0.7,
                     max_tokens=2048, client=None):
        gemini_call_count[0] += 1
        label = f"gemini_chat_s{gemini_call_count[0]}"
        t0 = time.monotonic()
        result = real_gemini_api(model, prompt, system_prompt, temperature,
                                max_tokens, client=client)
        t1 = time.monotonic()
        timer.record(label, model, t0, t1, output_preview=result)
        return result

    def timed_ollama(model, prompt, temperature=0.1, max_tokens=500):
        ollama_call_count[0] += 1
        label = f"ollama_extract_{ollama_call_count[0]}"
        t0 = time.monotonic()
        result = real_ollama(model, prompt, temperature, max_tokens)
        t1 = time.monotonic()
        timer.record(label, model, t0, t1, output_preview=result)
        return result

    with patch.object(llm, "call_gemini_api", side_effect=timed_gemini), \
         patch.object(llm, "call_ollama", side_effect=timed_ollama):

        # ── Session 1: introduce yourself ──
        session1 = Chat()

        t0 = time.monotonic()
        recall_1 = session1._memory.recall("My name is Dave")
        t1 = time.monotonic()
        timer.record("chromadb_recall_s1", "chromadb", t0, t1,
                      output_preview=str(recall_1))

        response1 = session1.send("Hi! My name is Dave and I work as a chef in Paris.")

        # Verify facts were stored
        stored = session1._memory.get_all()
        assert len(stored) >= 1, f"Expected extracted facts, got: {stored}"
        combined_facts = " ".join(stored).lower()
        assert "dave" in combined_facts, f"'dave' not in facts: {stored}"

        # ── Session 2: ask for the name back ──
        session2 = Chat()

        t0 = time.monotonic()
        recall_2 = session2._memory.recall("What is my name?")
        t1 = time.monotonic()
        timer.record("chromadb_recall_s2", "chromadb", t0, t1,
                      output_preview=str(recall_2))

        response2 = session2.send("What is my name and what do I do?")

    pipeline_end = time.monotonic()
    total_ms = int((pipeline_end - pipeline_start) * 1000)

    # ── Visualize + persist ──
    viz = timer.visualize("Remember name → Recall across sessions", total_ms)
    print(viz)
    timer.save_to_debug_log("Remember name → Recall across sessions", total_ms)

    # ── Assertions ──
    # Session 2 prompt must contain recalled memories about Dave
    assert len(recall_2) >= 1, f"ChromaDB recall returned nothing in session 2"
    recall_text = " ".join(fact for fact, _ in recall_2).lower()
    assert "dave" in recall_text, f"'dave' not in recalled memories: {recall_2}"

    # Both Gemini calls should have produced non-empty responses
    assert len(response1) > 0
    assert len(response2) > 0

    # Pipeline should have recorded at least: 2x gemini, 1-2x ollama, 2x chromadb
    labels = [s["label"] for s in timer.spans]
    assert any("gemini" in l for l in labels), f"No gemini spans: {labels}"
    assert any("ollama" in l for l in labels), f"No ollama spans: {labels}"
    assert any("chromadb" in l for l in labels), f"No chromadb spans: {labels}"


@requires_all_llms
def test_full_pipeline_multi_turn_memory():
    """Multi-turn conversation across 3 sessions with full LLM pipeline.

    Session 1 — share name
    Session 2 — share hobby
    Session 3 — ask about both → verify all facts recalled
    """
    import config
    import llm
    from chat import Chat

    # Restore real Gemini clients for integration test
    config._gemini_client = _real_gemini_client
    config._gemini_clients = _real_gemini_clients

    timer = PipelineTimer()
    pipeline_start = time.monotonic()

    real_gemini_api = llm.call_gemini_api
    real_ollama = llm.call_ollama
    call_idx = [0, 0]

    def timed_gemini(model, prompt, system_prompt="", temperature=0.7,
                     max_tokens=2048, client=None):
        call_idx[0] += 1
        t0 = time.monotonic()
        result = real_gemini_api(model, prompt, system_prompt, temperature,
                                max_tokens, client=client)
        t1 = time.monotonic()
        timer.record(f"gemini_{call_idx[0]}", model, t0, t1, output_preview=result)
        return result

    def timed_ollama(model, prompt, temperature=0.1, max_tokens=500):
        call_idx[1] += 1
        t0 = time.monotonic()
        result = real_ollama(model, prompt, temperature, max_tokens)
        t1 = time.monotonic()
        timer.record(f"ollama_{call_idx[1]}", model, t0, t1, output_preview=result)
        return result

    with patch.object(llm, "call_gemini_api", side_effect=timed_gemini), \
         patch.object(llm, "call_ollama", side_effect=timed_ollama):

        # ── Session 1: share name ──
        s1 = Chat()
        s1.send("Hey, my name is Elena and I'm from Barcelona.")

        # ── Session 2: share hobby ──
        s2 = Chat()
        s2.send("My biggest hobby is playing the violin, I've been doing it for 10 years.")

        # ── Session 3: ask about everything ──
        s3 = Chat()

        t0 = time.monotonic()
        recall_3 = s3._memory.recall("Tell me everything about me")
        t1 = time.monotonic()
        timer.record("chromadb_recall_s3", "chromadb", t0, t1,
                      output_preview=str(recall_3))

        response3 = s3.send("What do you know about me?")

    pipeline_end = time.monotonic()
    total_ms = int((pipeline_end - pipeline_start) * 1000)

    viz = timer.visualize("3-session accumulation test", total_ms)
    print(viz)
    timer.save_to_debug_log("3-session accumulation test", total_ms)

    # Session 3 should recall facts from both prior sessions
    all_mems = s3._memory.get_all()
    combined = " ".join(all_mems).lower()
    assert "elena" in combined, f"'elena' not in memories: {all_mems}"
    assert "violin" in combined, f"'violin' not in memories: {all_mems}"
    assert len(response3) > 0
