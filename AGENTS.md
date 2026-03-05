# AGENTS.md — Cortana / AI Companion

## Project summary

- **LLM**: Gemini API (google-genai) chat; optional Ollama local + fact extraction.
- **Memory**: ChromaDB. Fact extraction: Ollama first (`memory.ollama.extraction_model`), fallback Gemini 2.5 Flash Lite (`memory.extraction_fallback.model`, `api_key`) when Ollama unavailable. Recall: Ebbinghaus-style decay, significance, access count.
- **TTS**: Optional Unix socket server. Backends: Kokoro, ElevenLabs, Google, Gemini. `tts.py`; client in `chat.py`.
- **Service**: FastAPI :8391 (`CORTANA_PORT`). `/health`, `/models`, `POST /chat` (SSE), `POST /command`.
- **CLI**: `cli/main.go` — starts `service.py` + `tts.py`, REPL over HTTP. Optional `Cortana.app` same CLI.
- **Personality**: `config/personality.yaml` (`base`, `modes`). Name: Cortana.

## Architecture

```
User → Go CLI | Cortana.app | HTTP client
  → HTTP :8391 → service.py (FastAPI)
  → chat.py (Chat, LLM, slash cmds) | memory.py (ChromaDB, extract Ollama→Gemini) | tts client → tts.py
  → llm.py (Gemini, Ollama) | config.py (settings, personality)
```

- Config: `config/settings.yaml` (env `${VAR}`), `config/personality.yaml`. `.env` root (not committed).
- Data: `data/chromadb`, `data/logs/debug.log` (if debug).

## Key files

| Path | Role |
|------|------|
| `config.py` | load_config, env, Gemini clients, MODELS/ALIASES/DEFAULT_MODEL, memory/TTS/debug/personality constants |
| `config/settings.yaml` | api_keys, models, memory (chromadb, ollama, extraction_fallback, declutter), tts, debug, ui |
| `config/personality.yaml` | base system prompt, modes |
| `chat.py` | Chat: history, model, send/send_stream, handle_command (slash), emotion strip, TTS client |
| `llm.py` | call_gemini_api, stream_gemini_api, call_ollama; DebugLogger(LOG_FILE) |
| `memory.py` | Memory: ChromaDB, _extract_facts (Ollama then Gemini fallback), recall, store, declutter, backfill, auto_declutter |
| `tts.py` | Backends, TTSEngine, Unix server, TTS client |
| `service.py` | FastAPI app, /health /models /chat /command, Chat(), SSE stream |
| `lite.py` | Re-exports config, llm, Memory, Chat |
| `cli/main.go` | ProjectDir, StartProc service+tts, WaitForHealth, REPL, StreamChat/Command |
| `cli/client.go` | Health, Models, Command, StreamChat (SSE) |
| `cli/proc.go` | StartProc, Stop, ProjectDir, WaitForHealth |
| `cli/ui.go` | glamour, lipgloss panel |
| `tests/test_lite.py` | pytest, mock Gemini/Ollama, tmp_path ChromaDB, @requires_ollama / @requires_all_llms |

## Environment & config

- **Required**: `GEMINI_API_KEY` (or key in api_keys). `.env` or env.
- **Optional**: `GEMINI_PRO_API_KEY`, `ELEVENLABS_API_KEY`, `GOOGLE_TTS_API_KEY`, `CORTANA_PORT` (8391), `CORTANA_URL` (http://127.0.0.1:8391).
- Config: `load_dotenv(BASE_DIR/.env)`, `load_config(settings.yaml)`, `${VAR}` → `os.environ.get("VAR","")`.

## Conventions

- Python 3, type hints, `from __future__ import annotations`. Run from project root; tests `sys.path.insert(0, parent)`.
- Slash commands: `chat.handle_command`; extend there, update `/help` and README.
- Emotion: LLM prefix e.g. `[happy]`; chat strips and passes to TTS.

## Testing

- `pytest tests/` from root. Mock `llm.call_gemini_api`, `llm.call_ollama`; patch config CHROMA_PATH, COLLECTION_NAME, _gemini_client. ChromaDB: fake embedding. Integration: @requires_ollama, @requires_all_llms (skip if unavailable).

## Deployment

1. **CLI**: `pip install -r requirements.txt`, set GEMINI_API_KEY, `cd cli && go run .` or binary. Starts service + tts.
2. **Service only**: `python service.py` or uvicorn. CORTANA_PORT. Clients POST /chat, POST /command.
3. **TTS**: CLI starts tts.py; or `python tts.py [--socket ...] [--backend ...]`. No TTS → chat text-only.
4. **macOS**: Cortana.app runs CLI; needs project dir so service.py, tts.py found.
5. **Ollama**: Optional. memory.ollama (base_url, extraction_model). If absent → extraction_fallback (Gemini). /remember always works.

## Common tasks

- **Slash command**: `chat.handle_command` in chat.py, /help, README.
- **Config**: settings.yaml key → config.py constant → use in module.
- **TTS backend**: TTSBackend in tts.py, create_backend, BACKENDS, settings + config TTS_*.
- **Personality**: config/personality.yaml base/modes.
- **Debug**: settings debug: true or /debug; data/logs/debug.log JSON-lines.

## File layout

```
.  config/ settings.yaml personality.yaml
   config.py chat.py llm.py memory.py tts.py service.py lite.py
   cli/ main.go client.go proc.go ui.go go.mod go.sum
   tests/ test_lite.py
   data/ chromadb/ logs/   (gitignored)
   Cortana.app/ Contents/MacOS/launcher
   .env requirements.txt README.md AGENTS.md  .env data/ gitignored
```
