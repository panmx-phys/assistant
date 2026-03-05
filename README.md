# Cortana — AI Companion

A personal AI companion that runs on your machine. It remembers your conversations, supports multiple LLM backends (Gemini API, optional Ollama), and can speak replies via optional text-to-speech (Kokoro, ElevenLabs, Google, or Gemini TTS).

---

## Quick start

### 1. Prerequisites

- **Python 3** with pip  
- **Go 1.25+** (for the CLI)  
- **Gemini API key** ([Google AI Studio](https://aistudio.google.com/apikey))

### 2. Setup

```bash
# Clone or open the project, then:
cd /path/to/assitant

# Python dependencies
pip install -r requirements.txt

# API key (required for chat)
echo "GEMINI_API_KEY=your_key_here" > .env
```

### 3. Run

**Option A — Go CLI (recommended)**  
Starts the backend and optional TTS, then opens an interactive chat:

```bash
go run ./cli
```

**Option B — Backend only**  
Run the HTTP service yourself; use any client that can call the API:

```bash
python service.py
# Service: http://127.0.0.1:8391
# Try: curl -X POST http://127.0.0.1:8391/chat -H "Content-Type: application/json" -d '{"message":"Hello"}' --no-buffer
```

**Option C — macOS app**  
If you have the `Cortana.app` bundle, run it. Ensure the app can find the project directory (e.g. keep the app next to the project or set the working directory accordingly).

### 4. In the chat

- Type a message and press Enter for a streaming reply.  
- Use **/help** for all commands.  
- **/model** — list or switch models (e.g. `flash` / `pro`).  
- **/remember** — store a fact. **/memories** — list stored memories.  
- **\\p** and **\\f** — switch to Pro or Flash model quickly.

---

## Deployment

### Minimal (service only)

1. On the host: install Python 3 and dependencies (`pip install -r requirements.txt`).  
2. Set `GEMINI_API_KEY` (in `.env` or environment).  
3. Run the service:

   ```bash
   python service.py
   ```

   Default port is **8391**. Override with:

   ```bash
   CORTANA_PORT=9000 python service.py
   ```

4. Point clients at `http://<host>:8391` (or your port).  
   - **Health**: `GET /health`  
   - **Models**: `GET /models`  
   - **Chat (streaming)**: `POST /chat` with `{"message": "..."}`; response is SSE.  
   - **Commands**: `POST /command` with `{"command": "/model flash"}` (or any slash command).

### With TTS (voice)

- The **Go CLI** starts both the HTTP service and the TTS server by default.  
- **Standalone TTS**: run `python tts.py` (optionally with `--socket`, `--backend`, `--voice`, `--lang`, `--speed`). Default socket: `/tmp/tts.sock`.  
- If the TTS server is not running, chat still works; replies are text-only.

### With the Go CLI

- Build: `go build -o cortana ./cli` (or build from your CI).  
- Run: `./cortana` (from a directory where it can find the project, or set `CORTANA_URL` if the service is already running elsewhere).  
- The CLI expects `service.py` and `tts.py` relative to the project root (it locates the project from the executable path or current working directory).

### Optional: Ollama (local fact extraction)

- Fact extraction (automatic memory from conversations) tries **Ollama** first (e.g. `qwen3:1.7b`).  
- If Ollama is not installed or unavailable, it **falls back to Gemini 2.5 Flash Lite** (same API key as chat). No need to install Ollama to get automatic memory.  
- To use Ollama: install and run [Ollama](https://ollama.com), then pull the extraction model (e.g. `ollama pull qwen3:1.7b`). Config: `config/settings.yaml` → `memory.ollama` (base_url, extraction_model).  
- Fallback model and API key: `memory.extraction_fallback` (default model: `gemini-2.5-flash-lite`, api_key: `gemini`).  
- You can always use **/remember** and **/memories** regardless of Ollama.

### Environment variables (summary)

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Required for Gemini chat (and default TTS if using Gemini TTS). |
| `GEMINI_PRO_API_KEY` | Optional; used for the “Pro” model if configured. |
| `CORTANA_PORT` | HTTP service port (default 8391). |
| `CORTANA_URL` | Base URL for the CLI (default `http://127.0.0.1:8391`). |
| `ELEVENLABS_API_KEY` | For ElevenLabs TTS backend. |
| `GOOGLE_TTS_API_KEY` | For Google Cloud TTS backend. |

---

## Configuration

- **Main config**: `config/settings.yaml` — models, memory (ChromaDB path, Ollama, **extraction_fallback** for when Ollama is unavailable, declutter), TTS backend and voice, debug.  
- **Personality**: `config/personality.yaml` — system prompt and companion name (Cortana).  
- **Secrets**: Put API keys in `.env` in the project root; the app loads them via `python-dotenv`. Do not commit `.env`.

---

## Commands (in-chat)

| Command | Description |
|---------|-------------|
| `/model [name]` | List models or switch (e.g. `flash`, `pro`). |
| `/remember <fact>` | Store a fact in long-term memory. |
| `/memories` | List stored memories. |
| `/clearmemories` | Delete all memories. |
| `/clear` | Clear conversation history (memories kept). |
| `/tts` | Toggle text-to-speech on/off. |
| `/ttsbackend [name]` | Switch TTS backend (kokoro, elevenlabs, google, gemini). |
| `/voice [preset]` | Change TTS voice (e.g. en, zh). |
| `/declutter` | Merge duplicates and prune low-value memories. |
| `/backfill` | Score all memories by significance. |
| `/debug` | Toggle debug logging (writes to `data/logs/debug.log`). |
| `/help` | Show command list. |
| `/quit` | Exit. |

---

## Project layout

- **Python**: `config.py` (config), `chat.py` (engine), `llm.py` (Gemini/Ollama), `memory.py` (ChromaDB + extraction), `tts.py` (TTS server and backends), `service.py` (FastAPI).  
- **CLI**: `cli/main.go` (entry), `cli/client.go` (HTTP), `cli/proc.go` (processes), `cli/ui.go` (terminal UI).  
- **Config**: `config/settings.yaml`, `config/personality.yaml`.  
- **Tests**: `tests/test_lite.py` (pytest; run with `pytest tests/` from project root).

For detailed architecture and conventions for contributors or AI agents, see **AGENTS.md**.
# assistant
