package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	webview "github.com/webview/webview_go"
)

type proc struct {
	cmd *exec.Cmd
}

func startProc(script string, args ...string) (*proc, error) {
	cmdArgs := append([]string{script}, args...)
	pythonExec := strings.TrimSpace(os.Getenv("CORTANA_PYTHON"))
	if pythonExec == "" {
		pythonExec = "python3"
	}
	cmd := exec.Command(pythonExec, cmdArgs...)
	cmd.Dir = filepath.Dir(script)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &proc{cmd: cmd}, nil
}

func (p *proc) stop() {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return
	}
	_ = p.cmd.Process.Signal(os.Interrupt)
	done := make(chan error, 1)
	go func() { done <- p.cmd.Wait() }()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		_ = p.cmd.Process.Kill()
		<-done
	}
}

func projectDir() string {
	// Try from current working directory and its ancestors.
	cwd, _ := os.Getwd()
	cur := cwd
	for i := 0; i < 5; i++ {
		if _, err := os.Stat(filepath.Join(cur, "service.py")); err == nil {
			return cur
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			break
		}
		cur = parent
	}

	// Fallback to executable location and its ancestors.
	exe, err := os.Executable()
	if err == nil {
		cur = filepath.Dir(exe)
		for i := 0; i < 6; i++ {
			if _, err := os.Stat(filepath.Join(cur, "service.py")); err == nil {
				return cur
			}
			parent := filepath.Dir(cur)
			if parent == cur {
				break
			}
			cur = parent
		}
	}
	return cwd
}

func waitForHealth(url string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 1 * time.Second}
	for time.Now().Before(deadline) {
		resp, err := client.Get(url + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return nil
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
	return fmt.Errorf("service did not become healthy within %s", timeout)
}

type chatRequest struct {
	Message string `json:"message"`
}

type commandRequest struct {
	Command string `json:"command"`
}

type commandResponse struct {
	Result string `json:"result"`
	Action string `json:"action,omitempty"`
}

type sseEvent struct {
	Chunk string `json:"chunk,omitempty"`
	Done  bool   `json:"done"`
	Model string `json:"model,omitempty"`
	Error string `json:"error,omitempty"`
}

type modelInfo struct {
	Description string `json:"description"`
}

type modelsResponse struct {
	CurrentDescription string               `json:"current_description"`
	Models             map[string]modelInfo `json:"models"`
}

type apiClient struct {
	baseURL string
	http    *http.Client
}

func newClient(baseURL string) *apiClient {
	return &apiClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 0},
	}
}

func (c *apiClient) models() (*modelsResponse, error) {
	resp, err := c.http.Get(c.baseURL + "/models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *apiClient) command(cmd string) (*commandResponse, error) {
	body, _ := json.Marshal(commandRequest{Command: cmd})
	resp, err := c.http.Post(c.baseURL+"/command", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out commandResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *apiClient) streamChat(msg string, onChunk func(string), onDone func(string), onErr func(string)) {
	body, _ := json.Marshal(chatRequest{Message: msg})
	req, err := http.NewRequest("POST", c.baseURL+"/chat", bytes.NewReader(body))
	if err != nil {
		onErr(err.Error())
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.http.Do(req)
	if err != nil {
		onErr(err.Error())
		return
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := line[6:]
		var evt sseEvent
		if err := json.Unmarshal([]byte(data), &evt); err != nil {
			continue
		}
		if evt.Error != "" {
			onErr(evt.Error)
			return
		}
		if evt.Done {
			onDone(evt.Model)
			return
		}
		if evt.Chunk != "" {
			onChunk(evt.Chunk)
		}
	}
	if err := scanner.Err(); err != nil {
		onErr(err.Error())
		return
	}
	onDone("")
}

type bridge struct {
	w          webview.WebView
	client     *apiClient
	mu         sync.Mutex
	msgCounter int64
}

func newBridge(w webview.WebView, c *apiClient) *bridge {
	return &bridge{w: w, client: c}
}

func (b *bridge) nextID() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.msgCounter++
	return "msg-" + strconv.FormatInt(b.msgCounter, 10)
}

func quoteJS(v string) string {
	return strconv.Quote(v)
}

func (b *bridge) SendMessage(text string) string {
	input := strings.TrimSpace(text)
	if input == "" {
		return ""
	}
	if strings.HasPrefix(input, "/") || input == `\p` || input == `\f` {
		go b.handleCommand(input)
		return ""
	}
	replyID := b.nextID()
	b.w.Dispatch(func() {
		b.w.Eval("window.cortana.addAssistantMessage(" + quoteJS(replyID) + ");")
		b.w.Eval("window.cortana.setThinking(true);")
	})
	go b.client.streamChat(
		input,
		func(chunk string) {
			b.w.Dispatch(func() {
				b.w.Eval("window.cortana.appendAssistantChunk(" + quoteJS(replyID) + "," + quoteJS(chunk) + ");")
			})
		},
		func(model string) {
			b.w.Dispatch(func() {
				b.w.Eval("window.cortana.setThinking(false);")
				b.w.Eval("window.cortana.setModel(" + quoteJS(model) + ");")
			})
		},
		func(errMsg string) {
			b.w.Dispatch(func() {
				b.w.Eval("window.cortana.setThinking(false);")
				b.w.Eval("window.cortana.showError(" + quoteJS(errMsg) + ");")
			})
		},
	)
	return ""
}

func (b *bridge) handleCommand(cmd string) {
	resp, err := b.client.command(cmd)
	if err != nil {
		b.w.Dispatch(func() {
			b.w.Eval("window.cortana.showError(" + quoteJS(err.Error()) + ");")
		})
		return
	}
	b.w.Dispatch(func() {
		if resp.Result != "" {
			b.w.Eval("window.cortana.addSystemMessage(" + quoteJS(resp.Result) + ");")
		}
		if resp.Action == "quit" {
			b.w.Terminate()
		}
	})
}

func main() {
	baseURL := os.Getenv("CORTANA_URL")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:8391"
	}

	project := projectDir()
	servicePath := filepath.Join(project, "service.py")
	ttsPath := filepath.Join(project, "tts.py")

	serviceProc, err := startProc(servicePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to start service: %v\n", err)
		os.Exit(1)
	}
	defer serviceProc.stop()

	ttsProc, err := startProc(ttsPath)
	if err == nil {
		defer ttsProc.stop()
	}

	if err := waitForHealth(baseURL, 15*time.Second); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	client := newClient(baseURL)

	title := "Cortana"
	if models, err := client.models(); err == nil && strings.TrimSpace(models.CurrentDescription) != "" {
		title = "Cortana - " + models.CurrentDescription
	}

	w := webview.New(true)
	defer w.Destroy()
	w.SetTitle(title)
	w.SetSize(980, 700, webview.HintNone)

	ui := newBridge(w, client)
	if err := w.Bind("sendMessage", ui.SendMessage); err != nil {
		fmt.Fprintf(os.Stderr, "failed to bind sendMessage: %v\n", err)
		os.Exit(1)
	}

	w.SetHtml(html)
	w.Run()
}

const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cortana</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #111830;
      --muted: #96a2d5;
      --text: #e6ebff;
      --line: #2a3566;
      --accent: #5f84ff;
      --error: #ff5d73;
      --sys: #6ec6ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: linear-gradient(160deg, #090e1d, #121c3a);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(17, 24, 48, 0.75);
      backdrop-filter: blur(6px);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .title { font-weight: 700; letter-spacing: 0.2px; }
    .model { color: var(--muted); font-size: 13px; }
    .thinking {
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
      padding: 8px 16px 0 16px;
    }
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .msg {
      max-width: 80%;
      padding: 10px 12px;
      border-radius: 12px;
      white-space: pre-wrap;
      line-height: 1.4;
      border: 1px solid var(--line);
    }
    .user {
      align-self: flex-end;
      background: #1a2756;
      border-color: #3954ad;
    }
    .assistant {
      align-self: flex-start;
      background: var(--panel);
    }
    .system {
      align-self: center;
      width: 90%;
      max-width: 90%;
      background: rgba(110, 198, 255, 0.1);
      border-color: rgba(110, 198, 255, 0.35);
      color: var(--sys);
      font-size: 13px;
    }
    .error {
      align-self: center;
      width: 90%;
      max-width: 90%;
      background: rgba(255, 93, 115, 0.1);
      border-color: rgba(255, 93, 115, 0.4);
      color: #ffc2cc;
      font-size: 13px;
    }
    form {
      display: flex;
      gap: 10px;
      padding: 14px 16px;
      border-top: 1px solid var(--line);
      background: rgba(17, 24, 48, 0.78);
    }
    input {
      flex: 1;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #0f1530;
      color: var(--text);
      padding: 12px 14px;
      font-size: 15px;
      outline: none;
    }
    input:focus { border-color: var(--accent); }
    button {
      border: 0;
      border-radius: 10px;
      background: var(--accent);
      color: white;
      padding: 0 18px;
      font-weight: 600;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.55;
      cursor: default;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      padding: 0 16px 10px 16px;
    }
  </style>
</head>
<body>
  <header>
    <div class="title">Cortana</div>
    <div id="model" class="model">model: connecting...</div>
  </header>
  <div id="thinking" class="thinking"></div>
  <div id="chat"></div>
  <form id="form">
    <input id="input" placeholder="Message Cortana... (/help for commands)" autocomplete="off" />
    <button id="send" type="submit">Send</button>
  </form>
  <div class="hint">Slash commands are supported (example: /model, /remember, /tts, /help).</div>
  <script>
    const chat = document.getElementById("chat");
    const modelEl = document.getElementById("model");
    const form = document.getElementById("form");
    const input = document.getElementById("input");
    const sendBtn = document.getElementById("send");
    const thinking = document.getElementById("thinking");
    const messages = new Map();

    function scrollBottom() {
      chat.scrollTop = chat.scrollHeight;
    }

    function createMsg(text, cls) {
      const el = document.createElement("div");
      el.className = "msg " + cls;
      el.textContent = text;
      chat.appendChild(el);
      scrollBottom();
      return el;
    }

    window.cortana = {
      addAssistantMessage(id) {
        const el = createMsg("", "assistant");
        messages.set(id, el);
      },
      appendAssistantChunk(id, chunk) {
        const el = messages.get(id);
        if (!el) return;
        el.textContent += chunk;
        scrollBottom();
      },
      addSystemMessage(text) {
        createMsg(text, "system");
      },
      showError(text) {
        createMsg(text, "error");
      },
      setThinking(on) {
        thinking.textContent = on ? "Cortana is thinking..." : "";
        sendBtn.disabled = on;
      },
      setModel(model) {
        if (!model) return;
        modelEl.textContent = "model: " + model;
      }
    };

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      createMsg(text, "user");
      input.value = "";
      try {
        await sendMessage(text);
      } catch (err) {
        window.cortana.showError(String(err));
      }
      input.focus();
    });

    window.cortana.addSystemMessage("Connected. Start chatting, or use /help for commands.");
    input.focus();
  </script>
</body>
</html>`
