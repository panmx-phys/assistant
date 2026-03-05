package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Client talks to the Python service.
type Client struct {
	baseURL string
	http    *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 0}, // no timeout for streaming
	}
}

// ── Request / Response types ─────────────────────────────────────────

type ChatRequest struct {
	Message string `json:"message"`
}

type CommandRequest struct {
	Command string `json:"command"`
}

type CommandResponse struct {
	Result     string `json:"result"`
	Action     string `json:"action,omitempty"`
	NotCommand bool   `json:"not_command,omitempty"`
}

type SSEEvent struct {
	Chunk string `json:"chunk,omitempty"`
	Done  bool   `json:"done"`
	Model string `json:"model,omitempty"`
	Error string `json:"error,omitempty"`
}

type ModelInfo struct {
	Description string `json:"description"`
	Provider    string `json:"provider"`
	Model       string `json:"model"`
}

type ModelsResponse struct {
	Models             map[string]ModelInfo `json:"models"`
	Aliases            map[string]string    `json:"aliases"`
	Current            string               `json:"current"`
	CurrentDescription string               `json:"current_description"`
}

// ── API calls ────────────────────────────────────────────────────────

func (c *Client) Health() error {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(c.baseURL + "/health")
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("health check returned %d", resp.StatusCode)
	}
	return nil
}

func (c *Client) Models() (*ModelsResponse, error) {
	resp, err := c.http.Get(c.baseURL + "/models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var m ModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}
	return &m, nil
}

func (c *Client) Command(cmd string) (*CommandResponse, error) {
	body, _ := json.Marshal(CommandRequest{Command: cmd})
	resp, err := c.http.Post(c.baseURL+"/command", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var cr CommandResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return nil, err
	}
	return &cr, nil
}

// StreamChat sends a message and calls onChunk for each streamed text chunk.
// Returns the model label when done.
func (c *Client) StreamChat(msg string, onChunk func(string)) (string, error) {
	body, _ := json.Marshal(ChatRequest{Message: msg})
	req, err := http.NewRequest("POST", c.baseURL+"/chat", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("connection failed: %w", err)
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	var model string
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := line[6:]
		var evt SSEEvent
		if err := json.Unmarshal([]byte(data), &evt); err != nil {
			continue
		}
		if evt.Done {
			model = evt.Model
			break
		}
		if evt.Error != "" {
			onChunk(evt.Error)
			continue
		}
		if evt.Chunk != "" {
			onChunk(evt.Chunk)
		}
	}
	return model, scanner.Err()
}
