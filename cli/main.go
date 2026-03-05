package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/peterh/liner"
)

var commands = []string{
	"/model", "/remember", "/memories", "/clearmemories",
	"/clear", "/tts", "/debug", "/help", "/quit",
}

func main() {
	baseURL := os.Getenv("CORTANA_URL")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:8391"
	}

	projectDir := ProjectDir()

	// ── Launch backend processes ──────────────────────────────────────
	fmt.Println("Starting services...")

	servicePath := filepath.Join(projectDir, "service.py")
	ttsPath := filepath.Join(projectDir, "tts.py")

	serviceProc, err := StartProc("service", servicePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to start service: %v\n", err)
		os.Exit(1)
	}
	defer serviceProc.Stop()

	ttsProc, err := StartProc("tts", ttsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: TTS server failed to start: %v\n", err)
		// Non-fatal — TTS is optional
	} else {
		defer ttsProc.Stop()
	}

	// Wait for the service to be ready
	fmt.Print("Waiting for service...")
	if err := WaitForHealth(baseURL, 15*time.Second); err != nil {
		fmt.Fprintf(os.Stderr, "\n%v\n", err)
		os.Exit(1)
	}
	fmt.Print("\r\033[2K") // clear "Waiting..." line

	// ── Connect ──────────────────────────────────────────────────────
	client := NewClient(baseURL)

	modelsResp, err := client.Models()
	greeting := "Cortana"
	if err == nil && modelsResp != nil {
		greeting = fmt.Sprintf("Cortana — %s", modelsResp.CurrentDescription)
	}

	homeDir, _ := os.UserHomeDir()
	historyFile := filepath.Join(homeDir, ".cortana_history")

	line := liner.NewLiner()
	defer line.Close()

	line.SetCtrlCAborts(false)
	line.SetCompleter(func(input string) []string {
		var completions []string
		for _, cmd := range commands {
			if strings.HasPrefix(cmd, strings.ToLower(input)) {
				completions = append(completions, cmd)
			}
		}
		return completions
	})

	if f, err := os.Open(historyFile); err == nil {
		line.ReadHistory(f)
		f.Close()
	}

	fmt.Printf("\n%s\n", greeting)
	fmt.Println("Type /help for commands, /quit to exit.\n")

	for {
		input, err := line.Prompt("You: ")
		if err != nil {
			if err == liner.ErrPromptAborted || err == io.EOF {
				fmt.Println("\nBye!")
			}
			break
		}

		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}

		line.AppendHistory(input)

		// Model shortcuts
		if input == `\p` || input == `\f` {
			resp, err := client.Command(input)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				continue
			}
			fmt.Printf("\n%s\n\n", resp.Result)
			continue
		}

		// Slash commands
		if strings.HasPrefix(input, "/") {
			resp, err := client.Command(input)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				continue
			}
			if resp.Action == "quit" {
				fmt.Println("\nBye!")
				break
			}
			if resp.Result != "" {
				fmt.Printf("\n%s\n\n", resp.Result)
			}
			continue
		}

		// Streaming chat — collect chunks, show a spinner, then render panel
		fmt.Print("\n\033[38;5;74m⠋ thinking...\033[0m")
		spinFrames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
		spinIdx := 0
		var accumulated strings.Builder
		model, err := client.StreamChat(input, func(chunk string) {
			accumulated.WriteString(chunk)
			spinIdx = (spinIdx + 1) % len(spinFrames)
			fmt.Printf("\r\033[2K\033[38;5;74m%s streaming...\033[0m", spinFrames[spinIdx])
		})
		fmt.Print("\r\033[2K") // clear spinner line
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}

		text := accumulated.String()
		if text != "" {
			fmt.Println(renderPanel(text, "Cortana", model))
		}
	}

	// Save history
	if f, err := os.Create(historyFile); err == nil {
		line.WriteHistory(f)
		f.Close()
	}

	fmt.Println("Shutting down services...")
}
