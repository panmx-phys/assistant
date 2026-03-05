package main

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// Proc manages a child process.
type Proc struct {
	name string
	cmd  *exec.Cmd
}

// StartProc launches a python script as a background process.
// Output goes to stderr so it doesn't interfere with the CLI's terminal.
func StartProc(name, script string, args ...string) (*Proc, error) {
	cmdArgs := append([]string{script}, args...)
	cmd := exec.Command("python3", cmdArgs...)
	cmd.Dir = filepath.Dir(script)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start %s: %w", name, err)
	}
	return &Proc{name: name, cmd: cmd}, nil
}

// Stop sends SIGINT then waits briefly, falls back to SIGKILL.
func (p *Proc) Stop() {
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

// ProjectDir finds the project root (parent of the cli/ directory).
// Looks relative to the executable path first, then falls back to cwd.
func ProjectDir() string {
	// Try relative to executable
	exe, err := os.Executable()
	if err == nil {
		dir := filepath.Dir(exe)
		// If exe is in cli/, go up one level
		candidate := filepath.Join(dir, "..", "service.py")
		if _, err := os.Stat(candidate); err == nil {
			return filepath.Dir(candidate)
		}
		// If exe is in project root
		candidate = filepath.Join(dir, "service.py")
		if _, err := os.Stat(candidate); err == nil {
			return dir
		}
	}
	// Fall back to cwd
	cwd, _ := os.Getwd()
	candidate := filepath.Join(cwd, "service.py")
	if _, err := os.Stat(candidate); err == nil {
		return cwd
	}
	// Try parent of cwd (if running from cli/)
	parent := filepath.Dir(cwd)
	return parent
}

// WaitForHealth polls the service health endpoint until ready or timeout.
func WaitForHealth(url string, timeout time.Duration) error {
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
