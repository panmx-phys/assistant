package main

import (
	"strings"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
)

var renderer *glamour.TermRenderer

var (
	borderColor  = lipgloss.Color("74") // steel blue
	subtextColor = lipgloss.Color("67") // dim blue

	panelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(borderColor).
			Padding(0, 1).
			Width(80)

	titleStyle = lipgloss.NewStyle().
			Foreground(borderColor).
			Bold(true)

	subtitleStyle = lipgloss.NewStyle().
			Foreground(subtextColor)
)

func init() {
	r, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(74), // 80 - 2 border - 2 padding
	)
	if err != nil {
		return
	}
	renderer = r
}

func renderMarkdown(text string) string {
	if renderer == nil || strings.TrimSpace(text) == "" {
		return text
	}
	out, err := renderer.Render(text)
	if err != nil {
		return text
	}
	return strings.TrimRight(out, "\n")
}

func renderPanel(content string, title string, subtitle string) string {
	rendered := renderMarkdown(content)

	panel := panelStyle.Render(rendered)

	header := titleStyle.Render("── " + title + " ")
	footer := subtitleStyle.Render(" " + subtitle + " ──")

	return lipgloss.JoinVertical(lipgloss.Left,
		header,
		panel,
		lipgloss.PlaceHorizontal(80, lipgloss.Right, footer),
	)
}
