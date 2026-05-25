package optional

func hintFor(o Optional) InstallHint {
	h := InstallHint{Description: o.Description, DocsURL: o.DocsURL}
	switch o.ID {
	case "yt-dlp":
		h.Commands = []OSCmd{
			{OS: "darwin", Label: "Homebrew", Command: "brew install yt-dlp"},
			{OS: "windows", Label: "winget", Command: "winget install yt-dlp"},
			{OS: "linux", Label: "pipx", Command: "pipx install yt-dlp"},
		}
	case "gallery-dl":
		h.Commands = []OSCmd{
			{OS: "darwin", Label: "Homebrew", Command: "brew install gallery-dl"},
			{OS: "windows", Label: "pip", Command: "pip install --user gallery-dl"},
			{OS: "linux", Label: "pipx", Command: "pipx install gallery-dl"},
		}
	case "ollama":
		h.Commands = []OSCmd{
			{OS: "darwin", Label: "Homebrew", Command: "brew install ollama"},
			{OS: "windows", Label: "Download", Command: "Download installer from https://ollama.com/download/windows"},
			{OS: "linux", Label: "Shell installer", Command: "curl -fsSL https://ollama.com/install.sh | sh"},
		}
	case "dce":
		h.Commands = []OSCmd{
			{OS: "darwin", Label: "Homebrew", Command: "brew install --cask discord-chat-exporter"},
			{OS: "windows", Label: "Releases", Command: "Download DiscordChatExporter.Cli from https://github.com/Tyrrrz/DiscordChatExporter/releases and add to PATH as 'dce'"},
			{OS: "linux", Label: "Releases", Command: "Download DiscordChatExporter.Cli from https://github.com/Tyrrrz/DiscordChatExporter/releases and add to PATH as 'dce'"},
		}
	}
	return h
}
