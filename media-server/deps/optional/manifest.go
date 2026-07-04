package optional

var Manifest = []Optional{
	{
		ID:          "yt-dlp",
		Name:        "yt-dlp",
		Binary:      "yt-dlp",
		VersionArgs: []string{"--version"},
		Feature:     "Import from YouTube & video sites",
		Description: "Video downloader. Required to import from YouTube, Twitch, and many other sites.",
		DocsURL:     "https://github.com/yt-dlp/yt-dlp#installation",
	},
	{
		ID:          "gallery-dl",
		Name:        "gallery-dl",
		Binary:      "gallery-dl",
		VersionArgs: []string{"--version"},
		Feature:     "Import from image galleries",
		Description: "Image gallery downloader for sites like DeviantArt, Pixiv, Reddit.",
		DocsURL:     "https://github.com/mikf/gallery-dl#installation",
	},
	{
		ID:          "ollama",
		Name:        "Ollama",
		Binary:      "ollama",
		VersionArgs: []string{"--version"},
		Feature:     "AI descriptions & vision tagging",
		Description: "Local large language model runtime. Enables AI captioning and chat features.",
		DocsURL:     "https://ollama.com/download",
	},
	{
		ID:          "dce",
		Name:        "DiscordChatExporter",
		Binary:      "dce",
		VersionArgs: nil,
		Feature:     "Import Discord channels",
		Description: "Discord Chat Exporter CLI. Required to ingest Discord channel exports.",
		DocsURL:     "https://github.com/Tyrrrz/DiscordChatExporter",
	},
}
