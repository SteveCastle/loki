package downloads

// DownloadStatus represents the current state of a download.
type DownloadStatus string

const (
	StatusPending     DownloadStatus = "pending"
	StatusDownloading DownloadStatus = "downloading"
	StatusExtracting  DownloadStatus = "extracting"
	StatusComplete    DownloadStatus = "complete"
	StatusError       DownloadStatus = "error"
	StatusCancelled   DownloadStatus = "cancelled"
)

// Progress represents the current progress of a single dependency download.
type Progress struct {
	DependencyID    string         `json:"dependency_id"`
	DependencyName  string         `json:"dependency_name"`
	Status          DownloadStatus `json:"status"`
	Message         string         `json:"message"`
	BytesDownloaded int64          `json:"bytes_downloaded"`
	TotalBytes      int64          `json:"total_bytes"`
	Percent         float64        `json:"percent"`
	Speed           int64          `json:"speed"` // bytes/sec
	Error           string         `json:"error,omitempty"`
}

// OverallProgress represents the combined progress of all dependency downloads.
type OverallProgress struct {
	TotalDeps      int        `json:"total_deps"`
	CompletedCount int        `json:"completed_count"`
	OverallPercent float64    `json:"overall_percent"`
	Dependencies   []Progress `json:"dependencies"`
	Installing     bool       `json:"installing"`
}

// ProgressCallback is a function called to report download progress.
type ProgressCallback func(Progress)

// ByteProgressCallback is a function called to report raw byte progress during download.
type ByteProgressCallback func(downloaded, total int64)
