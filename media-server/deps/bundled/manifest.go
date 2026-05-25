package bundled

import "runtime"

var Manifest = func() []Bundled {
	exe := ""
	if runtime.GOOS == "windows" {
		exe = ".exe"
	}
	libExt := ".so"
	switch runtime.GOOS {
	case "windows":
		libExt = ".dll"
	case "darwin":
		libExt = ".dylib"
	}

	entries := []Bundled{
		{ID: "ffmpeg", Name: "FFmpeg", RelPath: "ffmpeg" + exe, VersionArgs: []string{"-version"}},
		{ID: "ffprobe", Name: "FFprobe", RelPath: "ffprobe" + exe, VersionArgs: []string{"-version"}},
		{ID: "exiftool", Name: "ExifTool", RelPath: "exiftool" + exe, VersionArgs: []string{"-ver"}},
		{ID: "onnxtag", Name: "ONNX Tagger", RelPath: "onnxtag" + exe, VersionArgs: []string{"--version"}},
		{ID: "onnxruntime", Name: "ONNX Runtime", RelPath: "onnxruntime" + libExt, VersionArgs: nil},
	}
	if runtime.GOOS != "darwin" {
		entries = append(entries, Bundled{ID: "ffplay", Name: "FFplay", RelPath: "ffplay" + exe, VersionArgs: []string{"-version"}})
	}
	return entries
}()
