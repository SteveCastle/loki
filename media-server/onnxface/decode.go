package onnxface

import (
	"image"
	_ "image/gif"  // register decoders: the face task hands us whatever the
	_ "image/jpeg" // library holds (or an extracted video frame, which is jpeg)
	_ "image/png"
	"os"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
)

// decodeImageFile opens and decodes one image file.
func decodeImageFile(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

// fitWithin returns the scale factor (<= 1) that fits w×h within maxSide on
// its longest edge; 1 when it already fits or maxSide <= 0.
func fitWithin(w, h, maxSide int) float64 {
	if maxSide <= 0 {
		return 1
	}
	longest := w
	if h > longest {
		longest = h
	}
	if longest <= maxSide {
		return 1
	}
	return float64(maxSide) / float64(longest)
}
