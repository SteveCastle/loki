package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
	_ "modernc.org/sqlite"
)

func newFacesTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	return db
}

func TestFaceHitItemsCollapsesAndAttachesFace(t *testing.T) {
	db := newFacesTestDB(t)
	for _, p := range []string{"a.jpg", "b.jpg"} {
		if _, err := db.Exec(`INSERT INTO media (path, width, height) VALUES (?,?,?)`, p, 100, 200); err != nil {
			t.Fatal(err)
		}
	}
	hits := []tasks.FaceHit{
		{FaceID: 1, MediaPath: "a.jpg", Score: 0.5, X: 0.1, Y: 0.2, W: 0.3, H: 0.4},
		{FaceID: 2, MediaPath: "b.jpg", Score: 0.9, X: 0.5, Y: 0.5, W: 0.1, H: 0.1},
		{FaceID: 3, MediaPath: "a.jpg", Score: 0.7, X: 0.6, Y: 0.6, W: 0.2, H: 0.2},
	}
	items, err := faceHitItems(db, hits)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 {
		t.Fatalf("got %d items, want 2 (collapsed per path)", len(items))
	}
	if items[0]["path"] != "b.jpg" {
		t.Fatalf("first item = %v, want b.jpg (best score)", items[0]["path"])
	}
	// a.jpg carries its BEST face (id 3, score 0.7), not the weaker one.
	if items[1]["path"] != "a.jpg" || items[1]["faceId"] != int64(3) {
		t.Fatalf("a.jpg item = %+v, want faceId 3", items[1])
	}
	face, _ := items[1]["face"].(map[string]any)
	if face == nil || face["x"] != 0.6 {
		t.Fatalf("face bbox not attached: %+v", items[1]["face"])
	}
}

func TestFacesForPathHandler(t *testing.T) {
	db := newFacesTestDB(t)
	model := tasks.ActiveFaceModel()
	if _, err := media.ReplaceFaces(db, "a.jpg", model.ID, []media.NewFace{
		{X: 0.1, Y: 0.2, W: 0.3, H: 0.4, Score: 0.9, Vec: []float32{1, 0}},
	}, 1); err != nil {
		t.Fatal(err)
	}
	deps := &Dependencies{DB: db}

	rec := httptest.NewRecorder()
	facesForPathHandler(deps)(rec, httptest.NewRequest(http.MethodGet, "/api/faces?path=a.jpg", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Model   string           `json:"model"`
		Scanned bool             `json:"scanned"`
		Faces   []map[string]any `json:"faces"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Model != model.ID || !resp.Scanned || len(resp.Faces) != 1 {
		t.Fatalf("resp = %+v", resp)
	}

	// Unscanned path: scanned=false, empty faces.
	rec = httptest.NewRecorder()
	facesForPathHandler(deps)(rec, httptest.NewRequest(http.MethodGet, "/api/faces?path=never.jpg", nil))
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Scanned || len(resp.Faces) != 0 {
		t.Fatalf("unscanned resp = %+v", resp)
	}

	// Missing path param → 400.
	rec = httptest.NewRecorder()
	facesForPathHandler(deps)(rec, httptest.NewRequest(http.MethodGet, "/api/faces", nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing path status = %d", rec.Code)
	}
}

func TestFaceCropHandler(t *testing.T) {
	db := newFacesTestDB(t)
	model := tasks.ActiveFaceModel()

	// A 200×100 image, red in the face region, blue elsewhere.
	imgPath := filepath.Join(t.TempDir(), "face.png")
	img := image.NewRGBA(image.Rect(0, 0, 200, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 200; x++ {
			if x >= 100 && x < 160 && y >= 20 && y < 80 {
				img.Set(x, y, color.RGBA{255, 0, 0, 255})
			} else {
				img.Set(x, y, color.RGBA{0, 0, 255, 255})
			}
		}
	}
	fh, err := os.Create(imgPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(fh, img); err != nil {
		t.Fatal(err)
	}
	fh.Close()

	// Face bbox = the red region, relative: x=0.5 w=0.3, y=0.2 h=0.6.
	ids, err := media.ReplaceFaces(db, imgPath, model.ID, []media.NewFace{
		{X: 0.5, Y: 0.2, W: 0.3, H: 0.6, Score: 0.9, Vec: []float32{1, 0}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	deps := &Dependencies{DB: db}

	rec := httptest.NewRecorder()
	faceCropHandler(deps)(rec, httptest.NewRequest(http.MethodGet, "/media/facecrop?id="+jsonNum(ids[0]), nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "image/jpeg" {
		t.Fatalf("content type = %q", ct)
	}
	crop, err := jpegDecode(rec.Body.Bytes())
	if err != nil {
		t.Fatalf("crop not decodable: %v", err)
	}
	cb := crop.Bounds()
	if cb.Dx() < 1 || cb.Dx() > 160 || cb.Dy() < 1 || cb.Dy() > 160 {
		t.Fatalf("crop dims %dx%d out of range", cb.Dx(), cb.Dy())
	}
	// Center pixel must be from the face (red-ish), proving the bbox math.
	c := color.RGBAModel.Convert(crop.At(cb.Dx()/2, cb.Dy()/2)).(color.RGBA)
	if c.R < 150 || c.B > 100 {
		t.Fatalf("crop center = %+v, want red-ish face region", c)
	}

	// Unknown id → 404.
	rec = httptest.NewRecorder()
	faceCropHandler(deps)(rec, httptest.NewRequest(http.MethodGet, "/media/facecrop?id=999999", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unknown id status = %d", rec.Code)
	}
}

func jsonNum(v int64) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func jpegDecode(b []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(b))
	return img, err
}
