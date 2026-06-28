package onnxtag

import (
	"testing"

	"github.com/eliben/go-sentencepiece"
	"github.com/stevecastle/shrike/deps"
)

func TestBuildTextInputIDsMatchesReference(t *testing.T) {
	tk, err := deps.ModelPath("siglip2-base-patch16-224", "tokenizer.model")
	if err != nil {
		t.Skip("tokenizer not installed; skipping reference check")
	}
	proc, err := sentencepiece.NewProcessorFromPath(tk)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	ids := BuildTextInputIDs(proc, "a red square", 64)
	if len(ids) != 64 {
		t.Fatalf("len=%d want 64", len(ids))
	}
	// Confirmed reference: sp ids [235250,3118,7800] + eos(1) + pad(0)...
	want := []int64{235250, 3118, 7800, 1, 0, 0, 0, 0}
	for i, w := range want {
		if ids[i] != w {
			t.Errorf("ids[%d]=%d want %d (full head=%v)", i, ids[i], w, ids[:8])
		}
	}
}
