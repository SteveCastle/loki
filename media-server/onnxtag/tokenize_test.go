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

// TestBuildTextInputIDsEOSOnMaxLength verifies that a token sequence longer
// than seqLen-1 is truncated so EOS always occupies the last slot.
func TestBuildTextInputIDsEOSOnMaxLength(t *testing.T) {
	tk, err := deps.ModelPath("siglip2-base-patch16-224", "tokenizer.model")
	if err != nil {
		t.Skip("tokenizer not installed; skipping max-length EOS check")
	}
	proc, err := sentencepiece.NewProcessorFromPath(tk)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	// Use a short seqLen so a normal sentence exceeds it.
	const seqLen = 4
	// "a red square" tokenizes to 3 tokens; with seqLen=4 that fits (3 < 4-1=3 is false, exactly 3=3),
	// so use a 4-token phrase to guarantee overflow.
	ids := BuildTextInputIDs(proc, "a red square field", seqLen)
	if len(ids) != seqLen {
		t.Fatalf("len=%d want %d", len(ids), seqLen)
	}
	info := proc.ModelInfo()
	if ids[seqLen-1] != int64(info.EndOfSentenceID) {
		t.Errorf("ids[seqLen-1]=%d want EOS=%d (full=%v)", ids[seqLen-1], info.EndOfSentenceID, ids)
	}
}
