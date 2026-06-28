package onnxtag

import (
	"strings"

	"github.com/eliben/go-sentencepiece"
)

// BuildTextInputIDs tokenizes text per the SigLIP 2 recipe: lowercase →
// SentencePiece encode → append EOS → pad/truncate to seqLen with PAD (no BOS).
// Returns int64 ids of length seqLen. EOS/PAD are read from the model.
func BuildTextInputIDs(proc *sentencepiece.Processor, text string, seqLen int) []int64 {
	info := proc.ModelInfo()
	toks := proc.Encode(strings.ToLower(text))
	ids := make([]int64, 0, seqLen)
	for _, t := range toks {
		ids = append(ids, int64(t.ID))
	}
	if len(ids) > seqLen-1 {
		ids = ids[:seqLen-1]
	}
	ids = append(ids, int64(info.EndOfSentenceID))
	for len(ids) < seqLen {
		ids = append(ids, int64(info.PadID))
	}
	return ids
}
