package tasks

import "testing"

// fakeFaceIDs builds n distinct face ids starting at base, for cadence tests.
func fakeFaceIDs(base int64, n int) []int64 {
	ids := make([]int64, n)
	for i := range ids {
		ids[i] = base + int64(i)
	}
	return ids
}

// TestNoteFacesScanned pins the in-scan clustering cadence: a pass becomes
// due once clusterEvery faces accumulate (across models), the due pass
// carries every model's batch of new face ids (so the pass can restrict
// itself to them), and counters reset afterward.
func TestNoteFacesScanned(t *testing.T) {
	st := &facesOpState{clusterEvery: 100}

	if due, _ := st.noteFacesScanned("sface", fakeFaceIDs(1, 99)); due {
		t.Fatal("99 faces must not trigger a pass")
	}
	due, batches := st.noteFacesScanned("ccip", fakeFaceIDs(1000, 1))
	if !due {
		t.Fatal("the 100th face must trigger a pass")
	}
	if len(batches) != 2 {
		t.Fatalf("pass must cover every model with new faces, got %v", batches)
	}
	if len(batches["sface"]) != 99 || len(batches["ccip"]) != 1 {
		t.Fatalf("batches must carry each model's new face ids: sface=%d ccip=%d",
			len(batches["sface"]), len(batches["ccip"]))
	}
	if batches["sface"][0] != 1 || batches["ccip"][0] != 1000 {
		t.Fatalf("batch ids mismatch: %v", batches)
	}

	// Counters reset: the next 99 faces don't retrigger.
	if due, _ := st.noteFacesScanned("sface", fakeFaceIDs(200, 99)); due {
		t.Fatal("counters must reset after a pass")
	}
	// The tail is visible for the Finalize pass.
	if residue := st.takeDirtyBatches(); len(residue) != 1 || len(residue["sface"]) != 99 {
		t.Fatalf("expected 99-face sface residue for the final pass, got %v", residue)
	}
	// takeDirtyBatches drains.
	if residue := st.takeDirtyBatches(); len(residue) != 0 {
		t.Fatalf("expected drained residue, got %v", residue)
	}
}

// TestIncrementalClusterParamsAreStricter pins the false-positive guard:
// mid-scan passes run many times per scan on small batches, so every
// acceptance knob must be strictly tighter than the one-shot defaults.
func TestIncrementalClusterParamsAreStricter(t *testing.T) {
	m := FaceModel{ID: "test", MatchThreshold: 0.42}
	def := defaultClusterParams(m)
	inc := incrementalClusterParams(m)
	if inc.joinThreshold <= def.joinThreshold {
		t.Errorf("incremental joinThreshold %v must exceed default %v", inc.joinThreshold, def.joinThreshold)
	}
	if inc.formThreshold <= def.formThreshold {
		t.Errorf("incremental formThreshold %v must exceed default %v", inc.formThreshold, def.formThreshold)
	}
	if inc.minCluster <= def.minCluster {
		t.Errorf("incremental minCluster %d must exceed default %d", inc.minCluster, def.minCluster)
	}
	if inc.minQuality < def.minQuality {
		t.Errorf("incremental minQuality %v must not be below default %v", inc.minQuality, def.minQuality)
	}
	if inc.passes >= def.passes {
		t.Errorf("incremental passes %d must be below default %d (no repeated transitivity)", inc.passes, def.passes)
	}
}

// TestNoteFacesScannedDisabled: cluster-every=0 never triggers in-scan passes
// (the Finalize path queues the classic faces-cluster job instead), but still
// tracks residue so nothing is lost if re-enabled.
func TestNoteFacesScannedDisabled(t *testing.T) {
	st := &facesOpState{clusterEvery: 0}
	for i := 0; i < 5; i++ {
		if due, _ := st.noteFacesScanned("sface", fakeFaceIDs(int64(i*100), 100)); due {
			t.Fatal("disabled mode must never trigger an in-scan pass")
		}
	}
	// Zero-face items never count.
	st2 := &facesOpState{clusterEvery: 1}
	if due, _ := st2.noteFacesScanned("sface", nil); due {
		t.Fatal("items with no faces must not count toward the cadence")
	}
}
