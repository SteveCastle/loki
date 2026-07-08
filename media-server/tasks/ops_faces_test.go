package tasks

import "testing"

// TestNoteFacesScanned pins the in-scan clustering cadence: a pass becomes
// due once clusterEvery faces accumulate (across models), the due pass
// carries every model with new faces, and counters reset afterward.
func TestNoteFacesScanned(t *testing.T) {
	st := &facesOpState{clusterEvery: 100}

	if due, _, _ := st.noteFacesScanned("sface", 99); due {
		t.Fatal("99 faces must not trigger a pass")
	}
	due, rebuild, models := st.noteFacesScanned("ccip", 1)
	if !due {
		t.Fatal("the 100th face must trigger a pass")
	}
	if rebuild {
		t.Fatal("rebuilds are disabled (rebuildEvery=0); the pass must be incremental")
	}
	if len(models) != 2 {
		t.Fatalf("pass must cover every model with new faces, got %v", models)
	}

	// Counters reset: the next 99 faces don't retrigger.
	if due, _, _ := st.noteFacesScanned("sface", 99); due {
		t.Fatal("counters must reset after a pass")
	}
	// The tail is visible for the Finalize pass.
	if residue := st.takeDirtyModels(); len(residue) != 1 || residue[0] != "sface" {
		t.Fatalf("expected sface residue for the final pass, got %v", residue)
	}
	// takeDirtyModels drains.
	if residue := st.takeDirtyModels(); len(residue) != 0 {
		t.Fatalf("expected drained residue, got %v", residue)
	}
}

// TestNoteFacesScannedRebuildCadence pins the periodic-rebuild cadence: every
// rebuildEvery faces the due pass is a REBUILD (reset + full regroup), it
// takes priority over a simultaneously-due incremental pass, and the two
// counters run independently.
func TestNoteFacesScannedRebuildCadence(t *testing.T) {
	st := &facesOpState{clusterEvery: 100, rebuildEvery: 300}

	// Faces 1-100, 101-200: ordinary incremental passes.
	for pass := 1; pass <= 2; pass++ {
		due, rebuild, _ := st.noteFacesScanned("sface", 100)
		if !due || rebuild {
			t.Fatalf("pass %d: due=%v rebuild=%v, want incremental", pass, due, rebuild)
		}
	}
	// Face 300: both cadences hit — the rebuild wins.
	due, rebuild, models := st.noteFacesScanned("sface", 100)
	if !due || !rebuild {
		t.Fatalf("300th face: due=%v rebuild=%v, want a rebuild", due, rebuild)
	}
	if len(models) != 1 || models[0] != "sface" {
		t.Fatalf("rebuild models = %v", models)
	}
	// Cadence restarts cleanly after the rebuild.
	if due, _, _ := st.noteFacesScanned("sface", 99); due {
		t.Fatal("counters must reset after the rebuild")
	}
	due, rebuild, _ = st.noteFacesScanned("sface", 1)
	if !due || rebuild {
		t.Fatalf("post-rebuild 100th face: due=%v rebuild=%v, want incremental", due, rebuild)
	}

	// Rebuild-only mode (incremental preview off) still rebuilds on cadence.
	st2 := &facesOpState{clusterEvery: 0, rebuildEvery: 200}
	if due, _, _ := st2.noteFacesScanned("sface", 199); due {
		t.Fatal("rebuild-only: 199 faces must not trigger")
	}
	due, rebuild, _ = st2.noteFacesScanned("sface", 1)
	if !due || !rebuild {
		t.Fatalf("rebuild-only 200th face: due=%v rebuild=%v, want a rebuild", due, rebuild)
	}
}

// TestIncrementalClusterParamsAreStricter pins the false-positive guard:
// mid-scan passes run dozens of times per scan on small batches, so every
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

// TestNoteFacesScannedDisabled: cluster-every=0 + rebuild-every=0 never
// triggers in-scan passes (the Finalize path queues the classic faces-cluster
// job instead), but still tracks residue so nothing is lost if re-enabled.
func TestNoteFacesScannedDisabled(t *testing.T) {
	st := &facesOpState{clusterEvery: 0, rebuildEvery: 0}
	for i := 0; i < 5; i++ {
		if due, _, _ := st.noteFacesScanned("sface", 100); due {
			t.Fatal("disabled mode must never trigger an in-scan pass")
		}
	}
	// Zero-face items never count.
	st2 := &facesOpState{clusterEvery: 1}
	if due, _, _ := st2.noteFacesScanned("sface", 0); due {
		t.Fatal("items with no faces must not count toward the cadence")
	}
}
