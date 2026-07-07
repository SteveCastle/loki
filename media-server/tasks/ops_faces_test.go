package tasks

import "testing"

// TestNoteFacesScanned pins the in-scan clustering cadence: a pass becomes
// due once clusterEvery faces accumulate (across models), the due pass
// carries every model with new faces, and counters reset afterward.
func TestNoteFacesScanned(t *testing.T) {
	st := &facesOpState{clusterEvery: 100}

	if due, _ := st.noteFacesScanned("sface", 99); due {
		t.Fatal("99 faces must not trigger a pass")
	}
	due, models := st.noteFacesScanned("ccip", 1)
	if !due {
		t.Fatal("the 100th face must trigger a pass")
	}
	if len(models) != 2 {
		t.Fatalf("pass must cover every model with new faces, got %v", models)
	}

	// Counters reset: the next 99 faces don't retrigger.
	if due, _ := st.noteFacesScanned("sface", 99); due {
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

// TestNoteFacesScannedDisabled: cluster-every=0 never triggers in-scan
// passes (the Finalize path queues the classic faces-cluster job instead),
// but still tracks residue so nothing is lost if re-enabled.
func TestNoteFacesScannedDisabled(t *testing.T) {
	st := &facesOpState{clusterEvery: 0}
	for i := 0; i < 5; i++ {
		if due, _ := st.noteFacesScanned("sface", 100); due {
			t.Fatal("disabled mode must never trigger an in-scan pass")
		}
	}
	// Zero-face items never count.
	st2 := &facesOpState{clusterEvery: 1}
	if due, _ := st2.noteFacesScanned("sface", 0); due {
		t.Fatal("items with no faces must not count toward the cadence")
	}
}
