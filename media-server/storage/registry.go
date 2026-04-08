package storage

import "sync"

// Registry routes paths to the correct Backend by consulting each backend's
// Contains method in order. Backends are checked in the order they were registered.
type Registry struct {
	mu       sync.RWMutex
	backends []Backend
}

// NewRegistry creates a Registry pre-populated with the provided backends.
func NewRegistry(backends []Backend) *Registry {
	cp := make([]Backend, len(backends))
	copy(cp, backends)
	return &Registry{backends: cp}
}

// BackendFor returns the first backend whose Contains method returns true for
// path, or nil if no backend claims the path.
func (r *Registry) BackendFor(path string) Backend {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, b := range r.backends {
		if b.Contains(path) {
			return b
		}
	}
	return nil
}

// AllRoots returns the Root entry from every registered backend.
func (r *Registry) AllRoots() []Entry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	roots := make([]Entry, len(r.backends))
	for i, b := range r.backends {
		roots[i] = b.Root()
	}
	return roots
}

// Replace atomically swaps the full backend list (used on config reload).
func (r *Registry) Replace(backends []Backend) {
	cp := make([]Backend, len(backends))
	copy(cp, backends)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.backends = cp
}

// AllBackends returns a snapshot copy of the current backend slice.
func (r *Registry) AllBackends() []Backend {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cp := make([]Backend, len(r.backends))
	copy(cp, r.backends)
	return cp
}
