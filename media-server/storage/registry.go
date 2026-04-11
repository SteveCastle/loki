package storage

import "sync"

// Registry routes paths to the correct Backend by consulting each backend's
// Contains method in order. Backends are checked in the order they were registered.
type Registry struct {
	mu         sync.RWMutex
	backends   []Backend
	defaultIdx int // index of the default backend, or -1 if none explicitly set
}

// NewRegistry creates a Registry pre-populated with the provided backends.
func NewRegistry(backends []Backend) *Registry {
	cp := make([]Backend, len(backends))
	copy(cp, backends)
	return &Registry{backends: cp, defaultIdx: -1}
}

// DefaultBackend returns the backend designated as the default destination for
// uploads and downloads. Falls back to the first backend if none is explicitly
// marked. Returns nil if the registry is empty.
func (r *Registry) DefaultBackend() Backend {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if len(r.backends) == 0 {
		return nil
	}
	if r.defaultIdx >= 0 && r.defaultIdx < len(r.backends) {
		return r.backends[r.defaultIdx]
	}
	return r.backends[0]
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
	r.ReplaceWithDefault(backends, -1)
}

// ReplaceWithDefault atomically swaps the backend list and sets the default index.
func (r *Registry) ReplaceWithDefault(backends []Backend, defaultIdx int) {
	cp := make([]Backend, len(backends))
	copy(cp, backends)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.backends = cp
	r.defaultIdx = defaultIdx
}

// AllBackends returns a snapshot copy of the current backend slice.
func (r *Registry) AllBackends() []Backend {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cp := make([]Backend, len(r.backends))
	copy(cp, r.backends)
	return cp
}
