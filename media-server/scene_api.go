package main

// scene_api.go — the Scene editor's API (/api/scenes/...).
//
// A scene is a layered composition over one still that is edited and
// animated a masked region at a time (see tasks/scene.go). The editor window
// drives everything through these routes; generations and exports are
// ordinary jobs on the queue.
//
//   GET    /api/scenes?source=P                     scenes started from P (all when empty)
//   POST   /api/scenes {source, time, fresh}         open the latest scene for source, or start one
//   GET    /api/scenes/{id}                          the document (finished jobs folded in)
//   PUT    /api/scenes/{id}                          save the editor's fields
//   DELETE /api/scenes/{id}                          delete the scene and its files
//   POST   /api/scenes/{id}/layers/{lid}/generate    queue the layer's next generation
//   POST   /api/scenes/{id}/layers/{lid}/cancel      cancel it
//   POST   /api/scenes/{id}/layers/{lid}/versions/{n}/loop {kind}  seamless-loop copy of a motion version
//                                                               (crossfade | natural | boomerang)
//   POST   /api/scenes/{id}/layers/{lid}/versions/{n}/motion    where a motion version moves {dataUrl, region}
//   DELETE /api/scenes/{id}/layers/{lid}/versions/{n}            drop a version
//   DELETE /api/scenes/{id}/layers/{lid}/versions?keep=N         drop every version but N
//   POST   /api/scenes/{id}/render {format}          export video | still beside the source

import (
	"errors"
	"net/http"
	"strconv"

	"github.com/stevecastle/shrike/renderer"
	"github.com/stevecastle/shrike/tasks"
)

func RegisterSceneRoutes(mux *http.ServeMux, deps *Dependencies) {
	mux.HandleFunc("/api/scenes", renderer.ApplyMiddlewares(scenesHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/scenes/{id}", renderer.ApplyMiddlewares(sceneHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/scenes/{id}/render", renderer.ApplyMiddlewares(sceneRenderHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/scenes/{id}/layers/{lid}/{action}", renderer.ApplyMiddlewares(sceneLayerActionHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/scenes/{id}/layers/{lid}/versions", renderer.ApplyMiddlewares(sceneKeepOnlyHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/scenes/{id}/layers/{lid}/versions/{n}", renderer.ApplyMiddlewares(sceneVersionHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/scenes/{id}/layers/{lid}/versions/{n}/{action}", renderer.ApplyMiddlewares(sceneVersionActionHandler(deps), renderer.RoleAdmin))
}

func sceneError(w http.ResponseWriter, err error) {
	code := http.StatusBadRequest
	switch {
	case errors.Is(err, tasks.ErrSceneNotFound), errors.Is(err, tasks.ErrLayerNotFound):
		code = http.StatusNotFound
	case errors.Is(err, tasks.ErrSceneBusy):
		code = http.StatusConflict
	}
	w.Header().Set("Content-Type", "application/json")
	httpError(w, err.Error(), code)
}

func sceneReply(w http.ResponseWriter, s *tasks.Scene, err error) {
	if err != nil {
		sceneError(w, err)
		return
	}
	writeJSON(w, s)
}

func scenesHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			writeJSON(w, tasks.ListScenes(r.URL.Query().Get("source")))
		case http.MethodPost:
			var req struct {
				Source string  `json:"source"`
				Time   float64 `json:"time"`
				Fresh  bool    `json:"fresh"`
			}
			if err := readJSONBody(r, &req); err != nil || req.Source == "" {
				sceneError(w, errors.New("needs a source path"))
				return
			}
			if !req.Fresh {
				if list := tasks.ListScenes(req.Source); len(list) > 0 {
					s, err := tasks.GetScene(deps.Queue, list[0].ID)
					sceneReply(w, s, err)
					return
				}
			}
			s, err := tasks.CreateScene(r.Context(), req.Source, req.Time)
			sceneReply(w, s, err)
		default:
			http.Error(w, "Use GET or POST", http.StatusMethodNotAllowed)
		}
	}
}

func sceneHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		switch r.Method {
		case http.MethodGet:
			s, err := tasks.GetScene(deps.Queue, id)
			sceneReply(w, s, err)
		case http.MethodPut:
			var in tasks.Scene
			if err := readJSONBody(r, &in); err != nil {
				sceneError(w, errors.New("bad json"))
				return
			}
			s, err := tasks.UpdateScene(deps.Queue, id, &in)
			sceneReply(w, s, err)
		case http.MethodDelete:
			if err := tasks.DeleteScene(deps.Queue, id); err != nil {
				sceneError(w, err)
				return
			}
			writeJSON(w, map[string]bool{"ok": true})
		default:
			http.Error(w, "Use GET, PUT or DELETE", http.StatusMethodNotAllowed)
		}
	}
}

func sceneRenderHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Format string `json:"format"`
		}
		_ = readJSONBody(r, &req)
		s, err := tasks.QueueSceneRender(deps.Queue, r.PathValue("id"), req.Format)
		sceneReply(w, s, err)
	}
}

func sceneLayerActionHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		id, lid := r.PathValue("id"), r.PathValue("lid")
		switch r.PathValue("action") {
		case "generate":
			s, err := tasks.GenerateLayer(r.Context(), deps.Queue, id, lid)
			sceneReply(w, s, err)
		case "cancel":
			s, err := tasks.CancelLayer(deps.Queue, id, lid)
			sceneReply(w, s, err)
		default:
			http.NotFound(w, r)
		}
	}
}

func sceneVersionN(w http.ResponseWriter, r *http.Request) (int, bool) {
	n, err := strconv.Atoi(r.PathValue("n"))
	if err != nil {
		sceneError(w, errors.New("bad version"))
		return 0, false
	}
	return n, true
}

func sceneVersionHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "Use DELETE", http.StatusMethodNotAllowed)
			return
		}
		n, ok := sceneVersionN(w, r)
		if !ok {
			return
		}
		s, err := tasks.DeleteVersion(r.PathValue("id"), r.PathValue("lid"), n)
		sceneReply(w, s, err)
	}
}

func sceneKeepOnlyHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "Use DELETE", http.StatusMethodNotAllowed)
			return
		}
		n, err := strconv.Atoi(r.URL.Query().Get("keep"))
		if err != nil {
			sceneError(w, errors.New("needs ?keep=<version index>"))
			return
		}
		s, err := tasks.KeepOnlyVersion(r.PathValue("id"), r.PathValue("lid"), n)
		sceneReply(w, s, err)
	}
}

func sceneVersionActionHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		n, ok := sceneVersionN(w, r)
		if !ok {
			return
		}
		id, lid := r.PathValue("id"), r.PathValue("lid")
		switch r.PathValue("action") {
		case "loop":
			var req struct {
				Kind string `json:"kind"`
			}
			_ = readJSONBody(r, &req)
			s, err := tasks.LoopVersion(r.Context(), id, lid, n, req.Kind)
			sceneReply(w, s, err)
		case "boomerang":
			s, err := tasks.LoopVersion(r.Context(), id, lid, n, tasks.LoopBoomerang)
			sceneReply(w, s, err)
		case "motion":
			url, region, err := tasks.VersionMotionMap(r.Context(), id, lid, n)
			if err != nil {
				sceneError(w, err)
				return
			}
			writeJSON(w, map[string]any{"dataUrl": url, "region": region})
		default:
			http.NotFound(w, r)
		}
	}
}
