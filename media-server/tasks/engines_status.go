package tasks

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/workerapi"
)

// EngineTargetStatus is one place engines run, as the GPU Engines config tab shows it.
type EngineTargetStatus struct {
	Key      string                  `json:"key"`
	Pool     string                  `json:"pool,omitempty"`
	Kind     string                  `json:"kind"`
	URL      string                  `json:"url,omitempty"`
	Healthy  bool                    `json:"healthy"`
	Checked  time.Time               `json:"checked,omitzero"`
	Error    string                  `json:"error,omitempty"`
	Running  int                     `json:"running"`
	Slots    int                     `json:"slots"`
	Info     *workerapi.Info         `json:"info,omitempty"`
	RunPod   *workerapi.RunPodHealth `json:"runpod,omitempty"`
	Engines  []string                `json:"engines,omitempty"`
	Transfer string                  `json:"transfer,omitempty"`
}

// EngineStatus is the dispatcher's state.
type EngineStatus struct {
	Strategy  string               `json:"strategy"`
	Routes    map[string]string    `json:"routes,omitempty"`
	Fallback  bool                 `json:"fallback"`
	LocalOnly bool                 `json:"localOnly"`
	Placement map[string]string    `json:"placement"`
	Targets   []EngineTargetStatus `json:"targets"`
}

// EnginesStatus reports every target and where each engine goes.
func EnginesStatus() EngineStatus {
	d := engineDisp
	d.mu.Lock()
	cfg := d.cfg
	targets := append([]*engineTarget{d.local}, d.targets...)
	d.mu.Unlock()
	st := EngineStatus{Strategy: cfg.Strategy, Routes: cfg.Routes, Fallback: cfg.Fallback, LocalOnly: cfg.LocalOnly, Placement: map[string]string{}}
	for _, e := range workerapi.Engines {
		st.Placement[e] = d.route(e)
	}
	for _, t := range targets {
		s := EngineTargetStatus{Key: t.key, Pool: t.pool, Kind: t.kind, Slots: t.slots(), Engines: t.engines}
		t.mu.Lock()
		s.Healthy = t.kind == "local" || t.healthy
		s.Checked, s.Error, s.Running, s.Info, s.RunPod = t.checked, t.lastErr, t.inflight, t.info, t.rpHealth
		t.mu.Unlock()
		if t.node != nil {
			s.URL = t.node.URL
		}
		if t.transfer != nil {
			switch {
			case t.transfer.Root != "":
				s.Transfer = "storage root " + t.transfer.Root
			case t.transfer.Bucket != "":
				s.Transfer = "bucket " + t.transfer.Bucket
			}
		} else if t.kind == "runpod" {
			s.Transfer = "inline (files up to 7 MB)"
		}
		st.Targets = append(st.Targets, s)
	}
	return st
}

// CheckEngines probes every worker now.
func CheckEngines(ctx context.Context) EngineStatus {
	engineDisp.checkAll(ctx)
	return EnginesStatus()
}

// WarmEngines asks the workers of a pool (every pool when empty) to fetch
// and load engines' models ahead of the first job. Serverless endpoints get
// a warm job (it also downloads the models to their volume).
func WarmEngines(ctx context.Context, pool string, engines []string, model string) []string {
	d := engineDisp
	d.mu.Lock()
	targets := append([]*engineTarget(nil), d.targets...)
	d.mu.Unlock()
	var notes []string
	for _, t := range targets {
		if pool != "" && !strings.EqualFold(t.pool, pool) {
			continue
		}
		var err error
		switch t.kind {
		case "node":
			err = t.node.Warm(ctx, workerapi.WarmRequest{Tools: engines, Model: model})
		case "runpod":
			_, err = t.rp.Run(ctx, workerapi.RunPodInput{Warm: &workerapi.WarmRequest{Tools: engines, Model: model}})
		}
		if err != nil {
			notes = append(notes, fmt.Sprintf("%s: %v", t.key, err))
		} else {
			notes = append(notes, t.key+": warming "+strings.Join(engines, ", "))
		}
	}
	return notes
}

// ProbeWorker checks a worker before it is added: a node's info, or a
// RunPod endpoint's health.
func ProbeWorker(ctx context.Context, kind, url, token, endpoint, apiKey string) (any, error) {
	if kind == appconfig.PoolKindRunPod {
		return (&workerapi.RunPod{Endpoint: endpoint, APIKey: apiKey}).Health(ctx)
	}
	n := &workerapi.Node{Name: url, URL: url, Token: token}
	return n.Info(ctx, false)
}
