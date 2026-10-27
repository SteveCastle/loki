package tasks

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/workerapi"
)

// remoteHelp returns the engine's --help as the leased worker prints it.
func remoteHelp(ctx context.Context, tool aiToolRun) (string, error) {
	return tool.lease.t.help(ctx, tool.lease.engine)
}

// transferFor builds a pool's transfer bucket (nil: files go inline).
func (t *engineTarget) transferFor(ctx context.Context) (workerapi.Transfer, error) {
	if t.transfer == nil {
		return nil, nil
	}
	t.mu.Lock()
	tr := t.tr
	t.mu.Unlock()
	if tr != nil {
		return tr, nil
	}
	c := workerapi.S3Config{Endpoint: t.transfer.Endpoint, Region: t.transfer.Region, Bucket: t.transfer.Bucket, Prefix: t.transfer.Prefix, AccessKey: t.transfer.AccessKey, SecretKey: t.transfer.SecretKey}
	if label := strings.TrimSpace(t.transfer.Root); label != "" {
		found := false
		for _, r := range appconfig.Get().Roots {
			if r.Type == "s3" && strings.EqualFold(r.Label, label) {
				c = workerapi.S3Config{Endpoint: r.Endpoint, Region: r.Region, Bucket: r.Bucket, Prefix: strings.Trim(r.Prefix+"/"+t.transfer.Prefix, "/"), AccessKey: r.AccessKey, SecretKey: r.SecretKey}
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("pool %s: no S3 storage root labelled %q for transfers", t.pool, label)
		}
	}
	if c.Bucket == "" {
		return nil, nil
	}
	s3t, err := workerapi.NewS3Transfer(ctx, c)
	if err != nil {
		return nil, err
	}
	t.mu.Lock()
	t.tr = s3t
	t.mu.Unlock()
	return s3t, nil
}

// runRemote runs one engine command line on the leased worker. A job the
// worker never accepted (unreachable, refused) moves to another target of
// the same engine once, and the lease follows it.
func runRemote(ctx context.Context, tool aiToolRun, args []string, light bool, onLog func(string), onStatus func(string)) ([]string, error) {
	l := tool.lease
	for attempt := 0; ; attempt++ {
		plan := workerapi.PlanArgs(args)
		hooks := workerapi.Hooks{Log: onLog, Status: onStatus}
		var res *workerapi.Result
		var err error
		switch l.t.kind {
		case "node":
			res, err = workerapi.RunOnNode(ctx, l.t.node, plan, l.engine, light, hooks)
		case "runpod":
			var tr workerapi.Transfer
			if tr, err = l.t.transferFor(ctx); err == nil {
				res, err = workerapi.RunOnRunPod(ctx, l.t.rp, plan, l.engine, light, tr, hooks)
			}
		default:
			return nil, fmt.Errorf("internal: remote run on the local target")
		}
		if err == nil {
			return res.Stdout, nil
		}
		if !errors.Is(err, workerapi.ErrNotSubmitted) || attempt > 0 || ctx.Err() != nil {
			return nil, err
		}
		// the worker never took the job: mark it down and try another
		l.t.markDown(err)
		if onStatus != nil {
			onStatus(fmt.Sprintf("%v; trying another worker", err))
		}
		l.tried[l.t] = true
		next, aerr := acquireEngine(ctx, l.engine, onStatus, l.tried)
		if aerr != nil {
			return nil, err
		}
		if next.local() {
			next.Release()
			return nil, err
		}
		l.Release()
		next.tried = l.tried
		*l = engineLease{t: next.t, engine: next.engine, tried: next.tried}
	}
}

// remotePreviewTool finds a worker that can answer a quick model-free call
// (--show-prompt) for an engine this machine does not have: a reachable node
// (serverless workers would cold-start for it). The lease is not counted
// against the worker's slots: such calls do not queue behind GPU work there.
func remotePreviewTool(spec aiToolSpec) (aiToolRun, bool) {
	engine := workerapi.EngineName(spec.Bin)
	if engineRunsLocalOnly(engine) {
		return aiToolRun{}, false
	}
	cands, _ := engineDisp.candidates(engine)
	for _, t := range cands {
		if t.kind == "node" {
			return aiToolRun{Bin: spec.Bin, lease: &engineLease{t: t, engine: engine, tried: map[*engineTarget]bool{}}, light: true}, true
		}
	}
	return aiToolRun{}, false
}
