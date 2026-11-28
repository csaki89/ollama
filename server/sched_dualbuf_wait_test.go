package server

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/multimodel"
)

// the pending and completed loops run; the loads are simulated: a load on gpu0 needs room while "busy" sits there

type waitSched struct {
	s     *Scheduler
	cfg   *multimodel.Config
	mu    sync.Mutex
	calls map[string]int
	order []string
	full  func(req *LlmRequest) bool // the load needs room
}

func newWaitSched(t *testing.T) *waitSched {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	s := InitScheduler(ctx)
	test := dualbufTestScheduler(t, sprintf(schedTestConfig, "false"))
	s.dualbuf = test.dualbuf
	s.dualbufInit()
	s.getGpuFn = func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		return []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, TotalMemory: 8 << 30, FreeMemory: 4 << 30}}
	}
	s.getSystemInfoFn = getSystemInfoFn
	s.waitForRecovery = 10 * time.Millisecond
	w := &waitSched{s: s, cfg: s.dualbuf.Config(), calls: map[string]int{}}
	s.loadFn = func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
		name := req.model.ShortName
		w.mu.Lock()
		w.calls[name]++
		w.mu.Unlock()
		if w.full != nil && w.full(req) {
			return true
		}
		r := &runnerRef{modelPath: req.model.ModelPath, modelKey: schedulerModelKey(req.model), llama: &mockLlm{}, sessionDuration: time.Minute, refCount: 1}
		r.model.Store(req.model)
		s.loadedMu.Lock()
		s.loaded[r.modelKey] = r
		s.loadedMu.Unlock()
		w.mu.Lock()
		w.order = append(w.order, name)
		w.mu.Unlock()
		req.successCh <- r
		return false
	}
	s.Run(ctx)
	return w
}

func (w *waitSched) loaded(key string) bool {
	w.s.loadedMu.Lock()
	defer w.s.loadedMu.Unlock()
	return w.s.loaded[key] != nil
}

func (w *waitSched) callsOf(name string) int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.calls[name]
}

// a busy (active) model on dev
func (w *waitSched) busy(name, dev string) *runnerRef {
	r := mkRunner(w.s, w.cfg, name, dev, 1, 1, time.Minute)
	r.llama = &mockLlm{}
	return r
}

func (w *waitSched) submit(name, dev string) *LlmRequest {
	req := pendingFor(w.cfg, name, dev)
	req.opts = api.DefaultOptions()
	req.successCh = make(chan *runnerRef, 1)
	req.errCh = make(chan error, 1)
	w.s.pendingReqCh <- req
	return req
}

func served(t *testing.T, req *LlmRequest, within time.Duration) {
	t.Helper()
	select {
	case <-req.successCh:
	case err := <-req.errCh:
		t.Fatalf("%s: %v", req.model.ShortName, err)
	case <-time.After(within):
		t.Fatalf("%s was not served", req.model.ShortName)
	}
}

func notServed(t *testing.T, req *LlmRequest, during time.Duration) {
	t.Helper()
	select {
	case <-req.successCh:
		t.Fatalf("%s must still wait", req.model.ShortName)
	case <-time.After(during):
	}
}

// the end of the busy model's request: the scheduler's finished event and the release of its place
func (w *waitSched) finish(r *runnerRef) {
	w.s.finishedReqCh <- &LlmRequest{model: r.model.Load()}
}

func TestDualbufWaitingLoadDoesNotHoldOthers(t *testing.T) {
	w := newWaitSched(t)
	busy := w.busy("busy", "gpu0")
	w.full = func(req *LlmRequest) bool { return req.model.dualbuf.placement == "gpu0" && w.loaded(busy.modelKey) }

	a := w.submit("a", "gpu0") // needs room on gpu0, only the active busy model is there: it waits aside
	b := w.submit("b", "gpu1") // another free device: it loads at once
	served(t, b, 2*time.Second)
	notServed(t, a, 50*time.Millisecond)
	// no polling: without an event the waiting load is not tried again
	n := w.callsOf("a")
	time.Sleep(150 * time.Millisecond)
	if w.callsOf("a") != n {
		t.Fatalf("the waiting load was retried without an event: %d -> %d", n, w.callsOf("a"))
	}
	// the busy model becomes idle (an event): it is the victim, unloaded, and a loads after it (one more event)
	w.finish(busy)
	served(t, a, 2*time.Second)
	if w.loaded(busy.modelKey) {
		t.Fatal("the victim was not unloaded")
	}
}

// the turn on a device: a later load on the waiting load's devices (the CPU of a split model too) waits after it,
// one on another device does not
func TestDualbufWaitingLoadKeepsItsTurn(t *testing.T) {
	w := newWaitSched(t)
	busy := w.busy("busy", "gpu0")
	w.s.dualbuf.SetSplit("/m/a", "gpu0", true) // a was seen split between gpu0 and the CPU
	w.full = func(req *LlmRequest) bool { return req.model.dualbuf.placement == "gpu0" && w.loaded(busy.modelKey) }

	a := w.submit("a", "gpu0")
	deadline := time.Now().Add(2 * time.Second)
	for w.callsOf("a") == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	c := w.submit("c", "cpu")  // the CPU is claimed by the waiting split load: c waits after it
	b := w.submit("b", "gpu1") // not claimed: b loads at once
	served(t, b, 2*time.Second)
	notServed(t, c, 80*time.Millisecond)
	if w.callsOf("c") != 0 {
		t.Fatal("c must not load before the earlier load on its device")
	}
	w.finish(busy)
	served(t, a, 2*time.Second)
	served(t, c, 2*time.Second)
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.order) != 3 || w.order[0] != "b" || w.order[1] != "a" || w.order[2] != "c" {
		t.Fatalf("load order %v, want b a c", w.order)
	}
}

// a waiting load whose request ends leaves its turn: the later one on its device goes
func TestDualbufWaitingLoadCancelled(t *testing.T) {
	w := newWaitSched(t)
	busy := w.busy("busy", "gpu0")
	w.full = func(req *LlmRequest) bool { return req.model.ShortName == "a" && w.loaded(busy.modelKey) }
	ctx, cancel := context.WithCancel(context.Background())
	a := pendingFor(w.cfg, "a", "gpu0")
	a.ctx, a.opts, a.successCh, a.errCh = ctx, api.DefaultOptions(), make(chan *runnerRef, 1), make(chan error, 1)
	w.s.pendingReqCh <- a
	deadline := time.Now().Add(2 * time.Second)
	for w.callsOf("a") == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	d := w.submit("d", "gpu0") // behind a
	notServed(t, d, 50*time.Millisecond)
	cancel()
	served(t, d, 2*time.Second)
}

// a runner reporting driver host memory the cgroup does not see (the Vulkan host buffers)
type unchargedLlm struct{ mockLlm }

func (u *unchargedLlm) HostUncharged() uint64 { return 512 << 20 }

// load runs dualbufServerConfig with loadedMu held: the uncharged host memory of the other runners is read without
// taking it again (a self-deadlock before), the launch gets it
func TestDualbufLoadDoesNotDeadlock(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 20*time.Second)
	defer done()
	s := InitScheduler(ctx)
	s.dualbuf = dualbufTestScheduler(t, sprintf(schedTestConfig, "true")).dualbuf
	cfg := s.dualbuf.Config()
	other := mkRunner(s, cfg, "other", "gpu1", 1, 0, time.Minute)
	other.llama = &unchargedLlm{}

	a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: time.Minute}, nil)
	a.req.model.ShortName = "small:latest"
	a.req.model.dualbuf = &dualbufLoad{cfg: cfg, eff: cfg.Resolve("small:latest", nil, nil, nil, nil), placement: multimodel.DeviceCPU, tiers: []string{"gpu0", "cpu"}}
	var got llm.LlamaServerConfig
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *gguf.Model, adapters []string, projectors []string, opts api.Options, numParallel int, config llm.LlamaServerConfig) (llm.LlamaServer, error) {
		got = config
		return a.newServer(systemInfo, gpus, model, f, adapters, projectors, opts, numParallel, config)
	}
	finished := make(chan bool, 1)
	go func() { finished <- s.load(a.req, getSystemInfoFn(), nil, false) }()
	select {
	case <-finished:
	case <-time.After(10 * time.Second):
		t.Fatal("load did not return: deadlock in the dualbuf launch path")
	}
	select {
	case <-a.req.successCh:
	case err := <-a.req.errCh:
		t.Fatalf("load: %v", err)
	case <-time.After(10 * time.Second):
		t.Fatal("the runner did not come up")
	}
	if got.ExtraEnv["LLAMA_ARG_RAM_UNCHARGED_MIB"] != "512" {
		t.Fatalf("the other runner's uncharged host memory goes to the launch: %v", got.ExtraEnv)
	}
}

// a placed load on a GPU whose free memory the stock estimate finds too small (the room was made by the dualbuf: a park,
// the rest of the model goes to the CPU) is launched: the stock "predicted to exceed available memory, evicting" branch
// does not run (it asked for an eviction with nothing left to evict, and the load waited for an event that never came)
func TestDualbufLoadSkipsStockPreflight(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 20*time.Second)
	defer done()
	s := InitScheduler(ctx)
	s.dualbuf = dualbufTestScheduler(t, sprintf(schedTestConfig, "true")).dualbuf
	s.dualbuf.NoteDevices([]ml.DeviceInfo{testGPU0, testGPU1})
	cfg := s.dualbuf.Config()
	other := mkRunner(s, cfg, "other", "gpu1", 1, 0, time.Minute)
	other.llama = &unchargedLlm{}

	a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: time.Minute}, nil)
	a.req.model.ShortName = "small:latest"
	a.req.model.dualbuf = &dualbufLoad{cfg: cfg, eff: cfg.Resolve("small:latest", nil, nil, nil, nil), placement: "gpu0", tiers: []string{"gpu0", "cpu"}}
	launched := false
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *gguf.Model, adapters []string, projectors []string, opts api.Options, numParallel int, config llm.LlamaServerConfig) (llm.LlamaServer, error) {
		launched = true
		return a.newServer(systemInfo, gpus, model, f, adapters, projectors, opts, numParallel, config)
	}
	gpu := testGPU0
	gpu.FreeMemory = 1 // far below any estimate
	finished := make(chan bool, 1)
	go func() { finished <- s.load(a.req, getSystemInfoFn(), []ml.DeviceInfo{gpu}, true) }()
	select {
	case evict := <-finished:
		if evict || !launched {
			t.Fatalf("the placed load is launched, no stock eviction: evict=%v launched=%v", evict, launched)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("load did not return")
	}
	if !dualbufPlaced(a.req) || dualbufPlaced(&LlmRequest{model: &Model{}}) {
		t.Fatal("placed: a dualbuf placement only")
	}
}
