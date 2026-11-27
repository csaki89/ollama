package server

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/multimodel"
)

// the multi-model scheduler rules (docs/MULTIMODEL-DESIGN.md 2.4-2.6) with simulated devices, runners and requests

const schedTestConfig = `
default:
  keep_alive: 5m
devices:
  gpu0: { backend: rocm, match: "aa:00.0", max_running: 1 }
  gpu1: { backend: vulkan, match: "Test GPU B", max_running: 1 }
  cpu:  { backend: cpu, max_running: 1 }
swap:
  enabled: %s
models:
  big:latest:
    devices: [gpu0, gpu1, cpu]
    on_busy: wait
    wait_timeout: -1
  small:latest:
    devices: [gpu0, cpu]
    on_busy: next
`

func schedTest(t *testing.T, swap string) (*Scheduler, *multimodel.Config) {
	t.Helper()
	s := dualbufTestScheduler(t, sprintf(schedTestConfig, swap))
	return s, s.dualbuf.Config()
}

func sprintf(f, a string) string {
	out := []byte{}
	for i := 0; i < len(f); i++ {
		if f[i] == '%' && i+1 < len(f) && f[i+1] == 's' {
			out = append(out, a...)
			i++
			continue
		}
		out = append(out, f[i])
	}
	return string(out)
}

// mkRunner: a loaded runner of a model placed on dev, last used at lastUsed, with refs active requests
func mkRunner(s *Scheduler, cfg *multimodel.Config, name, dev string, lastUsed int64, refs uint, keepAlive time.Duration) *runnerRef {
	m := &Model{ModelPath: "/m/" + name, ShortName: name}
	m.dualbuf = &dualbufLoad{cfg: cfg, eff: cfg.Resolve(name, nil, nil, nil, nil), placement: dev}
	r := &runnerRef{modelPath: m.ModelPath, modelKey: schedulerModelKey(m), refCount: refs, sessionDuration: keepAlive}
	r.model.Store(m)
	r.dualbufLastUsed.Store(lastUsed)
	s.loaded[r.modelKey] = r
	return r
}

func pendingFor(cfg *multimodel.Config, name, dev string) *LlmRequest {
	return &LlmRequest{ctx: context.Background(), model: &Model{ModelPath: "/m/" + name, ShortName: name, dualbuf: &dualbufLoad{cfg: cfg, placement: dev}}}
}

// a request of the model waits in the coordinator (its first tier is held by another model)
func waitingRequest(t *testing.T, s *Scheduler, key string, devs ...string) (chan string, func()) {
	t.Helper()
	ch := make(chan string, 1)
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		g, err := s.dualbuf.Coord.Acquire(ctx, multimodel.Request{Model: key, Key: key, Devices: devs, OnBusy: multimodel.OnBusyWait, WaitTimeout: time.Duration(math.MaxInt64)})
		if err != nil {
			ch <- "error"
			return
		}
		ch <- g.Device
		g.Release()
	}()
	deadline := time.Now().Add(2 * time.Second)
	for !s.dualbuf.Coord.HasWaiter(key) {
		if time.Now().After(deadline) {
			t.Fatal("no waiter")
		}
		time.Sleep(time.Millisecond)
	}
	return ch, cancel
}

func TestDualbufOneInstancePerModel(t *testing.T) {
	s, cfg := schedTest(t, "true")
	a := &Model{ModelPath: "/m/a", dualbuf: &dualbufLoad{cfg: cfg, placement: "gpu0"}}
	b := &Model{ModelPath: "/m/a", dualbuf: &dualbufLoad{cfg: cfg, placement: "cpu"}}
	if schedulerModelKey(a) != schedulerModelKey(b) || schedulerModelKey(a) != "/m/a" {
		t.Fatalf("one key per model: %q %q", schedulerModelKey(a), schedulerModelKey(b))
	}
	// another placement is a reload (or unpark), never a second runner
	if dualbufFingerprint(a) == dualbufFingerprint(b) {
		t.Fatal("the placement is part of the reload fingerprint")
	}
	_ = s
}

func TestDualbufVictimIdleLRUNeverActive(t *testing.T) {
	s, cfg := schedTest(t, "true")
	mkRunner(s, cfg, "on-gpu1", "gpu1", 1, 0, time.Minute)
	newer := mkRunner(s, cfg, "newer", "gpu0", 30, 0, time.Minute)
	older := mkRunner(s, cfg, "older", "gpu0", 20, 0, time.Minute)
	busy := mkRunner(s, cfg, "busy", "gpu0", 10, 1, time.Minute)
	if v := s.dualbufVictim(pendingFor(cfg, "want", "gpu0"), false); v != older || !older.dualbufPark.Load() || older.dualbufDemoteTo.Load() != "" {
		t.Fatalf("victim %v park %v", v.modelPath, v.dualbufPark.Load())
	}
	if newer.dualbufPark.Load() || busy.dualbufPark.Load() {
		t.Fatal("only the victim moves")
	}
	// only active models on the device (and none idle elsewhere): no victim, the load waits
	s2, cfg2 := schedTest(t, "true")
	mkRunner(s2, cfg2, "busy", "gpu0", 10, 1, time.Minute)
	ctx, cancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
	defer cancel()
	p := pendingFor(cfg2, "want", "gpu0")
	p.ctx = ctx
	if v := s2.dualbufVictim(p, false); v != nil {
		t.Fatalf("an active model is never a victim: %v", v.modelPath)
	}
	// active in the coordinator (between the place and the runner) counts too
	s3, cfg3 := schedTest(t, "true")
	r := mkRunner(s3, cfg3, "starting", "gpu0", 10, 0, time.Minute)
	g, _ := s3.dualbuf.Coord.Acquire(context.Background(), multimodel.Request{Model: "x", Key: r.modelKey, Devices: []string{"gpu1"}})
	defer g.Release()
	p = pendingFor(cfg3, "want", "gpu0")
	p.ctx = ctx
	if v := s3.dualbufVictim(p, false); v != nil {
		t.Fatal("a model with an active request is never a victim")
	}
}

func TestDualbufVictimDemoteWithWaitingRequest(t *testing.T) {
	s, cfg := schedTest(t, "true")
	v := mkRunner(s, cfg, "small:latest", "gpu0", 5, 0, time.Minute)
	hold, _ := s.dualbuf.Coord.Acquire(context.Background(), multimodel.Request{Model: "big", Key: "/m/big:latest", Devices: []string{"gpu0"}})
	ch, cancel := waitingRequest(t, s, v.modelKey, "gpu0", "cpu")
	defer cancel()
	if got := s.dualbufVictim(pendingFor(cfg, "big:latest", "gpu0"), false); got != v || v.dualbufDemoteTo.Load() != "cpu" {
		t.Fatalf("demote: %v to %q", got, v.dualbufDemoteTo.Load())
	}
	// once it left gpu0, its waiting request is served one tier down
	s.dualbufMoved(v)
	select {
	case d := <-ch:
		if d != "cpu" {
			t.Fatalf("served on %s", d)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the demoted model's request was not served")
	}
	hold.Release()
}

func TestDualbufVictimParkWithoutWaiter(t *testing.T) {
	s, cfg := schedTest(t, "true")
	v := mkRunner(s, cfg, "small:latest", "gpu0", 5, 0, time.Minute)
	if got := s.dualbufVictim(pendingFor(cfg, "big:latest", "gpu0"), false); got != v || v.dualbufDemoteTo.Load() != "" || !v.dualbufPark.Load() {
		t.Fatalf("park: demote %q park %v", v.dualbufDemoteTo.Load(), v.dualbufPark.Load())
	}
	// a waiting request but no lower tier on its list: park
	s2, cfg2 := schedTest(t, "true")
	low := mkRunner(s2, cfg2, "small:latest", "cpu", 5, 0, time.Minute)
	hold, _ := s2.dualbuf.Coord.Acquire(context.Background(), multimodel.Request{Model: "h", Key: "h", Devices: []string{"gpu0"}})
	defer hold.Release()
	_, cancel := waitingRequest(t, s2, low.modelKey, "gpu0")
	defer cancel()
	if got := s2.dualbufVictim(pendingFor(cfg2, "big:latest", "cpu"), false); got != low || low.dualbufDemoteTo.Load() != "" || !low.dualbufPark.Load() {
		t.Fatalf("lowest tier: demote %q park %v", low.dualbufDemoteTo.Load(), low.dualbufPark.Load())
	}
}

// chaining: the demoted model's load on the lower tier makes room there by the same rule
func TestDualbufVictimChain(t *testing.T) {
	s, cfg := schedTest(t, "true")
	first := mkRunner(s, cfg, "big:latest", "gpu0", 5, 0, time.Minute)
	second := mkRunner(s, cfg, "small:latest", "gpu1", 6, 0, time.Minute)
	hold, _ := s.dualbuf.Coord.Acquire(context.Background(), multimodel.Request{Model: "new", Key: "new", Devices: []string{"gpu0"}})
	defer hold.Release()
	_, c1 := waitingRequest(t, s, first.modelKey, "gpu0", "gpu1", "cpu")
	defer c1()
	if got := s.dualbufVictim(pendingFor(cfg, "new", "gpu0"), false); got != first || first.dualbufDemoteTo.Load() != "gpu1" {
		t.Fatalf("first step: %q", first.dualbufDemoteTo.Load())
	}
	// big now loads on gpu1, where small sits idle without a waiting request: small is parked
	if got := s.dualbufVictim(pendingFor(cfg, "big:latest", "gpu1"), false); got != second || second.dualbufDemoteTo.Load() != "" || !second.dualbufPark.Load() {
		t.Fatalf("second step: %v %q", got.modelPath, second.dualbufDemoteTo.Load())
	}
}

func TestDualbufVictimKeepAlive(t *testing.T) {
	// without swap an idle victim is unloaded, a keep_alive -1 one is parked (never unloaded)
	s, cfg := schedTest(t, "false")
	plain := mkRunner(s, cfg, "plain", "gpu0", 5, 0, 5*time.Minute)
	if got := s.dualbufVictim(pendingFor(cfg, "want", "gpu0"), false); got != plain || plain.dualbufPark.Load() {
		t.Fatal("no swap: unload")
	}
	delete(s.loaded, plain.modelKey)
	keep := mkRunner(s, cfg, "keep", "gpu0", 5, 0, time.Duration(math.MaxInt64))
	if got := s.dualbufVictim(pendingFor(cfg, "want", "gpu0"), false); got != keep || !keep.dualbufPark.Load() || !keep.dualbufForever.Load() {
		t.Fatal("keep_alive -1: parked, not unloaded")
	}
}

func TestDualbufPromoteOnRequest(t *testing.T) {
	s, cfg := schedTest(t, "true")
	// the model runs on its lower tier; a request finds gpu0 free: it is placed there (a reload, one instance)
	loaded := mkRunner(s, cfg, "small:latest", "cpu", 5, 0, time.Minute)
	m := &Model{ModelPath: "/m/small:latest", ShortName: "small:latest"}
	opts := api.DefaultOptions()
	m2, _, g, err := s.dualbufPrepare(context.Background(), m, nil, nil, &opts)
	if err != nil || g == nil || m2.dualbuf.placement != "gpu0" {
		t.Fatalf("promote: %+v %v", m2.dualbuf, err)
	}
	defer g.Release()
	if schedulerModelKey(m2) != loaded.modelKey || dualbufFingerprint(m2) == dualbufFingerprint(loaded.model.Load()) {
		t.Fatal("the same runner, reloaded on the higher tier")
	}
	// without a request nothing moves: the loaded runner keeps its placement
	if loaded.model.Load().dualbuf.placement != "cpu" {
		t.Fatal("moved without a request")
	}
}

func TestDualbufQueueLimit(t *testing.T) {
	s, _ := schedTest(t, "true")
	s.dualbuf.Coord.SetMaxQueue(1)
	hold, _ := s.dualbuf.Coord.Acquire(context.Background(), multimodel.Request{Model: "h", Key: "h", Devices: []string{"gpu0"}})
	defer hold.Release()
	_, cancel := waitingRequest(t, s, "w", "gpu0")
	defer cancel()
	m := &Model{ModelPath: "/m/big:latest", ShortName: "big:latest"}
	opts := api.DefaultOptions()
	if _, _, _, err := s.dualbufPrepare(context.Background(), m, nil, nil, &opts); !errors.Is(err, ErrMaxQueue) {
		t.Fatalf("a full queue is the stock ErrMaxQueue: %v", err)
	}
}

// GPU <-> CPU in process: a demote to the CPU parks the model in its process (whatever swap says); a GPU placement
// with another tier on its list keeps the park copies and does not map the file; a one-tier model is unchanged
func TestDualbufDemoteToCPUInProcess(t *testing.T) {
	s, cfg := schedTest(t, "false")
	v := mkRunner(s, cfg, "small:latest", "gpu0", 5, 0, time.Minute)
	hold, _ := s.dualbuf.Coord.Acquire(context.Background(), multimodel.Request{Model: "big", Key: "/m/big:latest", Devices: []string{"gpu0"}})
	defer hold.Release()
	_, cancel := waitingRequest(t, s, v.modelKey, "gpu0", "cpu")
	defer cancel()
	if got := s.dualbufVictim(pendingFor(cfg, "big:latest", "gpu0"), false); got != v || v.dualbufDemoteTo.Load() != "cpu" || !v.dualbufPark.Load() {
		t.Fatalf("demote to the cpu without swap: to %q park %v", v.dualbufDemoteTo.Load(), v.dualbufPark.Load())
	}

	for _, tc := range []struct {
		name, model string
		park        bool
	}{
		{"two tiers", "small:latest", true}, // [gpu0, cpu]
		{"one tier", "solo:latest", false},  // [gpu0]
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := dualbufTestScheduler(t, sprintf(schedTestConfig, "false")+"  solo:latest:\n    devices: [gpu0]\n")
			m := &Model{ModelPath: "/m/" + tc.model, ShortName: tc.model}
			opts := api.DefaultOptions()
			m2, _, g, err := s.dualbufPrepare(context.Background(), m, nil, nil, &opts)
			if err != nil || g == nil || m2.dualbuf.placement != "gpu0" {
				t.Fatalf("prepare: %+v %v", m2.dualbuf, err)
			}
			defer g.Release()
			if m2.dualbuf.movesInProcess() != tc.park || (opts.UseMMap != nil) != tc.park || (tc.park && *opts.UseMMap) {
				t.Fatalf("moves in process %v, use_mmap %v", m2.dualbuf.movesInProcess(), opts.UseMMap)
			}
			var config llm.LlamaServerConfig
			s.dualbufServerConfig(&LlmRequest{model: m2}, &config)
			if (config.ExtraEnv["LLAMA_ARG_PARK"] == "1") != tc.park {
				t.Fatalf("park copies: %v", config.ExtraEnv)
			}
		})
	}
}
