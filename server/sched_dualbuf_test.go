package server

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/multimodel"
)

func dualbufTestScheduler(t *testing.T, config string) *Scheduler {
	t.Helper()
	p := filepath.Join(t.TempDir(), multimodel.ConfigFileName)
	if config != "" {
		if err := os.WriteFile(p, []byte(config), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return &Scheduler{loaded: map[string]*runnerRef{}, dualbuf: multimodel.NewManager(p, func(string) string { return "" })}
}

const dualbufTestConfig = `
defaults:
  max_context: 8192
  keep_alive: 30m
devices:
  cpu: { max_running: 1 }
models:
  small:latest:
    devices: [cpu]
    max_context: 2048
    dual_buffer: false
`

func TestDualbufNoConfigIsNoop(t *testing.T) {
	s := dualbufTestScheduler(t, "")
	m := &Model{ModelPath: "/models/a", ShortName: "small:latest"}
	opts := api.DefaultOptions()
	m2, ka, g, err := s.dualbufPrepare(context.Background(), m, nil, nil, &opts)
	if err != nil || m2 != m || ka != nil || g != nil {
		t.Fatalf("no config must not change anything: %v %v %v %v", m2, ka, g, err)
	}
	if schedulerModelKey(m) != "/models/a" || dualbufFingerprint(m) != "" {
		t.Fatalf("key %q fp %q", schedulerModelKey(m), dualbufFingerprint(m))
	}
	var nilSched *Scheduler
	if nilSched.dualbufConfigFor(m) != nil {
		t.Fatal("nil scheduler")
	}
}

func TestDualbufPrepareAndQueue(t *testing.T) {
	s := dualbufTestScheduler(t, dualbufTestConfig)
	m := &Model{ModelPath: "/models/a", ShortName: "small:latest", Digest: "sha256:abc"}
	opts := api.DefaultOptions()
	m2, ka, g, err := s.dualbufPrepare(context.Background(), m, nil, nil, &opts)
	if err != nil || g == nil || g.Device != "cpu" {
		t.Fatalf("prepare: %+v %v", g, err)
	}
	if m2 == m || m2.dualbuf.placement != "cpu" || opts.NumGPU != 0 {
		t.Fatalf("cpu placement: %+v numgpu %d", m2.dualbuf, opts.NumGPU)
	}
	if ka == nil || ka.Duration != 30*time.Minute {
		t.Fatalf("keep alive from defaults: %+v", ka)
	}
	if k := schedulerModelKey(m2); k != "/models/a\x00@cpu" {
		t.Fatalf("key %q", k)
	}
	if dualbufFingerprint(m2) == "" {
		t.Fatal("dual_buffer false must be in the fingerprint")
	}
	if m2.dualbuf.launch.Env["LLAMA_ARG_DUAL_BUFFER"] != "0" {
		t.Fatalf("env %+v", m2.dualbuf.launch.Env)
	}

	// max_running 1 on the cpu: a second request waits until the first ends
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	opts2 := api.DefaultOptions()
	if _, _, _, err := s.dualbufPrepare(ctx, m, nil, nil, &opts2); err == nil {
		t.Fatal("the second request must wait for the cpu place")
	}
	g.Release()
	opts3 := api.DefaultOptions()
	_, _, g3, err := s.dualbufPrepare(context.Background(), m, nil, nil, &opts3)
	if err != nil || g3 == nil {
		t.Fatalf("after release: %v", err)
	}
	g3.Release()

	// a request keep_alive wins over the config
	reqKA := &api.Duration{Duration: time.Minute}
	_, ka, g4, _ := s.dualbufPrepare(context.Background(), m, nil, reqKA, &opts3)
	if ka.Duration != time.Minute {
		t.Fatalf("request keep alive: %v", ka)
	}
	g4.Release()
}

func TestDualbufModelOptionsPrecedence(t *testing.T) {
	sched := dualbufTestScheduler(t, dualbufTestConfig)
	s := &Server{sched: sched}
	unknown := &Model{ModelPath: "/models/b", ShortName: "other:latest"}
	opts, err := s.modelOptions(unknown, nil)
	if err != nil || opts.NumCtx != 8192 {
		t.Fatalf("defaults: %d %v", opts.NumCtx, err)
	}
	withModelfile := &Model{ModelPath: "/models/b", ShortName: "other:latest", Options: map[string]any{"num_ctx": float64(4096)}}
	if opts, _ := s.modelOptions(withModelfile, nil); opts.NumCtx != 4096 {
		t.Fatalf("Modelfile over defaults: %d", opts.NumCtx)
	}
	section := &Model{ModelPath: "/models/a", ShortName: "small:latest", Options: map[string]any{"num_ctx": float64(4096)}}
	if opts, _ := s.modelOptions(section, nil); opts.NumCtx != 2048 {
		t.Fatalf("model section over Modelfile: %d", opts.NumCtx)
	}
	if opts, _ := s.modelOptions(section, map[string]any{"num_ctx": float64(1024)}); opts.NumCtx != 1024 {
		t.Fatalf("request over the section: %d", opts.NumCtx)
	}
}

func TestDualbufVictimSameDevice(t *testing.T) {
	s := dualbufTestScheduler(t, "devices:\n  x080: { backend: rocm, match: \"03:00.0\" }\n  rx550: { backend: vulkan, match: \"RX 550\" }\nswap:\n  enabled: true\n")
	cfg := s.dualbuf.Config()
	mk := func(path, dev string, lastUsed int64, refs uint) *runnerRef {
		m := &Model{ModelPath: path, dualbuf: &dualbufLoad{cfg: cfg, placement: dev}}
		r := &runnerRef{modelPath: path, modelKey: schedulerModelKey(m), refCount: refs}
		r.model.Store(m)
		r.dualbufLastUsed.Store(lastUsed)
		s.loaded[r.modelKey] = r
		return r
	}
	mk("/m/old-on-rx550", "rx550", 1, 0)
	newer := mk("/m/new-on-x080", "x080", 30, 0)
	older := mk("/m/old-on-x080", "x080", 20, 0)
	busy := mk("/m/busy-on-x080", "x080", 10, 1)
	req := &LlmRequest{model: &Model{ModelPath: "/m/want", dualbuf: &dualbufLoad{cfg: cfg, placement: "x080"}}}
	v := s.dualbufVictim(req)
	if v != older || !older.dualbufPark.Load() {
		t.Fatalf("victim %s park %v", v.modelPath, v.dualbufPark.Load())
	}
	if newer.dualbufPark.Load() || busy.dualbufPark.Load() {
		t.Fatal("only the victim is parked")
	}
	// a victim chosen for RAM comes first
	req.dualbufVictim = busy
	if v := s.dualbufVictim(req); v != busy || req.dualbufVictim != nil {
		t.Fatalf("RAM victim: %v", v)
	}
}
