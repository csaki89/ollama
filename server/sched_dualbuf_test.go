package server

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
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
	m := multimodel.NewManager(p, func(string) string { return "" })
	if err := m.Load(); err != nil {
		t.Fatal(err)
	}
	return &Scheduler{loaded: map[string]*runnerRef{}, dualbuf: m}
}

const dualbufTestConfig = `
default:
  max_context: 8192
  keep_alive: 30m
devices:
  cpu: { backend: cpu, max_running: 1 }
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
	if k := schedulerModelKey(m2); k != "/models/a" {
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
	s := dualbufTestScheduler(t, "default: {}\ndevices:\n  gpu0: { backend: rocm, match: \"aa:00.0\" }\n  gpu1: { backend: vulkan, match: \"Test GPU B\" }\nswap:\n  enabled: true\n")
	cfg := s.dualbuf.Config()
	mk := func(path, dev string, lastUsed int64, refs uint) *runnerRef {
		m := &Model{ModelPath: path, dualbuf: &dualbufLoad{cfg: cfg, placement: dev}}
		r := &runnerRef{modelPath: path, modelKey: schedulerModelKey(m), refCount: refs}
		r.model.Store(m)
		r.dualbufLastUsed.Store(lastUsed)
		s.loaded[r.modelKey] = r
		return r
	}
	mk("/m/old-on-gpu1", "gpu1", 1, 0)
	newer := mk("/m/new-on-gpu0", "gpu0", 30, 0)
	older := mk("/m/old-on-gpu0", "gpu0", 20, 0)
	busy := mk("/m/busy-on-gpu0", "gpu0", 10, 1)
	req := &LlmRequest{model: &Model{ModelPath: "/m/want", dualbuf: &dualbufLoad{cfg: cfg, placement: "gpu0"}}}
	v := s.dualbufVictim(req, false)
	if v != older || !older.dualbufPark.Load() {
		t.Fatalf("victim %s park %v", v.modelPath, v.dualbufPark.Load())
	}
	if newer.dualbufPark.Load() || busy.dualbufPark.Load() {
		t.Fatal("only the victim is parked")
	}
	// a victim chosen for RAM comes first
	req.dualbufVictim = busy
	if v := s.dualbufVictim(req, false); v != busy || req.dualbufVictim != nil {
		t.Fatalf("RAM victim: %v", v)
	}
}

// with a config the global gfx override does not apply (a warning says where it goes), without one it is kept
func TestDualbufStartupGFXOverride(t *testing.T) {
	var buf bytes.Buffer
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(old); discover.DualbufROCm = nil })
	t.Setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

	dualbufStartup(dualbufTestScheduler(t, "").dualbuf)
	if os.Getenv("HSA_OVERRIDE_GFX_VERSION") != "10.3.0" || discover.DualbufROCm != nil {
		t.Fatal("no config: the stock behaviour, the variable applies")
	}
	cfg := "default: {}\ndevices:\n  gpu_rocm: { backend: rocm, match: \"0000:aa:00.0\", gfx_override: 10.3.0 }\n"
	dualbufStartup(dualbufTestScheduler(t, cfg).dualbuf)
	if os.Getenv("HSA_OVERRIDE_GFX_VERSION") != "" {
		t.Fatal("with a config the global override does not apply")
	}
	if !strings.Contains(buf.String(), "HSA_OVERRIDE_GFX_VERSION is ignored") || !strings.Contains(buf.String(), "gfx_override") {
		t.Fatalf("warning: %s", buf.String())
	}
}

// envValues are the values of a variable in a subprocess environment
func envValues(env []string, key string) []string {
	var out []string
	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok && k == key {
			out = append(out, v)
		}
	}
	return out
}

// the gfx override the discovery and runner subprocesses get (both are set up by llm.SetupLlamaServerCommandEnv):
// without a config the global HSA_OVERRIDE_GFX_VERSION; with one only the rocm device's gfx_override as
// HSA_OVERRIDE_GFX_VERSION_<its KFD node>, and the global one no longer
func TestDualbufSubprocessGFXOverride(t *testing.T) {
	t.Cleanup(func() { discover.DualbufROCm = nil })
	exe := filepath.Join(t.TempDir(), "llama-server")
	t.Setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

	// no config: the stock environment
	dualbufStartup(dualbufTestScheduler(t, "").dualbuf)
	stock := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}}
	cmd := exec.Command(exe)
	llm.SetupLlamaServerCommandEnv(cmd, exe, nil, ml.GetDevicesEnv([]ml.DeviceInfo{stock}))
	if got := envValues(cmd.Env, "HSA_OVERRIDE_GFX_VERSION"); len(got) != 1 || got[0] != "10.3.0" {
		t.Fatalf("no config: %v", got)
	}

	// config: the device's override for its KFD node (the other card, node 1, gets none)
	cfg := "default: {}\ndevices:\n  gpu_rocm: { backend: rocm, match: \"0000:aa:00.0\", gfx_override: 11.0.0 }\n"
	s := dualbufTestScheduler(t, cfg)
	dualbufStartup(s.dualbuf)
	cards := dualbufROCmCards(s.dualbuf.Config(), map[string]int{"0000:bb:00.0": 1, "0000:aa:00.0": 2}, nil)
	if cards == nil || len(cards.PCIs) != 1 || len(cards.Env) != 1 {
		t.Fatalf("cards %+v", cards)
	}
	// the discovery gets the cards' environment as its extra environment, the runner the device's overrides
	for name, extra := range map[string]map[string]string{
		"discovery": cards.Env,
		"runner":    ml.GetDevicesEnv([]ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0000:aa:00.0", Library: "ROCm"}, FilterID: "0", RunnerEnvOverrides: cards.Env}}),
	} {
		cmd := exec.Command(exe)
		llm.SetupLlamaServerCommandEnv(cmd, exe, nil, extra)
		if got := envValues(cmd.Env, "HSA_OVERRIDE_GFX_VERSION_2"); len(got) != 1 || got[0] != "11.0.0" {
			t.Fatalf("%s: %v", name, got)
		}
		for _, kv := range cmd.Env {
			if k, _, _ := strings.Cut(kv, "="); k == "HSA_OVERRIDE_GFX_VERSION" || k == "HSA_OVERRIDE_GFX_VERSION_1" || k == "LD_PRELOAD" {
				t.Fatalf("%s: only the card's own override: %s", name, kv)
			}
		}
	}
}
