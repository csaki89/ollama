package multimodel

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
)

const exampleConfig = `
defaults:
  dual_buffer: true
  dynamic_kv: true
  kv_type: f16
  keep_alive: 30m

devices:
  x080:  { backend: rocm,   match: "0000:03:00.0", max_running: 1 }
  rx550: { backend: vulkan, match: "RX 550", max_running: 1 }
  cpu:   { max_running: 1 }

swap:
  enabled: true
  keep_in_ram: pinned
  save_kv: true

models:
  qwen3-coder:30b:
    devices: [x080, cpu]
    on_busy: wait
    wait_timeout: 120s
    zones: auto
    max_context: 131072
  qwen3-vl:8b:
    devices: [x080, cpu]
    on_busy: next
    max_context: 16384
  qwen3-embedding:*:
    devices: [rx550, cpu]
    dual_buffer: false
    max_context: 4096
    keep_alive: -1
`

func mustParse(t *testing.T, s string) *Config {
	t.Helper()
	c, err := Parse([]byte(s))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return c
}

func TestParseExample(t *testing.T) {
	c := mustParse(t, exampleConfig)
	if c.MaxRunning("x080") != 1 || c.MaxRunning("cpu") != 1 || c.MaxRunning("nope") != 0 {
		t.Fatalf("max_running: %d %d", c.MaxRunning("x080"), c.MaxRunning("cpu"))
	}
	if !c.SwapEnabled() || !c.SwapSaveKV() || c.Swap.KeepInRAM != KeepInRAMPinned {
		t.Fatalf("swap: %+v", c.Swap)
	}
	if ka := c.Models["qwen3-embedding:*"].KeepAlive; ka == nil || !ka.Forever {
		t.Fatalf("keep_alive -1: %+v", ka)
	}
	if wt := c.Models["qwen3-coder:30b"].WaitTimeout; wt == nil || wt.Duration != 120*time.Second {
		t.Fatalf("wait_timeout: %+v", wt)
	}
}

func TestParseEmpty(t *testing.T) {
	c, err := Parse([]byte("  \n"))
	if err != nil || c == nil {
		t.Fatalf("empty: %v %v", c, err)
	}
}

func TestParseErrors(t *testing.T) {
	cases := map[string]string{
		"unknown key":      "defaults:\n  dual_bufer: true\n",
		"bad on_busy":      "models:\n  a:b:\n    on_busy: later\n",
		"unknown device":   "models:\n  a:b:\n    devices: [gpu9]\n",
		"bad backend":      "devices:\n  g: { backend: opencl, match: x }\n",
		"no match":         "devices:\n  g: { backend: rocm }\n",
		"bad kv type":      "defaults:\n  kv_type: q3_k\n",
		"zones number":     "defaults:\n  zones: 4\n",
		"bad duration":     "defaults:\n  keep_alive: soon\n",
		"max_running zero": "devices:\n  cpu: { max_running: 0 }\n",
		"bad keep_in_ram":  "swap:\n  keep_in_ram: disk\n",
		"bad ram_limit":    "ram_limit: lots\n",
		"bad fa":           "defaults:\n  flash_attention: maybe\n",
		"yaml syntax":      "defaults: [\n",
		"cpu with match":   "devices:\n  cpu: { backend: rocm, match: x }\n",
		"device twice":     "devices:\n  g: { backend: rocm, match: x }\nmodels:\n  a:b:\n    devices: [g, g]\n",
	}
	for name, s := range cases {
		if _, err := Parse([]byte(s)); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestSectionFor(t *testing.T) {
	c := mustParse(t, exampleConfig+"  qwen3-embedding:0.6b:\n    threads: 2\n  qwen3*:\n    threads: 4\n")
	cases := map[string]string{
		"qwen3-coder:30b": "qwen3-coder:30b",
		"registry.ollama.ai/library/qwen3-coder:30b": "qwen3-coder:30b",
		"QWEN3-VL:8B":          "qwen3-vl:8b",
		"qwen3-embedding:0.6b": "qwen3-embedding:0.6b", // exact beats the pattern
		"qwen3-embedding:4b":   "qwen3-embedding:*",    // longest pattern
		"qwen3:latest":         "qwen3*",
		"qwen3":                "qwen3*",
		"llama3:8b":            "",
		"registry.ollama.ai/library/qwen3-embedding:8b": "qwen3-embedding:*",
	}
	for name, want := range cases {
		got, _ := c.SectionFor(name)
		if got != want {
			t.Errorf("%s: got %q want %q", name, got, want)
		}
	}
}

func TestResolvePrecedence(t *testing.T) {
	c := mustParse(t, exampleConfig)
	env := func(k string) string {
		return map[string]string{
			"OLLAMA_KEEP_ALIVE":     "10m",
			"LLAMA_ARG_DUAL_BUFFER": "0",
			"OLLAMA_NUM_PARALLEL":   "2",
			"OLLAMA_CONTEXT_LENGTH": "8192",
		}[k]
	}
	// unknown model: env < defaults
	e := c.Resolve("llama3:8b", env, nil, nil, nil)
	if e.Section != "" || !e.DualBuffer.V || e.DualBuffer.Src != SrcDefaults {
		t.Fatalf("dual_buffer: %+v section %q", e.DualBuffer, e.Section)
	}
	if e.KeepAlive.V.Duration != 30*time.Minute || e.KeepAlive.Src != SrcDefaults {
		t.Fatalf("keep_alive: %+v", e.KeepAlive)
	}
	if e.Parallel.V != 2 || e.Parallel.Src != SrcEnv {
		t.Fatalf("parallel: %+v", e.Parallel)
	}
	if e.MaxContext.V != 8192 || e.MaxContext.Src != SrcEnv {
		t.Fatalf("max_context env: %+v", e.MaxContext)
	}

	// defaults < Modelfile < model section < request
	mf := map[string]any{"num_ctx": float64(32768), "num_thread": float64(8)}
	e = c.Resolve("qwen3-coder:30b", env, mf, nil, nil)
	if e.MaxContext.V != 131072 || e.MaxContext.Src != "model:qwen3-coder:30b" {
		t.Fatalf("max_context model: %+v", e.MaxContext)
	}
	if e.Threads.V != 8 || e.Threads.Src != SrcModelfile {
		t.Fatalf("threads modelfile: %+v", e.Threads)
	}
	req := map[string]any{"num_ctx": float64(4096), "dual_buffer": false, "kv_type": "Q8_0"}
	ka := 2 * time.Minute
	e = c.Resolve("qwen3-coder:30b", env, mf, req, &ka)
	if e.MaxContext.V != 4096 || e.MaxContext.Src != SrcRequest {
		t.Fatalf("max_context request: %+v", e.MaxContext)
	}
	if e.DualBuffer.V || e.DualBuffer.Src != SrcRequest || e.KVType.V != "q8_0" {
		t.Fatalf("request fork keys: %+v %+v", e.DualBuffer, e.KVType)
	}
	if e.KeepAlive.V.Duration != 2*time.Minute || e.KeepAlive.Src != SrcRequest {
		t.Fatalf("keep_alive request: %+v", e.KeepAlive)
	}
	neg := time.Duration(math.MaxInt64)
	e = c.Resolve("qwen3-coder:30b", env, nil, nil, &neg)
	if !e.KeepAlive.V.Forever {
		t.Fatalf("keep_alive forever: %+v", e.KeepAlive)
	}

	// pattern section, keep_alive -1, dual_buffer off
	e = c.Resolve("qwen3-embedding:0.6b", env, nil, nil, nil)
	if e.Section != "qwen3-embedding:*" || e.DualBuffer.V || !e.KeepAlive.V.Forever || e.KeepAlive.V.String() != "-1" {
		t.Fatalf("embedding: %+v %+v %+v", e.Section, e.DualBuffer, e.KeepAlive)
	}
	if got := strings.Join(e.Devices.V, ","); got != "rx550,cpu" {
		t.Fatalf("devices: %s", got)
	}
	line := e.String()
	for _, want := range []string{"devices=[rx550,cpu](model:qwen3-embedding:*)", "dual_buffer=false(model:qwen3-embedding:*)", "keep_alive=-1(model:qwen3-embedding:*)", "parallel=2(env)", "kv_type=f16(defaults)", "on_busy=wait(builtin)", "wait_timeout=1m0s(builtin)"} {
		if !strings.Contains(line, want) {
			t.Errorf("log line misses %q: %s", want, line)
		}
	}
	e = c.Resolve("qwen3-vl:8b", env, nil, nil, nil)
	if e.OnBusy.V != OnBusyNext || !strings.Contains(e.String(), "wait_timeout=ignored") {
		t.Fatalf("on_busy next: %s", e.String())
	}
}

func TestResolveNoConfig(t *testing.T) {
	var c *Config
	e := c.Resolve("qwen3:0.6b", nil, nil, nil, nil)
	l := e.Launch()
	if len(l.Env) != 0 || len(l.Args) != 0 || l.Parallel != 0 || l.KVType != "" || l.Fingerprint() != "" {
		t.Fatalf("no config must not change the launch: %+v", l)
	}
	if c.DefaultsOptions() != nil || c.ModelOptions("x") != nil || c.SetsContext("x") {
		t.Fatalf("nil config option maps")
	}
}

func TestLaunch(t *testing.T) {
	c := mustParse(t, exampleConfig+`  zoff:latest:
    zones: "off"
    parallel: 3
    n_ubatch: 256
    flash_attention: "on"
    kv_type: q8_0
`)
	l := c.Resolve("qwen3-coder:30b", nil, nil, nil, nil).Launch()
	if l.Env["LLAMA_ARG_DUAL_BUFFER"] != "1" || l.Env["LLAMA_ARG_AUTO_ZONES"] != "1" {
		t.Fatalf("coder env: %+v", l.Env)
	}
	if l.KVType != "f16" {
		t.Fatalf("kv: %q", l.KVType)
	}
	if l.Env["LLAMA_ARG_ZONES_FLEX"] != "1" {
		t.Fatalf("coder flex band: %+v", l.Env)
	}
	if nk := c.Resolve("x:y", nil, nil, map[string]any{"dynamic_kv": false}, nil).Launch(); nk.Env["LLAMA_ARG_AUTO_ZONES"] != "0" || nk.Env["LLAMA_ARG_ZONES_FLEX"] != "" {
		t.Fatalf("no dynamic KV: %+v", nk.Env)
	}
	l = c.Resolve("qwen3-embedding:0.6b", nil, nil, nil, nil).Launch()
	if l.Env["LLAMA_ARG_DUAL_BUFFER"] != "0" {
		t.Fatalf("embedding dual: %+v", l.Env)
	}
	l = c.Resolve("zoff", nil, nil, nil, nil).Launch()
	// zones: off keeps the dynamic KV (defaults: dynamic_kv true), without a flex band
	if l.Env["LLAMA_ARG_AUTO_ZONES"] != "1" || l.Env["LLAMA_ARG_ZONES_FLEX"] != "0" || len(l.Notes) != 0 {
		t.Fatalf("zones off: %+v", l)
	}
	if l.Parallel != 3 || l.KVType != "q8_0" || l.Args["-ub"][0] != "256" || l.Args["--flash-attn"][0] != "on" {
		t.Fatalf("zoff launch: %+v", l)
	}
	if fp := l.Fingerprint(); fp != "LLAMA_ARG_AUTO_ZONES=1;LLAMA_ARG_DUAL_BUFFER=1;LLAMA_ARG_ZONES_FLEX=0;--flash-attn=on;-ub=256;np=3;kv=q8_0" {
		t.Fatalf("fingerprint: %s", fp)
	}
	if m := c.ModelOptions("qwen3-vl:8b"); m["num_ctx"] != float64(16384) {
		t.Fatalf("model options: %+v", m)
	}
	if !c.SetsContext("qwen3-vl:8b") || c.SetsContext("llama3:8b") {
		t.Fatalf("SetsContext")
	}
}

func TestMatchDevices(t *testing.T) {
	c := mustParse(t, exampleConfig)
	found := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Name: "ROCm0", Description: "AMD Radeon RX 9070", PCIID: "0000:03:00.0"},
		{DeviceID: ml.DeviceID{ID: "0", Library: "Vulkan"}, Name: "Vulkan0", Description: "AMD Radeon RX 550 / 550 Series (RADV POLARIS12)", PCIID: "0000:86:00.0"},
	}
	dm := c.MatchDevices(found)
	if dm.ByName["x080"].Info.Name != "ROCm0" || dm.ByName["rx550"].Info.Name != "Vulkan0" || !dm.Present("cpu") {
		t.Fatalf("match: %s", dm)
	}
	if len(dm.Missing) != 0 || len(dm.Errors) != 0 {
		t.Fatalf("missing/errors: %v %v", dm.Missing, dm.Errors)
	}

	// the backend must agree: the RX 550 seen only by ROCm does not match a vulkan entry
	dm = c.MatchDevices(found[:1])
	if len(dm.Missing) != 1 || dm.Missing[0] != "rx550" {
		t.Fatalf("missing: %v", dm.Missing)
	}

	// two cards of one type: a name match is ambiguous, the PCI address is not
	two := mustParse(t, "devices:\n  a: { backend: rocm, match: \"RX 9070\" }\n  b: { backend: rocm, match: \"04:00.0\" }\n")
	found2 := append(found[:1:1], ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "1", Library: "ROCm"}, Name: "ROCm1", Description: "AMD Radeon RX 9070", PCIID: "0000:04:00.0"})
	dm = two.MatchDevices(found2)
	if len(dm.Errors) != 1 || dm.Present("a") || dm.ByName["b"].Info.Name != "ROCm1" {
		t.Fatalf("two cards: %s %v", dm, dm.Errors)
	}
	// two entries for one card
	dup := mustParse(t, "devices:\n  a: { backend: rocm, match: \"RX 9070\" }\n  b: { backend: rocm, match: \"03:00.0\" }\n")
	dm = dup.MatchDevices(found[:1])
	if len(dm.Errors) != 1 || dm.Present("a") || dm.Present("b") {
		t.Fatalf("dup: %s %v", dm, dm.Errors)
	}
}

func TestStoreLastValid(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, ConfigFileName)
	s := NewStore(p)
	if s.Current() != nil {
		t.Fatal("missing file must give nil")
	}
	write := func(body string, at time.Time) {
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(p, at, at); err != nil {
			t.Fatal(err)
		}
	}
	t0 := time.Now().Add(-time.Hour)
	write("defaults:\n  threads: 3\n", t0)
	c := s.Current()
	if c == nil || *c.Defaults.Threads != 3 {
		t.Fatalf("first: %+v", c)
	}
	write("defaults:\n  threads: [\n", t0.Add(time.Minute))
	if c2 := s.Current(); c2 != c {
		t.Fatalf("broken file must keep the last valid config")
	}
	write("defaults:\n  threads: 5\n", t0.Add(2*time.Minute))
	if c3 := s.Current(); c3 == nil || *c3.Defaults.Threads != 5 {
		t.Fatalf("fixed: %+v", c3)
	}
	os.Remove(p)
	if s.Current() != nil {
		t.Fatal("removed file must give nil")
	}
	if got := DefaultPath(func(k string) string { return "" }, dir); got != p {
		t.Fatalf("default path %s", got)
	}
	if got := DefaultPath(func(k string) string { return map[string]string{"OLLAMA_DUALBUF_CONFIG": "/x.yaml"}[k] }, dir); got != "/x.yaml" {
		t.Fatalf("env path %s", got)
	}
}

func TestParseBytes(t *testing.T) {
	for s, want := range map[string]uint64{"90GiB": 90 << 30, "512MiB": 512 << 20, "100G": 100 << 30, "1000": 1000, "1.5GB": 1500000000} {
		if got, err := ParseBytes(s); err != nil || got != want {
			t.Errorf("%s: %d %v", s, got, err)
		}
	}
}

func TestExampleFileParses(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "docs", "dualbuf.example.yaml"))
	if err != nil {
		t.Skip(err)
	}
	c := mustParse(t, string(data))
	if len(c.Devices) != 3 || len(c.Models) != 3 || !c.SwapEnabled() {
		t.Fatalf("example: %+v", c)
	}
}
