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
default:
  dual_buffer: true
  dynamic_kv: true
  kv_type: f16
  keep_alive: 30m

devices:
  gpu0:  { backend: rocm,   match: "0000:aa:00.0", max_running: 1 }
  gpu1: { backend: vulkan, match: "Test GPU B", max_running: 1 }
  cpu:   { backend: cpu, max_running: 1 }

swap:
  enabled: true
  keep_in_ram: pinned
  save_kv: true

models:
  qwen3-coder:30b:
    devices: [gpu0, cpu]
    on_busy: wait
    wait_timeout: 120s
    zones: auto
    max_context: 131072
  qwen3-vl:8b:
    devices: [gpu0, cpu]
    on_busy: next
    max_context: 16384
  qwen3-embedding:*:
    devices: [gpu1, cpu]
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
	if c.MaxRunning("gpu0") != 1 || c.MaxRunning("cpu") != 1 || c.MaxRunning("nope") != 0 {
		t.Fatalf("max_running: %d %d", c.MaxRunning("gpu0"), c.MaxRunning("cpu"))
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

func TestParseEmptyNeedsDefault(t *testing.T) {
	// an empty file has no default section: invalid (a missing file is no config)
	if _, err := Parse([]byte("  \n")); err == nil || !strings.Contains(err.Error(), "default: missing") {
		t.Fatalf("empty: %v", err)
	}
	if c, err := Parse([]byte("default: {}\n")); err != nil || c == nil {
		t.Fatalf("only an empty default section: %v", err)
	}
}

func TestParseErrors(t *testing.T) {
	// every case has a default section, so the error is the one named
	cases := map[string]struct{ body, want string }{
		"unknown key":        {"default:\n  dual_bufer: true\n", "dual_bufer"},
		"bad on_busy":        {"default: {}\nmodels:\n  a:b:\n    on_busy: later\n", "on_busy"},
		"unknown device":     {"default: {}\nmodels:\n  a:b:\n    devices: [gpu9]\n", "unknown device \"gpu9\""},
		"cpu not declared":   {"default:\n  devices: [cpu]\n", "unknown device \"cpu\""},
		"bad backend":        {"default: {}\ndevices:\n  g: { backend: opencl, match: x }\n", "unknown backend"},
		"no backend":         {"default: {}\ndevices:\n  g: { match: x }\n", "backend: missing"},
		"no match":           {"default: {}\ndevices:\n  g: { backend: rocm }\n", "match: missing"},
		"bad kv type":        {"default:\n  kv_type: q3_k\n", "kv_type"},
		"zones number":       {"default:\n  zones: 4\n", "zones"},
		"bad duration":       {"default:\n  keep_alive: soon\n", "invalid duration"},
		"max_running zero":   {"default: {}\ndevices:\n  cpu: { backend: cpu, max_running: 0 }\n", "max_running"},
		"negative reserve":   {"default: {}\ndevices:\n  g: { backend: rocm, match: x, reserve_mib: -1 }\n", "reserve_mib"},
		"bad keep_in_ram":    {"default: {}\nswap:\n  keep_in_ram: disk\n", "keep_in_ram"},
		"bad ram_limit":      {"default: {}\nram_limit: lots\n", "ram_limit"},
		"bad fa":             {"default:\n  flash_attention: maybe\n", "flash_attention"},
		"yaml syntax":        {"default: [\n", "yaml"},
		"cpu other backend":  {"default: {}\ndevices:\n  cpu: { backend: rocm, match: x }\n", "devices.cpu.backend"},
		"cpu with match":     {"default: {}\ndevices:\n  cpu: { backend: cpu, match: x }\n", "devices.cpu.match"},
		"cpu backend elsewh": {"default: {}\ndevices:\n  host: { backend: cpu }\n", "entry named cpu only"},
		"device twice":       {"default: {}\ndevices:\n  g: { backend: rocm, match: x }\nmodels:\n  a:b:\n    devices: [g, g]\n", "listed twice"},
		"no default":         {"devices:\n  cpu: { backend: cpu }\n", "default: missing"},
		"old defaults name":  {"default: {}\ndefaults:\n  threads: 2\n", "called default now"},
		"duplicate model":    {"default: {}\nmodels:\n  a:b: {}\n  a:b: {}\n", "already defined"},
		"bad gfx_override":   {"default: {}\ndevices:\n  g: { backend: rocm, match: x, gfx_override: gfx1030 }\n", "is not a gfx version"},
		"gfx_override vk":    {"default: {}\ndevices:\n  g: { backend: vulkan, match: x, gfx_override: 10.3.0 }\n", "only a rocm device"},
	}
	for name, tc := range cases {
		_, err := Parse([]byte(tc.body))
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %v, want %q in it", name, err, tc.want)
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
		"qwen3-embedding:4b":   "qwen3-embedding:*",    // the first matching pattern of the file
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

	// the order of the file decides between patterns, not their length
	o := mustParse(t, "default: {}\nmodels:\n  qwen3*:\n    threads: 1\n  qwen3-embedding:*:\n    threads: 2\n  qwen3-embedding:0.6b:\n    threads: 3\n")
	for name, want := range map[string]string{"qwen3-embedding:4b": "qwen3*", "qwen3-embedding:0.6b": "qwen3-embedding:0.6b", "llama3": ""} {
		if got, _ := o.SectionFor(name); got != want {
			t.Errorf("order: %s: got %q want %q", name, got, want)
		}
	}
	o = mustParse(t, "default: {}\nmodels:\n  qwen3-embedding:*:\n    threads: 2\n  qwen3*:\n    threads: 1\n")
	if got, _ := o.SectionFor("qwen3-embedding:4b"); got != "qwen3-embedding:*" {
		t.Errorf("order reversed: %q", got)
	}
}

func TestDefaultSection(t *testing.T) {
	c := mustParse(t, exampleConfig)
	// no section: the default only
	e := c.Resolve("llama3:8b", nil, nil, nil, nil)
	if e.Section != "" || !e.DualBuffer.V || e.DualBuffer.Src != SrcDefaults || e.KeepAlive.V.Duration != 30*time.Minute {
		t.Fatalf("default for an unknown model: %+v", e)
	}
	// a section: its fields over the default, the rest from the default
	e = c.Resolve("qwen3-embedding:0.6b", nil, nil, nil, nil)
	if e.DualBuffer.V || e.KVType.V != "f16" || e.KVType.Src != SrcDefaults {
		t.Fatalf("section over default: %+v %+v", e.DualBuffer, e.KVType)
	}
}

func TestWaitTimeoutForever(t *testing.T) {
	c := mustParse(t, exampleConfig+"  w:latest:\n    devices: [gpu0, cpu]\n    wait_timeout: -1\n")
	e := c.Resolve("w", nil, nil, nil, nil)
	if e.WaitTimeout.V != time.Duration(math.MaxInt64) || !strings.Contains(e.String(), "wait_timeout=-1(model:w:latest)") {
		t.Fatalf("wait_timeout -1: %v %s", e.WaitTimeout, e.String())
	}
}

func TestReserve(t *testing.T) {
	c := mustParse(t, exampleConfig)
	// no reserve_mib: the GPU keeps LLAMA_ARG_FIT_TARGET, the CPU max(1 GiB, 5%)
	if _, ok := c.GPUReserveMiB("gpu0"); ok {
		t.Fatal("gpu0 has no reserve_mib")
	}
	if got := c.RAMReserve(100 << 30); got != 5<<30 {
		t.Fatalf("cpu default 5%%: %d", got)
	}
	if got := c.RAMReserve(8 << 30); got != 1<<30 {
		t.Fatalf("cpu default 1 GiB: %d", got)
	}
	r := mustParse(t, "default: {}\ndevices:\n  gpu0: { backend: rocm, match: x, reserve_mib: 1536 }\n  gpu1: { backend: vulkan, match: y, reserve_mib: 0 }\n  cpu: { backend: cpu, reserve_mib: 2048 }\n")
	if mib, ok := r.GPUReserveMiB("gpu0"); !ok || mib != 1536 {
		t.Fatalf("gpu0 override: %d %v", mib, ok)
	}
	if mib, ok := r.GPUReserveMiB("gpu1"); !ok || mib != 0 {
		t.Fatalf("gpu1 zero override: %d %v", mib, ok)
	}
	if _, ok := r.GPUReserveMiB("cpu"); ok {
		t.Fatal("the cpu reserve is no fit target")
	}
	if got := r.RAMReserve(100 << 30); got != 2048<<20 {
		t.Fatalf("cpu override: %d", got)
	}
	var none *Config
	if _, ok := none.GPUReserveMiB("gpu0"); ok || none.RAMReserve(8<<30) != 1<<30 {
		t.Fatal("nil config")
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
	if got := strings.Join(e.Devices.V, ","); got != "gpu1,cpu" {
		t.Fatalf("devices: %s", got)
	}
	line := e.String()
	for _, want := range []string{"devices=[gpu1,cpu](model:qwen3-embedding:*)", "dual_buffer=false(model:qwen3-embedding:*)", "keep_alive=-1(model:qwen3-embedding:*)", "parallel=2(env)", "kv_type=f16(default)", "on_busy=wait(builtin)", "wait_timeout=1m0s(builtin)"} {
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
		{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Name: "ROCm0", Description: "AMD Radeon Test GPU A", PCIID: "0000:aa:00.0"},
		{DeviceID: ml.DeviceID{ID: "0", Library: "Vulkan"}, Name: "Vulkan0", Description: "AMD Radeon Test GPU B (RADV)", PCIID: "0000:bb:00.0"},
	}
	dm := c.MatchDevices(found)
	if dm.ByName["gpu0"].Info.Name != "ROCm0" || dm.ByName["gpu1"].Info.Name != "Vulkan0" || !dm.Present("cpu") {
		t.Fatalf("match: %s", dm)
	}
	if len(dm.Missing) != 0 || len(dm.Errors) != 0 {
		t.Fatalf("missing/errors: %v %v", dm.Missing, dm.Errors)
	}

	// the backend must agree: the Vulkan card seen only by ROCm does not match a vulkan entry
	dm = c.MatchDevices(found[:1])
	if len(dm.Missing) != 1 || dm.Missing[0] != "gpu1" {
		t.Fatalf("missing: %v", dm.Missing)
	}

	// two cards of one type: a name match is ambiguous, the PCI address is not
	two := mustParse(t, "default: {}\ndevices:\n  a: { backend: rocm, match: \"Test GPU A\" }\n  b: { backend: rocm, match: \"cc:00.0\" }\n")
	found2 := append(found[:1:1], ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "1", Library: "ROCm"}, Name: "ROCm1", Description: "AMD Radeon Test GPU A", PCIID: "0000:cc:00.0"})
	dm = two.MatchDevices(found2)
	if len(dm.Errors) != 1 || dm.Present("a") || dm.ByName["b"].Info.Name != "ROCm1" {
		t.Fatalf("two cards: %s %v", dm, dm.Errors)
	}
	// two entries for one card
	dup := mustParse(t, "default: {}\ndevices:\n  a: { backend: rocm, match: \"Test GPU A\" }\n  b: { backend: rocm, match: \"aa:00.0\" }\n")
	dm = dup.MatchDevices(found[:1])
	if len(dm.Errors) != 1 || dm.Present("a") || dm.Present("b") {
		t.Fatalf("dup: %s %v", dm, dm.Errors)
	}
}

func TestStoreLoadOnce(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, ConfigFileName)
	// no file: no config, no error
	if c, err := NewStore(p).Load(); c != nil || err != nil {
		t.Fatalf("missing file: %v %v", c, err)
	}
	if err := os.WriteFile(p, []byte("default:\n  threads: 3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := NewStore(p)
	if s.Current() != nil {
		t.Fatal("nothing is read before Load")
	}
	c, err := s.Load()
	if err != nil || c == nil || *c.Defaults.Threads != 3 || s.Current() != c {
		t.Fatalf("load: %+v %v", c, err)
	}
	// a change while running is not read
	if err := os.WriteFile(p, []byte("default:\n  threads: 5\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if c2, _ := s.Load(); c2 != c || *s.Current().Defaults.Threads != 3 {
		t.Fatal("the config is read once")
	}
	// a broken file: Load fails with the path and the reason (the server does not start)
	if err := os.WriteFile(p, []byte("default:\n  threads: [\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if c, err := NewStore(p).Load(); c != nil || err == nil || !strings.Contains(err.Error(), p) || !strings.Contains(err.Error(), "invalid") {
		t.Fatalf("broken file: %v %v", c, err)
	}
	if err := os.WriteFile(p, []byte("default:\n  devices: [gpu9]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	m := NewManager(p, nil)
	if err := m.Load(); err == nil || !strings.Contains(err.Error(), "gpu9") {
		t.Fatalf("manager load: %v", err)
	}
	if m.Config() != nil {
		t.Fatal("no config after a failed load")
	}
	if got := DefaultPath(func(k string) string { return "" }, dir); got != p {
		t.Fatalf("default path %s", got)
	}
	if got := DefaultPath(func(k string) string { return map[string]string{"OLLAMA_DUALBUF_CONFIG": "/x.yaml"}[k] }, dir); got != "/x.yaml" {
		t.Fatalf("env path %s", got)
	}
}

func TestUnavailableDeviceSkipped(t *testing.T) {
	m := NewManager(filepath.Join(t.TempDir(), ConfigFileName), nil)
	c := mustParse(t, exampleConfig)
	// only the ROCm card is seen: gpu1 is skipped, the lists keep their other devices in order
	m.NoteDevices([]ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Name: "ROCm0", Description: "AMD Radeon Test GPU A", PCIID: "0000:aa:00.0"}})
	if got := strings.Join(m.Present(c, []string{"gpu1", "cpu"}), ","); got != "cpu" {
		t.Fatalf("gpu1 must be skipped: %s", got)
	}
	if got := strings.Join(m.Present(c, []string{"gpu0", "gpu1", "cpu"}), ","); got != "gpu0,cpu" {
		t.Fatalf("present: %s", got)
	}
	if dm := m.Devices(c); len(dm.Missing) != 1 || dm.Missing[0] != "gpu1" {
		t.Fatalf("missing: %v", dm.Missing)
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
