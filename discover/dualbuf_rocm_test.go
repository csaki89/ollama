package discover

import (
	"context"
	"io"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// the discovery environment of the cards: their per-node gfx overrides, never the global one
func TestDualbufROCmEnv(t *testing.T) {
	cards := &DualbufROCmCards{Devices: []string{"gpu_rocm"}, PCIs: []string{"0000:aa:00.0"}, Env: map[string]string{"HSA_OVERRIDE_GFX_VERSION_2": "10.3.0", "HSA_OVERRIDE_GFX_VERSION": "10.3.0"}}
	env := dualbufROCmEnv(cards)
	if len(env) != 1 || env["HSA_OVERRIDE_GFX_VERSION_2"] != "10.3.0" {
		t.Fatalf("env %v", env)
	}
	if cards.Env["HSA_OVERRIDE_GFX_VERSION"] == "" {
		t.Fatal("the cards' own environment is not changed")
	}
	if normPCIID("AA:00.0") != "0000:aa:00.0" {
		t.Fatal("pci")
	}
}

// the ROCm part of an environment, as the logs show it
func TestDualbufROCmEnvSummary(t *testing.T) {
	env := map[string]string{"HSA_OVERRIDE_GFX_VERSION_2": "10.3.0", "ROCR_VISIBLE_DEVICES": "0", "OTHER": "y"}
	if got := DualbufROCmEnvSummary(env); got != "HSA_OVERRIDE_GFX_VERSION_2=10.3.0 ROCR_VISIBLE_DEVICES=0" {
		t.Fatalf("summary %q", got)
	}
	c := &DualbufROCmCards{Devices: []string{"a", "b"}, PCIs: []string{"0000:aa:00.0", "0000:cc:00.0"}}
	if dualbufCardList(c) != "a=0000:aa:00.0 b=0000:cc:00.0" {
		t.Fatal(dualbufCardList(c))
	}
}

// a GPU library pass without a device: its error lines are the status error (the "left out" warning shows them)
func TestDualbufLogNoDevice(t *testing.T) {
	out := "load_backend: loaded CPU backend\nload_backend: failed to find ggml_backend_init in /x/libggml-hip.so\nsystem_info: n_threads = 4\n"
	if got := discoveryErrorLines(out); len(got) != 1 || !strings.Contains(got[0], "failed to find ggml_backend_init") {
		t.Fatalf("error lines %v", got)
	}
	st := llm.NewStatusWriter(io.Discard)
	dualbufLogNoDevice([]string{"/lib", "/lib/rocm"}, nil, out, st)
	if !strings.Contains(st.LastError(), "ggml_backend_init") {
		t.Fatalf("status error %q", st.LastError())
	}
	st2 := llm.NewStatusWriter(io.Discard)
	dualbufLogNoDevice([]string{"/lib"}, nil, out, st2)
	if st2.LastError() != "" {
		t.Fatal("the CPU pass is not a GPU library")
	}
}

// an unsupported card first in the runtime's numbering: the runtime skips it, so the config card is HIP ordinal 0
// but ROCr index 1; every later ROCm process (verification, runner) selects it by the ROCr index and gets its
// per-card override
func TestDualbufDiscoverROCmSkippedCardFirst(t *testing.T) {
	old := dualbufBootstrap
	t.Cleanup(func() { dualbufBootstrap = old })
	var gotEnv map[string]string
	dualbufBootstrap = func(_ context.Context, dirs []string, env map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
		gotEnv = env
		return []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, PCIID: "0000:aa:00.0", LibraryPath: dirs}}, llm.NewStatusWriter(io.Discard), nil
	}
	cards := &DualbufROCmCards{
		Devices:   []string{"gpu_rocm"},
		PCIs:      []string{"0000:aa:00.0"},
		Env:       map[string]string{"HSA_OVERRIDE_GFX_VERSION_2": "10.3.0"},
		ROCrIndex: map[string]int{"0000:bb:00.0": 0, "0000:aa:00.0": 1},
	}
	devs := dualbufDiscoverROCm(context.Background(), time.Second, []string{"/lib", "/lib/rocm"}, cards)
	if len(devs) != 1 || devs[0].ID != "0000:aa:00.0" || devs[0].FilterID != "1" {
		t.Fatalf("devices %+v", devs)
	}
	if gotEnv["HSA_OVERRIDE_GFX_VERSION_2"] != "10.3.0" {
		t.Fatalf("discovery env %v", gotEnv)
	}
	env := ml.GetDevicesEnv(devs)
	visible := env["ROCR_VISIBLE_DEVICES"]
	if runtime.GOOS != "linux" {
		visible = env["HIP_VISIBLE_DEVICES"]
	}
	if visible != "1" || env["HSA_OVERRIDE_GFX_VERSION_2"] != "10.3.0" {
		t.Fatalf("process env %v", env)
	}

	// without a ROCr index (no topology): the HIP ordinal, as before
	cards.ROCrIndex = nil
	if devs := dualbufDiscoverROCm(context.Background(), time.Second, []string{"/lib", "/lib/rocm"}, cards); len(devs) != 1 || devs[0].FilterID != "0" {
		t.Fatalf("no index: %+v", devs)
	}
}
