package discover

import (
	"io"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
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
