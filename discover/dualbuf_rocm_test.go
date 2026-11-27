package discover

import (
	"os"
	"path/filepath"
	"testing"
)

// a view's ROCm environment gets the hiding library of its ROCm directory; without the library nothing is hidden
func TestDualbufROCmEnv(t *testing.T) {
	card := DualbufROCmView{Devices: []string{"gpu_rocm"}, PCIs: []string{"0000:aa:00.0"}, Env: map[string]string{"GGML_HIP_HIDE_RENDER_NODES": "renderD129", "HSA_OVERRIDE_GFX_VERSION": "10.3.0"}}
	dir := t.TempDir()
	if env := dualbufROCmEnv(card, dir); env["LD_PRELOAD"] != "" || env["GGML_HIP_HIDE_RENDER_NODES"] != "" || env["HSA_OVERRIDE_GFX_VERSION"] != "10.3.0" {
		t.Fatalf("no library: %v", env)
	}
	if err := os.WriteFile(filepath.Join(dir, hideLibrary), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LD_PRELOAD", "")
	env := dualbufROCmEnv(card, dir)
	if env["LD_PRELOAD"] != filepath.Join(dir, hideLibrary) || env["GGML_HIP_HIDE_RENDER_NODES"] != "renderD129" {
		t.Fatalf("with the library: %v", env)
	}
	if card.Env["LD_PRELOAD"] != "" {
		t.Fatal("the view's own environment is not changed")
	}
	if normPCIID("AA:00.0") != "0000:aa:00.0" {
		t.Fatal("pci")
	}
}

// the ROCm view part of an environment, as the logs show it
func TestDualbufROCmEnvSummary(t *testing.T) {
	env := map[string]string{"HSA_OVERRIDE_GFX_VERSION": "10.3.0", "GGML_HIP_HIDE_RENDER_NODES": "renderD129", "LD_PRELOAD": "/x/libggml-hide-dri.so", "OTHER": "y"}
	if got := DualbufROCmEnvSummary(env); got != "GGML_HIP_HIDE_RENDER_NODES=renderD129 HSA_OVERRIDE_GFX_VERSION=10.3.0 LD_PRELOAD=/x/libggml-hide-dri.so" {
		t.Fatalf("summary %q", got)
	}
	v := DualbufROCmView{Devices: []string{"a", "b"}, PCIs: []string{"0000:aa:00.0", "0000:cc:00.0"}}
	if dualbufViewCards(v) != "a=0000:aa:00.0 b=0000:cc:00.0" {
		t.Fatal(dualbufViewCards(v))
	}
}
