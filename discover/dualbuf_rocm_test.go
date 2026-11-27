package discover

import (
	"os"
	"path/filepath"
	"testing"
)

// a card's ROCm environment gets the hiding library of its ROCm directory; without the library nothing is hidden
func TestDualbufROCmEnv(t *testing.T) {
	card := DualbufROCmCard{Device: "gpu_rocm", PCI: "0000:aa:00.0", Env: map[string]string{"GGML_HIP_HIDE_RENDER_NODES": "renderD129", "HSA_OVERRIDE_GFX_VERSION": "10.3.0"}}
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
		t.Fatal("the card's own environment is not changed")
	}
	if normPCIID("AA:00.0") != "0000:aa:00.0" {
		t.Fatal("pci")
	}
}
