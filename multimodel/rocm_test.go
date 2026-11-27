package multimodel

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/ollama/ollama/ml"
)

const rocmTestConfig = `
default: {}
devices:
  gpu_rocm: { backend: rocm, match: "0000:aa:00.0", gfx_override: 10.3.0 }
  gpu_named: { backend: rocm, match: "Test GPU C" }
  gpu_vk: { backend: vulkan, match: "0000:bb:00.0" }
  cpu: { backend: cpu }
`

// fake sysfs: /sys/class/drm/renderD<N>/device -> the card's PCI directory
func fakeSysfs(t *testing.T, cards map[string]string) string {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("sysfs: Linux only")
	}
	root := t.TempDir()
	for node, pci := range cards {
		dev := filepath.Join(root, "devices", "pci0000:00", pci)
		if err := os.MkdirAll(dev, 0o755); err != nil {
			t.Fatal(err)
		}
		dir := filepath.Join(root, "class", "drm", node)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(dev, filepath.Join(dir, "device")); err != nil {
			t.Skipf("no symlinks here: %v", err)
		}
	}
	return root
}

func TestRenderNodes(t *testing.T) {
	root := fakeSysfs(t, map[string]string{"renderD128": "0000:aa:00.0", "renderD129": "0000:bb:00.0", "renderD130": "0000:cc:00.0"})
	n := RenderNodes(root)
	if len(n) != 3 || n["0000:aa:00.0"][0] != "renderD128" || n["0000:cc:00.0"][0] != "renderD130" {
		t.Fatalf("render nodes %v", n)
	}
	if !sysfsPresent(root) || sysfsPresent(filepath.Join(root, "none")) {
		t.Fatal("sysfs presence")
	}
}

// the rocm devices by PCI address, or a name resolved to one from the known devices; left out with a reason
func TestROCmCards(t *testing.T) {
	cfg, err := Parse([]byte(rocmTestConfig))
	if err != nil {
		t.Fatal(err)
	}
	nodes := map[string][]string{"0000:aa:00.0": {"renderD128"}, "0000:bb:00.0": {"renderD129"}, "0000:cc:00.0": {"renderD130"}, "0000:dd:00.0": {"renderD131"}}
	known := []ml.DeviceInfo{{Description: "Test GPU C", PCIID: "0000:cc:00.0"}, {Description: "Test GPU B", PCIID: "0000:bb:00.0"}}
	cards, notes := cfg.ROCmCards(nodes, known)
	if len(notes) != 0 || len(cards) != 2 || cards[0].Device != "gpu_named" || cards[0].PCI != "0000:cc:00.0" || cards[0].minor != 130 ||
		cards[1].Device != "gpu_rocm" || cards[1].PCI != "0000:aa:00.0" || cards[1].GFXOverride != "10.3.0" {
		t.Fatalf("cards %+v notes %v", cards, notes)
	}
	// a name no known card has, a card without a render node in the container
	cards, notes = cfg.ROCmCards(map[string][]string{"0000:bb:00.0": {"renderD129"}}, nil)
	if len(cards) != 0 || len(notes) != 2 {
		t.Fatalf("left out: %+v %v", cards, notes)
	}
	// an ambiguous name
	dup := []ml.DeviceInfo{{Description: "Test GPU C", PCIID: "0000:cc:00.0"}, {Description: "Test GPU C", PCIID: "0000:dd:00.0"}}
	if _, notes := cfg.ROCmCards(nodes, dup); len(notes) != 1 {
		t.Fatalf("ambiguous: %v", notes)
	}
	var nilCfg *Config
	if c, n := nilCfg.ROCmCards(nodes, nil); c != nil || n != nil {
		t.Fatal("no config")
	}
}

// one view per gfx_override value (normally one, with every rocm card): only the other cards hidden (another backend's
// card, one the config does not name; with several values the other values' cards too), the global
// HSA_OVERRIDE_GFX_VERSION of the value in the view, never the per-node form
func TestROCmViews(t *testing.T) {
	nodes := map[string][]string{"0000:aa:00.0": {"renderD128"}, "0000:bb:00.0": {"renderD129"}, "0000:cc:00.0": {"renderD130"}, "0000:dd:00.0": {"renderD131"}}
	same := []ROCmCard{{Device: "r1", PCI: "0000:aa:00.0", GFXOverride: "10.3.0"}, {Device: "r2", PCI: "0000:cc:00.0", GFXOverride: "10.3.0"}}
	views, note := ROCmViews(same, nodes)
	if note != "" || len(views) != 1 || len(views[0].Cards) != 2 {
		t.Fatalf("one view: %+v %q", views, note)
	}
	env := views[0].Env
	if env["GGML_HIP_HIDE_RENDER_NODES"] != "renderD129,renderD131" || env["HSA_OVERRIDE_GFX_VERSION"] != "10.3.0" || len(env) != 2 {
		t.Fatalf("view env: %v", env)
	}

	mixed := []ROCmCard{{Device: "r1", PCI: "0000:aa:00.0", GFXOverride: "10.3.0"}, {Device: "r2", PCI: "0000:cc:00.0", GFXOverride: "11.0.0"}, {Device: "r3", PCI: "0000:dd:00.0"}}
	views, note = ROCmViews(mixed, nodes)
	if note == "" || len(views) != 3 {
		t.Fatalf("split by value: %+v %q", views, note)
	}
	if views[0].Env["HSA_OVERRIDE_GFX_VERSION"] != "10.3.0" || views[0].Env["GGML_HIP_HIDE_RENDER_NODES"] != "renderD129,renderD130,renderD131" {
		t.Fatalf("first view: %v", views[0].Env)
	}
	if views[1].Env["HSA_OVERRIDE_GFX_VERSION"] != "11.0.0" {
		t.Fatalf("second view: %v", views[1].Env)
	}
	if _, ok := views[2].Env["HSA_OVERRIDE_GFX_VERSION"]; ok {
		t.Fatal("a card without an override gets none")
	}
	for _, v := range views {
		for k := range v.Env {
			if strings.HasPrefix(k, "HSA_OVERRIDE_GFX_VERSION_") {
				t.Fatalf("the per-node form is not used: %v", v.Env)
			}
		}
	}
	if v, _ := ROCmViews(nil, nodes); v != nil {
		t.Fatal("no cards, no view")
	}
}
