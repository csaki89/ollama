package multimodel

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
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

// every rocm device gets a ROCm view of its own: the other cards' render nodes hidden (another backend's card and one
// the config does not name, e.g. a card ROCm cannot use), its own gfx_override only; by PCI address, or a name resolved
// to one from the known devices
func TestROCmCards(t *testing.T) {
	cfg, err := Parse([]byte(rocmTestConfig))
	if err != nil {
		t.Fatal(err)
	}
	nodes := map[string][]string{"0000:aa:00.0": {"renderD128"}, "0000:bb:00.0": {"renderD129"}, "0000:cc:00.0": {"renderD130"}, "0000:dd:00.0": {"renderD131"}}
	known := []ml.DeviceInfo{{Description: "Test GPU C", PCIID: "0000:cc:00.0"}, {Description: "Test GPU B", PCIID: "0000:bb:00.0"}}
	cards, notes := cfg.ROCmCards(nodes, known)
	if len(notes) != 0 || len(cards) != 2 {
		t.Fatalf("cards %+v notes %v", cards, notes)
	}
	byName := map[string]ROCmCard{}
	for _, c := range cards {
		byName[c.Device] = c
	}
	r := byName["gpu_rocm"]
	if r.PCI != "0000:aa:00.0" || !slices.Equal(r.Hide, []string{"renderD129", "renderD130", "renderD131"}) {
		t.Fatalf("gpu_rocm %+v", r)
	}
	if env := r.Env(); env["HSA_OVERRIDE_GFX_VERSION"] != "10.3.0" || env["GGML_HIP_HIDE_RENDER_NODES"] != "renderD129,renderD130,renderD131" {
		t.Fatalf("env %v", env)
	}
	c := byName["gpu_named"]
	if c.PCI != "0000:cc:00.0" || slices.Contains(c.Hide, "renderD130") || !slices.Contains(c.Hide, "renderD128") {
		t.Fatalf("name match %+v", c)
	}
	if _, ok := c.Env()["HSA_OVERRIDE_GFX_VERSION"]; ok {
		t.Fatal("the override of one card does not apply to another")
	}

	// a name no known card has, a card without a render node in the container: left out with a reason
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
