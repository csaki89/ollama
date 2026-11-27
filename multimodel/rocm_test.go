package multimodel

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
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

// one view with every rocm card: only the other cards hidden (another backend's card, one the config does not name),
// each card's gfx_override as HSA_OVERRIDE_GFX_VERSION_<its node index in the view>; the thunk counts the CPU nodes
// and the reachable GPUs in the topology order
func TestROCmViews(t *testing.T) {
	nodes := map[string][]string{"0000:aa:00.0": {"renderD128"}, "0000:bb:00.0": {"renderD129"}, "0000:cc:00.0": {"renderD130"}, "0000:dd:00.0": {"renderD131"}}
	cards := []ROCmCard{
		{Device: "r1", PCI: "0000:aa:00.0", GFXOverride: "10.3.0", minor: 128},
		{Device: "r2", PCI: "0000:cc:00.0", GFXOverride: "11.0.0", minor: 130},
		{Device: "r3", PCI: "0000:dd:00.0", minor: 131},
	}
	// topology: CPU, the vulkan card (bb, hidden), r1, a card not in the container (minor 140), r2, r3
	kfd := []kfdNode{{cpu: true}, {minor: 129}, {minor: 128}, {minor: 140}, {minor: 130}, {minor: 131}}
	present := func(n string) bool { return n != "renderD140" }
	views, note := ROCmViews(cards, nodes, kfd, present)
	if note != "" || len(views) != 1 || len(views[0].Cards) != 3 {
		t.Fatalf("views %+v note %q", views, note)
	}
	env := views[0].Env
	if env["GGML_HIP_HIDE_RENDER_NODES"] != "renderD129" {
		t.Fatalf("only the non-rocm card is hidden: %v", env)
	}
	// node indices: CPU 0, r1 1 (bb hidden, 140 absent: not counted), r2 2, r3 3 (no override)
	if env["HSA_OVERRIDE_GFX_VERSION_1"] != "10.3.0" || env["HSA_OVERRIDE_GFX_VERSION_2"] != "11.0.0" || len(env) != 3 {
		t.Fatalf("per card overrides: %v", env)
	}
	if _, ok := env["HSA_OVERRIDE_GFX_VERSION"]; ok {
		t.Fatal("no global override")
	}

	// without a topology: cards with different overrides cannot share a view; one view per value, the global variable
	views, note = ROCmViews(cards, nodes, nil, present)
	if note == "" || len(views) != 3 {
		t.Fatalf("grouped: %+v %q", views, note)
	}
	if views[0].Env["HSA_OVERRIDE_GFX_VERSION"] != "10.3.0" || views[0].Env["GGML_HIP_HIDE_RENDER_NODES"] != "renderD129,renderD130,renderD131" {
		t.Fatalf("first group: %v", views[0].Env)
	}
	if _, ok := views[2].Env["HSA_OVERRIDE_GFX_VERSION"]; ok {
		t.Fatal("a card without an override gets none")
	}
	// the same override everywhere: one view, the global variable is the same thing
	same := []ROCmCard{{PCI: "0000:aa:00.0", GFXOverride: "10.3.0", minor: 128}, {PCI: "0000:cc:00.0", GFXOverride: "10.3.0", minor: 130}}
	if views, note := ROCmViews(same, nodes, nil, present); note != "" || len(views) != 1 || views[0].Env["HSA_OVERRIDE_GFX_VERSION"] != "10.3.0" {
		t.Fatalf("same override: %+v %q", views, note)
	}
	if v, _ := ROCmViews(nil, nodes, kfd, present); v != nil {
		t.Fatal("no cards, no view")
	}
}

func TestKFDNodes(t *testing.T) {
	root := t.TempDir()
	write := func(n int, gpu, props string) {
		d := filepath.Join(root, "class", "kfd", "kfd", "topology", "nodes", strconv.Itoa(n))
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(d, "gpu_id"), []byte(gpu+"\n"), 0o644)
		os.WriteFile(filepath.Join(d, "properties"), []byte(props), 0o644)
	}
	write(0, "0", "cpu_cores_count 8\n")
	write(2, "4321", "simd_count 80\ndrm_render_minor 129\n")
	write(10, "1234", "drm_render_minor 128\n")
	got := KFDNodes(root)
	if len(got) != 3 || !got[0].cpu || got[1].minor != 129 || got[2].minor != 128 {
		t.Fatalf("nodes %+v (numeric order)", got)
	}
	if KFDNodes(filepath.Join(root, "none")) != nil {
		t.Fatal("no topology")
	}
}
