package multimodel

import (
	"os"
	"path/filepath"
	"strconv"
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

// kfdNode is a node of a fake KFD topology: its gpu_id (0 = CPU), domain and location_id (bus<<8 | device<<3 | function)
type kfdNode struct {
	gpuID          int
	domain, locate int
}

// fakeKFD writes class/kfd/kfd/topology/nodes/<n>/{gpu_id,properties} under a temporary sysfs root
func fakeKFD(t *testing.T, nodes []kfdNode) string {
	t.Helper()
	root := t.TempDir()
	for i, n := range nodes {
		dir := filepath.Join(root, "class", "kfd", "kfd", "topology", "nodes", strconv.Itoa(i))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		props := "cpu_cores_count 0\nsimd_count 40\nlocation_id " + strconv.Itoa(n.locate) + "\ndrm_render_minor 128\ndomain " + strconv.Itoa(n.domain) + "\n"
		if n.gpuID == 0 {
			props = "cpu_cores_count 8\nsimd_count 0\nlocation_id 0\ndomain 0\n"
		}
		if err := os.WriteFile(filepath.Join(dir, "properties"), []byte(props), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "gpu_id"), []byte(strconv.Itoa(n.gpuID)+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// the PCI address of a KFD GPU node from its domain and location_id; CPU nodes are no card
func TestKFDNodes(t *testing.T) {
	root := fakeKFD(t, []kfdNode{
		{gpuID: 0},
		{gpuID: 1111, locate: 0xbb << 8},
		{gpuID: 2222, locate: 0xaa << 8},
		{gpuID: 3333, domain: 1, locate: 0xcc<<8 | 2<<3 | 1},
	})
	got := KFDNodes(root)
	if len(got) != 3 || got["0000:bb:00.0"] != 1 || got["0000:aa:00.0"] != 2 || got["0001:cc:02.1"] != 3 {
		t.Fatalf("kfd nodes %v", got)
	}
	if !sysfsPresent(root) || sysfsPresent(filepath.Join(root, "none")) || KFDNodes(filepath.Join(root, "none")) != nil {
		t.Fatal("sysfs presence")
	}
}

// the rocm devices by PCI address, or a name resolved to one from the known devices, with their KFD node; left out
// with a reason
func TestROCmCards(t *testing.T) {
	cfg, err := Parse([]byte(rocmTestConfig))
	if err != nil {
		t.Fatal(err)
	}
	kfd := map[string]int{"0000:bb:00.0": 1, "0000:aa:00.0": 2, "0000:cc:00.0": 3, "0000:dd:00.0": 4}
	known := []ml.DeviceInfo{{Description: "Test GPU C", PCIID: "0000:cc:00.0"}, {Description: "Test GPU B", PCIID: "0000:bb:00.0"}}
	cards, notes := cfg.ROCmCards(kfd, known)
	if len(notes) != 0 || len(cards) != 2 || cards[0].Device != "gpu_named" || cards[0].PCI != "0000:cc:00.0" || cards[0].Node != 3 ||
		cards[1].Device != "gpu_rocm" || cards[1].PCI != "0000:aa:00.0" || cards[1].Node != 2 || cards[1].GFXOverride != "10.3.0" {
		t.Fatalf("cards %+v notes %v", cards, notes)
	}
	// a name no known card has, a card without a KFD node
	cards, notes = cfg.ROCmCards(map[string]int{"0000:bb:00.0": 1}, nil)
	if len(cards) != 0 || len(notes) != 2 || !strings.Contains(notes[1], "KFD") {
		t.Fatalf("left out: %+v %v", cards, notes)
	}
	// an ambiguous name
	dup := []ml.DeviceInfo{{Description: "Test GPU C", PCIID: "0000:cc:00.0"}, {Description: "Test GPU C", PCIID: "0000:dd:00.0"}}
	if _, notes := cfg.ROCmCards(kfd, dup); len(notes) != 1 {
		t.Fatalf("ambiguous: %v", notes)
	}
	var nilCfg *Config
	if c, n := nilCfg.ROCmCards(kfd, nil); c != nil || n != nil {
		t.Fatal("no config")
	}
}

// each card's override is its own per-node variable, never the global one; a card without an override gets none
func TestROCmEnv(t *testing.T) {
	env := ROCmEnv([]ROCmCard{{Device: "r1", PCI: "0000:aa:00.0", GFXOverride: "10.3.0", Node: 2}, {Device: "r2", PCI: "0000:cc:00.0", GFXOverride: "11.0.0", Node: 3}, {Device: "r3", PCI: "0000:dd:00.0", Node: 4}})
	if len(env) != 2 || env["HSA_OVERRIDE_GFX_VERSION_2"] != "10.3.0" || env["HSA_OVERRIDE_GFX_VERSION_3"] != "11.0.0" {
		t.Fatalf("env %v", env)
	}
	if _, ok := env["HSA_OVERRIDE_GFX_VERSION"]; ok {
		t.Fatal("the global override is never set")
	}
	if len(ROCmEnv(nil)) != 0 {
		t.Fatal("no cards")
	}
}

// the runtime's own numbering counts every KFD GPU node in node order, a card it does not support included
func TestROCrIndices(t *testing.T) {
	got := ROCrIndices(map[string]int{"0000:aa:00.0": 2, "0000:bb:00.0": 1, "0000:cc:00.0": 4})
	if len(got) != 3 || got["0000:bb:00.0"] != 0 || got["0000:aa:00.0"] != 1 || got["0000:cc:00.0"] != 2 {
		t.Fatalf("indices %v", got)
	}
	if len(ROCrIndices(nil)) != 0 {
		t.Fatal("no topology")
	}
}
