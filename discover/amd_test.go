package discover

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestApplyLinuxROCmRefinement(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake Linux PCI sysfs paths use ':' which is not valid in Windows filenames")
	}

	tests := []struct {
		name           string
		nodes          []fakeROCmNode
		devices        []ml.DeviceInfo
		applied        bool
		wantIntegrated []bool
		wantPCIIDs     []string
	}{
		{
			name: "apu is integrated",
			nodes: []fakeROCmNode{{
				node:        1,
				renderMinor: 128,
				gfxVersion:  "90012",
				vramTotal:   2 << 30,
				gttTotal:    32 << 30,
			}},
			devices: []ml.DeviceInfo{{
				DeviceID:  ml.DeviceID{ID: "0", Library: "ROCm"},
				Name:      "ROCm0",
				GFXTarget: "gfx90c",
			}},
			applied:        true,
			wantIntegrated: []bool{true},
		},
		{
			name: "low vram dgpu is not integrated",
			nodes: []fakeROCmNode{{
				node:        1,
				renderMinor: 128,
				gfxVersion:  "100601",
				vramTotal:   4 << 30,
				gttTotal:    32 << 30,
				vramVendor:  true,
				boardInfo:   true,
			}},
			devices: []ml.DeviceInfo{{
				DeviceID:  ml.DeviceID{ID: "0", Library: "ROCm"},
				Name:      "ROCm0",
				GFXTarget: "gfx1061",
			}},
			applied:        true,
			wantIntegrated: []bool{false},
		},
		{
			name: "mixed system follows kfd order not drm order",
			nodes: []fakeROCmNode{
				{
					node:        1,
					renderMinor: 129,
					gfxVersion:  "110000",
					vramTotal:   48 << 30,
					gttTotal:    64 << 30,
					vramVendor:  true,
					boardInfo:   true,
				},
				{
					node:        2,
					renderMinor: 128,
					gfxVersion:  "110003",
					vramTotal:   512 << 20,
					gttTotal:    32 << 30,
				},
			},
			devices: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Name: "ROCm0", GFXTarget: "gfx1100"},
				{DeviceID: ml.DeviceID{ID: "1", Library: "ROCm"}, Name: "ROCm1", GFXTarget: "gfx1103"},
			},
			applied:        true,
			wantIntegrated: []bool{false, true},
		},
		{
			name: "remapped visible order matches existing pci identity",
			nodes: []fakeROCmNode{
				{
					node:        1,
					renderMinor: 128,
					pciID:       "0000:e3:00.0",
					gfxVersion:  "110000",
					vramTotal:   48 << 30,
					gttTotal:    64 << 30,
					vramVendor:  true,
					boardInfo:   true,
				},
				{
					node:        2,
					renderMinor: 129,
					pciID:       "0000:c3:00.0",
					gfxVersion:  "120000",
					vramTotal:   2 << 30,
					gttTotal:    32 << 30,
				},
			},
			devices: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Name: "ROCm0", GFXTarget: "gfx1200", PCIID: "0000:c3:00.0"},
				{DeviceID: ml.DeviceID{ID: "1", Library: "ROCm"}, Name: "ROCm1", GFXTarget: "gfx1100", PCIID: "0000:e3:00.0"},
			},
			applied:        true,
			wantIntegrated: []bool{true, false},
			wantPCIIDs:     []string{"0000:c3:00.0", "0000:e3:00.0"},
		},
		{
			name: "remapped visible order matches unique gfx when pci is absent",
			nodes: []fakeROCmNode{
				{
					node:        1,
					renderMinor: 128,
					pciID:       "0000:e3:00.0",
					gfxVersion:  "110000",
					vramTotal:   48 << 30,
					gttTotal:    64 << 30,
					vramVendor:  true,
					boardInfo:   true,
				},
				{
					node:        2,
					renderMinor: 129,
					pciID:       "0000:c3:00.0",
					gfxVersion:  "120000",
					vramTotal:   2 << 30,
					gttTotal:    32 << 30,
				},
			},
			devices: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Name: "ROCm0", GFXTarget: "gfx1200"},
				{DeviceID: ml.DeviceID{ID: "1", Library: "ROCm"}, Name: "ROCm1", GFXTarget: "gfx1100"},
			},
			applied:        true,
			wantIntegrated: []bool{true, false},
			wantPCIIDs:     []string{"0000:c3:00.0", "0000:e3:00.0"},
		},
		{
			// the ROCm runtime leaves out a card it does not support: the reported one is paired by its PCI address
			name: "runtime reports fewer devices than kfd nodes",
			nodes: []fakeROCmNode{
				{
					node:        1,
					renderMinor: 128,
					pciID:       "0000:bb:00.0",
					gfxVersion:  "80003",
					vramTotal:   2 << 30,
					gttTotal:    32 << 30,
				},
				{
					node:        2,
					renderMinor: 129,
					pciID:       "0000:aa:00.0",
					gfxVersion:  "100301",
					vramTotal:   12 << 30,
					gttTotal:    16 << 30,
					vramVendor:  true,
					boardInfo:   true,
				},
			},
			devices: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Name: "ROCm0", GFXTarget: "gfx1030", PCIID: "0000:aa:00.0", Integrated: true},
			},
			applied:        true,
			wantIntegrated: []bool{false},
			wantPCIIDs:     []string{"0000:aa:00.0"},
		},
		{
			// with fewer reported devices there is no index or gfx fallback: without a PCI address nothing is paired
			name: "runtime reports fewer devices and no pci address",
			nodes: []fakeROCmNode{
				{node: 1, renderMinor: 128, pciID: "0000:bb:00.0", gfxVersion: "80003", vramTotal: 2 << 30, gttTotal: 32 << 30},
				{node: 2, renderMinor: 129, pciID: "0000:aa:00.0", gfxVersion: "100301", vramTotal: 12 << 30, gttTotal: 16 << 30, vramVendor: true, boardInfo: true},
			},
			devices: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Name: "ROCm0", GFXTarget: "gfx803", Integrated: true},
			},
			wantIntegrated: []bool{true},
			wantPCIIDs:     []string{""},
		},
		{
			name: "missing kfd data leaves devices unchanged",
			devices: []ml.DeviceInfo{{
				DeviceID:   ml.DeviceID{ID: "0", Library: "ROCm"},
				Name:       "ROCm0",
				Integrated: true,
			}},
			wantIntegrated: []bool{true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sysfsRoot := t.TempDir()
			for _, node := range tt.nodes {
				writeFakeROCmNode(t, sysfsRoot, node)
			}

			devices := append([]ml.DeviceInfo(nil), tt.devices...)
			applied := applyLinuxROCmRefinement(devices, sysfsRoot)
			if applied != tt.applied {
				t.Fatalf("applied = %v, want %v", applied, tt.applied)
			}
			for i, want := range tt.wantIntegrated {
				if devices[i].Integrated != want {
					t.Fatalf("device %d integrated = %v, want %v", i, devices[i].Integrated, want)
				}
			}
			for i, want := range tt.wantPCIIDs {
				if devices[i].PCIID != want {
					t.Fatalf("device %d PCIID = %q, want %q", i, devices[i].PCIID, want)
				}
			}
		})
	}
}

func TestSameRefreshDeviceMatchesROCmByPCI(t *testing.T) {
	updated := ml.DeviceInfo{
		DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"},
		PCIID:    "0000:c3:00.0",
	}
	existing := ml.DeviceInfo{
		DeviceID: ml.DeviceID{ID: "1", Library: "ROCm"},
		PCIID:    "0000:C3:00.0",
	}
	if !sameRefreshDevice(updated, existing) {
		t.Fatal("sameRefreshDevice did not match remapped ROCm device by PCI ID")
	}
}

func TestFilterUnsupportedROCmDevicesRespectsHSAOverride(t *testing.T) {
	t.Setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

	libDir := t.TempDir()
	rocblasDir := filepath.Join(libDir, "rocblas", "library")
	if err := os.MkdirAll(rocblasDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(rocblasDir, "TensileLibrary_lazy_gfx1030.dat"), nil, 0o644); err != nil {
		t.Fatal(err)
	}

	devices := filterUnsupportedROCmDevices([]ml.DeviceInfo{{
		DeviceID:     ml.DeviceID{ID: "0", Library: "ROCm"},
		Name:         "ROCm0",
		GFXTarget:    "gfx1031",
		ComputeMajor: 0x10,
		ComputeMinor: 0x31,
	}}, []string{libDir}, nil)
	if len(devices) != 1 {
		t.Fatalf("got %d devices, want 1", len(devices))
	}
	if got := devices[0].GFXTarget; got != "gfx1030" {
		t.Fatalf("GFXTarget = %q, want gfx1030", got)
	}
	if got := devices[0].Compute(); got != "gfx1030" {
		t.Fatalf("Compute() = %q, want gfx1030", got)
	}
}

type fakeROCmNode struct {
	node        int
	renderMinor int
	pciID       string
	gfxVersion  string
	vramTotal   uint64
	gttTotal    uint64
	vramVendor  bool
	boardInfo   bool
}

func writeFakeROCmNode(t *testing.T, sysfsRoot string, node fakeROCmNode) {
	t.Helper()

	nodeDir := filepath.Join(sysfsRoot, "class", "kfd", "kfd", "topology", "nodes", strconv.Itoa(node.node))
	if err := os.MkdirAll(nodeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	properties := "vendor_id 4098\n" +
		"device_id 1234\n" +
		"drm_render_minor " + strconv.Itoa(node.renderMinor) + "\n" +
		"gfx_target_version " + node.gfxVersion + "\n"
	if err := os.WriteFile(filepath.Join(nodeDir, "properties"), []byte(properties), 0o644); err != nil {
		t.Fatal(err)
	}

	deviceDir := filepath.Join(sysfsRoot, "class", "drm", "renderD"+strconv.Itoa(node.renderMinor), "device")
	if node.pciID != "" {
		targetDir := filepath.Join(sysfsRoot, "devices", node.pciID)
		if err := os.MkdirAll(targetDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Dir(deviceDir), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(targetDir, deviceDir); err != nil {
			t.Skipf("symlink unavailable for fake sysfs PCI path: %v", err)
		}
		deviceDir = targetDir
	} else if err := os.MkdirAll(deviceDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFakeSysfsFile(t, deviceDir, "vendor", "0x1002\n")
	writeFakeSysfsFile(t, deviceDir, "driver", "amdgpu\n")
	writeFakeSysfsFile(t, deviceDir, "mem_info_vram_total", strconv.FormatUint(node.vramTotal, 10)+"\n")
	writeFakeSysfsFile(t, deviceDir, "mem_info_gtt_total", strconv.FormatUint(node.gttTotal, 10)+"\n")
	if node.vramVendor {
		writeFakeSysfsFile(t, deviceDir, "mem_info_vram_vendor", "samsung\n")
	}
	if node.boardInfo {
		writeFakeSysfsFile(t, deviceDir, "board_info", "type : cem\n")
	}
}

func writeFakeSysfsFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// the gfx support check takes the override in effect for the card: its per-node HSA_OVERRIDE_GFX_VERSION_<KFD node>
// (from the discovery environment), else the global one; another node's override does not apply to it
func TestFilterUnsupportedROCmDevicesPerNodeOverride(t *testing.T) {
	t.Setenv("HSA_OVERRIDE_GFX_VERSION", "")
	libDir := t.TempDir()
	rocblasDir := filepath.Join(libDir, "rocblas", "library")
	if err := os.MkdirAll(rocblasDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(rocblasDir, "TensileLibrary_lazy_gfx1030.dat"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	card := func() []ml.DeviceInfo {
		return []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Name: "ROCm0", GFXTarget: "gfx1031", PCIID: "0000:aa:00.0"}}
	}
	nodes := map[string]int{"0000:bb:00.0": 1, "0000:aa:00.0": 2}

	got := filterUnsupportedROCmDevicesWith(card(), []string{libDir}, map[string]string{"HSA_OVERRIDE_GFX_VERSION_2": "10.3.0"}, nodes)
	if len(got) != 1 || got[0].GFXTarget != "gfx1030" {
		t.Fatalf("per-node override: %+v", got)
	}
	if got := filterUnsupportedROCmDevicesWith(card(), []string{libDir}, map[string]string{"HSA_OVERRIDE_GFX_VERSION_1": "10.3.0"}, nodes); len(got) != 0 {
		t.Fatalf("another node's override: %+v", got)
	}
	if got := filterUnsupportedROCmDevicesWith(card(), []string{libDir}, map[string]string{"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}, nodes); len(got) != 1 {
		t.Fatalf("global override: %+v", got)
	}
	if got := filterUnsupportedROCmDevicesWith(card(), []string{libDir}, nil, nodes); len(got) != 0 {
		t.Fatalf("no override: %+v", got)
	}
}

func TestROCmNodeGFXTarget(t *testing.T) {
	env := map[string]string{"HSA_OVERRIDE_GFX_VERSION_2": "10.3.0", "HSA_OVERRIDE_GFX_VERSION": "11.0.0"}
	getenv := func(k string) string { return env[k] }
	if got := rocmNodeGFXTarget(getenv, 2); got != "gfx1030" {
		t.Fatalf("node 2: %q", got)
	}
	if got := rocmNodeGFXTarget(getenv, 1); got != "gfx1100" {
		t.Fatalf("node 1 falls back to the global one: %q", got)
	}
	if got := rocmNodeGFXTarget(getenv, -1); got != "gfx1100" {
		t.Fatalf("unknown node: %q", got)
	}
}
