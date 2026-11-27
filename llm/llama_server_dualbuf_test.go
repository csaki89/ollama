package llm

import (
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestApplyArgOverrides(t *testing.T) {
	base := []string{"--model", "m", "-b", "512", "-ub", "512", "--flash-attn", "auto"}
	got := applyArgOverrides(slices.Clone(base), map[string][]string{"-ub": {"128"}, "--flash-attn": {"on"}, "-t": {"6"}})
	want := []string{"--model", "m", "-b", "512", "-ub", "128", "--flash-attn", "on", "-t", "6"}
	if !slices.Equal(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
	if got := applyArgOverrides(slices.Clone(base), nil); !slices.Equal(got, base) {
		t.Fatalf("nil overrides changed the args: %v", got)
	}
	// a flag at the end without its value is not overwritten past the end
	tail := []string{"--model", "m", "-ub"}
	if got := applyArgOverrides(slices.Clone(tail), map[string][]string{"-ub": {"64"}}); !slices.Equal(got, []string{"--model", "m", "-ub", "-ub", "64"}) {
		t.Fatalf("tail: %v", got)
	}
}

func TestOffloadsToCPU(t *testing.T) {
	s := &llamaServerRunner{}
	if s.OffloadsToCPU() {
		t.Fatal("a CPU-only runner does not offload to the CPU")
	}
	s.gpus = make([]ml.DeviceInfo, 1)
	s.totalLayers, s.gpuLayers = 29, 29
	if s.OffloadsToCPU() {
		t.Fatal("all layers on the GPU")
	}
	s.gpuLayers = 20
	if !s.OffloadsToCPU() {
		t.Fatal("20/29 layers")
	}
	s.gpuLayers, s.gpuLayerOverflow = 29, 3
	if !s.OffloadsToCPU() {
		t.Fatal("overflowing layers")
	}
}

func TestZonesStateLiveVRAM(t *testing.T) {
	s := &llamaServerRunner{vramByDevice: map[string]uint64{}, systemFreeAtLoad: map[string]uint64{}}
	s.gpus = []ml.DeviceInfo{{Name: "Vulkan0"}}
	w := &memoryParsingWriter{inner: io.Discard, runner: s}
	lines := "load_tensors:      Vulkan0 model buffer size =   934.69 MiB\n" +
		"llama_kv_cache:      Vulkan0 KV buffer size =   112.00 MiB\n" +
		"sched_reserve:      Vulkan0 compute buffer size =   300.00 MiB\n"
	if _, err := w.Write([]byte(lines)); err != nil {
		t.Fatal(err)
	}
	before := s.VRAMByGPU(s.gpus[0].DeviceID)
	mib := func(f float64) uint64 { return uint64(f * 1024 * 1024) }
	if before != mib(934.69)+mib(112)+mib(300) {
		t.Fatalf("load accounting %d", before)
	}
	// the KV grew to 224 MiB on the device (copied from the laptop log), an SWA part keeps 50 MiB
	z := "zones_log_state: zones_state (KV 4096 -> 8192 cells, 0 flex steps off, 0 back): weights GPU 935 MiB (88%), CPU 125 MiB; flex 0/0 layers on the GPU; KV 8192 cells (24 used, 224 MiB on the device); fixed on the device: SWA K/V 50 MiB; device free: booked -, driver 5899, sysfs - MiB (budget - MiB, margin 0 MiB)\n"
	if _, err := w.Write([]byte(z)); err != nil {
		t.Fatal(err)
	}
	want := uint64(935+224+50+300) << 20
	if got := s.VRAMByGPU(s.gpus[0].DeviceID); got != want {
		t.Fatalf("live %d want %d", got, want)
	}
	if _, vram := s.MemorySize(); vram != want {
		t.Fatalf("MemorySize vram %d want %d", vram, want)
	}
	// a later load line does not lose the live value
	if _, err := w.Write([]byte("sched_reserve:      Vulkan0 compute buffer size =   300.00 MiB\n")); err != nil {
		t.Fatal(err)
	}
	if got := s.VRAMByGPU(s.gpus[0].DeviceID); got != want {
		t.Fatalf("after another buffer line %d want %d", got, want)
	}
}

func TestHostUnchargedFromLog(t *testing.T) {
	s := &llamaServerRunner{}
	s.parseParkLocked([]byte("srv  server_run_i: host memory: 3120 MiB in backend host buffers, 3120 MiB not charged to the cgroup\n"))
	if got := s.HostUncharged(); got != 3120<<20 {
		t.Fatalf("got %d want %d", got, uint64(3120)<<20)
	}
	s.parseParkLocked([]byte("srv  server_run_i: host memory: 900 MiB in backend host buffers, 0 MiB not charged to the cgroup\n"))
	if got := s.HostUncharged(); got != 0 {
		t.Fatalf("a charged (ROCm) report must give 0, got %d", got)
	}
}

// a model that moves between GPUs in process: the other GPUs' backends load too, every card visible to its own backend
// only, the active device's backend first
func TestDualbufExtraDevices(t *testing.T) {
	rocmDir, vkDir := t.TempDir(), t.TempDir()
	for _, f := range []string{filepath.Join(rocmDir, "libggml-hip.so"), filepath.Join(vkDir, "libggml-vulkan.so")} {
		if err := os.WriteFile(f, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	active := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "1", Library: "ROCm"}, LibraryPath: []string{rocmDir}}
	other := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "2", Library: "Vulkan"}, LibraryPath: []string{vkDir}}
	libs0 := ml.LibraryPaths([]ml.DeviceInfo{active})
	env0 := ml.GetDevicesEnv([]ml.DeviceInfo{active})
	if libs, env := dualbufExtraDevices([]ml.DeviceInfo{active}, nil, libs0, env0); len(libs) != len(libs0) || len(env) != len(env0) {
		t.Fatal("without extra devices nothing changes")
	}
	libs, env := dualbufExtraDevices([]ml.DeviceInfo{active}, []ml.DeviceInfo{other}, libs0, env0)
	if !slices.Contains(libs, rocmDir) || !slices.Contains(libs, vkDir) {
		t.Fatalf("library paths %v", libs)
	}
	if env["GGML_VK_VISIBLE_DEVICES"] != "2" {
		t.Fatalf("vulkan sees only its card: %v", env)
	}
	rocmSeen := false
	for k, v := range env {
		if strings.Contains(k, "VISIBLE_DEVICES") && k != "GGML_VK_VISIBLE_DEVICES" && v == "1" {
			rocmSeen = true
		}
	}
	if !rocmSeen {
		t.Fatalf("rocm sees only its card: %v", env)
	}
	want := filepath.Join(rocmDir, "libggml-hip.so") + string(filepath.ListSeparator) + filepath.Join(vkDir, "libggml-vulkan.so")
	if env["GGML_BACKEND_PATH"] != want {
		t.Fatalf("backends %q, want the active one first %q", env["GGML_BACKEND_PATH"], want)
	}
}

func TestRenameDeviceBuffers(t *testing.T) {
	b := map[memoryBufferKey]memoryBuffer{
		{backend: "ROCm0", kind: "model"}:  {bytes: 10},
		{backend: "ROCm0", kind: "KV"}:     {bytes: 2},
		{backend: "CPU_REPACK", kind: "x"}: {bytes: 3},
	}
	renameDeviceBuffers(b, "ROCm0", "Vulkan0")
	if b[memoryBufferKey{backend: "Vulkan0", kind: "model"}].bytes != 10 || b[memoryBufferKey{backend: "Vulkan0", kind: "KV"}].bytes != 2 || len(b) != 3 {
		t.Fatalf("renamed: %v", b)
	}
}
