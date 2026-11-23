package llm

import (
	"io"
	"slices"
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
