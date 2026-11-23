package llm

import (
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
