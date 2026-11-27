package server

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/multimodel"
)

// the in-process tier changes (docs/MULTIMODEL-DESIGN.md 2.5): a move to another GPU of the list, a park for a CPU
// request, an unpark; simulated runners that park and move

type moveLlm struct {
	mockLlm
	canPark bool
	parked  bool
	moved   []ml.DeviceInfo
	failing bool
}

func (m *moveLlm) CanPark() bool      { return m.canPark }
func (m *moveLlm) ParkRefused() bool  { return false }
func (m *moveLlm) Parked() bool       { return m.parked }
func (m *moveLlm) ParkedVRAM() uint64 { return 1 << 30 }
func (m *moveLlm) Park(context.Context) (time.Duration, error) {
	m.parked = true
	return time.Millisecond, nil
}

func (m *moveLlm) Unpark(context.Context) (time.Duration, error) {
	m.parked = false
	return time.Millisecond, nil
}

func (m *moveLlm) Move(_ context.Context, to ml.DeviceInfo) (time.Duration, error) {
	if m.failing {
		return 0, errors.New("move failed")
	}
	m.moved = append(m.moved, to)
	m.parked = false
	return time.Millisecond, nil
}

var (
	testGPU0 = ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, PCIID: "0000:aa:00.0", Description: "Test GPU A", TotalMemory: 16 << 30, FreeMemory: 8 << 30}
	testGPU1 = ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "Vulkan"}, PCIID: "0000:bb:00.0", Description: "Test GPU B", TotalMemory: 8 << 30, FreeMemory: 6 << 30}
)

func moveSched(t *testing.T) (*Scheduler, *multimodel.Config) {
	t.Helper()
	s, cfg := schedTest(t, "false")
	s.dualbuf.NoteDevices([]ml.DeviceInfo{testGPU0, testGPU1})
	s.getGpuFn = func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		return []ml.DeviceInfo{testGPU0, testGPU1}
	}
	return s, cfg
}

// a runner of big:latest ([gpu0, gpu1, cpu]) loaded on dev, moving in process
func movableRunner(s *Scheduler, cfg *multimodel.Config, dev string, llama *moveLlm) *runnerRef {
	r := mkRunner(s, cfg, "big:latest", dev, 1, 0, time.Minute)
	r.model.Load().dualbuf.tiers = []string{"gpu0", "gpu1", "cpu"}
	r.llama = llama
	opts := api.DefaultOptions()
	r.Options = &opts
	r.contextShift = true
	if dev == "gpu0" {
		r.gpus = []ml.DeviceID{testGPU0.DeviceID}
	} else {
		r.gpus = []ml.DeviceID{testGPU1.DeviceID}
	}
	return r
}

func requestOn(cfg *multimodel.Config, dev string) *LlmRequest {
	req := pendingFor(cfg, "big:latest", dev)
	req.model.dualbuf.tiers = []string{"gpu0", "gpu1", "cpu"}
	req.opts = api.DefaultOptions()
	return req
}

func TestDualbufMoveToAnotherGPU(t *testing.T) {
	s, cfg := moveSched(t)
	llama := &moveLlm{canPark: true}
	r := movableRunner(s, cfg, "gpu0", llama)
	req := requestOn(cfg, "gpu1")
	if dualbufFingerprint(req.model) != dualbufFingerprint(r.model.Load()) || r.needsReload(context.Background(), req) {
		t.Fatal("a placement change on the list is no reload")
	}
	if !dualbufNeedsMove(r, req) || !dualbufNeedsChange(r, req) || dualbufNeedsMove(r, requestOn(cfg, "gpu0")) {
		t.Fatal("a request on the other GPU moves the runner, one on its own GPU does not")
	}
	if v, wait := s.dualbufUnparkFor(context.Background(), req, r, false); v != nil || wait {
		t.Fatalf("move: %v %v", v, wait)
	}
	if len(llama.moved) != 1 || llama.moved[0].PCIID != testGPU1.PCIID {
		t.Fatalf("moved to %+v", llama.moved)
	}
	if r.model.Load().dualbuf.placement != "gpu1" || len(r.gpus) != 1 || r.gpus[0] != testGPU1.DeviceID {
		t.Fatalf("the runner follows: %s %v", r.model.Load().dualbuf.placement, r.gpus)
	}
	if dualbufNeedsChange(r, requestOn(cfg, "gpu1")) {
		t.Fatal("on gpu1 now: no further change")
	}

	// parked (on the CPU), a request on the other GPU: the move brings it there (no unpark to the old device first)
	llama.parked = true
	req = requestOn(cfg, "gpu0")
	if v, wait := s.dualbufUnparkFor(context.Background(), req, r, false); v != nil || wait || llama.parked || len(llama.moved) != 2 {
		t.Fatalf("parked move: %v %v %v", v, wait, llama.moved)
	}
}

func TestDualbufMoveNotPossible(t *testing.T) {
	s, cfg := moveSched(t)
	// no park copies: the move is a reload (the runner itself is expired)
	r := movableRunner(s, cfg, "gpu0", &moveLlm{canPark: false})
	if v, wait := s.dualbufUnparkFor(context.Background(), requestOn(cfg, "gpu1"), r, false); v != r || !wait {
		t.Fatalf("no copies: %v %v", v, wait)
	}
	// the move fails: a reload too
	r2 := movableRunner(s, cfg, "gpu0", &moveLlm{canPark: true, failing: true})
	if v, wait := s.dualbufUnparkFor(context.Background(), requestOn(cfg, "gpu1"), r2, false); v != r2 || !wait {
		t.Fatalf("failed move: %v %v", v, wait)
	}
	if r2.model.Load().dualbuf.placement != "gpu0" {
		t.Fatal("a failed move keeps the placement")
	}
}

// the target GPU is held by an active model only: the move waits aside; an idle one there is the victim
func TestDualbufMoveMakesRoom(t *testing.T) {
	s, cfg := moveSched(t)
	r := movableRunner(s, cfg, "gpu0", &moveLlm{canPark: true})
	busy := mkRunner(s, cfg, "busy", "gpu1", 1, 1, time.Minute)
	req := requestOn(cfg, "gpu1")
	if v, wait := s.dualbufUnparkFor(context.Background(), req, r, true); v != nil || !wait || !req.dualbufDeferred {
		t.Fatalf("only active on the target: %v %v %v", v, wait, req.dualbufDeferred)
	}
	busy.refCount = 0
	req = requestOn(cfg, "gpu1")
	if v, wait := s.dualbufUnparkFor(context.Background(), req, r, true); v != busy || !wait {
		t.Fatalf("idle on the target: %v %v", v, wait)
	}
}

// a CPU request while the model's instance sits on its GPU: it parks there and serves the request on the CPU
func TestDualbufParkForCPURequest(t *testing.T) {
	s, cfg := moveSched(t)
	llama := &moveLlm{canPark: true}
	r := movableRunner(s, cfg, "gpu0", llama)
	if s.dualbufMovableFor(r.model.Load()) != r {
		t.Fatal("the loaded instance can serve a CPU request")
	}
	req := requestOn(cfg, "gpu0")
	req.model.dualbuf.viaParked = true
	if !dualbufNeedsPark(r, req) || r.needsReload(context.Background(), req) {
		t.Fatal("the CPU request parks the instance, no reload")
	}
	if v, wait := s.dualbufUnparkFor(context.Background(), req, r, false); v != nil || wait || !llama.parked {
		t.Fatalf("park for the cpu: %v %v %v", v, wait, llama.parked)
	}
	if dualbufNeedsChange(r, req) {
		t.Fatal("parked: the CPU request runs on it")
	}
	// a request on its GPU unparks it
	if !dualbufNeedsUnpark(r, requestOn(cfg, "gpu0")) {
		t.Fatal("a GPU request unparks")
	}
}
