package server

// Dual-buffer fork: a load that waits for room on its device does not hold the pending loop
// (docs/MULTIMODEL-DESIGN.md 2.6). It steps aside, keeps its turn on the devices it would use, and goes back to the
// pending queue on an event that can make room there: a request ends (a model becomes idle), a model moves (park,
// demote) or unloads. There is no polling.

import (
	"context"
	"log/slog"
	"slices"
	"sync"

	"github.com/ollama/ollama/multimodel"
)

type dualbufWaiting struct {
	mu   sync.Mutex
	gen  uint64           // events so far
	list []*dualbufWaiter // arrival order
}

type dualbufWaiter struct {
	req      *LlmRequest
	claims   []string // the devices the load keeps its turn on: its placement, and the CPU of a split model
	any      bool     // room from any device (OLLAMA_MAX_LOADED_MODELS): every event sends it back
	inflight bool     // sent back to the pending queue, not processed yet
}

// dualbufInit: a request that gives its places back is an event too (its model becomes idle).
func (s *Scheduler) dualbufInit() {
	if s.dualbuf != nil {
		s.dualbuf.Coord.SetOnRelease(func(devices []string) { s.dualbufWake(devices...) })
	}
}

func (s *Scheduler) dualbufEvents() uint64 {
	w := &s.dualbufWaiting
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.gen
}

// dualbufWake is an event on the devices (none: all of them): the loads waiting there go back to the pending queue,
// in their order.
func (s *Scheduler) dualbufWake(devices ...string) {
	w := &s.dualbufWaiting
	w.mu.Lock()
	w.gen++
	var send []*LlmRequest
	for _, x := range w.list {
		if x.inflight {
			continue
		}
		if len(devices) == 0 || x.any || slices.ContainsFunc(x.claims, func(d string) bool { return slices.Contains(devices, d) }) {
			x.inflight = true
			send = append(send, x.req)
		}
	}
	w.mu.Unlock()
	if len(send) > 0 {
		go func() {
			for _, r := range send {
				s.pendingReqCh <- r
			}
		}()
	}
}

// dualbufDone: the request is served (or ended); it no longer keeps a turn.
func (s *Scheduler) dualbufDone(req *LlmRequest) {
	w := &s.dualbufWaiting
	w.mu.Lock()
	n := len(w.list)
	w.list = slices.DeleteFunc(w.list, func(x *dualbufWaiter) bool { return x.req == req })
	removed := len(w.list) != n
	w.mu.Unlock()
	if removed {
		// the later loads on its devices may go now
		s.dualbufWake(s.dualbufClaims(req)...)
	}
}

// dualbufClaims are the devices a placed load keeps its turn on.
func (s *Scheduler) dualbufClaims(req *LlmRequest) []string {
	db := req.model.dualbuf
	if db == nil || db.placement == "" {
		return nil
	}
	out := []string{db.placement}
	if db.placement != multimodel.DeviceCPU && s.dualbuf != nil && s.dualbuf.Split(schedulerModelKeyBase(req.model), db.placement) {
		if _, ok := db.cfg.Devices[multimodel.DeviceCPU]; ok {
			out = append(out, multimodel.DeviceCPU)
		}
	}
	return out
}

// dualbufRunnerDevices are the devices a runner holds room on (its placement, the CPU when it is split there).
func dualbufRunnerDevices(r *runnerRef) []string {
	m := r.model.Load()
	if m == nil || m.dualbuf == nil || m.dualbuf.placement == "" {
		return nil
	}
	if m.dualbuf.placement == multimodel.DeviceCPU {
		return []string{multimodel.DeviceCPU}
	}
	return []string{m.dualbuf.placement, multimodel.DeviceCPU}
}

// dualbufMustWait: the load keeps waiting aside, without acting: its victim is still unloading, or an earlier load waits
// for room on one of its devices (its turn comes first). A request served by its loaded runner never waits here.
func (s *Scheduler) dualbufMustWait(ctx context.Context, pending *LlmRequest, runner *runnerRef) bool {
	db := pending.model.dualbuf
	if db == nil || db.placement == "" {
		return false
	}
	if v := pending.dualbufWaitFor; v != nil {
		s.loadedMu.Lock()
		still := s.loaded[v.modelKey] == v
		s.loadedMu.Unlock()
		if still {
			pending.dualbufDeferred = true
			return true
		}
		pending.dualbufWaitFor = nil
	}
	if runner != nil && !runner.needsReload(ctx, pending) && !dualbufNeedsChange(runner, pending) {
		return false
	}
	claims := s.dualbufClaims(pending)
	w := &s.dualbufWaiting
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, x := range w.list {
		if x.req == pending {
			return false
		}
		if slices.ContainsFunc(x.claims, func(d string) bool { return slices.Contains(claims, d) }) {
			slog.Debug("dualbuf: an earlier load waits on the device, this one keeps its turn after it", "model", dualbufModelName(pending.model), "device", db.placement)
			pending.dualbufDeferred = true
			return true
		}
	}
	return false
}

// dualbufAwaitUnload: a placed load does not wait for its victim's unload in the pending loop; the unload is the event
// that sends it back.
func (s *Scheduler) dualbufAwaitUnload(pending *LlmRequest, victim *runnerRef) bool {
	if db := pending.model.dualbuf; db == nil || db.placement == "" {
		return false
	}
	pending.dualbufWaitFor = victim
	pending.dualbufDeferred = true
	return true
}

// dualbufAfterPending: a load that waits keeps (or takes) its turn; any other leaves the list. An event that came while
// it decided to wait sends it back at once.
func (s *Scheduler) dualbufAfterPending(pending *LlmRequest, events uint64) {
	if !pending.dualbufDeferred || pending.ctx.Err() != nil {
		s.dualbufDone(pending)
		return
	}
	claims := s.dualbufClaims(pending)
	w := &s.dualbufWaiting
	w.mu.Lock()
	var x *dualbufWaiter
	for _, y := range w.list {
		if y.req == pending {
			x = y
			break
		}
	}
	first := x == nil
	if first {
		x = &dualbufWaiter{req: pending}
		w.list = append(w.list, x)
	}
	x.claims = claims
	x.any = pending.dualbufAnyDevice
	x.inflight = w.gen != events
	resend := x.inflight
	w.mu.Unlock()
	if first {
		slog.Info("dualbuf: the load waits for room on its device, the other requests go on", "model", dualbufModelName(pending.model),
			"device", pending.model.dualbuf.placement, "devices", x.claims)
		go func() {
			<-pending.ctx.Done()
			s.dualbufDone(pending)
		}()
	}
	if resend {
		go func() { s.pendingReqCh <- pending }()
	}
}
