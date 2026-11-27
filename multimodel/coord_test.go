package multimodel

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"
)

// simulated devices and requests: max_running 1 everywhere unless a test says otherwise

func limits(m map[string]int) func(string) int {
	return func(d string) int { return m[d] }
}

func one(devs ...string) func(string) int {
	m := map[string]int{}
	for _, d := range devs {
		m[d] = 1
	}
	return limits(m)
}

func mustAcquire(t *testing.T, c *Coordinator, r Request) *Grant {
	t.Helper()
	g, err := c.Acquire(context.Background(), r)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	return g
}

type result struct {
	g   *Grant
	err error
}

func acquireAsync(c *Coordinator, ctx context.Context, r Request) chan result {
	ch := make(chan result, 1)
	go func() {
		g, err := c.Acquire(ctx, r)
		ch <- result{g, err}
	}()
	return ch
}

func waitQueued(t *testing.T, c *Coordinator, n int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for c.Queued() != n {
		if time.Now().After(deadline) {
			t.Fatalf("queued %d, want %d", c.Queued(), n)
		}
		time.Sleep(time.Millisecond)
	}
}

func got(t *testing.T, ch chan result, want string) *Grant {
	t.Helper()
	select {
	case r := <-ch:
		if r.err != nil || r.g.Device != want {
			t.Fatalf("got %+v %v, want %s", r.g, r.err, want)
		}
		return r.g
	case <-time.After(2 * time.Second):
		t.Fatalf("no place, want %s", want)
	}
	return nil
}

func still(t *testing.T, ch chan result) {
	t.Helper()
	select {
	case r := <-ch:
		t.Fatalf("must still wait, got %+v %v", r.g, r.err)
	case <-time.After(30 * time.Millisecond):
	}
}

var gpuCPU = []string{"gpu0", "cpu"}

func req(key string, on string, wait time.Duration, devs ...string) Request {
	return Request{Model: key, Key: key, Devices: devs, OnBusy: on, WaitTimeout: wait}
}

const forever = time.Duration(math.MaxInt64)

func TestCoordFreeAndRelease(t *testing.T) {
	c := NewCoordinator(one("gpu0", "cpu"))
	g := mustAcquire(t, c, req("a", OnBusyWait, time.Minute, gpuCPU...))
	if g.Device != "gpu0" || g.Reason != "free" || g.Tier != 0 || c.Running("gpu0") != 1 || !c.Active("a") {
		t.Fatalf("grant %+v", g)
	}
	g.Release()
	g.Release()
	if c.Running("gpu0") != 0 || c.Active("a") {
		t.Fatalf("after release: %d", c.Running("gpu0"))
	}
	// no limit: always free
	u := NewCoordinator(limits(nil))
	g1, g2 := mustAcquire(t, u, req("a", OnBusyWait, 0, "gpu0")), mustAcquire(t, u, req("b", OnBusyWait, 0, "gpu0"))
	if g1.Device != "gpu0" || g2.Device != "gpu0" {
		t.Fatal("no limit")
	}
}

// an idle loaded model does not make a device busy: the coordinator only counts active requests
func TestCoordIdleModelNotBusy(t *testing.T) {
	c := NewCoordinator(one("gpu0", "cpu"))
	mustAcquire(t, c, req("a", OnBusyWait, time.Minute, gpuCPU...)).Release() // a stays loaded, idle
	if g := mustAcquire(t, c, req("b", OnBusyWait, time.Minute, gpuCPU...)); g.Device != "gpu0" || g.Reason != "free" {
		t.Fatalf("b: %+v", g)
	}
}

// a split model makes the GPU and the CPU busy while it works
func TestCoordSplitModelBusy(t *testing.T) {
	c := NewCoordinator(one("gpu0", "cpu"))
	split := req("a", OnBusyWait, time.Minute, gpuCPU...)
	split.Extra = func(d string) []string {
		if d == "gpu0" {
			return []string{"cpu"}
		}
		return nil
	}
	g := mustAcquire(t, c, split)
	if g.Device != "gpu0" || c.Running("cpu") != 1 || c.Running("gpu0") != 1 {
		t.Fatalf("split: %+v cpu %d", g, c.Running("cpu"))
	}
	b := acquireAsync(c, context.Background(), req("b", OnBusyNext, 0, "cpu"))
	still(t, b)
	g.Release()
	got(t, b, "cpu").Release()

	// found split only at the load: Also takes the CPU, waiting for the capacity
	g = mustAcquire(t, c, req("a", OnBusyWait, time.Minute, gpuCPU...))
	busy := mustAcquire(t, c, req("x", OnBusyWait, time.Minute, "cpu"))
	done := make(chan error, 1)
	go func() { done <- g.Also(context.Background(), "cpu") }()
	select {
	case <-done:
		t.Fatal("the CPU is busy")
	case <-time.After(30 * time.Millisecond):
	}
	busy.Release()
	if err := <-done; err != nil || c.Running("cpu") != 1 {
		t.Fatalf("also: %v %d", err, c.Running("cpu"))
	}
	g.Release()
	if c.Running("cpu") != 0 || c.Running("gpu0") != 0 {
		t.Fatal("release of the split places")
	}
}

// one model, one active request: the later requests of a working model wait, whatever their on_busy
func TestCoordOneModelOneRequest(t *testing.T) {
	c := NewCoordinator(limits(map[string]int{"gpu0": 2, "cpu": 2}))
	g := mustAcquire(t, c, req("a", OnBusyWait, time.Minute, gpuCPU...))
	second := acquireAsync(c, context.Background(), req("a", OnBusyNext, 0, gpuCPU...))
	still(t, second)
	if !c.HasWaiter("a") {
		t.Fatal("waiter of a")
	}
	// another model still runs next to it
	mustAcquire(t, c, req("b", OnBusyWait, time.Minute, gpuCPU...)).Release()
	g.Release()
	got(t, second, "gpu0").Release()
}

// per device arrival order: a later request does not take a device an earlier one waits for; a request that
// touches none of its devices goes ahead
func TestCoordArrivalOrderPerDevice(t *testing.T) {
	c := NewCoordinator(one("gpu0", "gpu1", "cpu"))
	hold := mustAcquire(t, c, req("h", OnBusyWait, time.Minute, "gpu0"))
	x := acquireAsync(c, context.Background(), req("x", OnBusyWait, forever, "gpu0", "cpu"))
	waitQueued(t, c, 1)
	y := acquireAsync(c, context.Background(), req("y", OnBusyNext, 0, "gpu0"))
	waitQueued(t, c, 2)
	// z uses only gpu1: not touched by the waiting ones, it starts at once
	if g := mustAcquire(t, c, req("z", OnBusyWait, time.Minute, "gpu1")); g.Device != "gpu1" {
		t.Fatalf("z: %+v", g)
	} else {
		g.Release()
	}
	// w wants the cpu: x accepts only gpu0 (wait_timeout -1), so the cpu is free for w
	if g := mustAcquire(t, c, req("w", OnBusyNext, 0, "cpu")); g.Device != "cpu" {
		t.Fatalf("w: %+v", g)
	} else {
		g.Release()
	}
	hold.Release()
	gx := got(t, x, "gpu0")
	still(t, y)
	gx.Release()
	got(t, y, "gpu0").Release()
}

// a waiting request that accepts the cpu (on_busy next) keeps it from a later one
func TestCoordWaiterKeepsAcceptedDevices(t *testing.T) {
	c := NewCoordinator(one("gpu0", "cpu"))
	g0 := mustAcquire(t, c, req("h0", OnBusyWait, time.Minute, "gpu0"))
	gc := mustAcquire(t, c, req("hc", OnBusyWait, time.Minute, "cpu"))
	x := acquireAsync(c, context.Background(), req("x", OnBusyNext, 0, gpuCPU...))
	waitQueued(t, c, 1)
	y := acquireAsync(c, context.Background(), req("y", OnBusyNext, 0, "cpu"))
	waitQueued(t, c, 2)
	gc.Release()
	got(t, x, "cpu").Release()
	got(t, y, "cpu").Release()
	g0.Release()
}

// on_busy wait: tier by tier after wait_timeout, the accepted set grows; a freed higher tier is taken first
func TestCoordWaitTimeoutTiers(t *testing.T) {
	c := NewCoordinator(one("gpu0", "gpu1", "cpu"))
	h0 := mustAcquire(t, c, req("h0", OnBusyWait, time.Minute, "gpu0"))
	h1 := mustAcquire(t, c, req("h1", OnBusyWait, time.Minute, "gpu1"))
	hc := mustAcquire(t, c, req("hc", OnBusyWait, time.Minute, "cpu"))
	start := time.Now()
	ch := acquireAsync(c, context.Background(), req("a", OnBusyWait, 40*time.Millisecond, "gpu0", "gpu1", "cpu"))
	// after one timeout gpu1 is accepted too, after two the cpu; all busy: it waits
	time.Sleep(120 * time.Millisecond)
	still(t, ch)
	// the cpu frees first: accepted by now, it is taken
	hc.Release()
	g := got(t, ch, "cpu")
	if g.Reason != "next" || g.Tier != 2 || time.Since(start) < 80*time.Millisecond {
		t.Fatalf("cpu after two timeouts: %+v", g)
	}
	g.Release()

	// before the second timeout only gpu0 and gpu1 are accepted: a free cpu is not taken
	ch = acquireAsync(c, context.Background(), req("b", OnBusyWait, 60*time.Millisecond, "gpu0", "gpu1", "cpu"))
	time.Sleep(80 * time.Millisecond) // one timeout: gpu0, gpu1
	still(t, ch)
	// two tiers free at once: the higher one wins
	c.mu.Lock()
	c.running["gpu0"]--
	c.running["gpu1"]--
	c.dispatchLocked()
	c.mu.Unlock()
	if g := got(t, ch, "gpu0"); g.Tier != 0 {
		t.Fatalf("highest free: %+v", g)
	}
	_ = h0
	_ = h1
}

// wait_timeout -1: the request never steps down
func TestCoordWaitForever(t *testing.T) {
	c := NewCoordinator(one("gpu0", "cpu"))
	h := mustAcquire(t, c, req("h", OnBusyWait, time.Minute, "gpu0"))
	ch := acquireAsync(c, context.Background(), req("a", OnBusyWait, forever, gpuCPU...))
	still(t, ch)
	h.Release()
	if g := got(t, ch, "gpu0"); g.Reason != "waited" {
		t.Fatalf("forever: %+v", g)
	}
}

// on_busy next: the next tier at once, without waiting
func TestCoordNext(t *testing.T) {
	c := NewCoordinator(one("gpu0", "gpu1", "cpu"))
	mustAcquire(t, c, req("h", OnBusyWait, time.Minute, "gpu0"))
	if g := mustAcquire(t, c, req("a", OnBusyNext, time.Hour, "gpu0", "gpu1", "cpu")); g.Device != "gpu1" || g.Reason != "next" || g.Waited > 20*time.Millisecond {
		t.Fatalf("next: %+v", g)
	}
	// every tier busy: at the end of the list it waits until one frees up
	ch := acquireAsync(c, context.Background(), req("b", OnBusyNext, 0, "gpu0", "gpu1"))
	still(t, ch)
}

// a cold start (nothing loaded yet) is placed by the same rule
func TestCoordColdStart(t *testing.T) {
	c := NewCoordinator(one("gpu0", "cpu"))
	if g := mustAcquire(t, c, req("new", OnBusyWait, time.Minute, gpuCPU...)); g.Device != "gpu0" || g.Reason != "free" {
		t.Fatalf("cold start: %+v", g)
	}
}

// demote: the waiting requests of a moved model accept its new tier
func TestCoordDemote(t *testing.T) {
	c := NewCoordinator(one("gpu0", "cpu"))
	h := mustAcquire(t, c, req("b", OnBusyWait, time.Minute, "gpu0"))
	ch := acquireAsync(c, context.Background(), req("v", OnBusyWait, forever, gpuCPU...))
	still(t, ch)
	c.Demote("v", "cpu")
	if g := got(t, ch, "cpu"); g.Tier != 1 {
		t.Fatalf("demoted: %+v", g)
	}
	h.Release()
}

// promote: a request of a model that ran on a lower tier goes to the highest free tier (not one step up)
func TestCoordPromoteHighestFree(t *testing.T) {
	c := NewCoordinator(one("gpu0", "gpu1", "cpu"))
	h := mustAcquire(t, c, req("h", OnBusyWait, time.Minute, "gpu0"))
	g := mustAcquire(t, c, req("a", OnBusyNext, 0, "gpu0", "gpu1", "cpu"))
	if g.Device != "gpu1" {
		t.Fatalf("a: %+v", g)
	}
	g.Release()
	h.Release()
	if g := mustAcquire(t, c, req("a", OnBusyNext, 0, "gpu0", "gpu1", "cpu")); g.Device != "gpu0" {
		t.Fatalf("promote to the highest free tier: %+v", g)
	}
}

func TestCoordCancelAndQueueLimit(t *testing.T) {
	c := NewCoordinator(one("gpu0"))
	c.SetMaxQueue(1)
	h := mustAcquire(t, c, req("h", OnBusyWait, time.Minute, "gpu0"))
	ctx, cancel := context.WithCancel(context.Background())
	ch := acquireAsync(c, ctx, req("a", OnBusyWait, forever, "gpu0"))
	waitQueued(t, c, 1)
	if _, err := c.Acquire(context.Background(), req("b", OnBusyWait, forever, "gpu0")); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("queue limit: %v", err)
	}
	cancel()
	if r := <-ch; !errors.Is(r.err, context.Canceled) {
		t.Fatalf("cancel: %+v", r)
	}
	waitQueued(t, c, 0)
	h.Release()
	if c.Running("gpu0") != 0 {
		t.Fatal("a cancelled waiter holds nothing")
	}
}

func TestCoordReleaseOnDone(t *testing.T) {
	c := NewCoordinator(one("gpu0"))
	ctx, cancel := context.WithCancel(context.Background())
	g := mustAcquire(t, c, req("a", OnBusyWait, time.Minute, "gpu0"))
	g.ReleaseOnDone(ctx)
	cancel()
	deadline := time.Now().Add(time.Second)
	for c.Running("gpu0") != 0 {
		if time.Now().After(deadline) {
			t.Fatal("not released")
		}
		time.Sleep(time.Millisecond)
	}
}

// the CPU keeps its strict order: a waiting request that claims it (a split model's CPU, or the CPU accepted after
// its wait_timeout) keeps it from every later request, a request on another free device starts at once
func TestCoordCPUStrictOrder(t *testing.T) {
	c := NewCoordinator(one("gpu0", "gpu1", "cpu"))
	g0 := mustAcquire(t, c, req("h0", OnBusyWait, time.Minute, "gpu0"))
	gc := mustAcquire(t, c, req("hc", OnBusyWait, time.Minute, "cpu"))
	// a split model on gpu0 (it also computes on the CPU): it waits for gpu0 and claims the cpu
	split := req("split", OnBusyWait, forever, "gpu0")
	split.Extra = func(d string) []string {
		if d == "gpu0" {
			return []string{"cpu"}
		}
		return nil
	}
	s := acquireAsync(c, context.Background(), split)
	waitQueued(t, c, 1)
	// the CPU accepted after one wait_timeout
	x := acquireAsync(c, context.Background(), req("x", OnBusyWait, 20*time.Millisecond, "gpu0", "cpu"))
	waitQueued(t, c, 2)
	time.Sleep(50 * time.Millisecond)
	y := acquireAsync(c, context.Background(), req("y", OnBusyNext, 0, "cpu"))
	waitQueued(t, c, 3)
	// gpu1 is claimed by none of them
	if g := mustAcquire(t, c, req("z", OnBusyWait, time.Minute, "gpu1")); g.Device != "gpu1" {
		t.Fatalf("z: %+v", g)
	} else {
		g.Release()
	}
	// the cpu frees: the split model still waits for gpu0, and it claims the cpu first: neither x nor y takes it
	gc.Release()
	still(t, x)
	still(t, y)
	g0.Release()
	gs := got(t, s, "gpu0")
	if len(gs.held) != 2 {
		t.Fatalf("the split model holds gpu0 and the cpu: %v", gs.held)
	}
	gs.Release()
	// in order: x (gpu0 first, it is free), then y on the cpu
	got(t, x, "gpu0").Release()
	got(t, y, "cpu").Release()
}

// a waiting request starts on the event itself: the release of a place, the expiry of its wait_timeout (no polling)
func TestCoordStartsOnEvents(t *testing.T) {
	c := NewCoordinator(one("gpu0", "cpu"))
	var released [][]string
	c.SetOnRelease(func(d []string) { released = append(released, d) })
	g0 := mustAcquire(t, c, req("h0", OnBusyWait, time.Minute, "gpu0"))
	ch := acquireAsync(c, context.Background(), req("a", OnBusyWait, forever, "gpu0"))
	waitQueued(t, c, 1)
	at := time.Now()
	g0.Release()
	ga := got(t, ch, "gpu0")
	if d := time.Since(at); d > 50*time.Millisecond {
		t.Fatalf("started %v after the release", d)
	}
	ga.Release()
	if len(released) != 2 || released[0][0] != "gpu0" {
		t.Fatalf("release callback: %v", released)
	}

	// the timeout: the next tier is accepted and checked at once
	g0 = mustAcquire(t, c, req("h0", OnBusyWait, time.Minute, "gpu0"))
	start := time.Now()
	ch = acquireAsync(c, context.Background(), req("b", OnBusyWait, 60*time.Millisecond, "gpu0", "cpu"))
	gb := got(t, ch, "cpu")
	if d := time.Since(start); d < 60*time.Millisecond || d > 150*time.Millisecond {
		t.Fatalf("started %v after the request, want right after its 60ms wait_timeout", d)
	}
	gb.Release()
	g0.Release()
}
