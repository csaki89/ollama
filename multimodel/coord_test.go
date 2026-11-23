package multimodel

import (
	"context"
	"testing"
	"time"
)

func limits(m map[string]int) func(string) int {
	return func(d string) int { return m[d] }
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

var gpuCPU = []string{"x080", "cpu"}

func TestCoordFreeAndNoLimit(t *testing.T) {
	c := NewCoordinator(limits(map[string]int{"x080": 1}))
	a := mustAcquire(t, c, Request{Devices: gpuCPU})
	if a.Device != "x080" || a.Reason != "free" {
		t.Fatalf("a: %+v", a)
	}
	// cpu has no limit: any number may run there
	b := mustAcquire(t, c, Request{Devices: []string{"cpu"}})
	d := mustAcquire(t, c, Request{Devices: []string{"cpu"}})
	if b.Device != "cpu" || d.Device != "cpu" || c.Running("cpu") != 2 {
		t.Fatalf("cpu: %+v %+v", b, d)
	}
	a.Release()
	a.Release() // once
	if c.Running("x080") != 0 {
		t.Fatalf("running after release: %d", c.Running("x080"))
	}
	if g := mustAcquire(t, c, Request{}); g.Device != "" {
		t.Fatalf("no devices: %+v", g)
	}
}

func TestCoordWaitThenGet(t *testing.T) {
	c := NewCoordinator(limits(map[string]int{"x080": 1, "cpu": 1}))
	a := mustAcquire(t, c, Request{Devices: gpuCPU})
	ch := acquireAsync(c, context.Background(), Request{Devices: gpuCPU, OnBusy: OnBusyWait, WaitTimeout: 2 * time.Second})
	waitQueued(t, c, 1)
	a.Release()
	r := <-ch
	if r.err != nil || r.g.Device != "x080" || r.g.Reason != "waited" {
		t.Fatalf("waited: %+v %v", r.g, r.err)
	}
}

func TestCoordWaitTimeoutNext(t *testing.T) {
	c := NewCoordinator(limits(map[string]int{"x080": 1, "cpu": 1}))
	mustAcquire(t, c, Request{Devices: gpuCPU})
	start := time.Now()
	g := mustAcquire(t, c, Request{Devices: gpuCPU, OnBusy: OnBusyWait, WaitTimeout: 50 * time.Millisecond})
	if g.Device != "cpu" || g.Reason != "next" || time.Since(start) < 50*time.Millisecond {
		t.Fatalf("timeout -> cpu: %+v after %v", g, time.Since(start))
	}
}

func TestCoordNextNoWait(t *testing.T) {
	c := NewCoordinator(limits(map[string]int{"x080": 1, "cpu": 1}))
	mustAcquire(t, c, Request{Devices: gpuCPU})
	start := time.Now()
	g := mustAcquire(t, c, Request{Devices: gpuCPU, OnBusy: OnBusyNext, WaitTimeout: time.Hour})
	if g.Device != "cpu" || g.Reason != "next" || time.Since(start) > 500*time.Millisecond {
		t.Fatalf("next: %+v after %v", g, time.Since(start))
	}
}

func TestCoordQueueArrivalOrder(t *testing.T) {
	c := NewCoordinator(limits(map[string]int{"x080": 1, "cpu": 1}))
	gpu := mustAcquire(t, c, Request{Devices: gpuCPU})
	cpu := mustAcquire(t, c, Request{Devices: []string{"cpu"}})
	d := acquireAsync(c, context.Background(), Request{Model: "d", Devices: gpuCPU, OnBusy: OnBusyNext})
	waitQueued(t, c, 1)
	e := acquireAsync(c, context.Background(), Request{Model: "e", Devices: gpuCPU, OnBusy: OnBusyNext})
	waitQueued(t, c, 2)
	// a later request that only wants the cpu does not overtake d and e
	f := acquireAsync(c, context.Background(), Request{Model: "f", Devices: []string{"cpu"}})
	waitQueued(t, c, 3)

	cpu.Release() // first freed device: cpu -> d (the earliest)
	rd := <-d
	if rd.g.Device != "cpu" || rd.g.Reason != "queued" {
		t.Fatalf("d: %+v", rd.g)
	}
	gpu.Release() // x080 -> e
	re := <-e
	if re.g.Device != "x080" {
		t.Fatalf("e: %+v", re.g)
	}
	select {
	case r := <-f:
		t.Fatalf("f must still wait for the cpu: %+v", r.g)
	case <-time.After(20 * time.Millisecond):
	}
	rd.g.Release()
	if rf := <-f; rf.g.Device != "cpu" {
		t.Fatalf("f: %+v", rf.g)
	}
}

func TestCoordWaiterKeepsItsTurn(t *testing.T) {
	// a request waiting for its first device is not overtaken there by a later one
	c := NewCoordinator(limits(map[string]int{"x080": 1, "cpu": 1}))
	a := mustAcquire(t, c, Request{Devices: gpuCPU})
	b := acquireAsync(c, context.Background(), Request{Model: "b", Devices: gpuCPU, OnBusy: OnBusyWait, WaitTimeout: 2 * time.Second})
	waitQueued(t, c, 1)
	late := acquireAsync(c, context.Background(), Request{Model: "late", Devices: []string{"x080"}})
	waitQueued(t, c, 2)
	a.Release()
	if r := <-b; r.g.Device != "x080" {
		t.Fatalf("b: %+v", r.g)
	}
	select {
	case r := <-late:
		t.Fatalf("late overtook: %+v", r.g)
	case <-time.After(20 * time.Millisecond):
	}
}

func TestCoordCancel(t *testing.T) {
	c := NewCoordinator(limits(map[string]int{"x080": 1}))
	mustAcquire(t, c, Request{Devices: []string{"x080"}})
	ctx, cancel := context.WithCancel(context.Background())
	ch := acquireAsync(c, ctx, Request{Devices: []string{"x080"}})
	waitQueued(t, c, 1)
	cancel()
	if r := <-ch; r.err == nil {
		t.Fatalf("cancel: %+v", r.g)
	}
	if c.Queued() != 0 {
		t.Fatalf("queue after cancel: %d", c.Queued())
	}
}

func TestCoordAlsoCPU(t *testing.T) {
	c := NewCoordinator(limits(map[string]int{"x080": 1, "cpu": 1}))
	cpuOnly := mustAcquire(t, c, Request{Devices: []string{"cpu"}})
	g := mustAcquire(t, c, Request{Devices: gpuCPU})
	done := make(chan error, 1)
	go func() { done <- g.Also(context.Background(), "cpu") }()
	select {
	case <-done:
		t.Fatal("the split model must wait for the cpu place")
	case <-time.After(20 * time.Millisecond):
	}
	cpuOnly.Release()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if c.Running("cpu") != 1 || c.Running("x080") != 1 {
		t.Fatalf("running: cpu %d x080 %d", c.Running("cpu"), c.Running("x080"))
	}
	g.Release()
	if c.Running("cpu") != 0 || c.Running("x080") != 0 {
		t.Fatalf("after release: cpu %d x080 %d", c.Running("cpu"), c.Running("x080"))
	}
}

func TestCoordReleaseOnDone(t *testing.T) {
	c := NewCoordinator(limits(map[string]int{"x080": 1}))
	ctx, cancel := context.WithCancel(context.Background())
	g := mustAcquire(t, c, Request{Devices: []string{"x080"}})
	g.ReleaseOnDone(ctx)
	cancel()
	deadline := time.Now().Add(time.Second)
	for c.Running("x080") != 0 {
		if time.Now().After(deadline) {
			t.Fatal("not released")
		}
		time.Sleep(time.Millisecond)
	}
}
