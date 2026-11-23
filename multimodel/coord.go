package multimodel

import (
	"context"
	"log/slog"
	"slices"
	"sync"
	"time"
)

// Coordinator keeps the running limit per device (max_running): at most that many requests generate on a
// device at once. A request takes a place before it gets its runner and keeps it until it ends. The waiting
// requests are served in arrival order.
type Coordinator struct {
	mu      sync.Mutex
	running map[string]int
	waiters []*waiter
	seq     uint64
	limit   func(device string) int // 0 = no limit
	now     func() time.Time
}

type waiter struct {
	seq   uint64
	wants []string // in preference order
	got   string
	ready chan struct{}
}

// NewCoordinator returns a coordinator; limit gives max_running of a device (0: no limit).
func NewCoordinator(limit func(device string) int) *Coordinator {
	return &Coordinator{running: map[string]int{}, limit: limit, now: time.Now}
}

// Request is what a request asks for: its model's device list (only present devices) and the busy rule.
type Request struct {
	Model       string
	Devices     []string
	OnBusy      string
	WaitTimeout time.Duration
}

// Grant is a place on a device, held until Release.
type Grant struct {
	Device string
	Reason string        // free, waited, next, queued
	Waited time.Duration // time spent waiting for the place
	c      *Coordinator
	held   []string
	once   sync.Once
}

// Acquire takes a place for the request:
//   - the first device of the list that is free wins;
//   - if the first device is busy: on_busy wait waits for it up to WaitTimeout, on_busy next goes on at once;
//   - the other devices of the list are taken only when free, without waiting;
//   - when no device is free the request is queued and starts on the first device of its list that frees up,
//     in arrival order. Only the context ends the wait (no error, no overload).
func (c *Coordinator) Acquire(ctx context.Context, r Request) (*Grant, error) {
	start := c.now()
	if len(r.Devices) == 0 {
		return &Grant{Reason: "no devices"}, nil
	}
	c.mu.Lock()
	c.seq++
	seq := c.seq

	if c.takeLocked(r.Devices[0], seq) {
		c.mu.Unlock()
		return c.grant(r, r.Devices[0], "free", start), nil
	}

	if r.OnBusy == OnBusyWait && r.WaitTimeout > 0 {
		w := c.addLocked(seq, r.Devices[:1])
		c.mu.Unlock()
		t := time.NewTimer(r.WaitTimeout)
		select {
		case <-w.ready:
			t.Stop()
			return c.grant(r, w.got, "waited", start), nil
		case <-ctx.Done():
			t.Stop()
			if c.cancel(w) {
				return nil, ctx.Err()
			}
			return c.grant(r, w.got, "waited", start), nil
		case <-t.C:
			if !c.cancel(w) {
				return c.grant(r, w.got, "waited", start), nil
			}
		}
		c.mu.Lock()
	}

	for _, d := range r.Devices[1:] {
		if c.takeLocked(d, seq) {
			c.mu.Unlock()
			return c.grant(r, d, "next", start), nil
		}
	}

	w := c.addLocked(seq, r.Devices)
	c.mu.Unlock()
	slog.Info("dualbuf: no device free, the request is queued", "model", r.Model, "devices", r.Devices)
	select {
	case <-w.ready:
		return c.grant(r, w.got, "queued", start), nil
	case <-ctx.Done():
		if c.cancel(w) {
			return nil, ctx.Err()
		}
		return c.grant(r, w.got, "queued", start), nil
	}
}

// Also takes a second place for a granted request (the CPU of a model split between the GPU and the CPU),
// waiting in arrival order; the places are always taken GPU first, then CPU, so no wait can go round.
func (g *Grant) Also(ctx context.Context, device string) error {
	c := g.c
	if c == nil || device == "" || slices.Contains(g.held, device) {
		return nil
	}
	c.mu.Lock()
	c.seq++
	if c.takeLocked(device, c.seq) {
		g.held = append(g.held, device)
		c.mu.Unlock()
		return nil
	}
	w := c.addLocked(c.seq, []string{device})
	c.mu.Unlock()
	select {
	case <-w.ready:
	case <-ctx.Done():
		if c.cancel(w) {
			return ctx.Err()
		}
	}
	c.mu.Lock()
	g.held = append(g.held, device)
	c.mu.Unlock()
	return nil
}

// Release gives the places back (once).
func (g *Grant) Release() {
	if g == nil || g.c == nil {
		return
	}
	g.once.Do(func() {
		c := g.c
		c.mu.Lock()
		for _, d := range g.held {
			c.running[d]--
		}
		c.dispatchLocked()
		c.mu.Unlock()
	})
}

// ReleaseOnDone releases the places when ctx ends.
func (g *Grant) ReleaseOnDone(ctx context.Context) {
	if g == nil || g.c == nil {
		return
	}
	go func() {
		<-ctx.Done()
		g.Release()
	}()
}

// Running is the number of requests running on a device.
func (c *Coordinator) Running(device string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.running[device]
}

// Queued is the number of waiting requests.
func (c *Coordinator) Queued() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.waiters)
}

func (c *Coordinator) grant(r Request, device, reason string, start time.Time) *Grant {
	g := &Grant{Device: device, Reason: reason, Waited: c.now().Sub(start), c: c, held: []string{device}}
	slog.Info("dualbuf placement", "model", r.Model, "device", device, "reason", reason, "waited", g.Waited.Round(time.Millisecond), "devices", r.Devices)
	return g
}

func (c *Coordinator) freeLocked(d string) bool {
	l := c.limit(d)
	return l <= 0 || c.running[d] < l
}

// takeLocked takes d when it is free and no earlier waiter wants it.
func (c *Coordinator) takeLocked(d string, seq uint64) bool {
	if !c.freeLocked(d) {
		return false
	}
	for _, w := range c.waiters {
		if w.seq < seq && slices.Contains(w.wants, d) {
			return false
		}
	}
	c.running[d]++
	return true
}

func (c *Coordinator) addLocked(seq uint64, wants []string) *waiter {
	w := &waiter{seq: seq, wants: append([]string(nil), wants...), ready: make(chan struct{})}
	i, _ := slices.BinarySearchFunc(c.waiters, seq, func(x *waiter, s uint64) int {
		switch {
		case x.seq < s:
			return -1
		case x.seq > s:
			return 1
		}
		return 0
	})
	c.waiters = slices.Insert(c.waiters, i, w)
	c.dispatchLocked()
	return w
}

// cancel removes a waiter; false if it got a place meanwhile (the caller then owns it).
func (c *Coordinator) cancel(w *waiter) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	select {
	case <-w.ready:
		return false
	default:
	}
	c.waiters = slices.DeleteFunc(c.waiters, func(x *waiter) bool { return x == w })
	c.dispatchLocked()
	return true
}

// dispatchLocked gives free places to the waiters in arrival order, each to the first free device of its list.
func (c *Coordinator) dispatchLocked() {
	for i := 0; i < len(c.waiters); {
		w := c.waiters[i]
		granted := false
		for _, d := range w.wants {
			if !c.freeLocked(d) {
				continue
			}
			blocked := false
			for _, e := range c.waiters[:i] {
				if slices.Contains(e.wants, d) {
					blocked = true
					break
				}
			}
			if blocked {
				continue
			}
			c.running[d]++
			w.got = d
			close(w.ready)
			c.waiters = slices.Delete(c.waiters, i, i+1)
			granted = true
			break
		}
		if !granted {
			i++
		}
	}
}
