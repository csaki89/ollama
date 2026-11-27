package multimodel

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"slices"
	"sync"
	"time"
)

// ErrQueueFull: the waiting requests reached the queue limit (OLLAMA_MAX_QUEUE); the server answers it as the
// stock scheduler does.
var ErrQueueFull = errors.New("dualbuf: the request queue is full")

// Coordinator decides where and when a request of a configured model runs (docs/MULTIMODEL-DESIGN.md 2.4):
//   - a device is busy while active requests run on it up to its max_running (active models: a model has one
//     active request at a time); a split model (GPU + CPU) makes every device it computes on busy, the CPU too;
//   - one model, one active request: the later requests of a working model wait, whatever their on_busy;
//   - per device arrival order: a later request does not take a device an earlier waiting request would use;
//     a request touching none of its devices may start before it;
//   - tiers: a request first accepts only the first device of its list; when it is busy, on_busy next accepts
//     the next tier at once, on_busy wait after wait_timeout (-1: never), tier by tier to the end of the
//     list; when any accepted tier is free the request starts there, on the highest one; at the end it waits
//     until one frees up. No error, except a full queue (OLLAMA_MAX_QUEUE).
type Coordinator struct {
	mu       sync.Mutex
	running  map[string]int  // active models per device
	active   map[string]bool // models with an active request
	waiters  []*waiter       // arrival order
	seq      uint64
	limit    func(device string) int // max_running, 0 = no limit
	maxQueue int                     // 0 = no limit
	now      func() time.Time
}

type waiter struct {
	seq    uint64
	key    string
	name   string
	tiers  []string
	accept int // the first accept tiers are acceptable
	extra  func(device string) []string
	start  time.Time
	got    string
	held   []string
	ready  chan struct{}
	steps  *time.Timer
	cont   bool // the CPU place of a request already running (Grant.Also): capacity only, before the others
}

// NewCoordinator returns a coordinator; limit gives max_running of a device (0: no limit).
func NewCoordinator(limit func(device string) int) *Coordinator {
	return &Coordinator{running: map[string]int{}, active: map[string]bool{}, limit: limit, now: time.Now}
}

// SetMaxQueue sets the limit of waiting requests (OLLAMA_MAX_QUEUE; 0 = none).
func (c *Coordinator) SetMaxQueue(n int) {
	c.mu.Lock()
	c.maxQueue = n
	c.mu.Unlock()
}

// Request is what a request asks for.
type Request struct {
	Model       string        // the model name (log lines)
	Key         string        // the model identity (one active request per key); "" = Model
	Devices     []string      // the model's present devices, highest tier first
	OnBusy      string        // wait or next
	WaitTimeout time.Duration // per tier for on_busy wait; math.MaxInt64 = never step down
	// Extra are the other devices the model computes on when placed on a device (the CPU of a split model)
	Extra func(device string) []string
}

// Grant is a place on a device, held until Release.
type Grant struct {
	Device string
	Tier   int           // index of Device in the model's list
	Reason string        // free (first tier at once), waited (first tier after a wait), next (a lower tier)
	Waited time.Duration // time spent waiting for the place
	c      *Coordinator
	key    string
	held   []string
	once   sync.Once
}

// Acquire waits for the request's place (see Coordinator). It returns ErrQueueFull when the queue is full and
// the context error when the request ends while it waits.
func (c *Coordinator) Acquire(ctx context.Context, r Request) (*Grant, error) {
	if len(r.Devices) == 0 {
		return &Grant{Reason: "no devices"}, nil
	}
	key := r.Key
	if key == "" {
		key = r.Model
	}
	c.mu.Lock()
	if c.maxQueue > 0 && len(c.waiters) >= c.maxQueue {
		c.mu.Unlock()
		return nil, ErrQueueFull
	}
	c.seq++
	w := &waiter{seq: c.seq, key: key, name: r.Model, tiers: append([]string(nil), r.Devices...), accept: 1, extra: r.Extra,
		start: c.now(), ready: make(chan struct{})}
	if r.OnBusy == OnBusyNext {
		w.accept = len(w.tiers)
	}
	c.waiters = append(c.waiters, w)
	c.dispatchLocked()
	select {
	case <-w.ready:
		c.mu.Unlock()
		return c.grant(r, w), nil
	default:
	}
	if r.OnBusy != OnBusyNext && r.WaitTimeout > 0 && r.WaitTimeout != time.Duration(math.MaxInt64) && len(w.tiers) > 1 {
		c.stepLocked(w, r.WaitTimeout)
	}
	c.mu.Unlock()
	slog.Info("dualbuf: the request waits for its place", "model", r.Model, "devices", r.Devices, "on_busy", r.OnBusy)
	select {
	case <-w.ready:
		return c.grant(r, w), nil
	case <-ctx.Done():
		if c.cancel(w) {
			return nil, ctx.Err()
		}
		return c.grant(r, w), nil
	}
}

// stepLocked makes the next tier acceptable after d, again until the end of the list.
func (c *Coordinator) stepLocked(w *waiter, d time.Duration) {
	w.steps = time.AfterFunc(d, func() {
		c.mu.Lock()
		defer c.mu.Unlock()
		if !c.waitingLocked(w) {
			return
		}
		if w.accept < len(w.tiers) {
			w.accept++
			slog.Info("dualbuf: wait_timeout passed, the next tier is accepted too", "model", w.name, "devices", w.tiers[:w.accept])
			c.dispatchLocked()
		}
		if c.waitingLocked(w) && w.accept < len(w.tiers) {
			c.stepLocked(w, d)
		}
	})
}

func (c *Coordinator) waitingLocked(w *waiter) bool {
	return slices.Contains(c.waiters, w)
}

func (c *Coordinator) grant(r Request, w *waiter) *Grant {
	tier := slices.Index(w.tiers, w.got)
	g := &Grant{Device: w.got, Tier: tier, Waited: c.now().Sub(w.start), c: c, key: w.key, held: w.held}
	switch {
	case tier > 0:
		g.Reason = "next"
	case g.Waited > time.Millisecond:
		g.Reason = "waited"
	default:
		g.Reason = "free"
	}
	slog.Info("dualbuf placement", "model", r.Model, "device", g.Device, "reason", g.Reason, "waited", g.Waited.Round(time.Millisecond), "devices", r.Devices)
	return g
}

// Also takes a further place for a granted request (the CPU of a model found to be split at its load); it goes
// before the waiting requests (the request already runs) and waits only for the capacity.
func (g *Grant) Also(ctx context.Context, device string) error {
	c := g.c
	if c == nil || device == "" {
		return nil
	}
	c.mu.Lock()
	if slices.Contains(g.held, device) {
		c.mu.Unlock()
		return nil
	}
	c.seq++
	w := &waiter{seq: 0, key: "", tiers: []string{device}, accept: 1, ready: make(chan struct{}), cont: true}
	c.waiters = slices.Insert(c.waiters, 0, w)
	c.dispatchLocked()
	c.mu.Unlock()
	select {
	case <-w.ready:
	case <-ctx.Done():
		if c.cancel(w) {
			return ctx.Err()
		}
	}
	c.mu.Lock()
	g.held = append(g.held, w.held...)
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
		delete(c.active, g.key)
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

// Running is the number of active models on a device.
func (c *Coordinator) Running(device string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.running[device]
}

// Active reports whether a model has an active request.
func (c *Coordinator) Active(key string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.active[key]
}

// Queued is the number of waiting requests.
func (c *Coordinator) Queued() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.waiters)
}

// HasWaiter reports whether a request of the model waits.
func (c *Coordinator) HasWaiter(key string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, w := range c.waiters {
		if w.key == key {
			return true
		}
	}
	return false
}

// Demote: the model was moved down to device to make room; its waiting requests accept that tier (and the
// ones above it) from now on, so they are served there.
func (c *Coordinator) Demote(key, device string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, w := range c.waiters {
		if w.key != key {
			continue
		}
		if i := slices.Index(w.tiers, device); i >= 0 && w.accept < i+1 {
			w.accept = i + 1
		}
	}
	c.dispatchLocked()
}

func (c *Coordinator) freeLocked(d string) bool {
	l := c.limit(d)
	return l <= 0 || c.running[d] < l
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
	if w.steps != nil {
		w.steps.Stop()
	}
	c.waiters = slices.DeleteFunc(c.waiters, func(x *waiter) bool { return x == w })
	c.dispatchLocked()
	return true
}

// needs is what a waiter takes on tier device d: d and the other devices the model computes on there.
func (w *waiter) needs(d string) []string {
	out := []string{d}
	if w.extra != nil {
		for _, e := range w.extra(d) {
			if !slices.Contains(out, e) {
				out = append(out, e)
			}
		}
	}
	return out
}

// dispatchLocked gives the free places to the waiters in arrival order: each one gets the highest acceptable
// tier whose devices are all free and none of them is wanted by an earlier waiting request.
func (c *Coordinator) dispatchLocked() {
	for {
		progress := false
		blocked := map[string]bool{}
		for i := 0; i < len(c.waiters); i++ {
			w := c.waiters[i]
			if w.cont {
				// the CPU place of a running request: capacity only
				if d := w.tiers[0]; c.freeLocked(d) {
					c.running[d]++
					w.got, w.held = d, []string{d}
					c.removeLocked(w)
					progress = true
					break
				}
				continue
			}
			if !c.active[w.key] {
				for _, d := range w.tiers[:w.accept] {
					need := w.needs(d)
					ok := true
					for _, n := range need {
						if blocked[n] || !c.freeLocked(n) {
							ok = false
							break
						}
					}
					if !ok {
						continue
					}
					for _, n := range need {
						c.running[n]++
					}
					c.active[w.key] = true
					w.got, w.held = d, need
					c.removeLocked(w)
					progress = true
					break
				}
				if progress {
					break
				}
			}
			// a waiting request keeps every device it would use from the later ones; one that waits for its own
			// working model waits for the model, not for a device (it keeps its turn once the model is free)
			if c.active[w.key] {
				continue
			}
			for _, d := range w.tiers[:w.accept] {
				for _, n := range w.needs(d) {
					blocked[n] = true
				}
			}
		}
		if !progress {
			return
		}
	}
}

func (c *Coordinator) removeLocked(w *waiter) {
	if w.steps != nil {
		w.steps.Stop()
	}
	c.waiters = slices.DeleteFunc(c.waiters, func(x *waiter) bool { return x == w })
	close(w.ready)
}
