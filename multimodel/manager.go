package multimodel

import (
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
)

// Manager ties the parts together for the scheduler: the config file, the running limits, the matched
// devices and the parked models.
type Manager struct {
	Store *Store
	Coord *Coordinator
	Env   Env

	mu       sync.Mutex
	limits   *Config // the config the running limits come from (the last one read)
	devs     []ml.DeviceInfo
	haveDevs bool
	dmFor    *Config
	dm       DeviceMap
	dmLogged string
	parked   map[string]*Parked
	hostSeen map[string]uint64 // measured host RAM of the last instance per key
	split    map[string]bool   // model key + device: the model computes on the CPU too when placed there
}

// Parked is a model without a process whose KV (and, with keep_in_ram: pinned, its model file pages) is in RAM.
type Parked struct {
	Key       string // the scheduler key of the instance
	Model     string
	Placement string
	Prefix    string // KV file prefix in the RAM directory ("" = no KV saved)
	Pinned    *Pinned
	Forever   bool
	LastUsed  time.Time
	Bytes     uint64 // RAM held: the pinned file and the KV files
	// KeepAlive: the parked model leaves the RAM when it is not used for this long (as the stock keep_alive); 0 or
	// Forever: no expiry
	KeepAlive time.Duration

	expiry *time.Timer
}

// NewManager makes a manager for the config at path.
func NewManager(path string, env Env) *Manager {
	m := &Manager{Store: NewStore(path), Env: env, parked: map[string]*Parked{}, hostSeen: map[string]uint64{}, split: map[string]bool{}}
	m.Coord = NewCoordinator(m.maxRunning)
	return m
}

// Load reads the config file once (at the server start); a broken file is an error and the server must not start.
func (m *Manager) Load() error {
	if m == nil {
		return nil
	}
	_, err := m.Store.Load()
	if err == nil {
		m.Config()
	}
	return err
}

// Config is the config read at the start (see Store.Current); it also becomes the source of the running limits.
func (m *Manager) Config() *Config {
	if m == nil {
		return nil
	}
	c := m.Store.Current()
	m.mu.Lock()
	m.limits = c
	m.mu.Unlock()
	return c
}

func (m *Manager) maxRunning(device string) int {
	// called with the coordinator lock held: no file access here
	m.mu.Lock()
	c := m.limits
	m.mu.Unlock()
	return c.MaxRunning(device)
}

// NoteDevices records the discovered devices (after every scheduler discovery).
func (m *Manager) NoteDevices(devs []ml.DeviceInfo) {
	if m == nil || len(devs) == 0 {
		return
	}
	m.mu.Lock()
	m.devs = append([]ml.DeviceInfo(nil), devs...)
	m.haveDevs = true
	m.dmFor = nil
	m.mu.Unlock()
}

// HaveDevices reports whether a discovery was recorded.
func (m *Manager) HaveDevices() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.haveDevs
}

// Devices matches the config devices against the last discovery, with one log line when the result changes.
func (m *Manager) Devices(c *Config) DeviceMap {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.dmFor == c && m.dmFor != nil {
		return m.dm
	}
	dm := c.MatchDevices(m.devs)
	m.dm, m.dmFor = dm, c
	if s := dm.String() + strings.Join(dm.Errors, ";"); s != m.dmLogged && c != nil && len(c.Devices) > 0 {
		m.dmLogged = s
		slog.Info("dualbuf devices", "matched", dm.String())
		for _, e := range dm.Errors {
			slog.Error("dualbuf device match", "error", e)
		}
		if len(dm.Missing) > 0 && m.haveDevs {
			slog.Warn("dualbuf: configured devices are not available (not in the machine, not visible in the container, or the driver does not see them): skipped, the models use the other devices of their list", "missing", dm.Missing)
		}
	}
	return dm
}

// Present filters a device list to the devices that were found (all of them before the first discovery).
func (m *Manager) Present(c *Config, list []string) []string {
	if !m.HaveDevices() {
		return list
	}
	dm := m.Devices(c)
	var out []string
	for _, d := range list {
		if dm.Present(d) {
			out = append(out, d)
		}
	}
	return out
}

// Park records a parked model (replacing an earlier record of the same key).
func (m *Manager) Park(p *Parked) {
	m.mu.Lock()
	old := m.parked[p.Key]
	m.parked[p.Key] = p
	if old != nil && old.expiry != nil {
		old.expiry.Stop()
	}
	if !p.Forever && p.KeepAlive > 0 {
		// keep_alive runs on while the model is parked: when it passes, the model leaves the RAM as the stock unload does
		left := time.Until(p.LastUsed.Add(p.KeepAlive))
		if left < 0 {
			left = 0
		}
		p.expiry = time.AfterFunc(left, func() {
			m.mu.Lock()
			same := m.parked[p.Key] == p
			m.mu.Unlock()
			if same {
				slog.Info("dualbuf: keep_alive of the parked model passed", "model", p.Model)
				m.Drop(p.Key)
			}
		})
	}
	m.mu.Unlock()
	if old != nil && old.Pinned != p.Pinned {
		old.Pinned.Release()
	}
	slog.Info("dualbuf: model parked", "model", p.Model, "placement", p.Placement, "kv", p.Prefix != "", "pinned", p.Pinned != nil, "ram", p.Bytes)
}

// SetPinned attaches the pinned model file to a parked record; if the model is no longer parked, the pin is released.
func (m *Manager) SetPinned(key string, pin *Pinned) {
	m.mu.Lock()
	p := m.parked[key]
	if p != nil && p.Pinned == nil {
		p.Pinned = pin
		p.Bytes += pin.Bytes
		m.mu.Unlock()
		slog.Info("dualbuf: model file pinned in RAM", "model", p.Model, "bytes", pin.Bytes)
		return
	}
	m.mu.Unlock()
	pin.Release()
}

// Unpark takes the record of a key (the model is being loaded again); nil if it was not parked.
func (m *Manager) Unpark(key string) *Parked {
	m.mu.Lock()
	defer m.mu.Unlock()
	p := m.parked[key]
	delete(m.parked, key)
	if p != nil && p.expiry != nil {
		p.expiry.Stop()
	}
	return p
}

// ParkedList is a snapshot of the parked models.
func (m *Manager) ParkedList() []*Parked {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*Parked, 0, len(m.parked))
	for _, p := range m.parked {
		c := *p
		c.expiry = nil
		out = append(out, &c)
	}
	return out
}

// Drop removes a parked model completely (RAM eviction): the pinned pages and the KV in RAM go.
func (m *Manager) Drop(key string) {
	p := m.Unpark(key)
	if p == nil {
		return
	}
	p.Pinned.Release()
	if p.Prefix != "" {
		RemoveKV(RAMDir(), p.Prefix)
	}
	slog.Info("dualbuf: parked model dropped from RAM", "model", p.Model, "placement", p.Placement, "ram", p.Bytes)
}

// NoteHost records the measured host RAM of an instance (for the next estimate of the same key).
func (m *Manager) NoteHost(key string, bytes uint64) {
	if bytes == 0 {
		return
	}
	m.mu.Lock()
	m.hostSeen[key] = bytes
	m.mu.Unlock()
}

// HostEstimate is the measured host RAM of the last instance of key, else fallback.
func (m *Manager) HostEstimate(key string, fallback uint64) uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if b, ok := m.hostSeen[key]; ok {
		return b
	}
	return fallback
}

// SetSplit records whether the model computes on the CPU too when placed on device (seen at its load).
func (m *Manager) SetSplit(key, device string, split bool) {
	m.mu.Lock()
	m.split[key+"\x00@"+device] = split
	m.mu.Unlock()
}

// Split reports whether the model was seen computing on the CPU too on device.
func (m *Manager) Split(key, device string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.split[key+"\x00@"+device]
}
