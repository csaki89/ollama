// Package multimodel holds the dual-buffer fork's multi-model settings
// (<models>/dualbuf.yaml): the devices (GPUs and the CPU) with their running limits and reserves, the
// default section and the per model sections (device list, busy rule, keep alive, the zones and KV
// switches), and the swap and KV saving rules. The file is read once, at the server start; a broken file
// stops the start. Without a config file nothing here changes the scheduler.
package multimodel

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"path"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Config is the parsed and validated dualbuf.yaml.
type Config struct {
	// Default is the required default section: the settings of the models without their own section, and the
	// base of the fields a model section leaves out
	Default *Settings           `yaml:"default"`
	Devices map[string]Device   `yaml:"devices"`
	Models  map[string]Settings `yaml:"models"`
	Swap    Swap                `yaml:"swap"`
	// RAMLimit caps the RAM the scheduler plans with below the container limit (e.g. "90GiB"); empty: the cgroup limit
	RAMLimit string `yaml:"ram_limit"`
	// OldDefaults is the earlier name of the default section: an error that names the new one
	OldDefaults any `yaml:"defaults"`

	// Defaults is the default section (empty when the config was not parsed)
	Defaults      Settings `yaml:"-"`
	modelOrder    []string // the models keys in file order (the first matching pattern wins)
	ramLimitBytes uint64
}

// Device is one entry of the devices section: a GPU (backend rocm, vulkan, cuda or metal, matched by its PCI
// address or a part of its name) or the CPU (the entry named "cpu", backend cpu, no match).
type Device struct {
	Backend    string `yaml:"backend"`
	Match      string `yaml:"match"`
	MaxRunning *int   `yaml:"max_running"`
	// ReserveMiB: on a GPU the VRAM kept free on it (the llama-server fit target, also the zones' margin; default
	// LLAMA_ARG_FIT_TARGET); on the CPU the reserve of the whole RAM budget (default max(1 GiB, 5% of the limit))
	ReserveMiB *int `yaml:"reserve_mib"`
}

// Settings are the per model fields; the defaults section has the same fields.
type Settings struct {
	Devices        []string  `yaml:"devices"`
	OnBusy         *string   `yaml:"on_busy"`
	WaitTimeout    *Duration `yaml:"wait_timeout"`
	Zones          *string   `yaml:"zones"`
	DualBuffer     *bool     `yaml:"dual_buffer"`
	DynamicKV      *bool     `yaml:"dynamic_kv"`
	KVType         *string   `yaml:"kv_type"`
	MaxContext     *int      `yaml:"max_context"`
	KeepAlive      *Duration `yaml:"keep_alive"`
	Parallel       *int      `yaml:"parallel"`
	Threads        *int      `yaml:"threads"`
	NBatch         *int      `yaml:"n_batch"`
	NUbatch        *int      `yaml:"n_ubatch"`
	FlashAttention *string   `yaml:"flash_attention"`
	KVSave         *bool     `yaml:"kv_save"`
}

// Swap is the swap section: parking an idle model when the device is needed, and saving the KV.
type Swap struct {
	Enabled   *bool  `yaml:"enabled"`
	KeepInRAM string `yaml:"keep_in_ram"` // pinned | none
	SaveKV    *bool  `yaml:"save_kv"`
	KVDir     string `yaml:"kv_dir"`
}

const (
	OnBusyWait = "wait"
	OnBusyNext = "next"

	ZonesAuto = "auto"
	ZonesOff  = "off"

	KeepInRAMPinned = "pinned"
	KeepInRAMNone   = "none"

	// DeviceCPU is the name of the CPU device entry (backend cpu)
	DeviceCPU = "cpu"
	// BackendCPU is the backend of the CPU device
	BackendCPU = "cpu"
)

// Duration accepts "30m", "120s", "-1" / -1 (forever) and a number of seconds. keep_alive -1: the model is never
// taken off the stack (never loaded again from nothing) but may be moved (parked, put on another device);
// wait_timeout -1: wait for the first device without a limit.
type Duration struct {
	time.Duration
	Forever bool
}

func (d *Duration) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind != yaml.ScalarNode {
		return fmt.Errorf("line %d: a duration must be a scalar", n.Line)
	}
	v, err := ParseDuration(n.Value)
	if err != nil {
		return fmt.Errorf("line %d: %w", n.Line, err)
	}
	*d = v
	return nil
}

// ParseDuration parses a keep_alive / wait_timeout value; a negative value means forever.
func ParseDuration(s string) (Duration, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return Duration{}, errors.New("empty duration")
	}
	if n, err := strconv.ParseFloat(s, 64); err == nil {
		if n < 0 {
			return Duration{Duration: time.Duration(math.MaxInt64), Forever: true}, nil
		}
		return Duration{Duration: time.Duration(n * float64(time.Second))}, nil
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return Duration{}, fmt.Errorf("invalid duration %q", s)
	}
	if d < 0 {
		return Duration{Duration: time.Duration(math.MaxInt64), Forever: true}, nil
	}
	return Duration{Duration: d}, nil
}

func (d Duration) String() string {
	if d.Forever {
		return "-1"
	}
	return d.Duration.String()
}

// Parse reads and validates a config file's content. Unknown keys are errors, the default section is required.
func Parse(data []byte) (*Config, error) {
	var c Config
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		if !errors.Is(err, io.EOF) {
			return nil, err
		}
		// an empty file: no default section, reported by validate
	}
	c.modelOrder = modelKeyOrder(data)
	if c.Default != nil {
		c.Defaults = *c.Default
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

// modelKeyOrder is the order of the models keys in the file (the map loses it).
func modelKeyOrder(data []byte) []string {
	var root yaml.Node
	if yaml.Unmarshal(data, &root) != nil || len(root.Content) == 0 || root.Content[0].Kind != yaml.MappingNode {
		return nil
	}
	top := root.Content[0].Content
	for i := 0; i+1 < len(top); i += 2 {
		if top[i].Value != "models" || top[i+1].Kind != yaml.MappingNode {
			continue
		}
		var keys []string
		for j := 0; j+1 < len(top[i+1].Content); j += 2 {
			keys = append(keys, top[i+1].Content[j].Value)
		}
		return keys
	}
	return nil
}

var kvTypes = []string{"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}

func (c *Config) validate() error {
	var errs []error
	if c.OldDefaults != nil {
		errs = append(errs, errors.New("defaults: the section is called default now"))
	}
	if c.Default == nil {
		errs = append(errs, errors.New("default: missing (required: the settings of the models without their own section)"))
	}
	devNames := make([]string, 0, len(c.Devices))
	for name := range c.Devices {
		devNames = append(devNames, name)
	}
	sort.Strings(devNames)
	for _, name := range devNames {
		d := c.Devices[name]
		if strings.TrimSpace(name) == "" {
			errs = append(errs, errors.New("devices: empty device name"))
			continue
		}
		if d.MaxRunning != nil && *d.MaxRunning < 1 {
			errs = append(errs, fmt.Errorf("devices.%s.max_running: must be at least 1", name))
		}
		if d.ReserveMiB != nil && *d.ReserveMiB < 0 {
			errs = append(errs, fmt.Errorf("devices.%s.reserve_mib: must not be negative", name))
		}
		cpu := strings.EqualFold(d.Backend, BackendCPU)
		if name == DeviceCPU || cpu {
			switch {
			case name != DeviceCPU:
				errs = append(errs, fmt.Errorf("devices.%s.backend: cpu is the backend of the entry named cpu only", name))
			case d.Backend != "" && !cpu:
				errs = append(errs, fmt.Errorf("devices.cpu.backend: %q (the CPU entry has backend cpu)", d.Backend))
			case d.Match != "":
				errs = append(errs, fmt.Errorf("devices.cpu.match: not used for the CPU"))
			}
			continue
		}
		switch strings.ToLower(d.Backend) {
		case "rocm", "vulkan", "cuda", "metal":
		case "":
			errs = append(errs, fmt.Errorf("devices.%s.backend: missing (rocm, vulkan, cuda, metal; cpu for the entry named cpu)", name))
		default:
			errs = append(errs, fmt.Errorf("devices.%s.backend: unknown backend %q (rocm, vulkan, cuda, metal; cpu for the entry named cpu)", name, d.Backend))
		}
		if strings.TrimSpace(d.Match) == "" {
			errs = append(errs, fmt.Errorf("devices.%s.match: missing (a name or a PCI address)", name))
		}
	}
	errs = append(errs, c.validateSettings("default", c.Defaults)...)
	names := make([]string, 0, len(c.Models))
	for k := range c.Models {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		if strings.TrimSpace(k) == "" {
			errs = append(errs, errors.New("models: empty model key"))
			continue
		}
		if _, err := path.Match(k, ""); err != nil {
			errs = append(errs, fmt.Errorf("models.%s: invalid pattern: %v", k, err))
		}
		errs = append(errs, c.validateSettings("models."+k, c.Models[k])...)
	}
	switch c.Swap.KeepInRAM {
	case "", KeepInRAMPinned, KeepInRAMNone:
	default:
		errs = append(errs, fmt.Errorf("swap.keep_in_ram: %q (pinned or none)", c.Swap.KeepInRAM))
	}
	if c.RAMLimit != "" {
		b, err := ParseBytes(c.RAMLimit)
		if err != nil {
			errs = append(errs, fmt.Errorf("ram_limit: %w", err))
		}
		c.ramLimitBytes = b
	}
	return errors.Join(errs...)
}

func (c *Config) validateSettings(where string, s Settings) []error {
	var errs []error
	seen := map[string]bool{}
	for _, d := range s.Devices {
		if _, ok := c.Devices[d]; !ok {
			errs = append(errs, fmt.Errorf("%s.devices: unknown device %q (not in the devices section)", where, d))
		}
		if seen[d] {
			errs = append(errs, fmt.Errorf("%s.devices: %q is listed twice", where, d))
		}
		seen[d] = true
	}
	if s.OnBusy != nil && *s.OnBusy != OnBusyWait && *s.OnBusy != OnBusyNext {
		errs = append(errs, fmt.Errorf("%s.on_busy: %q (wait or next)", where, *s.OnBusy))
	}
	if s.Zones != nil && *s.Zones != ZonesAuto && *s.Zones != ZonesOff {
		errs = append(errs, fmt.Errorf("%s.zones: %q (auto or off; the flex band size is computed, not set)", where, *s.Zones))
	}
	if s.KVType != nil && !slices.Contains(kvTypes, strings.ToLower(*s.KVType)) {
		errs = append(errs, fmt.Errorf("%s.kv_type: %q (one of %s)", where, *s.KVType, strings.Join(kvTypes, ", ")))
	}
	for _, f := range []struct {
		name string
		v    *int
		min  int
	}{{"max_context", s.MaxContext, 256}, {"parallel", s.Parallel, 1}, {"threads", s.Threads, 1}, {"n_batch", s.NBatch, 1}, {"n_ubatch", s.NUbatch, 1}} {
		if f.v != nil && *f.v < f.min {
			errs = append(errs, fmt.Errorf("%s.%s: %d (at least %d)", where, f.name, *f.v, f.min))
		}
	}
	if s.FlashAttention != nil {
		if _, err := normalizeFlashAttention(*s.FlashAttention); err != nil {
			errs = append(errs, fmt.Errorf("%s.flash_attention: %w", where, err))
		}
	}
	return errs
}

func normalizeFlashAttention(v string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "on", "true", "1", "yes", "enabled":
		return "on", nil
	case "off", "false", "0", "no", "disabled":
		return "off", nil
	case "auto":
		return "auto", nil
	}
	return "", fmt.Errorf("%q (on, off or auto)", v)
}

// ParseBytes parses "90GiB", "512MiB", "100G", "1000000" (bytes).
func ParseBytes(s string) (uint64, error) {
	s = strings.TrimSpace(s)
	units := []struct {
		suffix string
		mul    float64
	}{
		{"TiB", 1 << 40}, {"GiB", 1 << 30}, {"MiB", 1 << 20}, {"KiB", 1 << 10},
		{"TB", 1e12}, {"GB", 1e9}, {"MB", 1e6}, {"KB", 1e3},
		{"T", 1 << 40}, {"G", 1 << 30}, {"M", 1 << 20}, {"K", 1 << 10}, {"B", 1},
	}
	mul := 1.0
	for _, u := range units {
		if strings.HasSuffix(s, u.suffix) {
			mul = u.mul
			s = strings.TrimSpace(strings.TrimSuffix(s, u.suffix))
			break
		}
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f < 0 {
		return 0, fmt.Errorf("invalid size %q", s)
	}
	return uint64(f * mul), nil
}

// RAMLimitBytes is the parsed ram_limit, 0 when unset.
func (c *Config) RAMLimitBytes() uint64 {
	if c == nil {
		return 0
	}
	return c.ramLimitBytes
}

// GPUReserveMiB is reserve_mib of a GPU device; false when not set (then LLAMA_ARG_FIT_TARGET, or the
// llama-server default, applies) or when the device is the CPU or unknown.
func (c *Config) GPUReserveMiB(device string) (int, bool) {
	if c == nil || device == "" || device == DeviceCPU {
		return 0, false
	}
	if d, ok := c.Devices[device]; ok && d.ReserveMiB != nil {
		return *d.ReserveMiB, true
	}
	return 0, false
}

// RAMReserve is the RAM kept free of the budget (limit): reserve_mib of the CPU device, else max(1 GiB, 5%).
func (c *Config) RAMReserve(limit uint64) uint64 {
	if c != nil {
		if d, ok := c.Devices[DeviceCPU]; ok && d.ReserveMiB != nil {
			return uint64(*d.ReserveMiB) << 20
		}
	}
	return RAMReserve(limit)
}

// MaxRunning of a device; 0 means no limit (the device has no entry or no max_running).
func (c *Config) MaxRunning(device string) int {
	if c == nil {
		return 0
	}
	if d, ok := c.Devices[device]; ok && d.MaxRunning != nil {
		return *d.MaxRunning
	}
	return 0
}

// SwapEnabled reports whether swap.enabled is on (default off).
func (c *Config) SwapEnabled() bool {
	return c != nil && c.Swap.Enabled != nil && *c.Swap.Enabled
}

// SwapSaveKV reports whether swap.save_kv is on (default on when swap is enabled).
func (c *Config) SwapSaveKV() bool {
	if c == nil {
		return false
	}
	if c.Swap.SaveKV != nil {
		return *c.Swap.SaveKV
	}
	return c.SwapEnabled()
}
