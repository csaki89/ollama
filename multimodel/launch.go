package multimodel

import (
	"fmt"
	"strconv"
	"strings"
)

// Launch is what the effective settings change in the llama-server start. Only values that come
// from the config or the request are set: with no config the start is the same as without the fork.
type Launch struct {
	Env      map[string]string   // per model environment (LLAMA_ARG_*)
	Args     map[string][]string // command line flags to replace or add (e.g. "-ub": {"256"})
	Parallel int                 // 0 = Ollama's value
	KVType   string              // "" = Ollama's value (OLLAMA_KV_CACHE_TYPE)
	Notes    []string            // log notes on values that had to be adjusted
}

// FromConfig reports whether a value came from the config file or the request (not builtin or env).
func FromConfig(src string) bool {
	return src != "" && src != SrcBuiltin && src != SrcEnv
}

// AutoZones is the effective --auto-zones switch (the dynamic KV); zones: off keeps it without a flex band
// (--no-zones-flex). The flex band needs the dynamic KV: with dynamic_kv false there is none.
func (e Effective) AutoZones() bool {
	return e.DynamicKV.V
}

// FlexBand reports whether the flex band is made (zones: auto with the dynamic KV).
func (e Effective) FlexBand() bool {
	return e.DynamicKV.V && e.Zones.V != ZonesOff
}

// Launch maps the effective settings to the llama-server start.
func (e Effective) Launch() Launch {
	l := Launch{Env: map[string]string{}, Args: map[string][]string{}}
	if FromConfig(e.DualBuffer.Src) {
		l.Env["LLAMA_ARG_DUAL_BUFFER"] = boolEnv(e.DualBuffer.V)
	}
	if FromConfig(e.DynamicKV.Src) || FromConfig(e.Zones.Src) {
		l.Env["LLAMA_ARG_AUTO_ZONES"] = boolEnv(e.AutoZones())
		if e.DynamicKV.V {
			l.Env["LLAMA_ARG_ZONES_FLEX"] = boolEnv(e.FlexBand())
		} else if e.Zones.V == ZonesAuto && FromConfig(e.Zones.Src) {
			l.Notes = append(l.Notes, "zones: auto needs dynamic_kv: true, no flex band")
		}
	}
	if FromConfig(e.KVType.Src) {
		l.KVType = e.KVType.V
	}
	if FromConfig(e.Parallel.Src) {
		l.Parallel = e.Parallel.V
	}
	if FromConfig(e.NUbatch.Src) && e.NUbatch.V > 0 {
		l.Args["-ub"] = []string{strconv.Itoa(e.NUbatch.V)}
	}
	if FromConfig(e.FlashAttention.Src) {
		l.Args["--flash-attn"] = []string{e.FlashAttention.V}
		if e.FlashAttention.V == "off" && e.AutoZones() {
			l.Notes = append(l.Notes, "flash_attention: off keeps the V cache transposed, the dynamic KV stays off")
		}
	}
	if FromConfig(e.OnBusy.Src) && e.OnBusy.V == OnBusyNext && FromConfig(e.WaitTimeout.Src) {
		l.Notes = append(l.Notes, "on_busy: next ignores wait_timeout")
	}
	return l
}

// Fingerprint identifies the fork-side launch settings: a loaded runner with another fingerprint is reloaded
// (the Ollama options, num_ctx etc., are compared by the scheduler itself).
func (l Launch) Fingerprint() string {
	var parts []string
	for _, k := range sortedKeys(l.Env) {
		parts = append(parts, k+"="+l.Env[k])
	}
	for _, k := range sortedKeys(l.Args) {
		parts = append(parts, k+"="+strings.Join(l.Args[k], ","))
	}
	if l.Parallel > 0 {
		parts = append(parts, fmt.Sprintf("np=%d", l.Parallel))
	}
	if l.KVType != "" {
		parts = append(parts, "kv="+l.KVType)
	}
	return strings.Join(parts, ";")
}

func boolEnv(b bool) string {
	if b {
		return "1"
	}
	return "0"
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}

// OptionsMap is the Ollama runner options of a settings section, for api.Options.FromMap:
// max_context -> num_ctx, n_batch -> num_batch, threads -> num_thread.
func (s Settings) OptionsMap() map[string]any {
	m := map[string]any{}
	if s.MaxContext != nil {
		m["num_ctx"] = float64(*s.MaxContext)
	}
	if s.NBatch != nil {
		m["num_batch"] = float64(*s.NBatch)
	}
	if s.Threads != nil {
		m["num_thread"] = float64(*s.Threads)
	}
	return m
}

// DefaultsOptions is OptionsMap of the defaults section (nil config: empty).
func (c *Config) DefaultsOptions() map[string]any {
	if c == nil {
		return nil
	}
	return c.Defaults.OptionsMap()
}

// ModelOptions is OptionsMap of the model's section (none: empty).
func (c *Config) ModelOptions(name string) map[string]any {
	if c == nil {
		return nil
	}
	if key, ok := c.SectionFor(name); ok {
		return c.Models[key].OptionsMap()
	}
	return nil
}

// SetsContext reports whether the defaults or the model's section give max_context (then Ollama's automatic
// context does not apply).
func (c *Config) SetsContext(name string) bool {
	if c == nil {
		return false
	}
	if c.Defaults.MaxContext != nil {
		return true
	}
	if key, ok := c.SectionFor(name); ok {
		return c.Models[key].MaxContext != nil
	}
	return false
}

// SetsBatch reports whether the defaults or the model's section give n_batch.
func (c *Config) SetsBatch(name string) bool {
	if c == nil {
		return false
	}
	if c.Defaults.NBatch != nil {
		return true
	}
	if key, ok := c.SectionFor(name); ok {
		return c.Models[key].NBatch != nil
	}
	return false
}
