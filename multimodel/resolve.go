package multimodel

import (
	"fmt"
	"math"
	"path"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Source names where an effective value came from, weakest first:
// builtin < env < default < modelfile < model:<key> < request.
const (
	SrcBuiltin   = "builtin"
	SrcEnv       = "env"
	SrcDefaults  = "default"
	SrcModelfile = "modelfile"
	SrcRequest   = "request"
)

// Value is an effective setting with its source.
type Value[T any] struct {
	V   T
	Src string
}

func (v *Value[T]) set(x T, src string) { v.V, v.Src = x, src }

// Effective is the resolved setting set of one model load.
type Effective struct {
	Model   string // the model name it was resolved for
	Section string // the models key that matched, "" for the default section only

	Devices        Value[[]string]
	OnBusy         Value[string]
	WaitTimeout    Value[time.Duration]
	Zones          Value[string]
	DualBuffer     Value[bool]
	DynamicKV      Value[bool]
	KVType         Value[string] // "" = the llama-server default (f16)
	MaxContext     Value[int]    // 0 = Ollama's automatic context
	KeepAlive      Value[Duration]
	Parallel       Value[int]
	Threads        Value[int] // 0 = automatic
	NBatch         Value[int]
	NUbatch        Value[int] // 0 = same as n_batch
	FlashAttention Value[string]
	KVSave         Value[bool]
}

// Env reads a process environment variable; envconfig.Var in the server, a map in tests.
type Env func(key string) string

// DefaultWaitTimeout applies to on_busy: wait without a wait_timeout.
const DefaultWaitTimeout = 60 * time.Second

// SectionFor returns the models key for a model name: the exact name:tag, else the first matching pattern in
// the order of the file; false when none matches (the model gets the default section). The name is tried as
// given, without the default registry and namespace, and with ":latest" when it has no tag.
func (c *Config) SectionFor(name string) (string, bool) {
	if c == nil || len(c.Models) == 0 {
		return "", false
	}
	cands := nameCandidates(name)
	for _, n := range cands {
		if _, ok := c.Models[n]; ok {
			return n, true
		}
		for k := range c.Models {
			if strings.EqualFold(k, n) {
				return k, true
			}
		}
	}
	// the patterns in file order: the first that matches wins
	order := c.modelOrder
	if len(order) != len(c.Models) {
		// not from Parse (a config built in code): key order
		order = make([]string, 0, len(c.Models))
		for k := range c.Models {
			order = append(order, k)
		}
		sort.Strings(order)
	}
	var keys []string
	for _, k := range order {
		if _, ok := c.Models[k]; ok && strings.ContainsAny(k, "*?[") {
			keys = append(keys, k)
		}
	}
	for _, k := range keys {
		for _, n := range cands {
			if ok, _ := path.Match(strings.ToLower(k), strings.ToLower(n)); ok {
				return k, true
			}
		}
	}
	return "", false
}

// nameCandidates: the name, without the default registry/namespace, and with :latest added.
func nameCandidates(name string) []string {
	name = strings.TrimSpace(name)
	var out []string
	add := func(s string) {
		for _, o := range out {
			if o == s {
				return
			}
		}
		out = append(out, s)
	}
	add(name)
	short := strings.TrimPrefix(name, "registry.ollama.ai/")
	short = strings.TrimPrefix(short, "library/")
	add(short)
	for _, n := range []string{name, short} {
		base := n[strings.LastIndex(n, "/")+1:]
		if !strings.Contains(base, ":") {
			add(n + ":latest")
		}
	}
	return out
}

// Resolve computes the effective settings of a model: env < default < Modelfile < model section < request.
// modelfile and request are the raw option maps (num_ctx, num_batch, num_thread are read from them; the
// request may also carry the fork keys dual_buffer, dynamic_kv, zones, kv_type, parallel, n_ubatch,
// flash_attention, kv_save). reqKeepAlive is the request's keep_alive, nil when not given.
func (c *Config) Resolve(name string, env Env, modelfile, request map[string]any, reqKeepAlive *time.Duration) Effective {
	e := Effective{Model: name}
	if env == nil {
		env = func(string) string { return "" }
	}

	// builtin and env
	e.OnBusy.set(OnBusyWait, SrcBuiltin)
	e.WaitTimeout.set(DefaultWaitTimeout, SrcBuiltin)
	e.Zones.set(ZonesAuto, SrcBuiltin)
	e.KeepAlive.set(Duration{Duration: 5 * time.Minute}, SrcBuiltin)
	e.Parallel.set(1, SrcBuiltin)
	e.NBatch.set(512, SrcBuiltin)
	e.FlashAttention.set("auto", SrcBuiltin)
	if v := env("LLAMA_ARG_DUAL_BUFFER"); v != "" {
		e.DualBuffer.set(truthy(v), SrcEnv)
	} else {
		e.DualBuffer.Src = SrcBuiltin
	}
	if v := env("LLAMA_ARG_AUTO_ZONES"); v != "" {
		on := truthy(v)
		e.DynamicKV.set(on, SrcEnv)
		if on {
			e.Zones.set(ZonesAuto, SrcEnv)
		} else {
			e.Zones.set(ZonesOff, SrcEnv)
		}
	} else {
		e.DynamicKV.Src = SrcBuiltin
	}
	if v := env("OLLAMA_KV_CACHE_TYPE"); v != "" {
		e.KVType.set(strings.ToLower(v), SrcEnv)
	} else {
		e.KVType.Src = SrcBuiltin
	}
	if v := env("OLLAMA_CONTEXT_LENGTH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			e.MaxContext.set(n, SrcEnv)
		}
	}
	if e.MaxContext.Src == "" {
		e.MaxContext.Src = SrcBuiltin
	}
	if v := env("OLLAMA_KEEP_ALIVE"); v != "" {
		if d, err := ParseDuration(v); err == nil {
			e.KeepAlive.set(d, SrcEnv)
		}
	}
	if v := env("OLLAMA_NUM_PARALLEL"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			e.Parallel.set(n, SrcEnv)
		}
	}
	if v := env("OLLAMA_FLASH_ATTENTION"); v != "" {
		if truthy(v) {
			e.FlashAttention.set("on", SrcEnv)
		} else {
			e.FlashAttention.set("off", SrcEnv)
		}
	}
	if v := env("LLAMA_ARG_UBATCH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			e.NUbatch.set(n, SrcEnv)
		}
	}
	if e.NUbatch.Src == "" {
		e.NUbatch.Src = SrcBuiltin
	}
	e.Threads.Src = SrcBuiltin
	e.KVSave.Src = SrcBuiltin
	e.Devices.Src = SrcBuiltin

	if c != nil {
		e.apply(c.Defaults, SrcDefaults)
	}
	e.applyOptions(modelfile, SrcModelfile, false)
	if c != nil {
		if key, ok := c.SectionFor(name); ok {
			e.Section = key
			e.apply(c.Models[key], "model:"+key)
		}
	}
	e.applyOptions(request, SrcRequest, true)
	if reqKeepAlive != nil {
		d := Duration{Duration: *reqKeepAlive}
		if *reqKeepAlive < 0 || *reqKeepAlive == time.Duration(math.MaxInt64) {
			d = Duration{Duration: time.Duration(math.MaxInt64), Forever: true}
		}
		e.KeepAlive.set(d, SrcRequest)
	}
	return e
}

func (e *Effective) apply(s Settings, src string) {
	if s.Devices != nil {
		e.Devices.set(append([]string(nil), s.Devices...), src)
	}
	if s.OnBusy != nil {
		e.OnBusy.set(*s.OnBusy, src)
	}
	if s.WaitTimeout != nil {
		e.WaitTimeout.set(s.WaitTimeout.Duration, src)
	}
	if s.Zones != nil {
		e.Zones.set(*s.Zones, src)
	}
	if s.DualBuffer != nil {
		e.DualBuffer.set(*s.DualBuffer, src)
	}
	if s.DynamicKV != nil {
		e.DynamicKV.set(*s.DynamicKV, src)
	}
	if s.KVType != nil {
		e.KVType.set(strings.ToLower(*s.KVType), src)
	}
	if s.MaxContext != nil {
		e.MaxContext.set(*s.MaxContext, src)
	}
	if s.KeepAlive != nil {
		e.KeepAlive.set(*s.KeepAlive, src)
	}
	if s.Parallel != nil {
		e.Parallel.set(*s.Parallel, src)
	}
	if s.Threads != nil {
		e.Threads.set(*s.Threads, src)
	}
	if s.NBatch != nil {
		e.NBatch.set(*s.NBatch, src)
	}
	if s.NUbatch != nil {
		e.NUbatch.set(*s.NUbatch, src)
	}
	if s.FlashAttention != nil {
		fa, _ := normalizeFlashAttention(*s.FlashAttention)
		e.FlashAttention.set(fa, src)
	}
	if s.KVSave != nil {
		e.KVSave.set(*s.KVSave, src)
	}
}

// applyOptions takes the Ollama runner options (num_ctx, num_batch, num_thread) from a Modelfile or request
// map, and with fork set the fork keys a request may carry.
func (e *Effective) applyOptions(m map[string]any, src string, fork bool) {
	if len(m) == 0 {
		return
	}
	if n, ok := intOpt(m, "num_ctx"); ok && n > 0 {
		e.MaxContext.set(n, src)
	}
	if n, ok := intOpt(m, "num_batch"); ok && n > 0 {
		e.NBatch.set(n, src)
	}
	if n, ok := intOpt(m, "num_thread"); ok && n > 0 {
		e.Threads.set(n, src)
	}
	if !fork {
		return
	}
	if b, ok := boolOpt(m, "dual_buffer"); ok {
		e.DualBuffer.set(b, src)
	}
	if b, ok := boolOpt(m, "dynamic_kv"); ok {
		e.DynamicKV.set(b, src)
	}
	if b, ok := boolOpt(m, "kv_save"); ok {
		e.KVSave.set(b, src)
	}
	if s, ok := m["zones"].(string); ok && (s == ZonesAuto || s == ZonesOff) {
		e.Zones.set(s, src)
	}
	if s, ok := m["kv_type"].(string); ok && s != "" {
		e.KVType.set(strings.ToLower(s), src)
	}
	if n, ok := intOpt(m, "parallel"); ok && n > 0 {
		e.Parallel.set(n, src)
	}
	if n, ok := intOpt(m, "n_ubatch"); ok && n > 0 {
		e.NUbatch.set(n, src)
	}
	if v, ok := m["flash_attention"]; ok {
		var s string
		switch t := v.(type) {
		case string:
			s = t
		case bool:
			s = strconv.FormatBool(t)
		}
		if fa, err := normalizeFlashAttention(s); err == nil {
			e.FlashAttention.set(fa, src)
		}
	}
}

func intOpt(m map[string]any, k string) (int, bool) {
	switch t := m[k].(type) {
	case int:
		return t, true
	case int64:
		return int(t), true
	case float64:
		return int(t), true
	case string:
		n, err := strconv.Atoi(t)
		return n, err == nil
	}
	return 0, false
}

func boolOpt(m map[string]any, k string) (bool, bool) {
	switch t := m[k].(type) {
	case bool:
		return t, true
	case string:
		b, err := strconv.ParseBool(t)
		return b, err == nil
	}
	return false, false
}

func truthy(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "on", "enabled", "yes":
		return true
	}
	return false
}

// String is the one line "effective settings" form: every field with its value and source.
func (e Effective) String() string {
	var b strings.Builder
	w := func(k, v, src string) {
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "%s=%s(%s)", k, v, src)
	}
	devs := "-"
	if len(e.Devices.V) > 0 {
		devs = "[" + strings.Join(e.Devices.V, ",") + "]"
	}
	w("devices", devs, e.Devices.Src)
	w("on_busy", e.OnBusy.V, e.OnBusy.Src)
	if e.OnBusy.V == OnBusyWait && e.WaitTimeout.V == time.Duration(math.MaxInt64) {
		w("wait_timeout", "-1", e.WaitTimeout.Src)
	} else if e.OnBusy.V == OnBusyWait {
		w("wait_timeout", e.WaitTimeout.V.String(), e.WaitTimeout.Src)
	} else {
		w("wait_timeout", "ignored", e.WaitTimeout.Src)
	}
	w("zones", e.Zones.V, e.Zones.Src)
	w("dual_buffer", strconv.FormatBool(e.DualBuffer.V), e.DualBuffer.Src)
	w("dynamic_kv", strconv.FormatBool(e.DynamicKV.V), e.DynamicKV.Src)
	kv := e.KVType.V
	if kv == "" {
		kv = "f16"
	}
	w("kv_type", kv, e.KVType.Src)
	mc := "auto"
	if e.MaxContext.V > 0 {
		mc = strconv.Itoa(e.MaxContext.V)
	}
	w("max_context", mc, e.MaxContext.Src)
	w("keep_alive", e.KeepAlive.V.String(), e.KeepAlive.Src)
	w("parallel", strconv.Itoa(e.Parallel.V), e.Parallel.Src)
	th := "auto"
	if e.Threads.V > 0 {
		th = strconv.Itoa(e.Threads.V)
	}
	w("threads", th, e.Threads.Src)
	w("n_batch", strconv.Itoa(e.NBatch.V), e.NBatch.Src)
	ub := "n_batch"
	if e.NUbatch.V > 0 {
		ub = strconv.Itoa(e.NUbatch.V)
	}
	w("n_ubatch", ub, e.NUbatch.Src)
	w("flash_attention", e.FlashAttention.V, e.FlashAttention.Src)
	w("kv_save", strconv.FormatBool(e.KVSave.V), e.KVSave.Src)
	return b.String()
}
