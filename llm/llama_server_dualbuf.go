package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Dual-buffer fork (multimodel): per model llama-server settings and what the scheduler reads back.

// applyArgOverrides replaces the value(s) after a flag Ollama already passes, or adds the flag with its
// values. The flags are the forms Ollama itself uses (-ub, --flash-attn, -t, ...).
func applyArgOverrides(params []string, overrides map[string][]string) []string {
	if len(overrides) == 0 {
		return params
	}
	flags := make([]string, 0, len(overrides))
	for f := range overrides {
		flags = append(flags, f)
	}
	sort.Strings(flags)
	for _, flag := range flags {
		vals := overrides[flag]
		replaced := false
		for i := 0; i+len(vals) < len(params); i++ {
			if params[i] == flag {
				copy(params[i+1:i+1+len(vals)], vals)
				replaced = true
				break
			}
		}
		if !replaced {
			params = append(params, flag)
			params = append(params, vals...)
		}
	}
	return params
}

// OffloadsToCPU reports whether the loaded model also runs on the CPU: not every layer is on the GPU, or
// the fit moved parts of GPU layers to the CPU (MoE experts). A CPU-only runner reports false.
func (s *llamaServerRunner) OffloadsToCPU() bool {
	if len(s.gpus) == 0 {
		return false
	}
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()
	if s.totalLayers == 0 {
		return true
	}
	return s.gpuLayers < s.totalLayers || s.gpuLayerOverflow > 0
}

// SlotFile is one saved or restored llama-server slot.
type SlotFile struct {
	Slot   int    `json:"slot"`
	File   string `json:"file"`
	Tokens int    `json:"tokens"`
	Bytes  int64  `json:"bytes"`
}

// SlotSavePath is the llama-server --slot-save-path of this runner ("" = slot saving is off).
func (s *llamaServerRunner) SlotSavePath() string {
	return s.launch.config.ExtraEnv["LLAMA_ARG_SLOT_SAVE_PATH"]
}

// NumParallel is the number of llama-server slots.
func (s *llamaServerRunner) NumParallel() int {
	return s.launch.numParallel
}

// SaveSlots saves every slot with tokens to <slot save path>/<prefix>-slot<i>.bin (POST /slots/<i>?action=save).
// A slot without tokens is skipped (its file is removed).
func (s *llamaServerRunner) SaveSlots(ctx context.Context, prefix string) ([]SlotFile, error) {
	return s.slotAction(ctx, "save", prefix)
}

// RestoreSlots loads the slot files written by SaveSlots with the same prefix; missing files are skipped.
func (s *llamaServerRunner) RestoreSlots(ctx context.Context, prefix string) ([]SlotFile, error) {
	return s.slotAction(ctx, "restore", prefix)
}

func (s *llamaServerRunner) slotAction(ctx context.Context, action, prefix string) ([]SlotFile, error) {
	dir := s.SlotSavePath()
	if dir == "" {
		return nil, errors.New("slot saving is not enabled for this runner")
	}
	var out []SlotFile
	var errs []error
	for i := 0; i < max(s.launch.numParallel, 1); i++ {
		name := fmt.Sprintf("%s-slot%d.bin", prefix, i)
		path := filepath.Join(dir, name)
		if action == "restore" {
			if _, err := os.Stat(path); err != nil {
				continue
			}
		}
		body, _ := json.Marshal(map[string]string{"filename": name})
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/slots/%d?action=%s", s.port, i, action), bytes.NewReader(body))
		if err != nil {
			return out, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := s.httpClient().Do(req)
		if err != nil {
			errs = append(errs, fmt.Errorf("slot %d %s: %w", i, action, err))
			continue
		}
		data, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			errs = append(errs, fmt.Errorf("slot %d %s: %s: %s", i, action, resp.Status, strings.TrimSpace(string(data))))
			continue
		}
		var r struct {
			NSaved    int   `json:"n_saved"`
			NWritten  int64 `json:"n_written"`
			NRestored int   `json:"n_restored"`
			NRead     int64 `json:"n_read"`
		}
		_ = json.Unmarshal(data, &r)
		f := SlotFile{Slot: i, File: path, Tokens: r.NSaved + r.NRestored, Bytes: r.NWritten + r.NRead}
		if action == "save" && f.Tokens == 0 {
			_ = os.Remove(path)
			continue
		}
		out = append(out, f)
	}
	return out, errors.Join(errs...)
}

// zonesLive is the device memory of a runner as the fork's zones report it after every change (dynamic KV growth
// and shrink, flex layer moves): the weights on the GPU, the KV on the device and the parts of the memory that keep
// their size. Ollama otherwise only knows the buffers of the load.
type zonesLive struct {
	seen    bool
	weights uint64
	kv      uint64
	fixed   uint64
}

var (
	zonesStateRegex = regexp.MustCompile(`zones_state \([^)]*\): weights GPU (\d+) MiB.*?KV \d+ cells \(\d+ used, (\d+) MiB on the device\)(?:; fixed on the device: ([^;]*))?`)
	mibRegex        = regexp.MustCompile(`(\d+) MiB`)
)

// parseZonesStateLocked reads the last zones_state line of b; true if there was one.
func (s *llamaServerRunner) parseZonesStateLocked(b []byte) bool {
	all := zonesStateRegex.FindAllSubmatch(b, -1)
	if len(all) == 0 {
		return false
	}
	m := all[len(all)-1]
	w, _ := strconv.ParseUint(string(m[1]), 10, 64)
	kv, _ := strconv.ParseUint(string(m[2]), 10, 64)
	var fixed uint64
	for _, f := range mibRegex.FindAllSubmatch(m[3], -1) {
		n, _ := strconv.ParseUint(string(f[1]), 10, 64)
		fixed += n
	}
	s.dualbufZones = zonesLive{seen: true, weights: w << 20, kv: kv << 20, fixed: fixed << 20}
	return true
}

// applyZonesStateLocked replaces the device total of a single-GPU runner with the live one: the zones' weights,
// KV and fixed parts plus the other device buffers of the load (compute, output, inputs).
func (s *llamaServerRunner) applyZonesStateLocked(buffers map[memoryBufferKey]memoryBuffer) {
	z := s.dualbufZones
	if !z.seen || len(s.gpus) != 1 {
		return
	}
	dev := s.gpus[0].Name
	var other uint64
	for key, b := range buffers {
		if isGPUBuffer(key.backend) && deviceName(key.backend) == dev && key.kind != "model" && key.kind != "KV" && key.kind != "RS" {
			other += b.bytes
		}
	}
	live := z.weights + z.kv + z.fixed + other
	old := s.vramByDevice[dev]
	s.vramByDevice[dev] = live
	s.memGPU = s.memGPU - min(s.memGPU, old) + live
}
