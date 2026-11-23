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
	"sort"
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
