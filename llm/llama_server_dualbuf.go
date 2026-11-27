package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/ml"
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
	if s.dualbufPark.parked {
		other = 0 // parked: the compute buffers of the load are freed too
	}
	live := z.weights + z.kv + z.fixed + other
	old := s.vramByDevice[dev]
	s.vramByDevice[dev] = live
	s.memGPU = s.memGPU - min(s.memGPU, old) + live
}

// in-process park (llama-server --park, POST /park and /unpark): the process stays, the device memory is freed

var (
	parkCopiesRegex = regexp.MustCompile(`park: \d+ GPU weights in \d+ groups have a pinned GPU-format copy`)
	parkNoRegex     = regexp.MustCompile(`park: .*no in-process park|park: --park requires`)
	demotedRegex    = regexp.MustCompile(`park: demoted ([\d.]+) MiB`)
	droppedRegex    = regexp.MustCompile(`park: the demoted \(repacked\) copies were dropped`)
	hostMemRegex    = regexp.MustCompile(`host memory: [\d.]+ MiB in backend host buffers, ([\d.]+) MiB not charged to the cgroup`)
)

// parseParkLocked notes from the log whether the instance can park in process, and the RAM of its demoted copies.
func (s *llamaServerRunner) parseParkLocked(b []byte) {
	if parkCopiesRegex.Match(b) {
		s.dualbufPark.copies = true
	}
	if parkNoRegex.Match(b) {
		s.dualbufPark.refused = true
	}
	if m := demotedRegex.FindSubmatch(b); m != nil {
		if mib, err := strconv.ParseFloat(string(m[1]), 64); err == nil {
			s.dualbufPark.demoted += uint64(mib * 1024 * 1024)
		}
	}
	if droppedRegex.Match(b) {
		s.dualbufPark.demoted = 0
	}
	if all := hostMemRegex.FindAllSubmatch(b, -1); len(all) > 0 {
		if mib, err := strconv.ParseFloat(string(all[len(all)-1][1]), 64); err == nil {
			s.dualbufPark.hostUncharged = uint64(mib * 1024 * 1024)
			// the RAM plan adds this to the cgroup's usage (it is in the system's used memory already)
			slog.Info("dualbuf: runner driver host memory not seen by the memory cgroup", "uncharged_mib", int64(mib))
		}
	}
}

// dualbufParkState is the in-process park state of a runner.
type dualbufParkState struct {
	copies  bool   // the load made the pinned copies
	refused bool   // --park was asked, the copies did not fit
	parked  bool   // POST /park succeeded, no /unpark since
	vram    uint64 // the device memory before the park (to check the room for the unpark)
	demoted uint64 // bytes of the repacked (demoted) copies the instance holds, from the last park/unpark/drop answer
	// driver host memory the memory cgroup does not see (Vulkan host buffers): from the load log and the park answers
	hostUncharged uint64
}

// HostUncharged is the RAM the runner holds in driver host allocations that the cgroup's usage does not show (the
// llama-server measures it; 0 for ROCm/CUDA, whose pinned host memory the cgroup sees).
func (s *llamaServerRunner) HostUncharged() uint64 {
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()
	return s.dualbufPark.hostUncharged
}

// DemotedBytes is the RAM of the repacked copies the parked model made for the CPU (0: none).
func (s *llamaServerRunner) DemotedBytes() uint64 {
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()
	return s.dualbufPark.demoted
}

// DropDemoted frees the repacked copies (POST /park/drop); the parked model then computes from its pinned copies.
func (s *llamaServerRunner) DropDemoted(ctx context.Context) (uint64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/park/drop", s.port), nil)
	if err != nil {
		return 0, err
	}
	resp, err := s.httpClient().Do(req)
	if err != nil {
		return 0, err
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	var r struct {
		Demoted uint64 `json:"demoted_bytes"`
	}
	_ = json.Unmarshal(data, &r)
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("park/drop: %s: %s", resp.Status, strings.TrimSpace(string(data)))
	}
	s.memoryMu.Lock()
	s.dualbufPark.demoted = 0
	s.memoryMu.Unlock()
	return r.Demoted, nil
}

// CanPark reports whether the runner was loaded with park copies.
func (s *llamaServerRunner) CanPark() bool {
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()
	return s.dualbufPark.copies && !s.dualbufPark.refused
}

// ParkRefused reports whether --park was asked and the pinned copies did not fit the RAM.
func (s *llamaServerRunner) ParkRefused() bool {
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()
	return s.dualbufPark.refused
}

// Parked reports whether the runner is parked.
func (s *llamaServerRunner) Parked() bool {
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()
	return s.dualbufPark.parked
}

// ParkedVRAM is the device memory the runner had before its park.
func (s *llamaServerRunner) ParkedVRAM() uint64 {
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()
	return s.dualbufPark.vram
}

// Park frees the device memory of the instance (POST /park); the process stays and can still run on the CPU.
func (s *llamaServerRunner) Park(ctx context.Context) (time.Duration, error) {
	return s.parkAction(ctx, true)
}

// Unpark loads the weights and the K/V back to the device (POST /unpark).
func (s *llamaServerRunner) Unpark(ctx context.Context) (time.Duration, error) {
	return s.parkAction(ctx, false)
}

func (s *llamaServerRunner) parkAction(ctx context.Context, park bool) (time.Duration, error) {
	path := "unpark"
	if park {
		path = "park"
	}
	var vram uint64
	if park {
		s.memoryMu.RLock()
		vram = s.memGPU
		s.memoryMu.RUnlock()
	}
	start := time.Now()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/%s", s.port, path), nil)
	if err != nil {
		return 0, err
	}
	resp, err := s.httpClient().Do(req)
	if err != nil {
		return 0, err
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	var r struct {
		Success  bool    `json:"success"`
		Parked   bool    `json:"parked"`
		Demoted  uint64  `json:"demoted_bytes"`
		HostUnch *uint64 `json:"host_uncharged_bytes"`
	}
	_ = json.Unmarshal(data, &r)
	took := time.Since(start)
	s.memoryMu.Lock()
	s.dualbufPark.parked = r.Parked
	s.dualbufPark.demoted = r.Demoted
	if r.HostUnch != nil {
		s.dualbufPark.hostUncharged = *r.HostUnch
	}
	if park && r.Parked {
		s.dualbufPark.vram = vram
	}
	if s.output != nil && s.output.buffers != nil {
		s.output.updateRunnerMemoryLocked()
	}
	s.memoryMu.Unlock()
	if resp.StatusCode != http.StatusOK || !r.Success {
		return took, fmt.Errorf("%s: %s: %s", path, resp.Status, strings.TrimSpace(string(data)))
	}
	return took, nil
}

// Move puts the model's GPU part on another GPU of the process (POST /move): parked first if it is not, the K/V goes
// over through the host, the GPU weights load from the model file onto the new device, the CPU layers stay. to is the
// target's discovered device; its PCI bus id names it to llama-server (else its name). The runner's device memory is
// accounted on the new device from then on.
func (s *llamaServerRunner) Move(ctx context.Context, to ml.DeviceInfo) (time.Duration, error) {
	target := to.PCIID
	if target == "" {
		target = to.Name
	}
	body, _ := json.Marshal(map[string]string{"device": target})
	start := time.Now()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/move", s.port), bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.httpClient().Do(req)
	if err != nil {
		return 0, err
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	var r struct {
		Success  bool    `json:"success"`
		Parked   bool    `json:"parked"`
		Device   string  `json:"device"`
		From     string  `json:"from"`
		Message  string  `json:"message"`
		Demoted  uint64  `json:"demoted_bytes"`
		HostUnch *uint64 `json:"host_uncharged_bytes"`
	}
	_ = json.Unmarshal(data, &r)
	took := time.Since(start)
	s.memoryMu.Lock()
	s.dualbufPark.parked = r.Parked
	s.dualbufPark.demoted = r.Demoted
	if r.HostUnch != nil {
		s.dualbufPark.hostUncharged = *r.HostUnch
	}
	if r.Device != "" && r.Device != r.From {
		// the device the model is on now: its buffers and the VRAM accounting go there
		moved := to
		moved.Name = r.Device
		s.gpus = []ml.DeviceInfo{moved}
		if s.output != nil && s.output.buffers != nil {
			renameDeviceBuffers(s.output.buffers, r.From, r.Device)
		}
		if old, ok := s.vramByDevice[r.From]; ok {
			delete(s.vramByDevice, r.From)
			s.vramByDevice[r.Device] = old
		}
	}
	if s.output != nil && s.output.buffers != nil {
		s.output.updateRunnerMemoryLocked()
	}
	s.memoryMu.Unlock()
	if resp.StatusCode != http.StatusOK || !r.Success {
		return took, fmt.Errorf("move to %s: %s: %s", target, resp.Status, strings.TrimSpace(string(data)))
	}
	return took, nil
}

// renameDeviceBuffers moves the parsed buffer sizes of a device to its new name (a move to another GPU).
func renameDeviceBuffers(buffers map[memoryBufferKey]memoryBuffer, from, to string) {
	if from == "" || to == "" || from == to {
		return
	}
	for key, b := range buffers {
		if deviceName(key.backend) != from {
			continue
		}
		delete(buffers, key)
		nk := key
		nk.backend = to + strings.TrimPrefix(key.backend, from)
		buffers[nk] = b
	}
}

// dualbufExtraDevices: a model that can move between GPUs loads the backends of the other GPUs of its list too
// (llama-server --park, POST /move). Every card is visible to its own backend only, and the active device's backend
// comes first (it is device 0 for --main-gpu 0). Without extra devices it changes nothing.
func dualbufExtraDevices(gpus, extra []ml.DeviceInfo, gpuLibs []string, env map[string]string) ([]string, map[string]string) {
	if len(extra) == 0 || len(gpus) == 0 {
		return gpuLibs, env
	}
	all := append(slices.Clone(gpus), extra...)
	gpuLibs = ml.LibraryPaths(all)
	out := map[string]string{}
	// ROCm filters by itself (and keeps its ordinals); Vulkan and CUDA are filtered one device at a time
	var rocm []ml.DeviceInfo
	for _, d := range all {
		if d.Library == "ROCm" {
			rocm = append(rocm, d)
		}
	}
	for k, v := range ml.GetDevicesEnv(rocm) {
		out[k] = v
	}
	for _, d := range all {
		if d.Library == "ROCm" {
			continue
		}
		for k, v := range ml.GetDevicesEnv([]ml.DeviceInfo{d}) {
			if old, ok := out[k]; ok && (k == "GGML_VK_VISIBLE_DEVICES" || k == "CUDA_VISIBLE_DEVICES") {
				out[k] = old + "," + v
			} else {
				out[k] = v
			}
		}
	}
	var backends []string
	for _, dir := range gpuLibs {
		if dir == ml.LibOllamaPath {
			continue
		}
		if b := findLlamaServerGPUBackend(dir); b != "" && !slices.Contains(backends, b) {
			backends = append(backends, b)
		}
	}
	if len(backends) > 0 {
		out["GGML_BACKEND_PATH"] = strings.Join(backends, string(filepath.ListSeparator))
	}
	return gpuLibs, out
}

// dualbufLogROCmView logs the ROCm view a runner starts with (INFO): the hidden render nodes, the gfx override
// variables, the visible ROCm devices and the preload library, as the runtime gets them. Nothing without a view.
func dualbufLogROCmView(model string, env map[string]string) {
	if _, ok := env["GGML_HIP_HIDE_RENDER_NODES"]; !ok {
		return
	}
	var parts []string
	for k, v := range env {
		if strings.HasPrefix(k, "HSA_OVERRIDE_GFX_VERSION") || k == "ROCR_VISIBLE_DEVICES" || k == "HIP_VISIBLE_DEVICES" || k == "LD_PRELOAD" {
			parts = append(parts, k+"="+v)
		}
	}
	sort.Strings(parts)
	slog.Info("dualbuf: ROCm view of the runner", "model", model, "hidden_render_nodes", env["GGML_HIP_HIDE_RENDER_NODES"], "env", strings.Join(parts, " "))
}
