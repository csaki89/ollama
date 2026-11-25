package server

// Dual-buffer fork: the multi-model rules of <models>/dualbuf.yaml (docs/MULTIMODEL-DESIGN.md). The
// scheduler calls these from a few hooks; without a config file each of them is a no-op.

import (
	"context"
	"log/slog"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/multimodel"
)

// dualbufLoad travels on the request's copy of the Model: the config snapshot, the resolved settings and
// the device the request got.
type dualbufLoad struct {
	cfg         *multimodel.Config
	eff         multimodel.Effective
	launch      multimodel.Launch
	placement   string // config device name, "" = Ollama places the model
	firstDevice string // the first present device of the model's list
	viaParked   bool   // a CPU request served by the parked GPU instance of the model (no unpark, no CPU instance)
}

func newDualbufManager() *multimodel.Manager {
	return multimodel.NewManager(multimodel.DefaultPath(envconfig.Var, envconfig.Models()), envconfig.Var)
}

func dualbufModelName(m *Model) string {
	if m == nil {
		return ""
	}
	if m.ShortName != "" {
		return m.ShortName
	}
	return m.Name
}

func dualbufFingerprint(m *Model) string {
	if m == nil || m.dualbuf == nil {
		return ""
	}
	return m.dualbuf.launch.Fingerprint() + "@" + m.dualbuf.placement
}

// schedulerModelKey: one instance per model and placed device (the CPU instance of a GPU model is another
// runner); without a placement the key is Ollama's.
func schedulerModelKey(m *Model) string {
	key := schedulerModelKeyBase(m)
	if key != "" && m.dualbuf != nil && m.dualbuf.placement != "" {
		key += "\x00@" + m.dualbuf.placement
	}
	return key
}

// dualbufConfigFor is the config a request resolves with: while the model is loaded, the snapshot it was
// loaded with (a changed file applies from the next load and never reloads a running model), else the
// current file.
func (s *Scheduler) dualbufConfigFor(m *Model) *multimodel.Config {
	if s == nil || s.dualbuf == nil || m == nil {
		return nil
	}
	s.loadedMu.Lock()
	for _, r := range s.loaded {
		if lm := r.model.Load(); lm != nil && lm.ModelPath != "" && lm.ModelPath == m.ModelPath {
			s.loadedMu.Unlock()
			if lm.dualbuf == nil {
				return nil
			}
			return lm.dualbuf.cfg
		}
	}
	s.loadedMu.Unlock()
	return s.dualbuf.Config()
}

// dualbufPrepare resolves the settings of a request and takes its place on a device (it may wait there,
// see multimodel.Coordinator). It returns the model copy to schedule, the keep alive to use and the grant to
// release when the request ends (nil without a config or a device list).
func (s *Scheduler) dualbufPrepare(ctx context.Context, m *Model, requestOpts map[string]any, keepAlive *api.Duration, opts *api.Options) (*Model, *api.Duration, *multimodel.Grant, error) {
	cfg := s.dualbufConfigFor(m)
	if cfg == nil || m.ModelPath == "" {
		return m, keepAlive, nil, nil
	}
	name := dualbufModelName(m)
	var reqKA *time.Duration
	if keepAlive != nil {
		d := keepAlive.Duration
		reqKA = &d
	}
	eff := cfg.Resolve(name, envconfig.Var, m.Options, requestOpts, reqKA)
	db := &dualbufLoad{cfg: cfg, eff: eff, launch: eff.Launch()}

	var grant *multimodel.Grant
	if len(eff.Devices.V) > 0 {
		devices := s.dualbuf.Present(cfg, eff.Devices.V)
		if len(devices) == 0 {
			slog.Warn("dualbuf: none of the model's devices was found, Ollama places it", "model", name, "devices", eff.Devices.V)
		} else {
			g, err := s.dualbuf.Coord.Acquire(ctx, multimodel.Request{Model: name, Devices: devices, OnBusy: eff.OnBusy.V, WaitTimeout: eff.WaitTimeout.V})
			if err != nil {
				return nil, nil, nil, err
			}
			grant = g
			db.placement = g.Device
			db.firstDevice = devices[0]
			if g.Device == multimodel.DeviceCPU {
				if pr := s.dualbufParkedFor(m); pr != nil {
					// the parked GPU instance runs it on the CPU, without a reload
					if lm := pr.model.Load(); lm != nil && lm.dualbuf != nil {
						db.placement = lm.dualbuf.placement
						db.viaParked = true
						slog.Info("dualbuf: the CPU request runs on the parked instance", "model", name, "instance", db.placement)
					}
				} else {
					opts.NumGPU = 0
				}
			}
		}
	}
	mm := *m
	mm.dualbuf = db
	if keepAlive == nil && multimodel.FromConfig(eff.KeepAlive.Src) {
		keepAlive = &api.Duration{Duration: eff.KeepAlive.V.Duration}
	}
	return &mm, keepAlive, grant, nil
}

type cpuOffloader interface{ OffloadsToCPU() bool }

// dualbufStarted: a GPU instance that also runs on the CPU takes a CPU place too (after the GPU place, so no
// wait goes round), and the places are given back when the request ends.
func (s *Scheduler) dualbufStarted(ctx context.Context, m *Model, llama llm.LlamaServer, grant *multimodel.Grant) {
	if grant == nil {
		return
	}
	if m != nil && m.dualbuf != nil && m.dualbuf.placement != "" && m.dualbuf.placement != multimodel.DeviceCPU && m.dualbuf.cfg.MaxRunning(multimodel.DeviceCPU) > 0 {
		if o, ok := llama.(cpuOffloader); ok && o.OffloadsToCPU() {
			if err := grant.Also(ctx, multimodel.DeviceCPU); err != nil {
				slog.Debug("dualbuf: request ended while waiting for the CPU place", "error", err)
			}
		}
	}
	grant.ReleaseOnDone(ctx)
}

// dualbufPlacementGPUs picks the device the request was placed on from the discovered devices.
func (s *Scheduler) dualbufPlacementGPUs(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, predicted uint64) ([]ml.DeviceInfo, api.Options) {
	db := req.model.dualbuf
	opts := req.opts
	if db.placement == multimodel.DeviceCPU {
		opts.NumGPU = 0
		return []ml.DeviceInfo{}, opts
	}
	dm := s.dualbuf.Devices(db.cfg)
	if want, ok := dm.ByName[db.placement]; ok {
		for _, g := range gpus {
			if g.DeviceID == want.Info.DeviceID {
				sel, o := singleLlamaServerGPUPlacement(g, opts)
				slog.Info("dualbuf: model placed on its device", "model", dualbufModelName(req.model), "device", db.placement,
					"id", g.ID, "library", g.Library, "description", g.Description, "pci_id", g.PCIID, "free", format.HumanBytes2(g.FreeMemory))
				return sel, o
			}
		}
	}
	slog.Warn("dualbuf: the placed device is not among the discovered GPUs, Ollama places the model", "model", dualbufModelName(req.model), "device", db.placement)
	return selectLlamaServerPlacement(systemInfo, gpus, predicted, req.opts)
}

// dualbufNumParallel is the parallel setting of the model's config (completion models only).
func dualbufNumParallel(req *LlmRequest, numParallel int, completion bool) int {
	if db := req.model.dualbuf; db != nil && db.launch.Parallel > 0 && completion {
		return db.launch.Parallel
	}
	return numParallel
}

// dualbufServerConfig adds the per model llama-server settings and logs the effective ones.
// dualbufHostUncharged sums the driver host memory of the loaded runners that the memory cgroup does not see
// (llama-server: "host memory: ... not charged to the cgroup"), except the runner except.
func (s *Scheduler) dualbufHostUncharged(except *runnerRef) uint64 {
	type uncharged interface{ HostUncharged() uint64 }
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()
	var sum uint64
	for _, r := range runners {
		if r == except || r.llama == nil {
			continue
		}
		if u, ok := r.llama.(uncharged); ok {
			sum += u.HostUncharged()
		}
	}
	return sum
}

func (s *Scheduler) dualbufServerConfig(req *LlmRequest, config *llm.LlamaServerConfig) {
	db := req.model.dualbuf
	if db == nil {
		return
	}
	config.ExtraEnv = map[string]string{}
	for k, v := range db.launch.Env {
		config.ExtraEnv[k] = v
	}
	// the other runners' uncharged driver host memory for the llama-server RAM guard (its own it measures itself)
	if unch := s.dualbufHostUncharged(nil); unch > 0 {
		config.ExtraEnv["LLAMA_ARG_RAM_UNCHARGED_MIB"] = strconv.FormatUint(unch>>20, 10)
	}
	config.ArgOverrides = db.launch.Args
	config.KVCacheType = db.launch.KVType
	if db.cfg.SwapEnabled() && db.cfg.Swap.KeepInRAM == multimodel.KeepInRAMPinned && db.placement != multimodel.DeviceCPU {
		// in-process swap: pinned GPU-format copies of the GPU weights (llama-server POST /park, /unpark)
		config.ExtraEnv["LLAMA_ARG_PARK"] = "1"
	}
	if db.eff.KVSave.V || (db.cfg.SwapEnabled() && db.cfg.SwapSaveKV()) {
		dir := multimodel.RAMDir()
		if err := os.MkdirAll(dir, 0o755); err != nil {
			slog.Warn("dualbuf: no KV directory, KV saving is off", "dir", dir, "error", err)
		} else {
			config.ExtraEnv["LLAMA_ARG_SLOT_SAVE_PATH"] = dir
		}
	}
	slog.Info("dualbuf effective settings", "model", db.eff.Model, "section", db.eff.Section, "placement", db.placement, "settings", db.eff.String())
	for _, n := range db.launch.Notes {
		slog.Warn("dualbuf setting", "model", db.eff.Model, "note", n)
	}
}

func dualbufKVType(m *Model) string {
	if m != nil && m.dualbuf != nil && m.dualbuf.launch.KVType != "" {
		return m.dualbuf.launch.KVType
	}
	return envconfig.KvCacheType()
}

func dualbufPrefix(m *Model) string {
	placement := ""
	if m.dualbuf != nil {
		placement = m.dualbuf.placement
	}
	return multimodel.KVPrefix(m.Digest, placement, dualbufKVType(m))
}

type slotSaver interface {
	SaveSlots(ctx context.Context, prefix string) ([]llm.SlotFile, error)
	RestoreSlots(ctx context.Context, prefix string) ([]llm.SlotFile, error)
	SlotSavePath() string
}

// dualbufRoom makes RAM room for a load (docs/MULTIMODEL-DESIGN.md 2.7): parked models are dropped at once;
// when a loaded model has to go, it becomes the request's victim and the function returns true (the scheduler
// then expires it and tries again).
func (s *Scheduler) dualbufRoom(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo) bool {
	db := req.model.dualbuf
	if db == nil || s.dualbuf == nil {
		return false
	}
	key := schedulerModelKey(req.model)
	size := modelFileSize(req.model.modelPaths()...)
	need := size
	if db.placement != multimodel.DeviceCPU && len(gpus) > 0 {
		var free uint64
		for _, g := range gpus {
			free += g.FreeMemory
		}
		need = size - min(size, free) + format.GigaByte
		if db.eff.DualBuffer.V {
			need *= 2
		}
	}
	need = s.dualbuf.HostEstimate(key, need)
	ram := multimodel.PlanRAM(db.cfg, systemInfo.TotalMemory, systemInfo.FreeMemory)
	// the runners' driver host memory the cgroup does not see (Vulkan host buffers) is used all the same; a system reading has
	// it already (also under a ram_limit: the limit replaces the cgroup's, not the usage)
	if unch := s.dualbufHostUncharged(nil); unch > 0 && ram.UsedFrom == "cgroup" {
		ram.Used += unch
		ram.Source += " + uncharged driver host memory"
	}
	if v := envconfig.Var("OLLAMA_DUALBUF_RAM_FREE_MIB"); v != "" {
		// test aid (like the llama-server LLAMA_DUAL_BUFFER_FREE_MIB): the free RAM the plan assumes
		if mib, err := strconv.ParseUint(v, 10, 64); err == nil {
			ram.Limit, ram.Source = ram.Used+mib<<20, "env override"
		}
	}
	// a reserve like the llama-server RAM guard: the eviction comes before the cgroup limit is reached
	need += multimodel.RAMReserve(ram.Limit)

	var cands []multimodel.Candidate
	for _, p := range s.dualbuf.ParkedList() {
		if p.Key == key {
			continue
		}
		cands = append(cands, multimodel.Candidate{Key: p.Key, Model: p.Model, Kind: multimodel.KindParked, LastUsed: p.LastUsed, Forever: p.Forever, Bytes: p.Bytes})
	}
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()
	for _, r := range runners {
		if r.modelKey == key {
			continue
		}
		r.refMu.Lock()
		idle := r.refCount == 0 && !r.loading
		forever := r.sessionDuration == time.Duration(math.MaxInt64)
		host := r.totalSize - min(r.totalSize, r.vramSize)
		r.refMu.Unlock()
		if !idle {
			continue
		}
		kind := multimodel.KindIdle
		if runnerParked(r) {
			kind = multimodel.KindParked
		} else if lm := r.model.Load(); lm != nil && lm.dualbuf != nil && lm.dualbuf.placement == multimodel.DeviceCPU && lm.dualbuf.firstDevice != multimodel.DeviceCPU {
			kind = multimodel.KindCPUFallback
		}
		cands = append(cands, multimodel.Candidate{Key: r.modelKey, Model: r.modelPath, Kind: kind, LastUsed: time.Unix(0, r.dualbufLastUsed.Load()), Forever: forever, Bytes: host, Ref: r})
		// the repacked (demoted) copies a parked model made for the CPU: freed first, the model stays
		if d, ok := r.llama.(demoter); ok && d.DemotedBytes() > 0 {
			cands = append(cands, multimodel.Candidate{Key: r.modelKey + "#demoted", Model: r.modelPath, Kind: multimodel.KindCPUFallback,
				LastUsed: time.Unix(0, r.dualbufLastUsed.Load()), Bytes: d.DemotedBytes(), Ref: demotedCopies{r}})
		}
	}
	plan := multimodel.PlanEvictions(ram, need, cands)
	if len(plan) == 0 {
		return false
	}
	slog.Info("dualbuf: RAM is short for the load, freeing", "model", dualbufModelName(req.model), "need", format.HumanBytes2(need),
		"limit", format.HumanBytes2(ram.Limit), "used", format.HumanBytes2(ram.Used), "source", ram.Source, "candidates", len(plan))
	for _, c := range plan {
		if dc, ok := c.Ref.(demotedCopies); ok {
			if d, ok := dc.r.llama.(demoter); ok {
				ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
				freed, err := d.DropDemoted(ctx)
				cancel()
				slog.Info("dualbuf: demoted copies dropped for RAM", "model", c.Model, "freed", format.HumanBytes2(freed), "error", err)
			}
			continue
		}
		if c.Kind == multimodel.KindParked && c.Ref == nil {
			s.dualbuf.Drop(c.Key)
			continue
		}
		if r, ok := c.Ref.(*runnerRef); ok {
			slog.Info("dualbuf: unloading for RAM", "model", c.Model, "kind", c.Kind.String(), "ram", format.HumanBytes2(c.Bytes))
			req.dualbufVictim = r
			return true
		}
	}
	return false
}

// dualbufVictim is the runner to expire for a pending request: the RAM victim chosen by dualbufRoom, else
// (the model does not fit on its device) the least recently used idle model on the same device, which is
// parked when swap is on; otherwise Ollama's choice.
func (s *Scheduler) dualbufVictim(req *LlmRequest) *runnerRef {
	if v := req.dualbufVictim; v != nil {
		req.dualbufVictim = nil
		return v
	}
	db := req.model.dualbuf
	if db == nil || db.placement == "" || db.placement == multimodel.DeviceCPU {
		return s.findRunnerToUnload()
	}
	s.loadedMu.Lock()
	var same []*runnerRef
	for _, r := range s.loaded {
		if lm := r.model.Load(); lm != nil && lm.dualbuf != nil && lm.dualbuf.placement == db.placement && !runnerParked(r) {
			same = append(same, r)
		}
	}
	s.loadedMu.Unlock()
	var pick *runnerRef
	pickIdle := false
	for _, r := range same {
		r.refMu.Lock()
		idle := r.refCount == 0
		r.refMu.Unlock()
		if pick == nil || (idle && !pickIdle) || (idle == pickIdle && r.dualbufLastUsed.Load() < pick.dualbufLastUsed.Load()) {
			pick, pickIdle = r, idle
		}
	}
	if pick == nil {
		return s.findRunnerToUnload()
	}
	if db.cfg.SwapEnabled() {
		pick.dualbufPark.Store(true)
	}
	slog.Info("dualbuf: making room on the device", "device", db.placement, "for", dualbufModelName(req.model), "victim", pick.modelPath, "idle", pickIdle, "park", db.cfg.SwapEnabled())
	return pick
}

// dualbufBeforeUnload saves what has to survive an unload: on a swap (park) the KV goes to RAM and the model
// file is pinned; on a full unload the KV goes to the NVMe when kv_save is on for the model.
func (s *Scheduler) dualbufBeforeUnload(runner *runnerRef) {
	if s.dualbuf == nil || runner == nil || runner.llama == nil || runner.llama.HasExited() {
		return
	}
	m := runner.model.Load()
	if m == nil || m.dualbuf == nil {
		return
	}
	db := m.dualbuf
	park := runner.dualbufPark.Swap(false) && db.cfg.SwapEnabled()
	nvme := db.eff.KVSave.V
	saveRAM := park && db.cfg.SwapSaveKV()
	saver, ok := runner.llama.(slotSaver)
	prefix := dualbufPrefix(m)
	saved := false
	var kvBytes int64
	if ok && saver.SlotSavePath() != "" && (saveRAM || nvme) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		start := time.Now()
		files, err := saver.SaveSlots(ctx, prefix)
		cancel()
		if err != nil {
			slog.Warn("dualbuf: saving the KV failed", "model", dualbufModelName(m), "error", err)
		}
		if len(files) > 0 {
			meta := multimodel.KVMeta{Model: dualbufModelName(m), Digest: m.Digest, Placement: db.placement, KVType: dualbufKVType(m), SavedAt: time.Now(), Reason: "unload"}
			if runner.Options != nil {
				meta.NumCtx = runner.Options.NumCtx
			}
			meta.Parallel = runner.numParallel
			if park {
				meta.Reason = "swap"
			}
			for _, f := range files {
				meta.Slots = append(meta.Slots, f.Slot)
				meta.Tokens = append(meta.Tokens, f.Tokens)
				kvBytes += f.Bytes
			}
			meta.Bytes = kvBytes
			if err := multimodel.WriteMeta(saver.SlotSavePath(), prefix, meta); err != nil {
				slog.Warn("dualbuf: writing the KV record failed", "error", err)
			} else {
				saved = true
				slog.Info("dualbuf: KV saved to RAM", "model", meta.Model, "slots", meta.Slots, "tokens", meta.Tokens, "bytes", kvBytes, "duration", time.Since(start).Round(time.Millisecond))
			}
		}
	}
	if saved && nvme && !park {
		dir := db.cfg.NVMeDir(envconfig.Models())
		start := time.Now()
		if err := multimodel.CopyKV(saver.SlotSavePath(), dir, prefix); err != nil {
			slog.Warn("dualbuf: writing the KV to the disk failed", "dir", dir, "error", err)
		} else {
			slog.Info("dualbuf: KV saved to the disk", "model", dualbufModelName(m), "dir", dir, "duration", time.Since(start).Round(time.Millisecond))
		}
		multimodel.RemoveKV(saver.SlotSavePath(), prefix)
	}
	if !park {
		return
	}
	p := &multimodel.Parked{Key: runner.modelKey, Model: dualbufModelName(m), Placement: db.placement, LastUsed: time.Unix(0, runner.dualbufLastUsed.Load()),
		Forever: runner.sessionDuration == time.Duration(math.MaxInt64)}
	if saved {
		p.Prefix = prefix
		p.Bytes += uint64(kvBytes)
	}
	s.dualbuf.Park(p)
	if paths := m.modelPaths(); db.cfg.Swap.KeepInRAM == multimodel.KeepInRAMPinned && len(paths) > 0 {
		// pinning reads the whole file; the unload does not wait for it
		go func(key, path string) {
			pin, err := multimodel.PinFile(path)
			if err != nil {
				slog.Warn("dualbuf: the model file stays in the page cache only", "path", path, "error", err)
				return
			}
			s.dualbuf.SetPinned(key, pin)
		}(p.Key, paths[0])
	}
}

// dualbufAfterLoad runs when a new instance is ready: the KV of a swap (RAM) or of an earlier unload (disk) is
// restored, the parked record goes, and the measured host RAM is noted for the next estimate.
func (s *Scheduler) dualbufAfterLoad(runner *runnerRef) {
	if s.dualbuf == nil || runner == nil {
		return
	}
	runner.dualbufLastUsed.Store(time.Now().UnixNano())
	m := runner.model.Load()
	if m == nil || m.dualbuf == nil {
		return
	}
	s.dualbuf.NoteHost(runner.modelKey, runner.totalSize-min(runner.totalSize, runner.vramSize))
	parked := s.dualbuf.Unpark(runner.modelKey)
	if parked != nil {
		defer parked.Pinned.Release()
	}
	saver, ok := runner.llama.(slotSaver)
	if !ok || saver.SlotSavePath() == "" {
		return
	}
	dir := saver.SlotSavePath()
	prefix := dualbufPrefix(m)
	from := ""
	if parked != nil && parked.Prefix != "" {
		from = "ram"
	} else if m.dualbuf.eff.KVSave.V {
		nvme := m.dualbuf.cfg.NVMeDir(envconfig.Models())
		if _, err := multimodel.ReadMeta(nvme, prefix); err == nil {
			if err := multimodel.CopyKV(nvme, dir, prefix); err != nil {
				slog.Warn("dualbuf: reading the saved KV failed", "dir", nvme, "error", err)
				return
			}
			from = "disk"
			defer multimodel.RemoveKV(nvme, prefix)
		}
	}
	if from == "" {
		return
	}
	defer multimodel.RemoveKV(dir, prefix)
	meta, err := multimodel.ReadMeta(dir, prefix)
	if err != nil {
		return
	}
	if err := meta.Compatible(m.Digest, dualbufKVType(m)); err != nil {
		slog.Info("dualbuf: the saved KV is not restored", "model", dualbufModelName(m), "reason", err)
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	start := time.Now()
	files, err := saver.RestoreSlots(ctx, prefix)
	if err != nil {
		slog.Warn("dualbuf: restoring the KV failed (the prompt is computed again)", "model", dualbufModelName(m), "from", from, "error", err)
	}
	if len(files) > 0 {
		var tokens []int
		for _, f := range files {
			tokens = append(tokens, f.Tokens)
		}
		slog.Info("dualbuf: KV restored", "model", dualbufModelName(m), "from", from, "tokens", tokens, "saved_num_ctx", meta.NumCtx, "duration", time.Since(start).Round(time.Millisecond))
	}
}

// dualbufDropParked removes the parked instances of a model that was asked to unload (keep_alive 0).
func (s *Scheduler) dualbufDropParked(model *Model) {
	if s == nil || s.dualbuf == nil || model == nil {
		return
	}
	name := dualbufModelName(model)
	for _, p := range s.dualbuf.ParkedList() {
		if p.Model == name {
			s.dualbuf.Drop(p.Key)
		}
	}
}

// in-process park (llama-server --park): the process stays, its device memory is freed; see docs/MULTIMODEL-DESIGN.md 4.

type parker interface {
	CanPark() bool
	ParkRefused() bool
	Parked() bool
	ParkedVRAM() uint64
	Park(ctx context.Context) (time.Duration, error)
	Unpark(ctx context.Context) (time.Duration, error)
}

func runnerParker(r *runnerRef) (parker, bool) {
	if r == nil || r.llama == nil {
		return nil, false
	}
	p, ok := r.llama.(parker)
	return p, ok
}

func runnerParked(r *runnerRef) bool {
	p, ok := runnerParker(r)
	return ok && p.Parked()
}

// dualbufParkedCountLocked: the parked runners hold no device memory, they do not count for OLLAMA_MAX_LOADED_MODELS
// (the RAM budget limits them). Called with loadedMu held.
func (s *Scheduler) dualbufParkedCountLocked() int {
	n := 0
	for _, r := range s.loaded {
		if runnerParked(r) {
			n++
		}
	}
	return n
}

// dualbufParkedFor is a parked GPU instance of the model (for a request placed on the CPU: it runs there without a reload).
func (s *Scheduler) dualbufParkedFor(m *Model) *runnerRef {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	for _, r := range s.loaded {
		if lm := r.model.Load(); lm != nil && lm.ModelPath == m.ModelPath && lm.dualbuf != nil && lm.dualbuf.placement != multimodel.DeviceCPU && runnerParked(r) {
			return r
		}
	}
	return nil
}

// dualbufNeedsUnpark: the request wants the device of a parked runner (not a CPU request served by the parked process).
func dualbufNeedsUnpark(r *runnerRef, req *LlmRequest) bool {
	if !runnerParked(r) {
		return false
	}
	return req.model.dualbuf == nil || !req.model.dualbuf.viaParked
}

// dualbufTryPark parks a victim in process; false if it cannot (no park copies, busy, already parked, the park failed),
// then the scheduler expires it (with the process-level park of dualbufBeforeUnload when swap is on).
func (s *Scheduler) dualbufTryPark(r *runnerRef) bool {
	if r == nil || !r.dualbufPark.Load() {
		return false
	}
	p, ok := runnerParker(r)
	if !ok || p.Parked() {
		return false
	}
	name := r.modelPath
	if lm := r.model.Load(); lm != nil {
		name = dualbufModelName(lm)
	}
	if !p.CanPark() {
		if p.ParkRefused() {
			slog.Warn("dualbuf: no in-process park (the pinned copies of the GPU weights did not fit the RAM budget at the load), the swap restarts the process", "model", name)
		}
		return false
	}
	r.refMu.Lock()
	defer r.refMu.Unlock()
	if r.refCount > 0 {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	took, err := p.Park(ctx)
	if err != nil {
		slog.Warn("dualbuf: in-process park failed, the model is unloaded instead", "model", name, "error", err)
		return false
	}
	r.dualbufPark.Store(false)
	slog.Info("dualbuf: model parked in process", "model", name, "duration", took.Round(time.Millisecond), "vram_before", format.HumanBytes2(p.ParkedVRAM()))
	return true
}

// dualbufUnparkFor brings a parked runner back for a request that wants its device. It returns the runner to expire (or
// to park) first when the device has no room for it (wait = true); else the runner is unparked and usable.
func (s *Scheduler) dualbufUnparkFor(ctx context.Context, req *LlmRequest, r *runnerRef, full bool) (*runnerRef, bool) {
	if !dualbufNeedsUnpark(r, req) {
		return nil, false
	}
	p, _ := runnerParker(r)
	need := p.ParkedVRAM()
	s.loadedMu.Lock()
	snapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
	for _, x := range s.loaded {
		snapshot = append(snapshot, x)
	}
	s.loadedMu.Unlock()
	gpus := s.getGpuFn(ctx, snapshot)
	s.updateFreeSpace(gpus)
	var free uint64
	found := false
	for _, g := range gpus {
		for _, id := range r.gpus {
			if g.DeviceID == id {
				free += g.FreeMemory
				found = true
			}
		}
	}
	if full || (found && need > free) {
		// room on the device first: the least recently used idle model there (parked in process when it can)
		if v := s.dualbufVictimExcept(req, r); v != nil {
			slog.Info("dualbuf: room for an unpark", "model", r.modelPath, "need", format.HumanBytes2(need), "free", format.HumanBytes2(free), "victim", v.modelPath)
			return v, true
		}
	}
	r.refMu.Lock()
	uctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	took, err := p.Unpark(uctx)
	cancel()
	r.refMu.Unlock()
	if err != nil {
		slog.Warn("dualbuf: unpark failed, the model is loaded again", "model", r.modelPath, "error", err)
		return r, true
	}
	slog.Info("dualbuf: model unparked", "model", r.modelPath, "duration", took.Round(time.Millisecond), "vram", format.HumanBytes2(need))
	return nil, false
}

// dualbufVictimExcept is the same device victim of dualbufVictim, not the runner being unparked and not a parked one.
func (s *Scheduler) dualbufVictimExcept(req *LlmRequest, except *runnerRef) *runnerRef {
	db := req.model.dualbuf
	if db == nil || db.placement == "" || db.placement == multimodel.DeviceCPU {
		return nil
	}
	s.loadedMu.Lock()
	var pick *runnerRef
	for _, r := range s.loaded {
		if r == except || runnerParked(r) {
			continue
		}
		lm := r.model.Load()
		if lm == nil || lm.dualbuf == nil || lm.dualbuf.placement != db.placement {
			continue
		}
		r.refMu.Lock()
		idle := r.refCount == 0
		r.refMu.Unlock()
		if idle && (pick == nil || r.dualbufLastUsed.Load() < pick.dualbufLastUsed.Load()) {
			pick = r
		}
	}
	s.loadedMu.Unlock()
	if pick != nil && db.cfg.SwapEnabled() {
		pick.dualbufPark.Store(true)
	}
	return pick
}

// demoter: the repacked copies a parked instance made for its CPU runs (llama-server POST /park/drop frees them)
type demoter interface {
	DemotedBytes() uint64
	DropDemoted(ctx context.Context) (uint64, error)
}

// demotedCopies is the eviction candidate of the demoted copies of a runner (dropped, the runner stays).
type demotedCopies struct{ r *runnerRef }
