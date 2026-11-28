package discover

// Dual-buffer fork (multimodel): with a config that names rocm cards, the ROCm discovery runs once with every card
// visible and each card's gfx override as its own HSA_OVERRIDE_GFX_VERSION_<KFD node> (ROCm 10 skips a card it does not
// support); the global HSA_OVERRIDE_GFX_VERSION is not passed. The cards are chosen by their PCI address, not by an
// ordinal. Without such a config the stock discovery runs.

import (
	"context"
	"log/slog"
	"maps"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// DualbufROCmCards are the rocm cards of the multi-model config: their device names, PCI addresses, and the
// environment of every ROCm process (the per-card gfx overrides). ROCrIndex is each card's index in the ROCm runtime's
// numbering (ROCR_VISIBLE_DEVICES), by PCI address: it counts the cards the runtime skips, the HIP ordinals do not.
type DualbufROCmCards struct {
	Devices   []string
	PCIs      []string
	Env       map[string]string
	ROCrIndex map[string]int
}

// dualbufBootstrap runs a discovery pass (a test replaces it).
var dualbufBootstrap = bootstrapDevicesWithStatusWatchdog

// DualbufROCm is set by the server when the config names rocm cards; known are the devices the other libraries found
// (to resolve a name match to a PCI address). nil, or no cards: the stock ROCm discovery.
var DualbufROCm func(known []ml.DeviceInfo) *DualbufROCmCards

// dualbufROCmEnv is the discovery environment of the cards: their gfx overrides, never the global one.
func dualbufROCmEnv(cards *DualbufROCmCards) map[string]string {
	env := maps.Clone(cards.Env)
	if env == nil {
		env = map[string]string{}
	}
	delete(env, "HSA_OVERRIDE_GFX_VERSION")
	return env
}

// dualbufDiscoverROCm discovers the config's rocm cards; a card the ROCm runtime does not start is left out with a
// warning that names it and the reason. A device's ID is its PCI address (never an ordinal of this machine), its filter
// ID its ROCm ordinal (the same in every process, as every one sees every card).
func dualbufDiscoverROCm(ctx context.Context, timeout time.Duration, dirs []string, cards *DualbufROCmCards) []ml.DeviceInfo {
	var out []ml.DeviceInfo
	env := dualbufROCmEnv(cards)
	slog.Info("dualbuf: ROCm discovery", "cards", dualbufCardList(cards), "env", DualbufROCmEnvSummary(env))
	c, cancel := context.WithTimeout(ctx, timeout)
	devs, status, err := dualbufBootstrap(c, dirs, env)
	cancel()
	found := map[string]bool{}
	ordinal := 0
	for _, d := range devs {
		if d.Library != "ROCm" {
			continue
		}
		idx := ordinal
		ordinal++
		pci := normPCIID(d.PCIID)
		if pci == "" && len(cards.PCIs) == 1 && ordinal == 1 {
			pci = cards.PCIs[0]
		}
		i := slices.Index(cards.PCIs, pci)
		if i < 0 {
			slog.Info("dualbuf: a ROCm card that is not a rocm device of the config is not used", "pci", d.PCIID, "rocm_ordinal", idx)
			continue
		}
		// the filter ID selects the card in every later ROCm process (ROCR_VISIBLE_DEVICES): the runtime's own index,
		// which counts a skipped card; the HIP ordinal does not, and would select that card instead
		rocr, ok := cards.ROCrIndex[pci]
		if !ok {
			rocr = idx
		}
		d.ID = pci
		d.FilterID = strconv.Itoa(rocr)
		d.PCIID = pci
		if d.RunnerEnvOverrides == nil {
			d.RunnerEnvOverrides = map[string]string{}
		}
		maps.Copy(d.RunnerEnvOverrides, env)
		out = append(out, d)
		found[pci] = true
		slog.Info("dualbuf: ROCm card found", "device", cards.Devices[i], "pci", pci, "rocm_ordinal", idx, "rocr_index", rocr)
	}
	reason := lastDiscoveryStatusError(status)
	if reason == "" && err != nil {
		reason = err.Error()
	}
	if reason == "" {
		reason = "the ROCm runtime reports no usable device for it (a missing or wrong gfx_override, no /dev/kfd, or the card is not supported)"
	}
	for i, p := range cards.PCIs {
		if !found[p] {
			slog.Warn("dualbuf: the ROCm card does not start, it is left out", "device", cards.Devices[i], "pci", p, "reason", reason,
				"cards", dualbufCardList(cards), "env", DualbufROCmEnvSummary(env))
		}
	}
	return out
}

// dualbufCardList lists the cards as device=PCI.
func dualbufCardList(cards *DualbufROCmCards) string {
	parts := make([]string, 0, len(cards.PCIs))
	for i, p := range cards.PCIs {
		parts = append(parts, cards.Devices[i]+"="+p)
	}
	return strings.Join(parts, " ")
}

// DualbufROCmEnvSummary is the ROCm part of an environment, as the runtime gets it: the gfx override variables and the
// visible ROCm devices (sorted KEY=VALUE).
func DualbufROCmEnvSummary(env map[string]string) string {
	var parts []string
	for k, v := range env {
		if strings.HasPrefix(k, "HSA_OVERRIDE_GFX_VERSION") || k == "ROCR_VISIBLE_DEVICES" || k == "HIP_VISIBLE_DEVICES" {
			parts = append(parts, k+"="+v)
		}
	}
	slices.Sort(parts)
	return strings.Join(parts, " ")
}

func normPCIID(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if len(s) == len("aa:00.0") {
		s = "0000:" + s
	}
	return s
}

// dualbufLogNoDevice: a discovery pass of a GPU library that finds no device logs its output (DEBUG: all of it; INFO: its
// error lines), and the error lines become the pass' status error (the "left out" warning shows them).
func dualbufLogNoDevice(libDirs []string, devs []ml.DeviceInfo, output string, status *llm.StatusWriter) {
	if len(libDirs) < 2 || len(devs) > 0 {
		return
	}
	errs := discoveryErrorLines(output)
	slog.Debug("llama-server discovery: no device found", "libDirs", libDirs, "output", output)
	if len(errs) > 0 {
		slog.Info("llama-server discovery: no device found", "libDirs", libDirs, "errors", strings.Join(errs, " | "))
		if status != nil {
			status.AppendError(strings.Join(errs, " | "))
		}
	}
}

// discoveryErrorLines are the lines of a discovery output that tell why no device came up (at most 4).
func discoveryErrorLines(output string) []string {
	var out []string
	for _, line := range strings.Split(output, "\n") {
		l := strings.ToLower(line)
		if strings.Contains(l, "error") || strings.Contains(l, "failed") || strings.Contains(l, "unable") ||
			strings.Contains(l, "not found") || strings.Contains(l, "no rocm") || strings.Contains(l, "no cuda") || strings.Contains(l, "no vulkan") {
			out = append(out, strings.TrimSpace(line))
			if len(out) == 4 {
				break
			}
		}
	}
	return out
}
