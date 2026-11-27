package discover

// Dual-buffer fork (multimodel): with a config that names rocm cards, the ROCm discovery runs in a ROCm view of those
// cards only (the other cards' render nodes hidden from the ROCm runtime, each card's own gfx override), so an AMD card
// of another backend or one the ROCm runtime cannot use does not stop it. The cards are chosen by their PCI address,
// not by an ordinal. Without such a config the stock discovery runs.

import (
	"context"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/ml"
)

// DualbufROCmView is a set of rocm cards of the multi-model config a ROCm process sees together (normally every one):
// their device names and PCI addresses, and the environment of such a process (hidden render nodes, gfx overrides).
type DualbufROCmView struct {
	Devices []string
	PCIs    []string
	Env     map[string]string
}

// DualbufROCmViews is set by the server when the config names rocm cards; known are the devices the other libraries
// found (to resolve a name match to a PCI address). nil, or no views: the stock ROCm discovery.
var DualbufROCmViews func(known []ml.DeviceInfo) []DualbufROCmView

// hideLibrary is the preload library that hides render nodes while the HIP backend loads (llama/compat/ggml-hide-dri.c).
const hideLibrary = "libggml-hide-dri.so"

// dualbufROCmEnv adds the preload library of the ROCm library directory to a view's environment.
func dualbufROCmEnv(view DualbufROCmView, rocmDir string) map[string]string {
	env := maps.Clone(view.Env)
	if env == nil {
		env = map[string]string{}
	}
	lib := filepath.Join(rocmDir, hideLibrary)
	if _, err := os.Stat(lib); err != nil {
		slog.Warn("dualbuf: the render-node hiding library is missing, the ROCm runtime sees every card", "path", lib)
		delete(env, "GGML_HIP_HIDE_RENDER_NODES")
		return env
	}
	if pre := os.Getenv("LD_PRELOAD"); pre != "" {
		lib = lib + ":" + pre
	}
	env["LD_PRELOAD"] = lib
	return env
}

// dualbufDiscoverROCm discovers each view's cards in that view; a card the ROCm runtime does not start is left out with
// a warning that names it and the reason. A device's ID is its PCI address (never an ordinal of this machine), its
// filter ID its ROCm ordinal in the view (the same in every process of the view).
func dualbufDiscoverROCm(ctx context.Context, timeout time.Duration, dirs []string, views []DualbufROCmView) []ml.DeviceInfo {
	var out []ml.DeviceInfo
	for _, view := range views {
		env := dualbufROCmEnv(view, dirs[len(dirs)-1])
		c, cancel := context.WithTimeout(ctx, timeout)
		devs, status, err := bootstrapDevicesWithStatusWatchdog(c, dirs, env)
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
			if pci == "" && len(view.PCIs) == 1 {
				pci = view.PCIs[0]
			}
			i := slices.Index(view.PCIs, pci)
			if i < 0 {
				slog.Warn("dualbuf: the ROCm view shows a card that is not a rocm device of the config, it is not used", "pci", d.PCIID, "view", view.PCIs)
				continue
			}
			d.ID = pci
			d.FilterID = strconv.Itoa(idx)
			d.PCIID = pci
			if d.RunnerEnvOverrides == nil {
				d.RunnerEnvOverrides = map[string]string{}
			}
			maps.Copy(d.RunnerEnvOverrides, env)
			out = append(out, d)
			found[pci] = true
			slog.Info("dualbuf: ROCm card found in the ROCm view", "device", view.Devices[i], "pci", pci, "rocm_ordinal", idx,
				"hidden_render_nodes", env["GGML_HIP_HIDE_RENDER_NODES"])
		}
		reason := lastDiscoveryStatusError(status)
		if reason == "" && err != nil {
			reason = err.Error()
		}
		if reason == "" {
			reason = "the ROCm runtime reports no usable device for it (a missing or wrong gfx_override, no /dev/kfd, or the card is not supported)"
		}
		for i, p := range view.PCIs {
			if !found[p] {
				slog.Warn("dualbuf: the ROCm card does not start, it is left out", "device", view.Devices[i], "pci", p, "reason", reason)
			}
		}
	}
	return out
}

func normPCIID(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if len(s) == len("aa:00.0") {
		s = "0000:" + s
	}
	return s
}
