package discover

// Dual-buffer fork (multimodel): with a config that names rocm cards, the ROCm discovery runs once per card, in a ROCm
// view of that card only (the other cards' render nodes hidden from the ROCm runtime, the card's own gfx override), so
// an AMD card of another backend or one the ROCm runtime cannot use does not stop it. The card is chosen by its PCI
// address, not by an ordinal. Without such a config the stock discovery runs.

import (
	"context"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/ml"
)

// DualbufROCmCard is a rocm card of the multi-model config: its device name, PCI address and the environment of a ROCm
// process that uses it (hidden render nodes, gfx override).
type DualbufROCmCard struct {
	Device string
	PCI    string
	Env    map[string]string
}

// DualbufROCmCards is set by the server when the config names rocm cards; known are the devices the other libraries
// found (to resolve a name match to a PCI address). nil, or no cards: the stock ROCm discovery.
var DualbufROCmCards func(known []ml.DeviceInfo) []DualbufROCmCard

// hideLibrary is the preload library that hides render nodes while the HIP backend loads (llama/compat/ggml-hide-dri.c).
const hideLibrary = "libggml-hide-dri.so"

// dualbufROCmEnv adds the preload library of the ROCm library directory to a card's environment.
func dualbufROCmEnv(card DualbufROCmCard, rocmDir string) map[string]string {
	env := maps.Clone(card.Env)
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

// dualbufDiscoverROCm discovers each card in its own ROCm view; a card the ROCm runtime does not start is left out
// with a warning that names it and the reason.
func dualbufDiscoverROCm(ctx context.Context, timeout time.Duration, dirs []string, cards []DualbufROCmCard) []ml.DeviceInfo {
	var out []ml.DeviceInfo
	for _, card := range cards {
		env := dualbufROCmEnv(card, dirs[len(dirs)-1])
		c, cancel := context.WithTimeout(ctx, timeout)
		devs, status, err := bootstrapDevicesWithStatusWatchdog(c, dirs, env)
		cancel()
		found := false
		for _, d := range devs {
			if d.Library != "ROCm" {
				continue
			}
			if d.PCIID != "" && !strings.EqualFold(normPCIID(d.PCIID), card.PCI) {
				slog.Warn("dualbuf: the ROCm view of a card shows another card, it is not used", "device", card.Device, "pci", card.PCI, "seen", d.PCIID)
				continue
			}
			// one card per view: ordinal 0 there; the ID is its PCI address (unique, never an ordinal of this machine)
			d.ID = card.PCI
			d.FilterID = "0"
			if d.PCIID == "" {
				d.PCIID = card.PCI
			}
			if d.RunnerEnvOverrides == nil {
				d.RunnerEnvOverrides = map[string]string{}
			}
			maps.Copy(d.RunnerEnvOverrides, env)
			out = append(out, d)
			found = true
		}
		if found {
			slog.Info("dualbuf: ROCm card found in its own view", "device", card.Device, "pci", card.PCI,
				"gfx_override", env["HSA_OVERRIDE_GFX_VERSION"], "hidden_render_nodes", env["GGML_HIP_HIDE_RENDER_NODES"])
			continue
		}
		reason := lastDiscoveryStatusError(status)
		if reason == "" && err != nil {
			reason = err.Error()
		}
		if reason == "" {
			reason = "the ROCm runtime reports no usable device (a missing or wrong gfx_override, no /dev/kfd, or the card is not supported)"
		}
		slog.Warn("dualbuf: the ROCm card does not start, it is left out", "device", card.Device, "pci", card.PCI,
			"gfx_override", env["HSA_OVERRIDE_GFX_VERSION"], "reason", reason)
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
