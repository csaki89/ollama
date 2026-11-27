package multimodel

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/ml"
)

// The ROCm runtime opens every GPU of the KFD topology whose render node it can reach when it starts, and it fails as
// a whole on a card it cannot use (another backend's AMD card, an unsupported one), before ROCR_VISIBLE_DEVICES
// filters anything. So with a config every rocm device gets a ROCm process view of its own: the render nodes of every
// other card are hidden from the ROCm runtime while the HIP backend loads (GGML_HIP_HIDE_RENDER_NODES, the
// ggml-hide-dri preload library), and its gfx_override is the process' HSA_OVERRIDE_GFX_VERSION (docs/MULTIMODEL-DESIGN.md 2.3).

// ROCmCard is a rocm device of the config with its PCI address and the render nodes hidden from its ROCm runtime.
type ROCmCard struct {
	Device      string   // config device name
	PCI         string   // normalized PCI address
	GFXOverride string   // the device's gfx_override ("" = none)
	Hide        []string // the render nodes of the other cards (renderD<N>)
}

// Env is the environment of a ROCm process that uses the card.
func (r ROCmCard) Env() map[string]string {
	env := map[string]string{"GGML_HIP_HIDE_RENDER_NODES": strings.Join(r.Hide, ",")}
	if r.GFXOverride != "" {
		env["HSA_OVERRIDE_GFX_VERSION"] = r.GFXOverride
	}
	return env
}

// RenderNodes maps the PCI address of every DRM render node under sysfsRoot (normally /sys) to its node names.
func RenderNodes(sysfsRoot string) map[string][]string {
	out := map[string][]string{}
	nodes, _ := filepath.Glob(filepath.Join(sysfsRoot, "class", "drm", "renderD*"))
	for _, n := range nodes {
		dev, err := filepath.EvalSymlinks(filepath.Join(n, "device"))
		if err != nil {
			continue
		}
		pci := normPCI(filepath.Base(dev))
		if !pciRe.MatchString(pci) {
			continue
		}
		out[pci] = append(out[pci], filepath.Base(n))
	}
	for k := range out {
		sort.Strings(out[k])
	}
	return out
}

// ROCmCards are the rocm devices of the config, each by its PCI address: the match itself, or for a name match the
// PCI address of the one known (already discovered, e.g. by Vulkan) device whose name contains it. nodes are the render
// nodes by PCI address (RenderNodes). notes says why a device is left out (it is skipped with a warning).
func (c *Config) ROCmCards(nodes map[string][]string, known []ml.DeviceInfo) (cards []ROCmCard, notes []string) {
	if c == nil {
		return nil, nil
	}
	names := make([]string, 0, len(c.Devices))
	for n, d := range c.Devices {
		if strings.EqualFold(d.Backend, "rocm") {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	for _, name := range names {
		d := c.Devices[name]
		m := strings.TrimSpace(d.Match)
		pci := ""
		if pciRe.MatchString(m) {
			pci = normPCI(m)
		} else {
			lm := strings.ToLower(m)
			for _, k := range known {
				if k.PCIID == "" || !(strings.Contains(strings.ToLower(k.Description), lm) || strings.EqualFold(k.Name, m)) {
					continue
				}
				if p := normPCI(k.PCIID); pci != "" && p != pci {
					notes = append(notes, fmt.Sprintf("devices.%s: the name %q fits several cards (%s, %s): give its PCI address", name, m, pci, p))
					pci = "-"
					break
				} else {
					pci = p
				}
			}
			if pci == "" {
				notes = append(notes, fmt.Sprintf("devices.%s: no card with the name %q was found to tell its PCI address (give the PCI address)", name, m))
				continue
			}
			if pci == "-" {
				continue
			}
		}
		if len(nodes[pci]) == 0 {
			notes = append(notes, fmt.Sprintf("devices.%s: the card %s has no render node in the container (pass its /dev/dri/renderD* device)", name, pci))
			continue
		}
		card := ROCmCard{Device: name, PCI: pci, GFXOverride: d.GFXOverride}
		for p, ns := range nodes {
			if p != pci {
				card.Hide = append(card.Hide, ns...)
			}
		}
		sort.Strings(card.Hide)
		cards = append(cards, card)
	}
	return cards, notes
}

// SysfsRoot is the sysfs mount the render nodes are read from (a test sets another).
var SysfsRoot = "/sys"

// sysfsPresent: a Linux sysfs with DRM devices is there.
func sysfsPresent(root string) bool {
	_, err := os.Stat(filepath.Join(root, "class", "drm"))
	return err == nil
}

// HasSysfs reports whether the render nodes can be read (Linux).
func HasSysfs() bool { return sysfsPresent(SysfsRoot) }
