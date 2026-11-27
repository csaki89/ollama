package multimodel

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/ml"
)

// The ROCm runtime opens every GPU of the KFD topology whose render node it can reach when it starts, and it fails as
// a whole on a card it cannot use (another backend's AMD card, an unsupported one), before ROCR_VISIBLE_DEVICES
// filters anything. So with a config the ROCm runtime gets a view of the config's rocm cards only: the render nodes of
// every other card are hidden from it while the HIP backend loads (GGML_HIP_HIDE_RENDER_NODES, the ggml-hide-dri
// preload library). The gfx_override of the cards of a view is the process' HSA_OVERRIDE_GFX_VERSION: in the view it
// reaches only them. The per-node HSA_OVERRIDE_GFX_VERSION_<n> of the thunk is not used: with ROCm 7.2 it did not start
// a card that the same value in the global variable starts (the thunk keeps it apart, OverrideEngineId). So the cards
// with different overrides are in separate views (docs/MULTIMODEL-DESIGN.md 2.3).

// ROCmCard is a rocm device of the config with its PCI address.
type ROCmCard struct {
	Device      string // config device name
	PCI         string // normalized PCI address
	GFXOverride string // the device's gfx_override ("" = none)
	minor       int    // its render node minor (renderD<minor>)
}

// ROCmView is a set of rocm cards a ROCm process sees together, and the environment of such a process.
type ROCmView struct {
	Cards []ROCmCard
	Hide  []string          // the render nodes of every other card (renderD<N>)
	Env   map[string]string // GGML_HIP_HIDE_RENDER_NODES and the gfx overrides
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
			ambiguous := false
			for _, k := range known {
				if k.PCIID == "" || !(strings.Contains(strings.ToLower(k.Description), lm) || strings.EqualFold(k.Name, m)) {
					continue
				}
				p := normPCI(k.PCIID)
				if pci != "" && p != pci {
					notes = append(notes, fmt.Sprintf("devices.%s: the name %q fits several cards (%s, %s): give its PCI address", name, m, pci, p))
					ambiguous = true
					break
				}
				pci = p
			}
			if ambiguous {
				continue
			}
			if pci == "" {
				notes = append(notes, fmt.Sprintf("devices.%s: no card with the name %q was found to tell its PCI address (give the PCI address)", name, m))
				continue
			}
		}
		if len(nodes[pci]) == 0 {
			notes = append(notes, fmt.Sprintf("devices.%s: the card %s has no render node in the container (pass its /dev/dri/renderD* device)", name, pci))
			continue
		}
		minor, _ := strconv.Atoi(strings.TrimPrefix(nodes[pci][0], "renderD"))
		cards = append(cards, ROCmCard{Device: name, PCI: pci, GFXOverride: d.GFXOverride, minor: minor})
	}
	return cards, notes
}

// ROCmViews groups the rocm cards into the views a ROCm process sees: one view per gfx_override value (normally one,
// with every card), with the global HSA_OVERRIDE_GFX_VERSION of that value (none for the cards without an override).
// note says when the cards are split (a move between views is a reload).
func ROCmViews(cards []ROCmCard, nodes map[string][]string) (views []ROCmView, note string) {
	if len(cards) == 0 {
		return nil, ""
	}
	byValue := map[string][]ROCmCard{}
	var order []string
	for _, c := range cards {
		if _, ok := byValue[c.GFXOverride]; !ok {
			order = append(order, c.GFXOverride)
		}
		byValue[c.GFXOverride] = append(byValue[c.GFXOverride], c)
	}
	for _, o := range order {
		in := byValue[o]
		keep := map[string]bool{}
		for _, c := range in {
			keep[c.PCI] = true
		}
		v := ROCmView{Cards: in}
		for p, ns := range nodes {
			if !keep[p] {
				v.Hide = append(v.Hide, ns...)
			}
		}
		sort.Strings(v.Hide)
		v.Env = map[string]string{"GGML_HIP_HIDE_RENDER_NODES": strings.Join(v.Hide, ",")}
		if o != "" {
			v.Env["HSA_OVERRIDE_GFX_VERSION"] = o
		}
		views = append(views, v)
	}
	if len(views) > 1 {
		note = "the rocm cards have different gfx_override values: a ROCm process takes one, so each value has its own ROCm view (the other values' cards hidden there); a move between the views is a reload"
	}
	return views, note
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
