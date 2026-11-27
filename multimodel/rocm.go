package multimodel

import (
	"bufio"
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
// preload library). The gfx_override of each card is its own HSA_OVERRIDE_GFX_VERSION_<node> (the ROCm thunk reads
// that before the global HSA_OVERRIDE_GFX_VERSION; <node> is its node index in the view); when the node indices cannot
// be read and the overrides differ, the cards with the same override share a view (docs/MULTIMODEL-DESIGN.md 2.3).

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

// kfdNode is a node of the KFD topology: a CPU (gpu id 0) or a GPU with its render node minor.
type kfdNode struct {
	cpu   bool
	minor int
}

// KFDNodes reads the KFD topology nodes in their order (sysfsRoot/class/kfd/kfd/topology/nodes/<n>); nil if there is none.
func KFDNodes(sysfsRoot string) []kfdNode {
	dir := filepath.Join(sysfsRoot, "class", "kfd", "kfd", "topology", "nodes")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var ids []int
	for _, e := range entries {
		if n, err := strconv.Atoi(e.Name()); err == nil {
			ids = append(ids, n)
		}
	}
	sort.Ints(ids)
	var out []kfdNode
	for _, id := range ids {
		base := filepath.Join(dir, strconv.Itoa(id))
		gpu, err := os.ReadFile(filepath.Join(base, "gpu_id"))
		if err != nil {
			continue
		}
		if strings.TrimSpace(string(gpu)) == "0" {
			out = append(out, kfdNode{cpu: true})
			continue
		}
		minor := -1
		if f, err := os.Open(filepath.Join(base, "properties")); err == nil {
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				if fl := strings.Fields(sc.Text()); len(fl) == 2 && fl[0] == "drm_render_minor" {
					minor, _ = strconv.Atoi(fl[1])
				}
			}
			f.Close()
		}
		out = append(out, kfdNode{minor: minor})
	}
	return out
}

// thunkNodeIDs are the node indices the ROCm thunk gives the GPUs of a view: in the KFD topology order it counts the CPU
// nodes and the GPUs whose render node it can open (in the container and not hidden). minor -> node index.
func thunkNodeIDs(nodes []kfdNode, reachable func(minor int) bool) map[int]int {
	out := map[int]int{}
	n := 0
	for _, k := range nodes {
		switch {
		case k.cpu:
			n++
		case k.minor >= 0 && reachable(k.minor):
			out[k.minor] = n
			n++
		}
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

// ROCmViews groups the rocm cards into the views a ROCm process sees. Normally one view with every card, each card's
// gfx_override as HSA_OVERRIDE_GFX_VERSION_<its node index> (kfd: the topology, KFDNodes; present: the render nodes in
// the container, renderD<N>). Without a topology, cards with different overrides cannot share a view: one view per
// override value, with the global HSA_OVERRIDE_GFX_VERSION; note says so.
func ROCmViews(cards []ROCmCard, nodes map[string][]string, kfd []kfdNode, present func(node string) bool) (views []ROCmView, note string) {
	if len(cards) == 0 {
		return nil, ""
	}
	hideExcept := func(in []ROCmCard) []string {
		keep := map[string]bool{}
		for _, c := range in {
			keep[c.PCI] = true
		}
		var hide []string
		for p, ns := range nodes {
			if !keep[p] {
				hide = append(hide, ns...)
			}
		}
		sort.Strings(hide)
		return hide
	}
	values := map[string]bool{}
	for _, c := range cards {
		values[c.GFXOverride] = true
	}
	if len(kfd) > 0 || len(values) == 1 {
		v := ROCmView{Cards: cards, Hide: hideExcept(cards)}
		v.Env = map[string]string{"GGML_HIP_HIDE_RENDER_NODES": strings.Join(v.Hide, ",")}
		if len(kfd) > 0 {
			hidden := map[string]bool{}
			for _, h := range v.Hide {
				hidden[h] = true
			}
			ids := thunkNodeIDs(kfd, func(minor int) bool {
				n := "renderD" + strconv.Itoa(minor)
				return !hidden[n] && present(n)
			})
			for _, c := range cards {
				if c.GFXOverride == "" {
					continue
				}
				if id, ok := ids[c.minor]; ok {
					v.Env["HSA_OVERRIDE_GFX_VERSION_"+strconv.Itoa(id)] = c.GFXOverride
				}
			}
		} else if o := cards[0].GFXOverride; o != "" {
			// one override for every card of the view: the global variable is the same
			v.Env["HSA_OVERRIDE_GFX_VERSION"] = o
		}
		return []ROCmView{v}, ""
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
		v := ROCmView{Cards: in, Hide: hideExcept(in)}
		v.Env = map[string]string{"GGML_HIP_HIDE_RENDER_NODES": strings.Join(v.Hide, ",")}
		if o != "" {
			v.Env["HSA_OVERRIDE_GFX_VERSION"] = o
		}
		views = append(views, v)
	}
	return views, "no KFD topology to tell the node index of a card: the gfx_override is per process, only the rocm cards with the same gfx_override share a ROCm process (a move between the others is a reload)"
}

// SysfsRoot is the sysfs mount the render nodes and the KFD topology are read from; DevDRI the device directory.
var (
	SysfsRoot = "/sys"
	DevDRI    = "/dev/dri"
)

// sysfsPresent: a Linux sysfs with DRM devices is there.
func sysfsPresent(root string) bool {
	_, err := os.Stat(filepath.Join(root, "class", "drm"))
	return err == nil
}

// HasSysfs reports whether the render nodes can be read (Linux).
func HasSysfs() bool { return sysfsPresent(SysfsRoot) }

// DevicePresent reports whether a render node is in the container (DevDRI/<node>).
func DevicePresent(node string) bool {
	_, err := os.Stat(filepath.Join(DevDRI, node))
	return err == nil
}
