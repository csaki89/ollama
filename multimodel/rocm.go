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

// With ROCm 10 every ROCm process sees every card: the runtime starts with an AMD card it does not support in the
// container and skips it. The gfx_override of a rocm device is its own HSA_OVERRIDE_GFX_VERSION_<node>, where <node> is
// the card's KFD topology node (found by its PCI address); the global HSA_OVERRIDE_GFX_VERSION never goes to a ROCm
// process with a config, as it would apply to the unsupported card too (docs/MULTIMODEL-DESIGN.md 2.3).

// ROCmCard is a rocm device of the config with its PCI address and KFD node.
type ROCmCard struct {
	Device      string // config device name
	PCI         string // normalized PCI address
	GFXOverride string // the device's gfx_override ("" = none)
	Node        int    // its KFD topology node
}

// KFDNodes maps the PCI address of every GPU node of the KFD topology under sysfsRoot (normally /sys) to its node
// number (class/kfd/kfd/topology/nodes/<n>/properties: domain and location_id; a CPU node has neither a GPU id nor a
// location).
func KFDNodes(sysfsRoot string) map[string]int {
	dir := filepath.Join(sysfsRoot, "class", "kfd", "kfd", "topology", "nodes")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	out := map[string]int{}
	for _, e := range entries {
		n, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		if gpu, err := os.ReadFile(filepath.Join(dir, e.Name(), "gpu_id")); err != nil || strings.TrimSpace(string(gpu)) == "0" {
			continue
		}
		if pci := kfdNodePCI(filepath.Join(dir, e.Name(), "properties")); pci != "" {
			out[pci] = n
		}
	}
	return out
}

// kfdNodePCI is the PCI address of a KFD node from its properties: location_id is bus<<8 | device<<3 | function.
func kfdNodePCI(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	loc, domain := int64(-1), int64(0)
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fl := strings.Fields(sc.Text())
		if len(fl) != 2 {
			continue
		}
		v, err := strconv.ParseInt(fl[1], 10, 64)
		if err != nil {
			continue
		}
		switch fl[0] {
		case "location_id":
			loc = v
		case "domain":
			domain = v
		}
	}
	if loc <= 0 {
		return ""
	}
	return fmt.Sprintf("%04x:%02x:%02x.%x", domain, (loc>>8)&0xff, (loc>>3)&0x1f, loc&0x7)
}

// ROCmCards are the rocm devices of the config, each by its PCI address: the match itself, or for a name match the
// PCI address of the one known (already discovered, e.g. by Vulkan) device whose name contains it; and its KFD node
// (KFDNodes). notes says why a device is left out (it is skipped with a warning).
func (c *Config) ROCmCards(kfd map[string]int, known []ml.DeviceInfo) (cards []ROCmCard, notes []string) {
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
		node, ok := kfd[pci]
		if !ok {
			notes = append(notes, fmt.Sprintf("devices.%s: the card %s has no KFD topology node (pass /dev/kfd; is it an amdgpu card?)", name, pci))
			continue
		}
		cards = append(cards, ROCmCard{Device: name, PCI: pci, GFXOverride: d.GFXOverride, Node: node})
	}
	return cards, notes
}

// ROCrIndices maps the PCI address of every KFD GPU node to its index in the ROCm runtime's own GPU numbering (what
// ROCR_VISIBLE_DEVICES selects): the GPU nodes in KFD node order, a card the runtime does not support included (it
// skips that card later, so the HIP ordinals of a process differ from these indices).
func ROCrIndices(kfd map[string]int) map[string]int {
	type gpu struct {
		pci  string
		node int
	}
	gpus := make([]gpu, 0, len(kfd))
	for p, n := range kfd {
		gpus = append(gpus, gpu{p, n})
	}
	sort.Slice(gpus, func(i, j int) bool { return gpus[i].node < gpus[j].node })
	out := make(map[string]int, len(gpus))
	for i, g := range gpus {
		out[g.pci] = i
	}
	return out
}

// ROCmEnv is the gfx override environment of every ROCm process: HSA_OVERRIDE_GFX_VERSION_<node> for each card with a
// gfx_override, never the global HSA_OVERRIDE_GFX_VERSION.
func ROCmEnv(cards []ROCmCard) map[string]string {
	env := map[string]string{}
	for _, c := range cards {
		if c.GFXOverride != "" {
			env["HSA_OVERRIDE_GFX_VERSION_"+strconv.Itoa(c.Node)] = c.GFXOverride
		}
	}
	return env
}

// SysfsRoot is the sysfs mount the KFD topology is read from (a test sets another).
var SysfsRoot = "/sys"

// sysfsPresent: a Linux sysfs with a KFD topology is there.
func sysfsPresent(root string) bool {
	_, err := os.Stat(filepath.Join(root, "class", "kfd", "kfd", "topology", "nodes"))
	return err == nil
}

// HasSysfs reports whether the KFD topology can be read (Linux, /dev/kfd's driver loaded).
func HasSysfs() bool { return sysfsPresent(SysfsRoot) }
