package multimodel

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/ollama/ollama/ml"
)

// Matched is a config device found among the discovered devices.
type Matched struct {
	Name string        // the config name (x080, rx550)
	Info ml.DeviceInfo // the discovered device
}

// DeviceMap is the result of matching the devices section against the discovered devices.
type DeviceMap struct {
	ByName  map[string]Matched
	Missing []string // config devices with no discovered device
	Errors  []string // ambiguous matches
}

var pciRe = regexp.MustCompile(`^(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$`)

// normPCI makes "03:00.0" and "0000:03:00.0" equal.
func normPCI(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if len(s) == len("03:00.0") {
		s = "0000:" + s
	}
	return s
}

func backendLibrary(b string) string {
	switch strings.ToLower(b) {
	case "rocm":
		return "ROCm"
	case "vulkan":
		return "Vulkan"
	case "cuda":
		return "CUDA"
	case "metal":
		return "Metal"
	}
	return b
}

// matches reports whether a discovered device fits a config entry: the backend library must be the same, the
// match is a PCI address (exact) or a case-insensitive part of the description or the backend name (ROCm0).
func (d Device) matches(info ml.DeviceInfo) bool {
	if !strings.EqualFold(info.Library, backendLibrary(d.Backend)) {
		return false
	}
	m := strings.TrimSpace(d.Match)
	if pciRe.MatchString(m) {
		return info.PCIID != "" && normPCI(info.PCIID) == normPCI(m)
	}
	lm := strings.ToLower(m)
	return strings.Contains(strings.ToLower(info.Description), lm) || strings.EqualFold(info.Name, m)
}

// MatchDevices finds every GPU entry of the devices section among the discovered devices. One entry has to
// match exactly one device (two cards of the same type need the PCI address), and a device can belong to
// one entry only. The CPU entry is always present.
func (c *Config) MatchDevices(found []ml.DeviceInfo) DeviceMap {
	dm := DeviceMap{ByName: map[string]Matched{}}
	if c == nil {
		return dm
	}
	names := make([]string, 0, len(c.Devices))
	for n := range c.Devices {
		names = append(names, n)
	}
	sort.Strings(names)
	owner := map[string]string{} // device key -> config name
	for _, n := range names {
		if n == DeviceCPU {
			continue
		}
		d := c.Devices[n]
		var hits []ml.DeviceInfo
		for _, info := range found {
			if d.matches(info) {
				hits = append(hits, info)
			}
		}
		switch {
		case len(hits) == 0:
			dm.Missing = append(dm.Missing, n)
		case len(hits) > 1:
			var ids []string
			for _, h := range hits {
				ids = append(ids, describe(h))
			}
			dm.Errors = append(dm.Errors, fmt.Sprintf("devices.%s: %q matches %d devices (%s), use the PCI address", n, d.Match, len(hits), strings.Join(ids, "; ")))
		default:
			key := hits[0].Library + "/" + hits[0].ID
			if prev, ok := owner[key]; ok {
				dm.Errors = append(dm.Errors, fmt.Sprintf("devices.%s and devices.%s match the same device (%s)", prev, n, describe(hits[0])))
				delete(dm.ByName, prev)
				continue
			}
			owner[key] = n
			dm.ByName[n] = Matched{Name: n, Info: hits[0]}
		}
	}
	dm.ByName[DeviceCPU] = Matched{Name: DeviceCPU}
	return dm
}

func describe(info ml.DeviceInfo) string {
	pci := info.PCIID
	if pci == "" {
		pci = "no PCI id"
	}
	return fmt.Sprintf("%s %s %q %s", info.Library, info.Name, info.Description, pci)
}

// String is one log line per matched device.
func (dm DeviceMap) String() string {
	var parts []string
	names := make([]string, 0, len(dm.ByName))
	for n := range dm.ByName {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		m := dm.ByName[n]
		if n == DeviceCPU {
			parts = append(parts, "cpu=CPU")
			continue
		}
		parts = append(parts, n+"="+describe(m.Info))
	}
	for _, n := range dm.Missing {
		parts = append(parts, n+"=missing")
	}
	return strings.Join(parts, ", ")
}

// Present reports whether a config device was found (cpu always).
func (dm DeviceMap) Present(name string) bool {
	_, ok := dm.ByName[name]
	return ok
}
