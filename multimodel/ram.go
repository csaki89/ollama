package multimodel

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// RAM is the memory the scheduler plans with: the tightest cgroup limit (or ram_limit), and what is used of
// it without the reclaimable page cache. The same reading as the llama-server RAM guard (patch 003).
type RAM struct {
	Limit  uint64 // bytes, 0 = no limit known
	Used   uint64 // bytes
	Source string // "cgroup", "ram_limit", "system", "none"
	// UsedFrom is where Used was read: "cgroup" (the driver host memory the cgroup does not see is missing from it) or
	// "system" (the system's used memory has every allocation)
	UsedFrom string
}

// Free is Limit - Used (0 when over).
func (r RAM) Free() uint64 {
	if r.Used >= r.Limit {
		return 0
	}
	return r.Limit - r.Used
}

// cgroupRoot and selfCgroup are variables for the tests.
var (
	cgroupRoot = "/sys/fs/cgroup"
	selfCgroup = "/proc/self/cgroup"
)

func readU64(path string) (uint64, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	s := strings.TrimSpace(string(b))
	if s == "" || s == "max" {
		return 0, false
	}
	v, err := strconv.ParseUint(s, 10, 64)
	return v, err == nil
}

func statValue(path, key string) uint64 {
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), " ")
		if ok && k == key {
			n, _ := strconv.ParseUint(strings.TrimSpace(v), 10, 64)
			return n
		}
	}
	return 0
}

// CgroupRAM reads the tightest memory limit of this process' cgroups (v2, or v1 memory controller).
func CgroupRAM() (RAM, bool) {
	data, err := os.ReadFile(selfCgroup)
	if err != nil {
		return RAM{}, false
	}
	var best RAM
	found := false
	for _, line := range strings.Split(string(data), "\n") {
		parts := strings.SplitN(line, ":", 3)
		if len(parts) != 3 {
			continue
		}
		v2 := parts[1] == ""
		if !v2 && !strings.Contains(","+parts[1]+",", ",memory,") {
			continue
		}
		dir := cgroupRoot
		if !v2 {
			dir = filepath.Join(cgroupRoot, "memory")
		}
		rel := parts[2]
		for {
			p := dir + rel
			var lim, use uint64
			var ok1, ok2 bool
			cacheKey := "inactive_file"
			if v2 {
				lim, ok1 = readU64(p + "/memory.max")
				use, ok2 = readU64(p + "/memory.current")
			} else {
				lim, ok1 = readU64(p + "/memory.limit_in_bytes")
				use, ok2 = readU64(p + "/memory.usage_in_bytes")
				cacheKey = "total_inactive_file"
			}
			if ok1 && ok2 && lim < 1<<60 {
				cache := statValue(p+"/memory.stat", cacheKey)
				if use > cache {
					use -= cache
				} else {
					use = 0
				}
				r := RAM{Limit: lim, Used: use, Source: "cgroup", UsedFrom: "cgroup"}
				if !found || r.Free() < best.Free() {
					best = r
				}
				found = true
			}
			i := strings.LastIndex(rel, "/")
			if rel == "" || i < 0 {
				break
			}
			rel = rel[:i]
		}
	}
	return best, found
}

// PlanRAM combines the cgroup reading, the config's ram_limit and the system memory (total, free) as a
// fallback: the smaller limit wins, the used part stays the measured one.
func PlanRAM(cfg *Config, systemTotal, systemFree uint64) RAM {
	r, ok := CgroupRAM()
	if !ok {
		r = RAM{Source: "none"}
		if systemTotal > 0 {
			r = RAM{Limit: systemTotal, Used: systemTotal - min(systemFree, systemTotal), Source: "system", UsedFrom: "system"}
		}
	}
	if l := cfg.RAMLimitBytes(); l > 0 && (r.Limit == 0 || l < r.Limit) {
		r.Limit = l
		r.Source = "ram_limit"
	}
	return r
}

// RAMReserve is kept free besides a load: max(1 GiB, 5% of the limit), the same as the llama-server RAM guard.
func RAMReserve(limit uint64) uint64 {
	return max(uint64(1)<<30, limit/20)
}
