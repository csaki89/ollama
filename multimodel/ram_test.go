package multimodel

import (
	"os"
	"path/filepath"
	"testing"
)

func writeFile(t *testing.T, p, s string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestCgroupRAMv2(t *testing.T) {
	dir := t.TempDir()
	oldRoot, oldSelf := cgroupRoot, selfCgroup
	t.Cleanup(func() { cgroupRoot, selfCgroup = oldRoot, oldSelf })
	cgroupRoot = dir
	selfCgroup = filepath.Join(dir, "self")
	writeFile(t, selfCgroup, "0::/docker/abc\n")
	// container: 100 GiB limit, 30 GiB used of which 10 GiB inactive page cache
	writeFile(t, filepath.Join(dir, "docker/abc/memory.max"), "107374182400\n")
	writeFile(t, filepath.Join(dir, "docker/abc/memory.current"), "32212254720\n")
	writeFile(t, filepath.Join(dir, "docker/abc/memory.stat"), "anon 1\ninactive_file 10737418240\nactive_file 5\n")
	// parent without a limit
	writeFile(t, filepath.Join(dir, "docker/memory.max"), "max\n")
	writeFile(t, filepath.Join(dir, "docker/memory.current"), "1\n")
	r, ok := CgroupRAM()
	if !ok || r.Limit != 100<<30 || r.Used != 20<<30 || r.Free() != 80<<30 {
		t.Fatalf("v2: %+v %v", r, ok)
	}
	c := mustParse(t, "default: {}\nram_limit: 50GiB\n")
	if p := PlanRAM(c, 0, 0); p.Limit != 50<<30 || p.Source != "ram_limit" || p.Used != 20<<30 || p.UsedFrom != "cgroup" {
		t.Fatalf("ram_limit: %+v", p)
	}
	if p := PlanRAM(nil, 0, 0); p.Limit != 100<<30 || p.Source != "cgroup" {
		t.Fatalf("plain: %+v", p)
	}
}

func TestCgroupRAMv1AndNone(t *testing.T) {
	dir := t.TempDir()
	oldRoot, oldSelf := cgroupRoot, selfCgroup
	t.Cleanup(func() { cgroupRoot, selfCgroup = oldRoot, oldSelf })
	cgroupRoot = dir
	selfCgroup = filepath.Join(dir, "self")
	writeFile(t, selfCgroup, "4:cpu,cpuacct:/x\n3:memory:/x\n")
	writeFile(t, filepath.Join(dir, "memory/x/memory.limit_in_bytes"), "1000\n")
	writeFile(t, filepath.Join(dir, "memory/x/memory.usage_in_bytes"), "600\n")
	writeFile(t, filepath.Join(dir, "memory/x/memory.stat"), "total_inactive_file 100\n")
	if r, ok := CgroupRAM(); !ok || r.Limit != 1000 || r.Used != 500 {
		t.Fatalf("v1: %+v %v", r, ok)
	}
	selfCgroup = filepath.Join(dir, "missing")
	if _, ok := CgroupRAM(); ok {
		t.Fatal("no cgroup file")
	}
	if p := PlanRAM(nil, 1000, 300); p.Source != "system" || p.Free() != 300 || p.UsedFrom != "system" {
		t.Fatalf("system: %+v", p)
	}
	// a ram_limit over the system reading: the usage is still the system's (it has the driver host memory already)
	c := mustParse(t, "default: {}\nram_limit: 800B\n")
	if p := PlanRAM(c, 1000, 300); p.Source != "ram_limit" || p.Limit != 800 || p.UsedFrom != "system" {
		t.Fatalf("ram_limit over system: %+v", p)
	}
}
