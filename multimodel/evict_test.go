package multimodel

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestEvictionOrderAndPlan(t *testing.T) {
	t0 := time.Now()
	cands := []Candidate{
		{Key: "idle-old", Kind: KindIdle, LastUsed: t0.Add(-3 * time.Hour), Bytes: 10},
		{Key: "parked-new", Kind: KindParked, LastUsed: t0.Add(-time.Minute), Bytes: 5},
		{Key: "parked-old", Kind: KindParked, LastUsed: t0.Add(-time.Hour), Bytes: 5},
		{Key: "parked-forever", Kind: KindParked, LastUsed: t0.Add(-5 * time.Hour), Forever: true, Bytes: 50},
		{Key: "idle-forever", Kind: KindIdle, LastUsed: t0.Add(-5 * time.Hour), Forever: true, Bytes: 50},
		{Key: "cpu-forever", Kind: KindCPUFallback, LastUsed: t0, Forever: true, Bytes: 3},
		{Key: "cpu", Kind: KindCPUFallback, LastUsed: t0.Add(-time.Second), Bytes: 3},
	}
	var keys []string
	for _, c := range EvictionOrder(cands) {
		keys = append(keys, c.Key)
	}
	want := []string{"cpu", "cpu-forever", "parked-old", "parked-new", "idle-old"}
	if len(keys) != len(want) {
		t.Fatalf("order %v", keys)
	}
	for i := range want {
		if keys[i] != want[i] {
			t.Fatalf("order %v want %v", keys, want)
		}
	}
	ram := RAM{Limit: 100, Used: 95}
	if p := PlanEvictions(ram, 4, cands); p != nil {
		t.Fatalf("fits: %v", p)
	}
	p := PlanEvictions(ram, 14, cands) // free 5 + cpu 3 + cpu-forever 3 + parked-old 5 = 16
	if len(p) != 3 || p[2].Key != "parked-old" {
		t.Fatalf("plan %+v", p)
	}
	if p := PlanEvictions(ram, 1000, cands); len(p) != 5 {
		t.Fatalf("not enough: all candidates, got %d", len(p))
	}
	if p := PlanEvictions(RAM{}, 1000, cands); p != nil {
		t.Fatalf("no limit known: no plan")
	}
}

func TestKVFiles(t *testing.T) {
	from, to := t.TempDir(), t.TempDir()
	prefix := KVPrefix("sha256:0123456789abcdef0123", "gpu0", "")
	if prefix != "kv-0123456789abcdef-gpu0-f16" {
		t.Fatalf("prefix %s", prefix)
	}
	if err := os.WriteFile(filepath.Join(from, SlotFileName(prefix, 0)), []byte("kv"), 0o644); err != nil {
		t.Fatal(err)
	}
	meta := KVMeta{Digest: "sha256:0123", KVType: "f16", Slots: []int{0}, Tokens: []int{12}}
	if err := WriteMeta(from, prefix, meta); err != nil {
		t.Fatal(err)
	}
	if err := CopyKV(from, to, prefix); err != nil {
		t.Fatal(err)
	}
	got, err := ReadMeta(to, prefix)
	if err != nil || got.Tokens[0] != 12 {
		t.Fatalf("meta %+v %v", got, err)
	}
	if b, _ := os.ReadFile(filepath.Join(to, SlotFileName(prefix, 0))); string(b) != "kv" {
		t.Fatalf("slot file %q", b)
	}
	if err := got.Compatible("sha256:0123", "f16"); err != nil {
		t.Fatal(err)
	}
	if got.Compatible("sha256:0123", "q8_0") == nil || got.Compatible("sha256:9999", "f16") == nil {
		t.Fatal("incompatible KV accepted")
	}
	RemoveKV(to, prefix)
	if _, err := os.Stat(filepath.Join(to, SlotFileName(prefix, 0))); err == nil {
		t.Fatal("slot file not removed")
	}
}

func TestManagerParkAndPresent(t *testing.T) {
	m := NewManager(filepath.Join(t.TempDir(), ConfigFileName), nil)
	c := mustParse(t, exampleConfig)
	if got := m.Present(c, []string{"gpu0", "cpu"}); len(got) != 2 {
		t.Fatalf("before discovery all devices count as present: %v", got)
	}
	m.NoteDevices(nil)
	if m.HaveDevices() {
		t.Fatal("empty discovery is not recorded")
	}
	m.Park(&Parked{Key: "k", Model: "a"})
	if len(m.ParkedList()) != 1 {
		t.Fatal("park")
	}
	m.Drop("k")
	if len(m.ParkedList()) != 0 || m.Unpark("k") != nil {
		t.Fatal("drop")
	}
	m.NoteHost("k", 42)
	if m.HostEstimate("k", 7) != 42 || m.HostEstimate("other", 7) != 7 {
		t.Fatal("host estimate")
	}
}
