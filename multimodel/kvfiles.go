package multimodel

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// KVMeta describes a saved KV (one model instance, all its slots); it sits next to the slot files.
type KVMeta struct {
	Model     string    `json:"model"`
	Digest    string    `json:"digest"`
	Placement string    `json:"placement"`
	KVType    string    `json:"kv_type"`
	NumCtx    int       `json:"num_ctx"`
	Parallel  int       `json:"parallel"`
	Slots     []int     `json:"slots"`
	Tokens    []int     `json:"tokens"`
	Bytes     int64     `json:"bytes"`
	SavedAt   time.Time `json:"saved_at"`
	Reason    string    `json:"reason"` // swap, unload
}

var unsafeName = regexp.MustCompile(`[^A-Za-z0-9_.-]+`)

// KVPrefix is the file prefix of a model instance's saved KV: digest, placement and KV type, so a restore
// only finds a KV of the same model and KV type.
func KVPrefix(digest, placement, kvType string) string {
	d := strings.TrimPrefix(digest, "sha256:")
	if len(d) > 16 {
		d = d[:16]
	}
	if placement == "" {
		placement = "auto"
	}
	if kvType == "" {
		kvType = "f16"
	}
	return unsafeName.ReplaceAllString(fmt.Sprintf("kv-%s-%s-%s", d, placement, kvType), "_")
}

// RAMDir is the directory for the KV kept in RAM (tmpfs) and for the llama-server --slot-save-path.
func RAMDir() string {
	if st, err := os.Stat("/dev/shm"); err == nil && st.IsDir() {
		return "/dev/shm/ollama-dualbuf-kv"
	}
	return filepath.Join(os.TempDir(), "ollama-dualbuf-kv")
}

// NVMeDir is swap.kv_dir, or <models>/dualbuf-kv.
func (c *Config) NVMeDir(modelsDir string) string {
	if c != nil && c.Swap.KVDir != "" {
		return c.Swap.KVDir
	}
	return filepath.Join(modelsDir, "dualbuf-kv")
}

func metaPath(dir, prefix string) string { return filepath.Join(dir, prefix+".json") }

// WriteMeta writes the sidecar of a saved KV.
func WriteMeta(dir, prefix string, m KVMeta) error {
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(metaPath(dir, prefix), b, 0o644)
}

// ReadMeta reads the sidecar of a saved KV.
func ReadMeta(dir, prefix string) (KVMeta, error) {
	var m KVMeta
	b, err := os.ReadFile(metaPath(dir, prefix))
	if err != nil {
		return m, err
	}
	err = json.Unmarshal(b, &m)
	return m, err
}

// Compatible reports whether a saved KV may be restored into an instance: the same model digest and KV
// type. A smaller context is allowed by llama-server only for slots whose tokens fit (it refuses the rest).
func (m KVMeta) Compatible(digest, kvType string) error {
	if m.Digest != digest {
		return fmt.Errorf("saved for another model digest (%s)", m.Digest)
	}
	want := kvType
	if want == "" {
		want = "f16"
	}
	have := m.KVType
	if have == "" {
		have = "f16"
	}
	if have != want {
		return fmt.Errorf("saved with kv_type %s, the instance has %s", have, want)
	}
	return nil
}

// SlotFileName is the file of one slot, as llm.SaveSlots names it.
func SlotFileName(prefix string, slot int) string {
	return fmt.Sprintf("%s-slot%d.bin", prefix, slot)
}

// CopyKV copies a saved KV (sidecar and slot files) between directories, e.g. RAM -> NVMe before an unload,
// NVMe -> RAM before a restore. The files are written under a temporary name and renamed.
func CopyKV(fromDir, toDir, prefix string) error {
	m, err := ReadMeta(fromDir, prefix)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(toDir, 0o755); err != nil {
		return err
	}
	for _, s := range m.Slots {
		if err := copyFile(filepath.Join(fromDir, SlotFileName(prefix, s)), filepath.Join(toDir, SlotFileName(prefix, s))); err != nil {
			return err
		}
	}
	return copyFile(metaPath(fromDir, prefix), metaPath(toDir, prefix))
}

// RemoveKV deletes a saved KV.
func RemoveKV(dir, prefix string) {
	m, err := ReadMeta(dir, prefix)
	if err == nil {
		for _, s := range m.Slots {
			_ = os.Remove(filepath.Join(dir, SlotFileName(prefix, s)))
		}
	}
	_ = os.Remove(metaPath(dir, prefix))
}

func copyFile(from, to string) error {
	src, err := os.Open(from)
	if err != nil {
		return err
	}
	defer src.Close()
	tmp := to + ".tmp"
	dst, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, err = io.Copy(dst, src)
	err = errors.Join(err, dst.Close())
	if err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, to)
}
