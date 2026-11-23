//go:build linux

package multimodel

import (
	"fmt"
	"os"
	"syscall"
)

// Pinned is a model file mapped and locked in RAM (its page cache cannot be evicted), so that a parked
// model reloads from RAM. Needs RLIMIT_MEMLOCK (the compose sets memlock: -1).
type Pinned struct {
	Path  string
	Bytes uint64
	data  []byte
}

// PinFile maps the file read-only and locks its pages; this reads the whole file into the page cache.
func PinFile(path string) (*Pinned, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() == 0 {
		return &Pinned{Path: path}, nil
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	if err := syscall.Mlock(data); err != nil {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("mlock %s (%d bytes, check the memlock limit): %w", path, st.Size(), err)
	}
	return &Pinned{Path: path, Bytes: uint64(st.Size()), data: data}, nil
}

// Release unlocks and unmaps the file.
func (p *Pinned) Release() {
	if p == nil || p.data == nil {
		return
	}
	_ = syscall.Munlock(p.data)
	_ = syscall.Munmap(p.data)
	p.data = nil
}
