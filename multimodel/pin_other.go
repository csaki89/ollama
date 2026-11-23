//go:build !linux

package multimodel

import "errors"

// Pinned is a model file locked in RAM (Linux only).
type Pinned struct {
	Path  string
	Bytes uint64
}

// PinFile is not supported here: the parked model reloads from the page cache or the disk.
func PinFile(path string) (*Pinned, error) {
	return nil, errors.New("locking the model file in RAM is supported on Linux only")
}

// Release does nothing.
func (p *Pinned) Release() {}
