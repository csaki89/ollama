package multimodel

import (
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
)

// ConfigFileName is the file next to the models.
const ConfigFileName = "dualbuf.yaml"

// Store holds the config file read once, at the server start (Load). A missing file means no config; a broken
// file is an error of Load (the server does not start). The file is not read again while the server runs.
type Store struct {
	path string

	mu     sync.Mutex
	loaded bool
	cfg    *Config // nil = no config file
}

// NewStore returns a store for the file at path; nothing is read until Load.
func NewStore(path string) *Store {
	return &Store{path: path}
}

// DefaultPath is OLLAMA_DUALBUF_CONFIG, else <models dir>/dualbuf.yaml.
func DefaultPath(env Env, modelsDir string) string {
	if env != nil {
		if p := env("OLLAMA_DUALBUF_CONFIG"); p != "" {
			return p
		}
	}
	return filepath.Join(modelsDir, ConfigFileName)
}

// Path is the config file path.
func (s *Store) Path() string { return s.path }

// Load reads and validates the file (once; later calls return the first result). No file: no config, no error.
func (s *Store) Load() (*Config, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.loaded {
		return s.cfg, nil
	}
	data, err := os.ReadFile(s.path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			s.loaded = true
			slog.Info("dualbuf config: no file, the scheduler runs without it", "path", s.path)
			return nil, nil
		}
		return nil, fmt.Errorf("dualbuf config %s: %w", s.path, err)
	}
	c, err := Parse(data)
	if err != nil {
		return nil, fmt.Errorf("dualbuf config %s is invalid: %w", s.path, err)
	}
	s.loaded, s.cfg = true, c
	slog.Info("dualbuf config loaded", "path", s.path, "devices", len(c.Devices), "models", len(c.Models), "swap", c.SwapEnabled())
	return c, nil
}

// Current is the config read by Load (nil before Load, without a file, or after a failed Load).
func (s *Store) Current() *Config {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cfg
}
