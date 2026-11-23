package multimodel

import (
	"errors"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ConfigFileName is the file next to the models.
const ConfigFileName = "dualbuf.yaml"

// Store reads the config file when asked (at a model load), keeps the last valid version, and reports a
// broken file in the log. A missing file means no config.
type Store struct {
	path string

	mu      sync.Mutex
	modTime time.Time
	size    int64
	present bool
	valid   *Config // last valid config, nil = none yet
	lastErr string  // the error of the current (broken) file, "" when the file is valid
}

// NewStore returns a store for the file at path.
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

// Current returns the config to use now: the file if it is valid, else the last valid one (with an error
// line in the log every time the broken file is used), nil when there is no file and none was valid.
func (s *Store) Current() *Config {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	st, err := os.Stat(s.path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			if s.present {
				slog.Info("dualbuf config removed, the scheduler runs without it", "path", s.path)
			}
			s.present, s.valid, s.lastErr = false, nil, ""
			return nil
		}
		return s.brokenLocked(err.Error())
	}
	if s.present && st.ModTime().Equal(s.modTime) && st.Size() == s.size {
		if s.lastErr != "" {
			return s.brokenLocked(s.lastErr)
		}
		return s.valid
	}
	s.present, s.modTime, s.size = true, st.ModTime(), st.Size()
	data, err := os.ReadFile(s.path)
	if err != nil {
		return s.brokenLocked(err.Error())
	}
	c, err := Parse(data)
	if err != nil {
		return s.brokenLocked(err.Error())
	}
	s.valid, s.lastErr = c, ""
	slog.Info("dualbuf config loaded", "path", s.path, "devices", len(c.Devices), "models", len(c.Models), "swap", c.SwapEnabled())
	return c
}

func (s *Store) brokenLocked(msg string) *Config {
	s.lastErr = msg
	if s.valid != nil {
		slog.Error("DUALBUF CONFIG INVALID, the model loads with the last valid config", "path", s.path, "error", msg)
	} else {
		slog.Error("DUALBUF CONFIG INVALID, no valid config yet, the model loads without it", "path", s.path, "error", msg)
	}
	return s.valid
}
