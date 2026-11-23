package multimodel

import (
	"sort"
	"time"
)

// Kind is what a RAM eviction candidate is; the lower kinds go first.
type Kind int

const (
	KindCPUFallback Kind = iota // a CPU instance of a model whose first device is a GPU (the demoted copy)
	KindParked                  // a parked model: no process, its file pages and KV kept in RAM
	KindIdle                    // an idle loaded model
)

func (k Kind) String() string {
	switch k {
	case KindCPUFallback:
		return "cpu-fallback"
	case KindParked:
		return "parked"
	case KindIdle:
		return "idle"
	}
	return "unknown"
}

// Candidate is a model that could leave the RAM.
type Candidate struct {
	Key      string
	Model    string
	Kind     Kind
	LastUsed time.Time
	Forever  bool   // keep_alive -1: never unloaded completely
	Bytes    uint64 // host RAM it holds (measured)
	Ref      any    // the scheduler's object
}

// EvictionOrder: CPU fallback instances first, then parked, then idle models; least recently used first
// within a kind. keep_alive -1 models are left out, except their CPU fallback instances (the extra copy).
func EvictionOrder(cands []Candidate) []Candidate {
	out := make([]Candidate, 0, len(cands))
	for _, c := range cands {
		if c.Forever && c.Kind != KindCPUFallback {
			continue
		}
		out = append(out, c)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		return out[i].LastUsed.Before(out[j].LastUsed)
	})
	return out
}

// PlanEvictions picks candidates in EvictionOrder until need bytes are free (ram.Free() plus what the picked
// ones hold). It returns nil when the need already fits; when even all of them are not enough it returns
// all of them (the load goes ahead after that, the llama-server RAM guard trims its copies).
func PlanEvictions(ram RAM, need uint64, cands []Candidate) []Candidate {
	if ram.Limit == 0 || ram.Free() >= need {
		return nil
	}
	free := ram.Free()
	var picked []Candidate
	for _, c := range EvictionOrder(cands) {
		picked = append(picked, c)
		free += c.Bytes
		if free >= need {
			break
		}
	}
	return picked
}
