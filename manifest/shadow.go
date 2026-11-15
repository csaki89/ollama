package manifest

import (
	"log/slog"
	"strings"

	"github.com/ollama/ollama/types/model"
)

// dualbuf: the local compat migration keeps a llamacpp:<hex> legacy tag for
// every converted child (a downgrade guard). Nothing removed it with the model,
// so after `ollama rm` it kept the converted blobs alive.

// IsRunnerShadow reports whether name is the llamacpp:<hex> tag of the
// manifest with the given digest.
func IsRunnerShadow(name model.Name, digest string) bool {
	hex := strings.TrimPrefix(strings.ToLower(digest), "sha256:")
	return hex != "" && name.EqualFold(model.ParseName(RunnerLlamaCPP+":"+hex))
}

// ManifestListChildDigests returns the child digests of the named manifest
// list (none for other manifests).
func ManifestListChildDigests(n model.Name) []string {
	data, err := ReadManifestData(n)
	if err != nil {
		return nil
	}
	m, err := parseManifest(data)
	if err != nil || m.MediaType != MediaTypeManifestList {
		return nil
	}
	var digests []string
	for _, child := range m.Manifests {
		if digest, err := ChildManifestDigest(child); err == nil {
			digests = append(digests, digest)
		}
	}
	return digests
}

// RemoveOrphanRunnerShadows removes the llamacpp:<hex> tags whose child no
// manifest list references, and the blobs only they kept. With digests, only
// the tags of those children (a removed model); without, every orphan (at
// startup, when no migration is writing one).
func RemoveOrphanRunnerShadows(digests ...string) ([]string, error) {
	refs, err := namedManifestRefs(true)
	if err != nil {
		return nil, err
	}

	hexOf := func(d string) string { return strings.TrimPrefix(strings.ToLower(d), "sha256:") }
	listed := make(map[string]struct{})
	shadows := make(map[model.Name]string)
	for n, ref := range refs {
		data, _, digest, err := readVerifiedManifest(ref.path, ref.root)
		if err != nil {
			continue
		}
		if IsRunnerShadow(n, digest) {
			shadows[n] = hexOf(digest)
			continue
		}
		m, err := parseManifest(data)
		if err != nil || m.MediaType != MediaTypeManifestList {
			continue
		}
		for _, child := range m.Manifests {
			if d, err := ChildManifestDigest(child); err == nil {
				listed[hexOf(d)] = struct{}{}
			}
		}
	}

	only := make(map[string]struct{})
	for _, d := range digests {
		only[hexOf(d)] = struct{}{}
	}

	var removed []string
	for n, hex := range shadows {
		if _, ok := listed[hex]; ok {
			continue
		}
		if _, ok := only[hex]; len(only) > 0 && !ok {
			continue
		}
		blobs, err := RemoveNamed(n)
		removed = append(removed, blobs...)
		if err != nil {
			return removed, err
		}
		slog.Info("removed converted manifest shadow", "name", n.DisplayShortest(), "blobs", len(blobs))
	}
	return removed, nil
}
