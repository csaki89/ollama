package compatmigrate

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// dualbuf: after a local compat migration the converted child replaces the
// source. The manifest list keeps neither the source child nor the legacy
// downgrade anchor, so the source GGUF (often tens of GB) is removed unless
// another manifest still references it.

// replaceSourceWithConverted writes refs as the manifest list of name without
// the child of source, then removes the source blobs no manifest references.
func replaceSourceWithConverted(name model.Name, source *manifest.Manifest, refs []manifest.Manifest) (bool, error) {
	data, err := json.Marshal(source)
	if err != nil {
		return false, err
	}
	sourceDigest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))

	kept := make([]manifest.Manifest, 0, len(refs))
	for _, ref := range refs {
		if manifest.SameDigest(ref.BlobDigest(), sourceDigest) {
			continue
		}
		kept = append(kept, ref)
	}
	if _, err := manifest.WriteManifestList(name, kept); err != nil {
		return false, err
	}
	removeSourceBlobs(name, []*manifest.Manifest{source}, []string{sourceDigest})
	return true, nil
}

func removeSourceBlobs(name model.Name, sources []*manifest.Manifest, manifestDigests []string) {
	candidates := append([]string{}, manifestDigests...)
	for _, m := range sources {
		if m.Config.Digest != "" {
			candidates = append(candidates, m.Config.Digest)
		}
		for _, layer := range m.Layers {
			candidates = append(candidates, layer.Digest)
		}
	}
	removed, err := manifest.RemoveUnreferencedBlobs(candidates...)
	if err != nil {
		slog.Warn("could not remove the source blobs of the local compat migration", "model", name.DisplayShortest(), "error", err)
	}
	slog.Info("removed the source blobs of the local compat migration", "model", name.DisplayShortest(), "blobs", len(removed))
}

// CleanupMigratedSources applies the same to the models an older build
// migrated: a manifest list with a complete llamacpp child that has its
// llamacpp:<hex> shadow (written only by the local migration) drops its ggml
// GGUF children. Run at startup, before the blob prune.
func CleanupMigratedSources() {
	ms, err := manifest.Manifests(true)
	if err != nil {
		slog.Warn("could not list manifests for the compat migration cleanup", "error", err)
		return
	}
	for name, mf := range ms {
		if manifest.IsRunnerShadow(name, mf.Digest()) {
			continue
		}
		if err := cleanupMigratedSource(name); err != nil {
			slog.Warn("compat migration cleanup failed", "model", name.DisplayShortest(), "error", err)
		}
	}
}

func cleanupMigratedSource(name model.Name) error {
	unlock := lockCompatibilityMigration(name)
	defer unlock()

	data, err := manifest.ReadManifestData(name)
	if err != nil {
		return nil
	}
	var parent manifest.Manifest
	if err := json.Unmarshal(data, &parent); err != nil || parent.MediaType != manifest.MediaTypeManifestList {
		return nil
	}

	converted := false
	var kept []manifest.Manifest
	var sources []*manifest.Manifest
	var sourceDigests []string
	for _, child := range parent.Manifests {
		resolved, err := resolveChildManifest(child)
		if err != nil {
			kept = append(kept, child)
			continue
		}
		if isRunnerFormat(resolved, manifest.RunnerGGML, manifest.FormatGGUF) {
			sources = append(sources, resolved)
			sourceDigests = append(sourceDigests, child.BlobDigest())
			continue
		}
		if isRunnerFormat(resolved, manifest.RunnerLlamaCPP, manifest.FormatGGUF) && manifestBlobsExist(resolved) {
			if shadow, err := convertedLegacyShadowName(child.BlobDigest()); err == nil {
				if _, err := manifest.ReadManifestData(shadow); err == nil {
					converted = true
				}
			}
		}
		kept = append(kept, child)
	}
	if !converted || len(sources) == 0 {
		return nil
	}

	if _, err := manifest.WriteManifestList(name, kept); err != nil {
		return err
	}
	removeSourceBlobs(name, sources, sourceDigests)
	return nil
}
