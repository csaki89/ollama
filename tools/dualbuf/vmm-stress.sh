#!/usr/bin/env bash
# VMM stress matrix, every run in its own process (a GPU page fault kills the process).
#   impl:    vmm-stress-impl, the real ggml-cuda arena and growable buffers through the ggml backend API, per GGML_CUDA_VMM_SYNC mode
#   pattern: vmm-stress, the bare HIP pattern without our code
# On the NAS from the image (the binaries are in it, nothing to clone or build):
#   docker run --rm --device /dev/kfd --device /dev/dri/renderD129 --group-add video -e HSA_OVERRIDE_GFX_VERSION=10.3.0 --entrypoint /bin/bash <image> /usr/lib/ollama/dualbuf-test/vmm-stress.sh
# From a checkout (pattern only, compiled here):
#   docker run --rm --device /dev/kfd --device /dev/dri/renderD129 --group-add video -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -v "$PWD":/src -w /src rocm/dev-ubuntu-24.04:7.2.1 bash tools/dualbuf/vmm-stress.sh
# Env: TESTS ("impl pattern"), RUNS (3), WAIT_MS (5), TARGET_GIB (6)
#      impl: IMPL_SYNCS ("auto none wait barrier both"), IMPL_CASES ("a b c d"), IMPL_CHUNK (64)
#      pattern: ARCH (gfx1030), SYNCS ("none wait barrier both"), CASES ("a b c d"), CHUNKS ("2 64 256 1024")
set -u
cd "$(dirname "$0")"
HERE=$PWD
export LD_LIBRARY_PATH="/usr/lib/ollama/rocm_v7_2:/usr/lib/ollama:${LD_LIBRARY_PATH:-}"
RUNS=${RUNS:-3}; WAIT_MS=${WAIT_MS:-5}; TARGET_GIB=${TARGET_GIB:-6}
SUM=/tmp/vmm-stress-summary.txt; : > "$SUM"

# impl_run <case> <sync>
impl_run() {
  local c=$1 s=$2 ok=0 r log rc info="" env_sync=()
  [ "$s" != auto ] && env_sync=("GGML_CUDA_VMM_SYNC=$s")
  for r in $(seq 1 "$RUNS"); do
    log="/tmp/vmm-impl-$c-$s-$r.log"
    env "${env_sync[@]}" GGML_CUDA_VMM_WAIT_MS="$WAIT_MS" "$HERE/vmm-stress-impl" "$c" --chunk-mib "${IMPL_CHUNK:-64}" --target-gib "$TARGET_GIB" > "$log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] && ok=$((ok + 1))
    if [ "$r" -eq 1 ]; then grep -E 'impl case|VRAM|filler|section at|FAIL|Memory access fault|ggml_cuda_vmm_sync' "$log"; fi
    [ "$rc" -ne 0 ] && info="last failure: exit $rc, $(grep -m1 -E 'FAIL|Memory access fault' "$log" | cut -c1-90)"
  done
  printf 'impl    %-5s %-8s %d/%d  %s\n' "$c" "$s" "$ok" "$RUNS" "$info" >> "$SUM"
}

# pattern_run <case> <sync> <chunk_mib> <batch>
pattern_run() {
  local c=$1 s=$2 ch=$3 b=$4 ok=0 r log rc info=""
  for r in $(seq 1 "$RUNS"); do
    log="/tmp/vmm-stress-$c-$s-$ch-$b-$r.log"
    "$PATTERN" "$c" --sync "$s" --wait-ms "$WAIT_MS" --chunk-mib "$ch" --batch "$b" --target-gib "$TARGET_GIB" > "$log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] && ok=$((ok + 1))
    if [ "$r" -eq 1 ]; then cat "$log"; fi
    [ "$rc" -ne 0 ] && info="last failure: exit $rc, $(grep -m1 -E 'FAIL|Memory access fault' "$log" | cut -c1-90)"
  done
  local peak; peak=$(grep -E 'VRAM (grown|grown again)' "/tmp/vmm-stress-$c-$s-$ch-$b-1.log" | sed -E 's/.*used\(sysfs\) *([0-9]+) MiB.*/\1/' | sort -n | tail -1)
  printf 'pattern %-5s %-8s %6s %6s  %d/%d  peak VRAM %6s MiB  %s\n' "$c" "$s" "$ch" "$b" "$ok" "$RUNS" "${peak:-?}" "$info" >> "$SUM"
}

for t in ${TESTS:-impl pattern}; do
  if [ "$t" = impl ]; then
    if [ ! -x "$HERE/vmm-stress-impl" ]; then echo "vmm-stress-impl is not here (it is built in the ROCm image), skipping impl"; continue; fi
    for s in ${IMPL_SYNCS:-auto none wait barrier both}; do
      for c in ${IMPL_CASES:-a b c d}; do impl_run "$c" "$s"; done
    done
  else
    PATTERN=$HERE/vmm-stress
    if [ ! -x "$PATTERN" ]; then
      HIPCC=/opt/rocm/bin/hipcc; [ -x "$HIPCC" ] || HIPCC=hipcc
      PATTERN=/tmp/vmm-stress
      "$HIPCC" -O2 --offload-arch="${ARCH:-gfx1030}" -o "$PATTERN" vmm-stress.cpp || { echo "build failed"; exit 1; }
    fi
    for s in ${SYNCS:-none wait barrier both}; do
      for c in ${CASES:-a b c d}; do
        case $c in
          a) for ch in ${CHUNKS:-2 64 256 1024}; do
               [ "$ch" -eq 2 ] && [ "$TARGET_GIB" -gt 1 ] && { TARGET_SAVE=$TARGET_GIB; TARGET_GIB=1; pattern_run a "$s" "$ch" 1; TARGET_GIB=$TARGET_SAVE; continue; }
               pattern_run a "$s" "$ch" 1
             done
             pattern_run a "$s" 64 8 ;;
          b) pattern_run b "$s" 64 1; pattern_run b "$s" 256 1 ;;
          c) pattern_run c "$s" 64 1 ;;
          d) pattern_run d "$s" 4 1; pattern_run d "$s" 4 8 ;;
        esac
      done
    done
  fi
done

echo
echo "=== summary: impl: case, sync, successful runs; pattern: case, sync, chunk MiB, batch, successful runs, peak VRAM (a 2 MiB chunk run is limited to 1 GiB)"
cat "$SUM"
