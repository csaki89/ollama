#!/usr/bin/env bash
# VMM stress matrix, every run in its own process (a GPU page fault kills the process).
#   impl:    vmm-stress-impl, the real ggml-cuda arena and growable buffers through the ggml backend API, per GGML_CUDA_VMM_SYNC mode
#   pattern: vmm-stress, the bare HIP pattern without our code (same cases, same steps)
# In the running app container (unload the models first, the test fills the free VRAM):
#   docker exec ollama-dualbuf bash /usr/lib/ollama/dualbuf-test/vmm-stress.sh
#   docker exec -e QUICK=1 ollama-dualbuf bash /usr/lib/ollama/dualbuf-test/vmm-stress.sh   (one run per case)
# From a checkout (pattern only, compiled here):
#   docker run --rm --device /dev/kfd --device /dev/dri/renderD129 --group-add video -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -v "$PWD":/src -w /src rocm/dev-ubuntu-24.04:7.2.1 bash tools/dualbuf/vmm-stress.sh
# Env: QUICK (1: one run per case), RUNS (3), TESTS ("impl pattern"), SYNCS ("none wait barrier both"), IMPL_CASES ("a b c d e"), CASES (pattern, "a b c d"),
#      WAIT_MS (5), CHUNK_MIB (64), TARGET_GIB (6), ARCH (gfx1030, only for compiling the pattern)
set -u
cd "$(dirname "$0")"
HERE=$PWD
export LD_LIBRARY_PATH="/usr/lib/ollama/rocm_v7_2:/usr/lib/ollama:${LD_LIBRARY_PATH:-}"
unset GGML_CUDA_VMM_DEBUG GGML_CUDA_VMM_SYNC GGML_CUDA_VMM_WAIT_MS  # logging would delay the first access; the mode is set per run
RUNS=${RUNS:-3}; [ "${QUICK:-0}" = 1 ] && RUNS=1
WAIT_MS=${WAIT_MS:-5}; CHUNK_MIB=${CHUNK_MIB:-64}; TARGET_GIB=${TARGET_GIB:-6}
SYNCS=${SYNCS:-none wait barrier both}; CASES=${CASES:-a b c d}; IMPL_CASES=${IMPL_CASES:-a b c d e}; TESTS=${TESTS:-impl pattern}
SUM=/tmp/vmm-stress-summary.txt; : > "$SUM"

PATTERN=$HERE/vmm-stress
if [[ " $TESTS " == *" pattern "* ]] && [ ! -x "$PATTERN" ]; then
  HIPCC=/opt/rocm/bin/hipcc; [ -x "$HIPCC" ] || HIPCC=hipcc
  PATTERN=/tmp/vmm-stress
  "$HIPCC" -O2 --offload-arch="${ARCH:-gfx1030}" -o "$PATTERN" vmm-stress.cpp || { echo "build failed"; exit 1; }
fi
if [[ " $TESTS " == *" impl "* ]] && [ ! -x "$HERE/vmm-stress-impl" ]; then
  echo "vmm-stress-impl is not here (it is built in the ROCm image), skipping impl"; TESTS=${TESTS//impl/}
fi

n_syncs=$(echo $SYNCS | wc -w); total=0
for t in $TESTS; do
  if [ "$t" = impl ]; then cs=$IMPL_CASES; else cs=$CASES; fi
  total=$((total + n_syncs*$(echo $cs | wc -w)*RUNS))
done
echo "=== $total runs ($TESTS; modes: $SYNCS; cases: impl $IMPL_CASES, pattern $CASES; $RUNS each), estimated $((total*3/60 + 1)) min at ~3 s per run"
[ "$RUNS" -gt 1 ] && echo "    (QUICK=1: one run per case, about $((total/RUNS*3/60 + 1)) min)"
T0=$(date +%s); done_runs=0

# run <test> <case> <sync>
run() {
  local t=$1 c=$2 s=$3 ok=0 r log rc info="" gap="" back=""
  for r in $(seq 1 "$RUNS"); do
    log="/tmp/vmm-$t-$c-$s-$r.log"
    if [ "$t" = impl ]; then
      GGML_CUDA_VMM_SYNC=$s GGML_CUDA_VMM_WAIT_MS=$WAIT_MS "$HERE/vmm-stress-impl" "$c" --chunk-mib "$CHUNK_MIB" --target-gib "$TARGET_GIB" > "$log" 2>&1
    else
      "$PATTERN" "$c" --sync "$s" --wait-ms "$WAIT_MS" --chunk-mib "$CHUNK_MIB" --batch 4 --target-gib "$TARGET_GIB" > "$log" 2>&1
    fi
    rc=$?
    done_runs=$((done_runs + 1))
    if [ "$rc" -eq 0 ]; then
      ok=$((ok + 1))
      gap=$(grep -m1 'mapping end -> first access' "$log" | sed -E 's/.*access: //')
      back=$(grep -m1 -E 'came back in [0-9]+ ms; run time' "$log" | sed -E 's/.*came back in ([0-9]+) ms; run time (.*)/VRAM back \1 ms, \2/')
    else
      info="last failure: exit $rc, $(grep -m1 -E 'FAIL|Memory access fault|Aborted' "$log" | cut -c1-100)"
    fi
  done
  printf '%-7s %-1s %-8s %d/%d  %s\n' "$t" "$c" "$s" "$ok" "$RUNS" "${info:-$back}" | tee -a "$SUM"
  [ -n "$gap" ] && printf '                     gap: %s\n' "$gap" | tee -a "$SUM"
  printf '%s %s %s %d %d\n' "$t" "$c" "$s" "$ok" "$RUNS" >> /tmp/vmm-stress-counts.txt
  echo "    ($done_runs/$total runs, $(( $(date +%s) - T0 )) s)"
}

: > /tmp/vmm-stress-counts.txt
for t in $TESTS; do
  for s in $SYNCS; do
    if [ "$t" = impl ]; then cs=$IMPL_CASES; else cs=$CASES; fi
    for c in $cs; do run "$t" "$c" "$s"; done
  done
done

echo
echo "=== summary ($(( $(date +%s) - T0 )) s): test, case, sync mode, successful runs, VRAM return / run time or the failure; gap: mapping end -> first access"
cat "$SUM"
for t in $TESTS; do
  if grep -q "^$t [a-e] none " /tmp/vmm-stress-counts.txt; then
    if awk -v t="$t" '$1 == t && $3 == "none" && $4 != $5 { f = 1 } END { exit f ? 1 : 0 }' /tmp/vmm-stress-counts.txt; then
      echo "WARNING ($t): the none mode had no failure, so the test is not sensitive enough here: the success of the other modes proves nothing"
    else
      echo "($t: the none mode failed, the test is sensitive; the other modes decide)"
    fi
  fi
done
