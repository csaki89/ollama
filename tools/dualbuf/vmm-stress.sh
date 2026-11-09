#!/usr/bin/env bash
# Builds tools/dualbuf/vmm-stress.cpp and runs the matrix, every run in its own process (a GPU page fault kills the process).
# On the NAS, from a checkout of the dual-buffer branch, in one command:
#   docker run --rm --device /dev/kfd --device /dev/dri/renderD129 --group-add video -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -v "$PWD":/src -w /src rocm/dev-ubuntu-24.04:7.2.1 bash tools/dualbuf/vmm-stress.sh
# Env: ARCH (gfx1030), RUNS (3), SYNCS ("none wait barrier both"), WAIT_MS (5), CASES ("a b c d"), CHUNKS ("2 64 256 1024"), TARGET_GIB (6)
set -u
cd "$(dirname "$0")"
HIPCC=/opt/rocm/bin/hipcc; [ -x "$HIPCC" ] || HIPCC=hipcc
OUT=/tmp/vmm-stress
"$HIPCC" -O2 --offload-arch="${ARCH:-gfx1030}" -o "$OUT" vmm-stress.cpp || { echo "build failed"; exit 1; }
RUNS=${RUNS:-3}; WAIT_MS=${WAIT_MS:-5}; TARGET_GIB=${TARGET_GIB:-6}
SUM=/tmp/vmm-stress-summary.txt; : > "$SUM"

# run <case> <sync> <chunk_mib> <batch>
run() {
  local c=$1 s=$2 ch=$3 b=$4 ok=0 r log rc info=""
  for r in $(seq 1 "$RUNS"); do
    log="/tmp/vmm-stress-$c-$s-$ch-$b-$r.log"
    "$OUT" "$c" --sync "$s" --wait-ms "$WAIT_MS" --chunk-mib "$ch" --batch "$b" --target-gib "$TARGET_GIB" > "$log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] && ok=$((ok + 1))
    if [ "$r" -eq 1 ]; then cat "$log"; fi
    [ "$rc" -ne 0 ] && info="last failure: exit $rc, $(grep -m1 -E 'FAIL|Memory access fault' "$log" | cut -c1-90)"
  done
  local peak; peak=$(grep -E 'VRAM (grown|grown again)' "/tmp/vmm-stress-$c-$s-$ch-$b-1.log" | sed -E 's/.*used\(sysfs\) *([0-9]+) MiB.*/\1/' | sort -n | tail -1)
  printf '%-5s %-8s %6s %6s  %d/%d  peak VRAM %6s MiB  %s\n' "$c" "$s" "$ch" "$b" "$ok" "$RUNS" "${peak:-?}" "$info" >> "$SUM"
}

for s in ${SYNCS:-none wait barrier both}; do
  for c in ${CASES:-a b c d}; do
    case $c in
      a) for ch in ${CHUNKS:-2 64 256 1024}; do
           [ "$ch" -eq 2 ] && [ "$TARGET_GIB" -gt 1 ] && { TARGET_SAVE=$TARGET_GIB; TARGET_GIB=1; run a "$s" "$ch" 1; TARGET_GIB=$TARGET_SAVE; continue; }
           run a "$s" "$ch" 1
         done
         run a "$s" 64 8 ;;
      b) run b "$s" 64 1; run b "$s" 256 1 ;;
      c) run c "$s" 64 1 ;;
      d) run d "$s" 4 1; run d "$s" 4 8 ;;
    esac
  done
done

echo
echo "=== summary: case, sync, chunk MiB, batch, successful runs, peak VRAM (a 2 MiB chunk run is limited to 1 GiB)"
cat "$SUM"
