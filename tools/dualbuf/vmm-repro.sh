#!/usr/bin/env bash
# Builds tools/dualbuf/vmm-repro.cpp and runs the four cases, each in its own process (a GPU page fault kills the process).
# On the NAS, from a checkout of the dual-buffer branch, in one command:
#   docker run --rm --device /dev/kfd --device /dev/dri/renderD129 --group-add video -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
#     -v "$PWD":/src -w /src rocm/dev-ubuntu-24.04:7.2.1 bash tools/dualbuf/vmm-repro.sh
# Env: ARCH (gfx1030), ARGS (extra arguments: sections section_mib chunk_mib, default 96 32 4)
set -u
cd "$(dirname "$0")"
HIPCC=/opt/rocm/bin/hipcc; [ -x "$HIPCC" ] || HIPCC=hipcc
OUT=/tmp/vmm-repro
"$HIPCC" -O2 --offload-arch="${ARCH:-gfx1030}" -o "$OUT" vmm-repro.cpp || { echo "build failed"; exit 1; }
declare -A RES
for c in a b c d; do
  echo "=== case $c"
  "$OUT" "$c" ${ARGS:-} 2>&1 | tee "/tmp/vmm-repro-$c.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then RES[$c]="OK"; else RES[$c]="FAILED (exit $rc)"; fi
done
echo
echo "=== summary"
echo "a  old: 96 ranges, 4 MiB chunk, async zeroing on an own stream : ${RES[a]}"
echo "b  new: one range, sections, 2 MiB aligned chunks, sync zeroing : ${RES[b]}"
echo "c  old with synchronous zeroing                                : ${RES[c]}"
echo "d  old with alignment max(recommended, 2 MiB)                  : ${RES[d]}"
