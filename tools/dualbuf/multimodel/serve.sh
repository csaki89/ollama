#!/bin/bash
# start the laptop ollama (multimodel build) on 127.0.0.1:11500, log to serve.log
S=$(cd "$(dirname "$0")" && pwd)
export OLLAMA_MODELS=$(cygpath -w $S/models) OLLAMA_HOST=127.0.0.1:11500 LLAMA_ARG_LOAD_MODE=none OLLAMA_NOPRUNE=1 OLLAMA_IGPU_ENABLE=1
for kv in "$@"; do export "$kv"; done
$S/ollama.exe serve > $S/serve.log 2>&1 &
echo $! > $S/serve.pid
until curl -s localhost:11500/api/version >/dev/null; do sleep 0.5; done
echo started
