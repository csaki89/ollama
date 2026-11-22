#!/usr/bin/env bash
# Correctness test of a llama-server build on CPU, started with the args Ollama 0.40.16 uses.
# Usage: ct.sh <bin dir with llama-server.exe> <label> [mode...]   modes: base fork (default both)
# Results: ../results/<label>/<mode>/{chat-*.json,emb-*.json,*.log}
set -u
BIN=$1; LABEL=$2; shift 2; MODES=${*:-base fork}
HERE=$(cd "$(dirname "$0")" && pwd); M=${MODELS:-$HERE/models}; R=${RESULTS:-$HERE/results}/$LABEL
P=${PORT:-18111}
COMMON=(--port $P --host 127.0.0.1 --no-webui --offline -np 1 --load-mode none --flash-attn auto -ngl 0
        --log-verbosity 4 --no-log-prefix --no-log-timestamps)

start() { # $1 log, rest: args
  local log=$1; shift
  "$BIN/llama-server.exe" "${COMMON[@]}" "$@" > "$log" 2>&1 &
  PID=$!
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$P/health" > /dev/null && return 0
    kill -0 $PID 2>/dev/null || { echo "server died, see $log"; return 1; }
    sleep 1
  done
  echo "server not ready"; return 1
}
stop() { kill $PID 2>/dev/null; wait $PID 2>/dev/null; taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 1; }

post() { # $1 path, $2 json, $3 out
  curl -s "http://127.0.0.1:$P$1" -H 'Content-Type: application/json' --data-binary "$2" > "$3"
}

for mode in $MODES; do
  D=$R/$mode; mkdir -p "$D"
  if [ "$mode" = fork ]; then
    export LLAMA_ARG_DUAL_BUFFER=1 LLAMA_ARG_LOAD_MODE=none LLAMA_ARG_AUTO_ZONES=1
  else
    unset LLAMA_ARG_DUAL_BUFFER LLAMA_ARG_LOAD_MODE LLAMA_ARG_AUTO_ZONES
  fi
  echo "== $LABEL / $mode"

  # chat model: greedy, fixed seed, token ids
  if start "$D/chat.log" --model "$M/qwen3.gguf" -c 4096; then
    i=0
    for prompt in \
      '<|im_start|>user\nWhat is the capital of France? Answer in one sentence. /no_think<|im_end|>\n<|im_start|>assistant\n' \
      '<|im_start|>user\nWrite a haiku about a lighthouse. /no_think<|im_end|>\n<|im_start|>assistant\n' \
      'Once upon a time, in a small village by the sea,'; do
      i=$((i+1))
      body=$(node -e 'process.stdout.write(JSON.stringify({prompt:process.argv[1].replace(/\\n/g,"\n"),n_predict:48,temperature:0,top_k:1,seed:42,cache_prompt:false,return_tokens:true}))' "$prompt")
      post /completion "$body" "$D/chat-$i.json"
    done
    stop
  fi

  # embedding models, as Ollama: --embedding -b/-ub, /v1/embeddings with "input"
  for em in embeddinggemma-2 qwen3-embedding; do
    if start "$D/emb-$em.log" --model "$M/$em.gguf" -c 2048 --embedding -b 2048 -ub 2048; then
      i=0
      for text in "The quick brown fox jumps over the lazy dog." \
                  "Ollama runs large language models locally on CPUs and GPUs." \
                  "A dual buffer keeps two copies of the CPU weights."; do
        i=$((i+1))
        body=$(node -e 'process.stdout.write(JSON.stringify({input:process.argv[1]}))' "$text")
        post /v1/embeddings "$body" "$D/emb-$em-$i.json"
      done
      stop
    fi
  done
done
