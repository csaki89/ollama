#!/usr/bin/env bash
# Dynamic KV paths on CPU: a ~5400 token prompt (beyond the initial 4096 cells) must grow the KV (auto zones) and give the same tokens as the static KV.
# Usage: kv.sh <bin dir> <label>
set -u
BIN=$1; LABEL=$2
HERE=$(cd "$(dirname "$0")" && pwd); M=${MODELS:-$HERE/models}; R=${RESULTS:-$HERE/results}/$LABEL/kv; mkdir -p "$R"
P=18112
PROMPT=$(node -e 'let s="";for(let i=0;i<230;i++)s+=`Paragraph ${i}: the lighthouse keeper wrote down the weather, the ships and the colour of the sea. `;process.stdout.write(s+"\nSummarize the notes above in one sentence:")')
BODY=$(node -e 'process.stdout.write(JSON.stringify({prompt:process.argv[1],n_predict:32,temperature:0,top_k:1,seed:42,cache_prompt:false,return_tokens:true}))' "$PROMPT")
run() { # $1 name, env assignments follow
  local n=$1; shift
  env "$@" "$BIN/llama-server.exe" --model "$M/qwen3.gguf" --port $P --host 127.0.0.1 --no-webui --offline -np 1 \
      --load-mode none --flash-attn auto -ngl 0 -c 8192 --log-verbosity 4 --no-log-prefix --no-log-timestamps > "$R/$n.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$P/health" > /dev/null && break; sleep 1; done
  curl -s "http://127.0.0.1:$P/completion" -H 'Content-Type: application/json' --data-binary "$BODY" > "$R/$n.json"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 1
  node -e 'const r=JSON.parse(require("fs").readFileSync(process.argv[1]));console.log(process.argv[2].padEnd(10),"prompt_n="+r.timings.prompt_n,"tokens="+r.tokens.join(",").slice(0,60))' "$R/$n.json" "$n"
  sed 's/\x1b\[[0-9;]*m//g' "$R/$n.log" | grep -iE "dynamic KV|resize|grow|kv.*(cells|MiB).*->|zones_state|vmm|fallback" | sort | uniq -c | head -8
}
run static
run auto      LLAMA_ARG_AUTO_ZONES=1
run auto_novmm LLAMA_ARG_AUTO_ZONES=1 LLAMA_KV_NO_VMM=1
run sim1      LLAMA_ARG_AUTO_ZONES=1 LLAMA_KV_VMM_SIM=1
run sim2      LLAMA_ARG_AUTO_ZONES=1 LLAMA_KV_VMM_SIM=2
run sim3      LLAMA_ARG_AUTO_ZONES=1 LLAMA_KV_VMM_SIM=3
run dual_auto LLAMA_ARG_AUTO_ZONES=1 LLAMA_ARG_DUAL_BUFFER=1 LLAMA_ZONES_SHRINK=0 LLAMA_ZONES_SHRINK_IDLE=5
