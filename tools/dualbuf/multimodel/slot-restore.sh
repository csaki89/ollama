#!/bin/bash
# slot save in one llama-server, restore in a new one with another -c (and auto zones)
BIN=${BIN:?set BIN to the llama-server binary}
M=${MODEL:?set MODEL to a GGUF (qwen3:0.6b)}
D=$(pwd)/slots; mkdir -p $D
P=18123
PROMPT=$(for i in $(seq 1 60); do printf 'The quick brown fox jumps over the lazy dog number %d. ' $i; done)
start() { # $1 ctx, rest env
  local c=$1; shift
  env "$@" $BIN -m $M --port $P -c $c -np 1 -ngl 0 --load-mode none --slot-save-path "$(cygpath -w $D)" --no-webui > server-$c.log 2>&1 &
  SP=$!
  until curl -s localhost:$P/health | grep -q ok; do sleep 0.5; done
}
stop() { kill $SP; wait $SP 2>/dev/null; }
req() { curl -s localhost:$P/completion -d "{\"prompt\":\"$PROMPT\",\"n_predict\":16,\"temperature\":0,\"cache_prompt\":true}" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const j=JSON.parse(s);console.log("prompt_n",j.timings.prompt_n,"cache_n",j.timings.cache_n,"text",JSON.stringify(j.content))})'; }
start 4096
echo "== first (c=4096)"; req
curl -s -X POST "localhost:$P/slots/0?action=save" -d '{"filename":"t.bin"}'; echo
stop
for c in 4096 8192 1024; do
  start $c
  echo "== restore into c=$c"; curl -s -X POST "localhost:$P/slots/0?action=restore" -d '{"filename":"t.bin"}'; echo
  req; stop
done
start 8192 LLAMA_ARG_AUTO_ZONES=1 LLAMA_ARG_ZONES_CTX_INIT=512
echo "== restore into c=8192 auto zones, KV starts at 512"; curl -s -X POST "localhost:$P/slots/0?action=restore" -d '{"filename":"t.bin"}'; echo
req; stop
grep -h "set_dynamic\|relayout\|resize" server-8192.log | tail -5
