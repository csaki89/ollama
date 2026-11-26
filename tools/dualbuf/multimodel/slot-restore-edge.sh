BIN=${BIN:?set BIN to the llama-server binary}
M=${MODEL:?set MODEL to a GGUF (qwen3:0.6b)}
D=$(pwd)/slots; P=18124
PROMPT=$(for i in $(seq 1 60); do printf 'The quick brown fox jumps over the lazy dog number %d. ' $i; done)
run() { # name ctx extra-args...
  local n=$1 c=$2; shift 2
  $BIN -m $M --port $P -c $c -np 1 -ngl 0 --load-mode none --slot-save-path "$(cygpath -w $D)" --no-webui "$@" > s-$n.log 2>&1 & local sp=$!
  until curl -s localhost:$P/health | grep -q ok; do sleep 0.5; kill -0 $sp 2>/dev/null || { echo "$n: server died"; return; }; done
  echo "== $n: $(grep -o 'n_ctx_slot = [0-9]*' s-$n.log | head -1); $(grep -o 'K (q8_0)\|K (f16)' s-$n.log | head -1)"
  echo "restore: $(curl -s -X POST "localhost:$P/slots/0?action=restore" -d '{"filename":"t.bin"}')"
  curl -s localhost:$P/completion -d "{\"prompt\":\"$PROMPT\",\"n_predict\":16,\"temperature\":0,\"cache_prompt\":true}" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{const j=JSON.parse(s);console.log("prompt_n",j.timings.prompt_n,"cache_n",j.timings.cache_n,JSON.stringify(j.content))}catch(e){console.log(s.slice(0,300))}})'
  kill $sp; wait $sp 2>/dev/null
}
run c512 512
run q8 4096 --cache-type-k q8_0 --cache-type-v q8_0 -fa on
