#!/bin/bash
# running limit / on_busy / queue: A holds igpu, B (next) -> cpu, C (wait 3s) waits then queues, D (cpu only) queues after C
q() { # name model prompt num_predict
  local t0=$(date +%s.%N)
  local r=$(curl -s localhost:${PORT:-11434}/api/generate -d "{\"model\":\"$2\",\"prompt\":\"$3\",\"stream\":false,\"options\":{\"temperature\":0,\"seed\":1,\"num_predict\":$4}}")
  local t1=$(date +%s.%N)
  echo "$1 $2 start=$(echo "$t0-$T0"|bc) end=$(echo "$t1-$T0"|bc) $(echo "$r" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const j=JSON.parse(s);console.log(j.error?"ERROR "+j.error:("eval "+j.eval_count+" load "+(j.load_duration/1e9).toFixed(1)+"s"))})')"
}
emb() { local t0=$(date +%s.%N); curl -s localhost:${PORT:-11434}/api/embed -d '{"model":"qwen3-embedding:0.6b","input":"hello"}' -o /dev/null; local t1=$(date +%s.%N); echo "D embed start=$(echo "$t0-$T0"|bc) end=$(echo "$t1-$T0"|bc)"; }
T0=$(date +%s.%N)
q A qwen25:1.5b "Write a long story about a fox." 300 &
sleep 1; q B qwen3:0.6b "Write a long story about a dog." 150 &
sleep 1; q C qwen25:1.5b "Write a poem about the sea." 20 &
sleep 1; emb &
wait
