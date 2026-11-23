#!/bin/bash
# gen.sh model prompt [extra json] -> one line: load/prompt/eval stats and the text
m=$1; p=$2; x=${3:-}
curl -s localhost:11500/api/generate -d "{\"model\":\"$m\",\"prompt\":\"$p\",\"stream\":false,\"options\":{\"temperature\":0,\"seed\":1,\"num_predict\":24}$x}" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{const j=JSON.parse(s);if(j.error){console.log("ERROR",j.error);return}console.log(`load ${(j.load_duration/1e9).toFixed(2)}s prompt ${j.prompt_eval_count} eval ${j.eval_count} total ${(j.total_duration/1e9).toFixed(2)}s | ${JSON.stringify(j.response.slice(0,80))}`)}catch(e){console.log("BAD",s.slice(0,200))}})'
