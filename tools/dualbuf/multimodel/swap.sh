S=$(cd "$(dirname "$0")" && pwd)
PA=$(for i in $(seq 1 40); do printf 'Item %d is a red apple on a wooden table. ' $i; done)
PB=$(for i in $(seq 1 30); do printf 'Car %d is a blue car in a garage. ' $i; done)
echo "== A first";  bash $S/gen.sh qwen25:1.5b "$PA Summarize:"
echo "== B (swap A out)"; bash $S/gen.sh qwen3:0.6b "$PB Summarize:"
echo "== A again (swap back, KV from RAM)"; bash $S/gen.sh qwen25:1.5b "$PA Summarize:"
echo "== B again"; bash $S/gen.sh qwen3:0.6b "$PB Summarize:"
