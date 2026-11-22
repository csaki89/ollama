# CPU correctness check (laptop, no GPU)

Compares two llama-server builds on CPU, started with the arguments Ollama 0.40.16 uses
(`--load-mode none --flash-attn auto -ngl 0`, embeddings with `--embedding -b/-ub` and `/v1/embeddings`).

- `ct.sh <bin dir> <label> [base] [fork]`: qwen3:0.6b greedy (seed 42, temperature 0, token ids), embeddinggemma-2:270m
  and qwen3-embedding:0.6b (3 texts each). `fork` sets the compose switches (`LLAMA_ARG_DUAL_BUFFER=1`,
  `LLAMA_ARG_LOAD_MODE=none`, `LLAMA_ARG_AUTO_ZONES=1`).
- `kv.sh <bin dir> <label>`: a ~5400 token prompt grows the dynamic KV (4096 -> 5632 cells); static, auto zones,
  `LLAMA_KV_NO_VMM=1`, `LLAMA_KV_VMM_SIM=1/2/3` and dual + `LLAMA_ZONES_SHRINK=0` must give the same tokens.
- `node compare.js results/<a>/<mode> results/<b>/<mode>`: chat tokens identical, embedding dimension, norm, cosine.

Models: `models/qwen3.gguf`, `models/embeddinggemma-2.gguf`, `models/qwen3-embedding.gguf` (or `MODELS=<dir>`), the
model layer blob of the Ollama registry manifest (`https://registry.ollama.ai/v2/library/<name>/manifests/<tag>`,
then `/v2/library/<name>/blobs/<digest>`; check the sha256).

The GPU paths (VMM growth in place, flex layers, dual-buffer host copy) are not reachable on CPU: on CPU the
dual-buffer turns itself off (no device host buffer) and the KV grows by copy.
