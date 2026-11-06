# Laptopos próba (Vulkan, 3. fázis)

## Mit sikerült és mit nem
- **Sikerült:** a `003-dual-buffer.patch`-csel patch-elt llama-server lefordult Windowson (MSVC + Ninja, CPU + Vulkan, `tools/dualbuf/build-llama-server-win.bat`) az Ollama saját build-útvonalán (`cmake -S llama/server --preset ...`, FetchContent b11351 + `001`, `002`, `003` patch).
- **Nem sikerült: az `ollama` binárissal végzett próba (`ollama serve`, Modelfile-import).** A laptopon nincs Go (a `go.mod` 1.26.0-t kér), és a telepítés a `.claude/settings.json` szerint tiltott (`msiexec`, `winget`), a feladat szerint pedig semmit nem telepítek. A Go 1.26.10 MSI le van töltve és az SHA256 ellenőrizve (`go1.26.10.windows-amd64.msi`, 59 MB, `baf134644006fc229c7a119ca6248eaccab8472f10d718bdde71ad1de2fb0109`), a telepítést a felhasználónak kell elindítania. Emiatt nem futott: `PARAMETER num_gpu 14` + `PARAMETER use_mmap false` Modelfile, az `LLAMA_ARG_LOAD_MODE`-os globális mmap-kikapcsolás és az Ollama VRAM-számlálója a `Vulkan_Host`/`CPU_REPACK` sorokkal.
- **Helyette:** a patch-elt `llama-server.exe`-t közvetlenül indítottam ugyanazokkal a kapcsolókkal, amelyeket az Ollama ad (`llm/llama_server.go`: `--model --port --host --no-webui --offline -c -np --log-verbosity 4 --no-log-prefix --no-log-timestamps --load-mode none --flash-attn auto -b 512 -ub 512 -ngl 14`), `GGML_BACKEND_PATH`-szel a Vulkan backendre, és az env-változókkal. A llama-server oldala (amire a dual-buffer hat) így ugyanaz; az Ollama-oldal (kapcsoló-átadás, env-öröklés) a forrásból igazolt (DESIGN.md 2., 4.), futtatással nem.

## Beállítás
Qwen2.5-1.5B Q4_K_M (`../llama-dualbuf/models`), Intel Iris Xe iGPU (Vulkan0), 4 szál, `-ngl 14`, `--load-mode none`, 2251 tokenes prompt (ismételt mondat), `cache_prompt=false`, `temperature=0`, 48 generált token, 1 bemelegítő + 3 mért kérés. Eszköz: `tools/dualbuf/bench-server.ps1`. Módok:
- `host` = alapviselkedés, **LLAMA_ARG_DUAL_BUFFER nélkül** (a host-puffer nyer, a CPU-s súly `Vulkan_Host`-ban);
- `repack` = `LLAMA_ARG_NO_HOST=1` (a CPU-s súly `CPU_REPACK`-ben, nincs op offload a repackelt súlyokra);
- `dual` = `LLAMA_ARG_DUAL_BUFFER=1`.

## Eredmény
| mód | prompt tok/s átlag (min-max) | generálás tok/s | kimenet md5 (első 8) | pufferek (MiB) |
|---|---|---|---|---|
| host (dual-buffer nélkül) | 330,3 (328,6-332,5) | 16,0 | 5452B26B | Vulkan0 526,55; Vulkan_Host 533,33 |
| repack | 121,4 (120,6-122,8) | 17,1 | 5452B26B | CPU 225,15; CPU_REPACK 308,18; Vulkan0 526,55 |
| **dual** | **324,8 (323,7-326,2)** | 18,0 | 5452B26B | CPU 225,15; CPU_REPACK 308,18; Vulkan0 526,55; Vulkan_Host 308,18 |

## Igazolás
1. **A naplóban megjelenik a sor és a két puffer** `LLAMA_ARG_DUAL_BUFFER=1` mellett: `load_tensors: dual-buffer: 87 weights also stored in Vulkan_Host (308.18 MiB)`, `CPU_REPACK model buffer size = 308.18 MiB` és `Vulkan_Host model buffer size = 308.18 MiB`. A többlet RAM a `CPU_REPACK` méretével egyezik (308 MiB).
2. **`LLAMA_ARG_DUAL_BUFFER` nélkül az eredeti viselkedés:** a `host` és `repack` mód puffereiben nincs `dual-buffer` sor, a pufferek az eredetiek (a ../llama-dualbuf `test-results.md` ugyanezeket a számokat adja: `Vulkan0 526,55 / Vulkan_Host 533,33`, illetve `CPU 225,15 / CPU_REPACK 308,18`), és a kimenet md5 mindhárom módban azonos.
3. **A prompt-sebesség a host-puffer szintjén van:** `dual` 324,8 vs `host` 330,3 tok/s (98%), és 2,7x a `repack` 121,4 tok/s-ánál. Összevetés a ../llama-dualbuf `test-results.md`-vel (1611 tokenes prompt, azonos modell): ott `dual` 304,5 vs `host` 307,0 (99%) vs `repack` 129,5, vagyis a viszony (host-szint, ~2,3-2,7x a repack fölött) egyezik. Az abszolút számok eltérnek (más prompthossz és más futás); az összevetés alapja az azonos futáson belüli arány.
4. **Generálás:** nincs mérhető különbség a módok között a laptopon (16,0 / 17,1 / 18,0 tok/s, a szórás ekkora), ahogy a ../llama-dualbuf mérésekben sem; a laptop iGPU-ja és a CPU közös RAM-ot használ. A generálási nyereség a NAS-on mérendő.

## Nem tesztelt
ROCm/HIP (az image csak a GitHub Actionsben épül), MoE modell (qwen3-coder:30b), Ollama-compat blobok (qwen3-vl:8b), az `ollama` bináris és a Modelfile-útvonal, a `--fit` automatikus placement (`-ngl` nélkül).
