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

## Az `ollama serve`-es próba: akadály (Go telepítve, de nincs C fordító)
- Go 1.26.10 telepítve (`go version go1.26.10 windows/amd64`).
- `go build .` (CGO_ENABLED=0, az alapértelmezés itt) hibával áll le: `mlx\nn.go: undefined: Array` stb. Az `mlx` csomag `Array` típusa cgo-s fájlban van (`// #include "generated.h"`), CGO nélkül kimarad; az Ollama build-útvonala (`cmake/local.cmake`) is `CGO_ENABLED=1`-gyel épít.
- A cgo-hoz GCC-kompatibilis C fordító kell Windowson (MinGW-w64 gcc vagy llvm-mingw). A gépen nincs: a PATH-on nincs `gcc`/`clang`/`cc`, a devkitPro csak ARM/PPC keresztfordítókat tartalmaz, a Visual Studio-ban csak `clang-format`/`clang-tidy` van, az MSVC `cl` nem használható cgo-hoz.
- Telepíteni nem szabad, ezért az `ollama.exe` nem készült el, és az `ollama serve`, a Modelfile-import, valamint az `--load-mode none` indítási sor ellenőrzése az Ollama naplójában nem futott. A llama-server oldali eredmények (fent) érvényesek.
- Hiányzó eszköz és telepítés (egyik elég): **llvm-mingw** (zip, ~100 MB letöltés, ~700 MB kicsomagolva, PATH-ra tenni), vagy **WinLibs GCC** (`winget install BrechtSanders.WinLibs.POSIX.UCRT`, ~250 MB letöltés, ~1 GB telepítve), vagy MSYS2 `pacman -S mingw-w64-ucrt-x86_64-gcc` (~250 MB). Utána: `set CC=gcc` (vagy clang), `set CGO_ENABLED=1`, `go build -o build\ollama.exe .`, indítás a repó gyökeréből (a `dist\lib\ollama` a munkakönyvtárból megtalálható), `OLLAMA_DEBUG=1`.

## Rugalmas sáv a b11351-es patch-ben: futási próba a laptopon (Vulkan)
`dist\lib\ollama\llama-server.exe` (az Ollama fordítási útja, a 003-as patch az 410eb0e8d tartalmával), Qwen2.5-1.5B Q4_K_M, `-ngl 14`, `--load-mode none`, 2 kérés (a másodikat mértük), 40 token, `temperature=0`.

| eset | prompt tok/s | gen tok/s | kimenet |
|---|---|---|---|
| alap | 305,5 | 18,6 | azonos |
| `LLAMA_ARG_FLEX_LAYERS=2`, nincs mozgatás | 309,5 | 19,7 | azonos |
| `--flex-layers 2`, `LLAMA_FLEX_TEST=1:take,1:take,100:put,100:put` | 306,7 | 19,8 | azonos |
| `LLAMA_ARG_DUAL_BUFFER=1` | 313,7 | 19,3 | azonos |

A napló: `flex layers 15-16: 24 weights with a CPU copy (53,72 MiB), 12 with a host copy (42,61 MiB)`; a 15. és 16. réteg súlya és K/V-je a CPU-ra ment (0,9 / 0,0 ms súly, 0,4 ms K/V), majd visszatérve a Vulkan0-ra (3,9–6,1 ms súly, 0,6–1,1 ms K/V). A RAM-védelem sora a sávra is kiírja a becslést (`flex: free RAM … flex copies 107 MiB`). Ez a mérés a portolás helyességét igazolja (a mozgatás működik, a kimenet azonos), nem teljesítménymérés. A NAS-os mérés: `tools/dualbuf/nas-flex-bench.py` (SUMMARY.md).

# Többmodelles munka: laptopos tesztek (multimodel branch)

## KV-mentés és -visszatöltés (llama-server slot save/restore), 2026-11-23
Fork llama-server (b12177 + 001 + 003 + 004, CPU, `C:\b\c16`), qwen3:0.6b, `-np 1 -ngl 0 --load-mode none`, 832 tokenes prompt, 16 generált token, `temperature 0`. Mentés `-c 4096`-tal, visszatöltés új folyamatba. Szkriptek: `tools/dualbuf/multimodel/slot-restore.sh`, `slot-restore-edge.sh` (`MODEL=<gguf>`, `BIN=<llama-server>`).

| visszatöltés ide | eredmény | a következő azonos kérés | kimenet |
|---|---|---|---|
| `-c 4096` (azonos) | 847 token, 97 MB, 42 ms | `prompt_n 1, cache_n 831` | azonos |
| `-c 8192` (nagyobb) | 847 token, 59 ms | `prompt_n 1, cache_n 831` | azonos |
| `-c 1024` (kisebb, a tokenek elférnek) | 847 token, 42 ms | `prompt_n 1, cache_n 831` | azonos |
| `-c 8192`, auto zónák, a KV 512 cellával indul (`LLAMA_ARG_ZONES_CTX_INIT=512`) | 847 token, 105 ms (a dinamikus KV megnő) | `prompt_n 1, cache_n 831` | azonos |
| `-c 512` (a tokenek nem férnek el) | **elutasítva** (`No available space in KV cache`), a kérés hibával jön, mert a prompt maga sem fér el | – | – |
| `-c 4096`, `q8_0` KV (a mentés f16) | **elutasítva**, a következő kérés a promptot újraszámolja (`prompt_n 832`) | | azonos |

Következtetés: azonos modellre és KV-típusra eltérő max kontextusnál is visszatölthető, ha a mentett tokenek elférnek; a dinamikus KV-val is (a KV a visszatöltéskor nő). Eltérő KV-típusnál a llama-server maga is elutasítja; a fork ezt előre szűri (`KVMeta.Compatible`, a fájlnévben is benne a KV-típus), hogy a hibasor el se jöjjön.

## Az Ollama a multimodel-kóddal, laptopon (Vulkan iGPU + CPU), 2026-11-23
- Bináris: `ollama.exe` a `multimodel-ci` workflow Windows-buildjéből (`0.40.16-mm-3d5b17c`, majd `-0789900`), mellette `lib/ollama`: a fork CPU-s llama-servere (b12177 + 001 + 003 + 004, `C:\b\c16`) és a hozzá fordított `vulkan/ggml-vulkan.dll` (`C:\b\v16`). Intel Iris Xe (Vulkan0, `OLLAMA_IGPU_ENABLE=1`, Windowson PCI-cím nélkül), 16 GB RAM (a teszt alatt ~0,5–2 GB szabad).
- Modellek (`ollama create` a GGUF-okból): `qwen25:1.5b` (Q4_K_M), `qwen3:0.6b`, `qwen3-embedding:0.6b`. Config: `tools/dualbuf/multimodel/laptop-dualbuf.yaml`; szkriptek: `serve.sh`, `gen.sh`, `busy.sh`, `swap.sh` (ugyanott).

| eset | eredmény |
|---|---|
| config nélkül | az indítósor a gyári: `-c 4096 -np 1 ... --flash-attn auto -b 512 -ub 512 --context-shift --keep 4`; nincs `dualbuf` naplósor |
| config megjelenése futó modell mellett | a config nélkül betöltött példány változatlanul szolgál (0,01 s „betöltés”); a config a kirakás utáni betöltéskor lép életbe |
| erősorrend, naplósor | `dualbuf effective settings ... devices=[igpu,cpu](model:qwen25:1.5b) ... dual_buffer=true(defaults) ... max_context=8192(model:...) keep_alive=10m0s(defaults) parallel=1(builtin) ...`; a llama-server: `-c 8192`, `--split-mode none --main-gpu 0`, a naplóban `auto zones`, `dual-buffer`, `set_dynamic: dynamic KV: 4096 -> 8192 cells ... 28 by copy` (Vulkanon másolással nő) |
| eszközazonosítás | `dualbuf devices matched="cpu=CPU, igpu=Vulkan Vulkan0 \"Intel(R) Iris(R) Xe Graphics\" no PCI id"` (név szerint) |
| futási korlát, `on_busy: next` | A (qwen25) az igpu-n generál; B (qwen3, `next`) 0 s várakozással a CPU-ra kerül, `-ngl 0`-s CPU-s példány indul (repack) |
| `on_busy: wait`, lejár | C (qwen25, `wait_timeout 3s`) 3,04 s után továbblép; ha a CPU is foglalt, sorba áll, és az elsőként felszabaduló eszközt kapja (`reason=queued waited=17.571s`) |
| `on_busy: wait`, időben felszabadul | C 1,5 s várakozás után az igpu-t kapja (`reason=waited`) |
| érkezési sorrend | D (embedding, csak cpu) C után érkezett, a CPU-t C után, a B végén kapta |
| KV az NVMe-re (`kv_save`) | kirakáskor `KV saved to RAM ... tokens=[568] 92ms`, `KV saved to the disk 75ms` (16 MB); újratöltés után `KV restored from=disk 69ms`, a llama-server az azonos 521 tokenes promptból **1 tokent** számol, a kimenet azonos |
| csere (park) | `OLLAMA_MAX_LOADED_MODELS=1`: `making room on the device ... park=true`, `KV saved to RAM ... 160ms`, `model parked`; visszacserénél `KV restored from=ram 28ms`, a következő kérés **1 tokent** számol, a kimenet azonos; a csere a laptopon 3–4,4 s (folyamat-újraindítás); a fájlzárolás Windowson nincs (`supported on Linux only`) |
| RAM-hiány, CPU-tartalék | a szűk RAM miatt az embedding betöltése előtt a qwen3 CPU-s példánya rakódott ki elsőként (`kind=cpu-fallback`) |
| RAM-hiány, parkoló, keep_alive -1 | `ram_limit: 1GiB`: `parked model dropped from RAM` (qwen25), a `keep_alive: -1`-es embedding betöltve marad (`ollama ps`: `Forever`) |

Nem tesztelt a laptopon: a VRAM-hiány miatti csere (az iGPU szabad memóriája nem változik, ezért a `MAX_LOADED_MODELS` váltotta ki; ugyanaz az áldozatválasztás fut), két GPU, a GPU+CPU megosztott modell CPU-helye (egységteszt: `TestCoordAlsoCPU`), a fájlzárolás (Linux), ROCm. Ezek a `NAS-TESTS.md`-ben.

## Dinamikus KV Vulkanon az Ollamán át (laptop, Iris Xe), 2026-11-23
`qwen25:1.5b`, config: `max_context: 8192`, `dynamic_kv: true` (defaults). 7128 tokenes prompt („What color is the frog?”).
- Dinamikus KV: `set_dynamic: dynamic KV: 4096 -> 8192 cells ... 28 by copy`, a promptnál `relayout: KV 4096 -> 8192 cells (... 28 new by copy) 160.4 ms: alloc 119.7, copy 36.3, free 4.0`, `zones_state ... KV 8192 cells (224 MiB on the device)`. Válasz: „The color of the frog is green.”
- Ugyanez a kérés `options.dynamic_kv: false`-szal: a naplósor `dynamic_kv=false(request)`, az ujjlenyomat eltér, újratöltés; az új folyamatban nincs `auto zones`/`set_dynamic` sor (statikus KV). Válasz: ugyanaz.
- Következtetés: a dinamikus KV Vulkanon a másolásos úton működik (a helyben növelés, VMM/sparse, csak CUDA/HIP-en van).

# Többmodelles munka, 2. kör: Vulkan sparse KV és folyamaton belüli csere (laptop, 2026-11-23)

## Vulkan sparse KV (llama-server közvetlenül; csaki89/llama.cpp `multimodel`, Iris Xe)
qwen2.5-1.5B Q4_K_M, `--auto-zones -fa on --load-mode none`; szkript: llama.cpp `tools/dualbuf/vk-sparse-grow.sh`.
- `ggml_backend_vk_vmm_arena_new: Vulkan0: sparse KV buffers: block 65536, chunk 2097152 bytes`; `set_dynamic: ... growth on the device: 28/28 layers in place (VMM)`.
- `-c 8192`, ~7000 tokenes prompt: `relayout: KV 4096 -> 8192 cells (... 28 grown in place, 0 new by copy ...) 171.7 ms`; új rövid kérés: `KV 8192 -> 4096 cells (... 28 shrunk in place ...) 32.9 ms`. A válasz azonos a `GGML_VK_SPARSE=0`-s (másolásos, `no growable (sparse) KV buffers: GGML_VK_SPARSE=0; the KV grows by copy`) futáséval.
- `-c 32768`, ~20 000 tokenes prompt: 4096 → 24576 cella helyben (930 ms, 0 MiB másolva), vissza 4096-ra helyben (117 ms); a meghajtó szerinti szabad memória (`zones_state ... driver`) 5555 → 6045 MiB (a felszabadított 560 MiB-ból +490 MiB; az iGPU közös memóriáján más folyamatok is mozgatják).

## Folyamaton belüli park/unpark (llama-server közvetlenül)
Ugyanaz a modell, `--park`; szkriptek: llama.cpp `tools/dualbuf/park.sh`, `park-demote.sh`. A laptopon kevés a szabad RAM (577 MiB), a RAM-védelem ezért elsőre elutasította a pinned példányokat (`park: the pinned copies of the GPU weights (935 MiB) do not fit the free RAM (577 MiB, system ...), no in-process park`); a mérésekhez `LLAMA_DUAL_BUFFER_FREE_MIB=4096/6000`.
- 3 kör oda-vissza: minden unpark után a válasz bájtra azonos a referenciával, a következő kérés `prompt_n=1` (a KV megmaradt). Parkolt állapotban a CPU-n kiszolgál.
- Idők (935 MiB súly, 112 MiB KV): park súlyok 70–162 ms, K/V 88–197 ms, scheduler 17–45 ms; unpark súlyok 189–526 ms (1,9–5,2 GB/s), K/V 87–141 ms, scheduler 36–86 ms. A szabad memória a park után 6025 MiB (indulás 7384; a maradék ~1360 MiB-ban az iGPU közös heapjén a pinned példányok és a host-K/V is benne van, `OPEN-QUESTIONS.md` 1.).
- Flex sávval (`--fit-target 6600`: 1. zóna 15 réteg, flex 11–13 kint): park/unpark 3×, a kimenet azonos, a kint lévő flex rétegek a helyükön maradnak.
- Demote: parkolt állapotban az első CPU-s kérésnél `park: demoted 596.53 MiB (repacked ...) in 1795 ms`, unpark/park után megmarad, `POST /park/drop` felszabadítja (`dropped (596.53 MiB)`); minden unpark után azonos kimenet. A laptopon a repackelt és a pinned példány CPU-s generálási sebessége nem különíthető el (11–13 tok/s mindkettő, mérési zaj).

## Az Ollamán át (`ollama.exe 0.40.16-mm-2f4d126`, a végleges patch 003-mal fordított llama-server, Vulkan iGPU)
Config: `tools/dualbuf/multimodel/laptop-dualbuf.yaml` (`swap: { enabled: true, keep_in_ram: pinned }`), `OLLAMA_MAX_LOADED_MODELS=1`; szkript: `swap.sh` (A = qwen25:1.5b, B = qwen3:0.6b, kétszer).
- `LLAMA_DUAL_BUFFER_FREE_MIB=6000 OLLAMA_DUALBUF_RAM_FREE_MIB=6000`: 8 csere, mindegyik folyamaton belül (`dualbuf: model parked in process ... duration=226–540ms`, `model unparked ... duration=292–892ms`), csak 2 llama-server folyamat indult az egész futás alatt, az első kiértékelés után minden ismételt prompt 1 tokenből (a KV megmaradt), a válaszok minden körben azonosak.
- `LLAMA_DUAL_BUFFER_FREE_MIB=500` (a pinned példányok nem férnek el): `park: ... no in-process park` a llama-serverben, az Ollamában `dualbuf: no in-process park (...), the swap restarts the process`, a folyamatszintű csere fut (KV a RAM-ba és vissza: `KV restored from=ram`).
- Valódi laptop-RAM (felülírás nélkül): az Ollama RAM-terve szűknek látta a RAM-ot, a parkolt modell (LRU) teljesen kirakódott (KV az NVMe-re), a következő kérésnél visszatöltődött (`KV restored from=disk`).
- Parkolt modell CPU-s kérése: az igpu foglalt (qwen3 generál), a qwen25-kérés `on_busy: wait 3s` után `device=cpu reason=next`, `dualbuf: the CPU request runs on the parked instance`, demote (597 MiB, 1,7 s), helyes válasz; nincs új `-ngl 0`-s folyamat (összesen 2 llama-server).
- Javított hiba a mérés közben: a parkolt példány RAM-áldozatként busy-loopot okozott (az ütemező a parkoltakat a betöltött-számlálóból is kivonta); most csak a `MAX_LOADED_MODELS`-összevetésből maradnak ki.
