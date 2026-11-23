# Nyitott kérdések (többmodelles munka)

Amit nem tudtam eldönteni vagy megvalósítani; mindegyiknél a pontos akadály és a mostani állapot.

## Lezárva (2. kör)
- ~~A GPU-csere folyamatszintű~~ → **folyamaton belüli park/unpark kész** (llama.cpp `--park`, `POST /park`, `/unpark`; az Ollama ezt használja, a folyamatszintű csere csak akkor marad, ha a pinned példányok nem fértek a RAM-keretbe, naplósorral). Laptop: `test-results.md` „Folyamaton belüli park”.
- ~~`zones: off` a dinamikus KV-t is kikapcsolja~~ → két külön kapcsoló: `zones: off` = `LLAMA_ARG_ZONES_FLEX=0` (`--no-zones-flex`), a dinamikus KV marad.
- ~~Vulkan helyben növelés~~ → **kész** (sparse binding a ggml-vulkanban; tartalék csak hiányzó képességnél, a naplósor megnevezi).

## Nyitva
1. **A parkoló folyamat VRAM-maradéka diszkrét GPU-n nem mért.** A laptop iGPU-ján a heap közös, a pinned host-példányok (935 MiB) és a host-K/V is a „device” heap-et terhelik, ezért a park után a szabad memória (6025 MiB) nem adja ki tisztán a HIP/Vulkan-kontextus maradékát (a kezdeti 7384 MiB-hoz képest 1359 MiB, ebből ~1047 MiB a pinned példány és a K/V, a maradék ~310 MiB a Vulkan-kontextus és a staging pufferek becsülhetően). Mérés: `NAS-TESTS.md` 6. (a sysfs `mem_info_vram_used` az X080-on).
2. **A projektor (mmproj) és a rekurrens állapot parkoláskor a helyén marad.** A `llama_park` a `llama_context` és a modell foglalásait szabadítja fel; a llama-server multimodális projektora külön mtmd/clip kontextusban él (saját backend-pufferekkel), a hibrid modellek rekurrens állapota (`llama_memory_recurrent`) nincs a `park_kv`-ben (a napló: `the recurrent state stays where it is`). A qwen3-vl:8b-nél ez a projektor VRAM-ja (~0,6–1 GB) – a park után ennyi marad. Megoldás: az mtmd kontextus újralétrehozása unparkkor, illetve a rekurrens állapot `relayout`-szerű áthelyezése; még nincs megcsinálva.
3. **Parkolt állapotban a CPU-s kimenet eltérhet a GPU-stól.** A parkolt modell a CPU-n az 1. zóna eredeti formájú (nem repackelt) pinned példányaival számol, a numerikus út más, a mintavétel `temperature 0` mellett is eltérhet (a laptopon „Paris. The capital of Italy is Rome...” vs. „Paris, and the capital of Italy is Rome...”). Unpark után a kimenet bájtra azonos a park előttivel (mérve). A kért „demote” (repack a CPU-s futáshoz) parkolt modellnél nincs: a CPU a pinned példányból számol, lassabban; a repackelt CPU-tartalék külön folyamatként (`-ngl 0`) marad meg, ha a modellnek nincs parkolt példánya.
4. **Az Ollama-oldali park a laptopon csak kis modellekkel igazolható**; a 10 GB-os csereidő (cél ~1 s, PCIe ~11,6 GB/s) a NAS-on mérendő (`NAS-TESTS.md` 6.).
5. **`on_busy: wait` `wait_timeout` nélkül**: 60 s beépített alapérték.
6. **A futási hely a kérés teljes idejére szól**, a betöltéssel (és az unparkkal) együtt.
7. **A CPU-tartalék példány keep_alive-ja** a modellé; RAM-hiánynál elsőként megy.
8. **Egy kártya két backenden** (X080 ROCm-ként és Vulkanként is látszik): az Ollama a ROCm-ot tartja meg, egy `backend: vulkan` bejegyzés erre a kártyára `missing` lesz (szándékos).
