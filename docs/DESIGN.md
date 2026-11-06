# Dual-buffer az Ollama 0.40.2-ben: felderítés és terv

Állapot: 1. fázis (felderítés), a kód ennél a commitnál módosítatlan. A hivatkozások a `v0.40.2` (`b061384d`) állapotára vonatkoznak, a llama.cpp oldalon a `b11351` (`631109b34`) és a `e14a77971` (`b12140-23`) tagre/commitra. A dual-buffer forrása: `../llama-dualbuf`, `dual-buffer` branch (csak olvasva).

## 1. Hogyan kerül a llama.cpp az Ollamába

- A `LLAMA_CPP_VERSION` fájl rögzíti a verziót (`b11351`). A `llama/server/CMakeLists.txt` `FetchContent`-tel tölti le a `ggml-org/llama.cpp` forrást (`GIT_SHALLOW`), és a `llama-server` célt építi (`LLAMA_BUILD_SERVER=ON`, `LLAMA_BUILD_COMMON=ON`, CURL/OpenSSL/WebUI ki). Az Ollama Go kódja ebből semmit nem fordít be: a llama-server külön processz.
- Az Ollama-saját módosítások két helyen élnek:
  - `llama/compat/*.cpp,*.h` (a `target_sources()` köti a letöltött llama.cpp célokhoz): a régi, Ollama-formátumú modell-blobok (pl. a könyvtári `qwen3-vl`) fordítása memóriában llama.cpp-kompatibilis GGUF-fá.
  - `llama/compat/*.patch`: az `apply-git-patches.cmake` számsorrendben (`001-…`, `002-…`) `git apply`-olja a letöltött forrásra, idempotensen (a már alkalmazottat a `--reverse --check` felismeri és átugorja). `001-llama-cpp-hooks.patch`: hívási pontok a `llama-model-loader.cpp`-ben és a `clip.cpp`-ben; `002-clef.patch`: a `clef` (`common.cpp/.h`, `llama.h`, `llama-context.cpp`, `qwen35.cpp`, szerver-fájlok).
  - A compat réteg a lefordított modellnél `use_mmap = false`-ra állítja a loadert (`001`, `translate_metadata` után).
- Helyi fejlesztéshez az `OLLAMA_LLAMA_CPP_SOURCE` környezeti változó egy helyi llama.cpp fát ad meg (ilyenkor a patcheket nem alkalmazza automatikusan, `OLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH` szabályozza).
- A `llama/README.md` verziófrissítési munkafolyamata: új tag a `LLAMA_CPP_VERSION`-be, `cmake -S llama/server --preset cpu`, ami letölti és patcheli a forrást; ha egy patch beszúrási pontja elmozdult, a patchet **újra kell generálni** egy friss checkouton.

## 2. Hogyan épül és indul a llama-server

- Build: `cmake -S llama/server --preset <cpu|rocm_v7_2_linux|vulkan|…>` (`llama/server/CMakePresets.json`), `BUILD_SHARED_LIBS=ON`, `GGML_BACKEND_DL=ON` (a GPU-backendek külön `.so`-k az `<runner-dir>/` alatt). A Linux ROCm cél: `rocm_v7_2_linux` (AMDGPU_TARGETS: gfx908…gfx1201, benne gfx1030), a `Dockerfile` `llama-server-rocm_v7_2` stage-e; a végső image az `ubuntu:24.04` alapú utolsó stage (`FLAVOR=rocm` választja a `rocm` assembly stage-et: cpu + rocm).
- Indítás (`llm/llama_server.go`, `startLlamaServer`): az Ollama minden modellbetöltéshez új `llama-server` processzt indít `127.0.0.1:<szabad port>`-on:
  `--model --port --host --no-webui --offline -c <num_ctx*parallel> -np <parallel> --log-verbosity 4 --no-log-prefix --no-log-timestamps` + jinja/chat-template, `--mmproj`, speculative kapcsolók, `--lora`, `--load-mode` (lásd 4.), `--cache-type-k/-v`, `--flash-attn on|off|auto`, `-b/-ub`, `-ngl N` (csak ha a `num_gpu` ≥ 0 meg van adva; -1 esetén a llama-server `--fit` automatikája dönt), `-t`, `--split-mode none --main-gpu`, `--context-shift`/`--keep`.
- Környezet (`SetupLlamaServerCommandEnv`): az Ollama **teljes környezetét** (`os.Environ()`) örökli a processz, ehhez jön a könyvtárkeresési út, `GGML_BACKEND_PATH`, `LLAMA_MEDIA_MARKER`, az eszközválasztó változók (`ml.GetDevicesEnv`) és a `LLAMA_ARG_FIT_TARGET` (pad). Ezért **minden `LLAMA_ARG_*` változó átmegy** a llama-serverre, és érvényesül, ha a megfelelő kapcsolót az Ollama nem adja át a parancssorban (a llama.cpp-ben a CLI erősebb az envnél). Az `envconfig` csak a `LLAMA_ARG_FIT`-et és a `LLAMA_ARG_FIT_TARGET`-et sorolja fel a súgóban, de a többit sem szűri. A `LLAMA_ARG_DUAL_BUFFER` tehát módosítás nélkül átmegy.

## 3. Llama-server vagy az Ollama saját motorja?

- A 0.40.2-ben **minden GGUF modell a llama-serveren fut** (`llm/server.go`: „All GGUF models are served via the upstream llama-server subprocess”; a `server/sched.go` ezt hívja, `newServerFn = llm.NewLlamaServer`). A régi saját Go-motor GGUF-ra már nincs: a `model/` csak `parsers` és `renderers`, az `ml/` csak eszközleíró.
- Az egyetlen másik útvonal a `safetensors` formátumú modell (`Model.IsMLX()`), amit az `mlxrunner` futtat (MLX: Apple Metal, CUDA 13). A NAS ROCm image-ében ehhez nincs payload, a dual-buffer ott nem releváns.
- Következmény: a **qwen3-vl:8b, qwen3-coder:30b és minden más GGUF modell a llama-serveren fut**, a dual-buffer hat rájuk. A könyvtári blobokat (Ollama-formátum) a compat réteg fordítja, ami a loadert `use_mmap = false`-ra állítja, vagyis náluk a mmap-feltétel eleve teljesül (de ettől független a `use_mmap: false` beállítása, lásd 4.). A kérdés, hogy a mérés megerősíti-e: a naplóban a `using llama-server for model` sor.

## 4. A `use_mmap: false` tartós beállítása

Ollama-oldali útvonal (`llm/llama_server.go: appendLoadModeArgs`, `server/sched.go`):

- `opts.UseMMap != nil && !*UseMMap` → `--load-mode none`. Integrált GPU-n Linuxon (CUDA/ROCm `Integrated`) → `--load-mode dio` (az erősebb). Egyébként az Ollama **nem ad át** `--load-mode`-ot.
- A scheduler automatikusan is állíthatja `UseMMap = false`-ra (`applyLlamaServerMmapDefaults`): ha nincs GPU/`num_gpu 0` („cpu”), Windows+CUDA esetén, Metal részleges offloadnál, vagy Linux + diszkrét GPU esetén ha a gazda-RAM szűkös (`maybeDisableMmapForHostPressure`, ha `UseMMap == nil`). A szabad RAM a NAS-on (128 GB) általában elég, így ez nem kapcsol be: **az alapérték mmap marad**.

Tartós beállítás:

1. **Modelfile, modellenként:** `PARAMETER use_mmap false` (az `api.FormatParams` a `*bool` mezőt a json tag alapján elfogadja; a `FromMap` ugyanígy). A kérésben `"options": {"use_mmap": false}` is megy. Ez minden modellnél külön kell, és egy kérés `true`-ja felülírja.
2. **Globálisan, forrás alapján:** `LLAMA_ARG_LOAD_MODE=none` (a llama.cpp `-lm/--load-mode` kapcsolójának env-je, `common/arg.cpp`, b11351: `.set_env("LLAMA_ARG_LOAD_MODE")`). Mivel az Ollama a `--load-mode`-ot csak a fenti két esetben adja át, és minden más esetben a környezetet örökli a llama-server, az env **globális alapértelmezésként érvényesül** a Modelfile nélküli modellekre is. Ahol az Ollama átad `--load-mode`-ot (`none` vagy `dio`), az felülírja az env-et. Az Ollama kódját nem kell módosítani. (Nem futtatással, hanem a forrásból következtetve; a laptopos próbán ellenőrizzük.)
   - Mellékhatás: ha az `UseMMap` nincs beállítva, az Ollama belső számolása (`runnerUsesMmap`) továbbra is mmap-ot feltételez. Ez csak a „gazda-nyomás” heurisztikát és a betöltött mmap-modellek méretének összegzését befolyásolja, a betöltést nem.

## 5. Saját RAM-becslés az Ollamában?

- **VRAM-becslés van, gazda-RAM-becslés nincs.** `llm.PredictServerVRAM` = a modellfájl mérete + KV-cache (f16 képlet). A scheduler ezt az elérhető (GPU) memóriával veti össze (80%-os küszöb, csak ha már van betöltött modell és nincs explicit részleges offload), a kötegméret (`automaticGenerationBatch`) és a környezethossz-csökkentés (OOM utáni újrapróbálkozás) kiválasztásához.
- A betöltés közben az `memoryParsingWriter` a llama-server `… buffer size = X MiB` sorait olvassa (`bufferSizeRegex`), és `memTotal` / `vramByDevice` számlálókat vezet. A `*_Host` és `CPU*` pufferek nem számítanak GPU-nak (`isGPUBuffer`), tehát a **dual-buffer többlete (`ROCm_Host` iker, +`CPU_REPACK`-nyi RAM) nem kerül a VRAM-számlálóba**, és a becslés (modellfájl mérete) sem tartalmazza.
- A gazda-RAM egyetlen ellenőrzése a mmap-nyomás-heurisztika (`mmapHostPressureHeadroom`: max(8 GB, a RAM 10%-a)), ami csak `UseMMap == nil` esetén dönt a mmap kikapcsolásáról.
- **Korlát:** az Ollama nem véd a gazda-RAM elfogyásától. A dual-buffer a repackelt CPU-s súlyok méretét **még egyszer** foglalja a pinned host-memóriában. Nagy modellnél (pl. a gpt-oss:120b ~57 GB CPU-s része) ez nem fér el, és az OOM a gazdát érheti. A védekezés a llama.cpp `--dual-buffer-limit` / `--dual-buffer-pattern` kapcsolója (`LLAMA_ARG_DUAL_BUFFER_LIMIT` / `…_PATTERN`, az env-en át elérhető). A 128 GB-os NAS-on a tervezett modellek (qwen3-vl:8b, qwen3-coder:30b ≈ 18 GB) belefér.
- A „`--fit`” a llama.cpp-ben fut (`-ngl` nélkül), és ugyanúgy nem számol a host-iker többletével (lásd a `../llama-dualbuf/docs/DESIGN.md` 2.3).

## 6. b11351 és e14a77971 összevetése az érintett fájlokban

A dual-buffer 11 fájlt érint (151 sor a `docs/`, `tools/` nélkül, az `e14a77971`-hez képest). Az Ollama `b11351`-e (`b12140`-nél ~790 build régebbi) és az `e14a77971` közti különbség ugyanezekben a fájlokban **2687 beszúrt / 682 törölt sor** (`git diff --stat b11351 e14a77971`):

| fájl | b11351→e14a77971 |
|---|---|
| `src/llama-model.cpp` | +919 |
| `src/llama-context.cpp` | +671 |
| `ggml/src/ggml-backend.cpp` | +496 |
| `common/common.cpp` | +507 |
| `src/llama-model.h` | +229 |
| `common/arg.cpp` | 212 |
| `src/llama-model-loader.cpp` | +136 |
| `common/common.h` | 124 |
| `ggml-backend.h` / `llama.h` / `llama-model-loader.h` | 40 / 20 / 15 |

Kísérlet: a dual-buffer diffet (`git diff e14a77971 dual-buffer -- common ggml include src`, 344 sor) a `b11351` fára, rá az Ollama `001`+`002` patchével `git apply`-oltam: **7 hunk tisztán (offsettel) felmegy, 4 elutasított** (`common/common.h`, `ggml/src/ggml-backend.cpp`, `src/llama-context.cpp`, `src/llama-model.cpp`: csak környezeti sorok tértek el, pl. a `struct ggml_backend_sched` mezői, a `sched_new` szomszédos sorai, a `load_tensors` vége). A logika a b11351-ben is ugyanott él (`make_cpu_buft_list`, a loader `select_weight_buft`/`create_tensor`, a scheduler 1. passa), a `load_tensors` pedig a `llama_model_base`-be költözött; a b11351 `LLAMA_LOAD_MODE_AUTO` ugyanúgy mmap-ot jelent (`use_mmap = AUTO || MMAP`), a „host-puffer mmap esetén kimarad” szabály (`llama-model-loader.cpp:1269-1271`) is megvan. Az átültetés tehát kézi, de kicsi (4 hunk + az automatikusan felment hunkok szemantikai ellenőrzése).

## 7. Döntés: hogyan kerüljön be

Két lehetőség:

- **A) `llama/compat/003-dual-buffer.patch`**, az Ollama meglévő mechanizmusával. Az `apply-git-patches.cmake` automatikusan, `002` után alkalmazza; a Dockerfile minden stage-e (`COPY llama/compat`) és a laptopos build is kiveszi. Egy repóban él (csaki89/ollama `dual-buffer` branch), nincs külső függőség az image buildben.
- **B) a `FetchContent` forrása a csaki89/llama.cpp egy `dual-buffer-b11351` branche**, vagyis a `GIT_REPOSITORY`/`GIT_TAG` átírása. Ehhez a b11351-re átültetett branch kell a llama.cpp forkban, amit most nem írhatok (a `../llama-dualbuf` csak olvasható, és push a feladatban nincs), a GitHub Actions buildhez pedig a másik repóhoz is hozzáférés kellene, és a `001`/`002` patchek is az így módosított fára mennének.

**Választás: A).** Indok az újraalkalmazhatóságra:
- Az upstream Ollama-frissítésnél (új `LLAMA_CPP_VERSION`) a `llama/README.md` munkafolyamata már előírja a patchek újragenerálását; a `003` ebbe a folyamatba illeszkedik, nincs külön branch-karbantartás és külön repó.
- A merge-konfliktus az Ollama-oldalon (`LLAMA_CPP_VERSION`, a `Dockerfile`/CMake apró módosításai) kicsi marad: a `003-dual-buffer.patch` új fájl, nem ütközik az upstream változásaival. Ha a patch egy frissítésnél nem megy fel, az `apply-git-patches.cmake` a konfigurálásnál hibával megáll, nem csendben romlik.
- A `002-clef` ugyanazokat a fájlokat is érinti (`common.cpp/.h`, `llama.h`, `llama-context.cpp`); a `003`-at ezért a `001`+`002` **után** generálom, és a fájl elején jelzem a függőséget.
- Hosszabb távon, ha upstreamelni akarnánk a funkciót, a `../llama-dualbuf` e14a77971-es változata marad a forrás; a `003` az Ollama-verzióhoz igazított átültetés.

A patch tartalma a ../llama-dualbuf 11 fájlja (a `docs/`, `tools/` nélkül), b11351-re igazítva. Eltérések a DECISIONS.md-ben.

## 8. Következő lépések

2. A `003-dual-buffer.patch` előállítása (átültetés, `git apply --check`, kézi hunkok), fordítás Vulkannal a laptopon.
3. Laptopos próba az `ollama serve`-vel (Qwen2.5-1.5B, `num_gpu 14`, `use_mmap false`).
4. GitHub Actions + Dockerfile (csak gfx1030), 5. compose, 6. SUMMARY.

Ismert akadály: a laptopon nincs Go (a `go.mod` 1.26.0-t kér), és a telepítés a `settings.json` szerint tiltott. Ha a telepítés nem történik meg, a 3. pont (az `ollama` binárissal végzett próba) nem végezhető el; ekkor a llama-server önállóan fordul és tesztelhető (ugyanazok a kapcsolók), és ezt a SUMMARY.md rögzíti.
