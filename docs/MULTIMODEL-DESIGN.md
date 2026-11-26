# Többmodelles működés: a réteg kiválasztása és a terv (2. fázis)

Alap: `docs/MULTIMODEL-RESEARCH.md`. A szempontok sorrendje: (a) a viselkedés teljesen megvalósítható, (b) a kliensek változatlanok, (c) kicsi rebase-felület, (d) a meglévő mechanizmusok újrahasznosítása.

## 1. Döntés

**A többmodelles logika az Ollama Go-ütemezője mellé kerül, egy új, önálló csomagba (`multimodel/`), vékony horgokkal a `server/sched.go`-ban, a `server/routes.go`-ban és az `llm/llama_server.go`-ban. A llama.cpp-ben csak a folyamaton belüli mechanizmusok maradnak (flex rétegek, dinamikus KV, zónák, slot-mentés, a Vulkan sparse KV).** Kombináció tehát, de a döntések (melyik eszköz, ki fut, ki várjon, kit kell kirakni) egy helyen, a Go-oldalon születnek.

### Miért nem a router mód (elvetett út)
- **(a) nem valósítható meg vele teljesen:** a router modellenként egy fix argumentum-készletet (preset) ismer; nincs benne eszközválasztás kérésenként, eszközönkénti futási korlát, on_busy/várakozás/sorrend, kérésenkénti CPU-tartalék, csere RAM-ban tartással, RAM-keretre épülő LRU és keep_alive -1 védelem. Ezeket a router C++ kódjába kellene írni (`server-models.cpp`, 2800 sor, gyakran változik: csak 2026-ban a sor, a memória-margó, a stdin-parancsok átírása).
- **(b) a kliensek:** az Ollama API-t (és a `/v1`, Anthropic végpontokat) az Ollama fordítja a llama-server hívásaira, modellenkénti opciókkal (num_ctx → újratöltés, keep_alive). A router mellett ezt a fordítást is meg kellene tartani, és a router egy második, párhuzamos modellkezelő lenne az Ollamáé mellett (két helyen dőlne el, mi van betöltve).
- **(c) rebase:** a router-módosítás a llama.cpp leggyakrabban változó részébe nyúlna (patch 003 mellé egy nagy, sűrűn ütköző hunk), és az Ollama ütemezőjét is ki kellene kerülni.
- **(d) újrahasznosítás:** a router saját mérő-gyermeke és LRU-ja ütközne az Ollama könyvelésével; a mi mechanizmusaink (flex, zónák, dinamikus KV) folyamaton belüliek, a routertől függetlenek.

### Miért a Go-oldal
- Az Ollama már modellenként külön folyamatot indít, eszközt szűr (`GetDevicesEnv`), mér (naplóból), kirak és sorba állít; a hiányzó szabályok ezekre épülnek.
- A modellenkénti beállítás a `extraEnvs`-en (minden `LLAMA_ARG_*`) és néhány indítási kapcsolón át adható át, a llama-server kódjához nem kell nyúlni.
- A rebase-felület: egy új csomag (új fájlok, nem ütközik), és néhány soros horgok a meglévő függvényekben. A horgok config nélkül nem csinálnak semmit (a viselkedés bájtra a régi).

## 2. Részek

### 2.1 `multimodel/config.go`: a beállítófájl
- Út: `OLLAMA_DUALBUF_CONFIG`, alapból `<OLLAMA_MODELS>/dualbuf.yaml` (a NAS-on `/models/dualbuf.yaml`, a modellek mellett). Hiányzó fájl = nincs config = a régi viselkedés.
- Olvasás minden betöltés előtt (mtime/méret alapú gyorsítótárral); **a futó modellt nem érinti**: a betöltéskor használt pillanatkép a futóhoz rögzül, a további kérései azzal oldódnak fel (különben egy config-változás a `needsReload`-on át újratöltést váltana ki).
- Hibás fájl (YAML-hiba, ismeretlen mező, rossz érték): az utolsó érvényes pillanatkép marad, `level=ERROR` naplósor: `dualbuf config invalid, using the last valid one` a hibával és a fájl útjával (ha még nem volt érvényes: config nélkül).
- Szigorú olvasás (`yaml.v3` `KnownFields`): az elírt kulcs hiba, nem csendes kihagyás.

### 2.2 Erősorrend és forrás (`multimodel/resolve.go`)
Mezőnként: környezeti változó < `defaults` < Modelfile `PARAMETER` < a modell saját szakasza < a kérés opciói. A modell szakasza: pontos `név:tag`, különben a leghosszabb illeszkedő helyettesítő minta (`path.Match`: `qwen3-embedding:*`); tag nélküli kérésnél a `:latest` kerül a névhez. Ismeretlen modell: csak a `defaults`.
- Az Ollama saját opcióinál (`num_ctx` = `max_context`, `num_batch` = `n_batch`, `num_thread` = `threads`) a `defaults` a `DefaultOptions` (env) után, a Modelfile előtt, a modell szakasza a Modelfile után, a kérés előtt kerül az `api.Options`-ba (`routes.go` `modelOptionsWithEmbeddingBatchDefault`, két `FromMap`). A kérés betöltési opciói így a gyári módon viselkednek (eltérés → újratöltés).
- A többi mező env-je: `keep_alive` → `OLLAMA_KEEP_ALIVE`, `parallel` → `OLLAMA_NUM_PARALLEL`, `kv_type` → `OLLAMA_KV_CACHE_TYPE`, `flash_attention` → `OLLAMA_FLASH_ATTENTION`, `dual_buffer` → `LLAMA_ARG_DUAL_BUFFER`, `dynamic_kv` → `LLAMA_ARG_AUTO_ZONES`, `n_ubatch` → `LLAMA_ARG_UBATCH`. Modelfile-ban ezek nem adhatók meg (az Ollama nem ismeri őket), a kérésben a `keep_alive` mellett az `options` alatt is (`dual_buffer`, `kv_type`, ...; az Ollama ezekre csak figyelmeztet, a fork olvassa).
- Egy naplósor betöltéskor (`level=INFO msg="dualbuf effective settings"`): minden mező értéke és forrása (`env`, `defaults`, `modelfile`, `model:<kulcs>`, `request`, `builtin`).
- A rugalmas sáv mérete (`flex_layers`) nem mező: a program számolja (`--auto-zones`).

### 2.3 Eszközök (`multimodel/devices.go`)
- `devices.<név>: {backend, match, max_running}`; a `cpu` beépített név (backend nélkül).
- `match`: PCI-cím (pontos, kis-nagybetű független, a `0000:` tartomány elhagyható), különben a leírás (vagy a backend-név) részsztringje. A `backend` (`rocm`, `vulkan`, `cuda`) a felderített `Library`-vel egyezzen. Két találat ugyanarra a configbejegyzésre = hiba (két gpu0-nál PCI-cím kell); egy kártya két bejegyzésre = hiba.
- A párosítás a felderített eszközlistán (`discover.GPUDevices`) történik; a hiányzó eszköz `unavailable` (a modell listájában átugorja), egy naplósor jelzi.

### 2.4 Elhelyezés és futási korlát (`multimodel/coord.go`)
- Egy kérés a scheduleRunner elején **futási helyet foglal** (`Acquire`), és a kérés végéig tartja (a kérés context-jének vége, ahogy az ütemező is figyeli).
- Szabály: a modell eszközlistáján sorban az első szabad eszköz nyer (szabad: `running < max_running`, és jelen van). Az első eszköz foglalt: `on_busy: wait` → legfeljebb `wait_timeout`-ig vár rá (érkezési sorrendben), utána továbblép; `on_busy: next` → azonnal továbblép (a `wait_timeout`-ot figyelmen kívül hagyja, naplósorral). A lista többi eszközénél nincs várakozás. Ha sehol nincs hely, a kérés sorba áll, és a lista elsőként felszabaduló eszközén indul, érkezési sorrendben; hiba és túlterhelés nincs (a sor nem dob, csak a kérés context-je szakíthatja meg).
- A GPU-n futó, a CPU-val megosztott modell a CPU-n is helyet foglal: a GPU helye után, a betöltés ismeretében (a `-ngl`/fit szerinti osztás a naplóból: nem minden réteg a GPU-n). A foglalás sorrendje mindig GPU → CPU, így körkörös várakozás nem lehet.
- A tartalékeszköz csak az adott kérésre szól: a következő kérés újra a lista elejéről indul.
- Config nélkül, vagy ha a modellnek nincs `devices` listája: nincs foglalás, a régi viselkedés.

### 2.5 Példányok eszközönként
- Az ütemező kulcsa (`schedulerModelKey`) az elhelyezett eszközzel bővül (`<út>\x00@<eszköz>`), így ugyanaz a modell egyszerre lehet betöltve a gpu0-n és a CPU-n (CPU-s példány). Config nélkül a kulcs változatlan.
- GPU-s elhelyezés: a felderített eszközök közül csak a megfelelő (`selectLlamaServerPlacement` helyett), `main_gpu` 0, a fit a GPU és a CPU között oszt (zónák). CPU-s elhelyezés: `num_gpu 0`, eszközlista üres; a betöltés repackel (ez a „demote”).

### 2.6 Csere a fő GPU-n
- Ha a kért eszközön a becslés szerint nincs hely, az ütemező az **ugyanazon az eszközön** ülő tétlen modellek közül a legrégebben használtat választja (nem a globális, keep_alive szerinti sorrendet).
- **A csere ebben a változatban folyamatszintű (park):** a tétlen modell KV-je slot-mentéssel a RAM-ba kerül (tmpfs: `/dev/shm`), a folyamat leáll, a VRAM felszabadul; a modellfájl lapjai `keep_in_ram: pinned` esetén zárolva a RAM-ban maradnak (Linuxon `mmap` + `mlock` az Ollama folyamatában, a compose `memlock: -1`-ével), így a visszatöltés a RAM-ból olvas. A modell „parkoló” állapotba kerül (nincs folyamata, de a RAM-ban van, és a következő kérése után a KV visszatöltődik).
- **Eltérés a kért viselkedéstől:** a kért „~1 s / 10 GB” a GPU-formájú, rögzített (pinned) host-példányból való feltöltést feltételezi, ami folyamaton belüli mechanizmus (a fork flex sávjának kiterjesztése minden GPU-s rétegre és a kimeneti rétegre, a compute puffer elengedésével). A folyamatszintű csere ehhez képest a llama-server indítását, a fit-et, a CPU-rész repackjét is újra elvégzi (becslés: néhány másodperc; a NAS-on mérendő). A folyamaton belüli csere terve: 4. pont, a nyitott kérdés: `OPEN-QUESTIONS.md`.

### 2.7 RAM-keret és LRU
- Keret: a konténer cgroup-korlátja (`memory.max`, v1-nél `memory.limit_in_bytes`), ha a configban kisebb `ram_limit` van, az. Használat: a cgroup `memory.current` − `inactive_file` (a fork RAM-védelmével azonos értelmezés), cgroup nélkül a rendszer szabad memóriája.
- Új betöltés előtt: ha a becsült gazda-RAM-igény (az előző azonos betöltés mért értéke, különben a modellfájl mérete) nem fér el, kirakás ebben a sorrendben, amíg elfér: (1) tétlen CPU-s példányok, LRU; (2) parkoló modellek, LRU; (3) tétlen betöltött modellek, LRU. A `keep_alive: -1`-es modell soha nem rakódik ki teljesen (parkolni lehet). Ha nincs mit kirakni, a betöltés megy (a llama-server RAM-védelme a dual-buffer másolatait magától csökkenti).

### 2.8 KV-mentés az NVMe-re
- Teljes kirakás előtt (lejárat, LRU), ha a modellnél be van kapcsolva (`kv_save: true` és a `swap.save_kv`): `POST /slots/{i}?action=save` minden nem üres slotra, a `swap.kv_dir` (alap: `<modellek>/dualbuf-kv`) alá, egy kísérő JSON-nal (modell digest, kv_type, n_ctx, slot). A llama-server a `--slot-save-path`-ot az `extraEnvs`-ből kapja (`LLAMA_ARG_SLOT_SAVE_PATH`).
- Betöltés után a futó a kísérő alapján visszatölt: csak azonos modell-digestnél és KV-típusnál; eltérő max kontextusnál a viselkedést a laptopos teszt dönti el (`test-results.md`).
- Csere (2.6): ugyanez, de a RAM-ba (tmpfs).

### 2.9 Vulkan
- Image: a `rocm` cél a `llama-server-vulkan` stage-et is másolja, és a végső stage-be egy build-argumentummal (`EXTRA_APT_PACKAGES`, alap üres) a `mesa-vulkan-drivers` kerül (a fork workflow-ja adja át). Az upstream build ugyanaz marad.
- A Vulkan-konténernek (és a közös konténernek) a render node mellett a `/dev/kfd` is kell: a Vulkan-backend AMD-kártyán egy üres KFD compute-sorral kéri a compute-teljesítményállapotot, amit a ROCm-folyamat automatikusan kap (DECISIONS 95–96); nélküle a modell fut, de a GPU split-generálás közben alacsony órajelen marad.
- Dinamikus KV Vulkanon: ma másolással nő (backend-független út). Helyben növelés: a ggml-vulkan bővítése sparse-binding pufferrel (a fork llama.cpp `multimodel` branchén, a VMM-es utat követő API-val), ha az eszköz jelzi (`sparseBinding` + `sparseResidencyBuffer` + sparse sor); különben a másolásos út marad, egy naplósorral. (Ha a másolásos út sem megy, a fix KV a tartalék, a terv szerint.)

## 3. A horgok listája (rebase-felület)
| fájl | horog |
|---|---|
| `server/routes.go` | `modelOptionsWithEmbeddingBatchDefault`: két `FromMap` (defaults, modell szakasza); `scheduleRunnerForModel`: `Acquire` + az elhelyezés átadása; `usesAutomaticNumCtx`: a configban adott `max_context` nem automatikus |
| `server/sched.go` | `LlmRequest`/`runnerRef`: egy-egy mező (a pillanatkép, az elhelyezés); `schedulerModelKey`: az elhelyezés a kulcsban; `load`: eszközszűrés, `numParallel`, a llama-server beállításai, naplósor; `processPending`: RAM-keret, eszközre szűrt áldozat; `unload`: KV-mentés; `needsReload`: a fork-mezők ujjlenyomata |
| `server/images.go` (`Model`) | egy mező: az elhelyezés |
| `llm/server.go`, `llm/llama_server.go` | `LlamaServerConfig`: `ExtraEnv`, `ArgOverrides`, `KVCacheType`; `startLlamaServer`: a kapcsolók felülírása; `SaveSlots`/`RestoreSlots` |
| `Dockerfile` | a `rocm` cél + 1 `COPY`, a végső stage + 1 `ARG` |

## 4. A folyamaton belüli csere (következő lépés, nem ebben a körben)
- Betöltéskor (`keep_in_ram: pinned`): minden GPU-s súly (rétegek és a kimeneti réteg) kap egy eredeti formájú host-példányt a rögzített host-pufferben (a dual-buffer ikerképzésének mintájára, a flex `host` mezővel).
- Park (új llama-server végpont, csak tétlen állapotban): a KV a rögzített RAM-ba másolódik, a GPU-s súly-, KV- és compute pufferek felszabadulnak. Unpark: azonos méretű pufferek, a tenzorok ugyanabba a sorrendbe (`ggml_tallocr`), másolás a host-példányokból (DMA), KV vissza, a scheduler újrafoglalása.
- Ami kockázatos: a compute puffer elengedése (a schedulert ma csak a context létrehozásakor lehet felépíteni), a nézet-tenzorok újrakötése, HIP-gráfok a címváltozás után. A NAS-on a ROCm-mal kell igazolni.
