# Többmodelles működés: haladás

## Összefoglaló (reggelre)

**Elkészült** (branch: `multimodel` a csaki89/ollama-ban, pusholva; a csaki89/llama.cpp `multimodel` branche helyben létrejött, de llama.cpp-módosítás nem kellett, ezért nincs rajta új commit és nincs pusholva):
- Felderítés (`MULTIMODEL-RESEARCH.md`) és terv (`MULTIMODEL-DESIGN.md`): a logika az Ollama Go-ütemezője mellett, egy új `multimodel/` csomagban, vékony horgokkal; a llama-server router módja elvetve (indoklás a tervben).
- `/models/dualbuf.yaml`: olvasás minden betöltéskor, szigorú ellenőrzés (elírt kulcs = hiba), hibás fájlnál az utolsó érvényes + `ERROR` sor, a futó modellt nem érinti; erősorrend env < defaults < Modelfile < modell szakasza < kérés; minták (`qwen3-embedding:*`); egy naplósor betöltéskor minden mezővel és forrással.
- Eszközazonosítás PCI-cím vagy név szerint, backend-ellenőrzéssel (két azonos kártyánál PCI kell); eszközönkénti példány (a GPU-s és a CPU-s példány külön futó).
- Futási korlát (`max_running`), `on_busy: wait/next`, sor érkezési sorrendben, a megosztott (GPU+CPU) futó CPU-helye, kérésenkénti tartalékeszköz.
- CPU-tartalék példány (`-ngl 0`, repack), RAM-hiánynál elsőként megy.
- Csere (park) folyamatszinten: KV a RAM-ba (tmpfs), a modellfájl zárolva a RAM-ban (Linux), visszatöltéskor a KV visszajön.
- RAM-keret a cgroupból (vagy `ram_limit`), LRU-kirakás (CPU-tartalék → parkoló → tétlen), `keep_alive: -1` soha.
- KV-mentés az NVMe-re teljes kirakás előtt (`kv_save`), visszatöltés a következő betöltéskor (azonos digest és KV-típus).
- Az ütemező a fork mért memóriájával számol: a VRAM a `zones_state` sorokból futás közben is frissül.
- Image: `rocm-vulkan` cél (ROCm + Vulkan backend + RADV), `docker-compose.multimodel.yaml` (port 30071, `shm_size`), CI (`multimodel-ci`: vet + `-race` tesztek Linuxon, Windowsos `ollama.exe`).

**Bizonyítottan működik** (laptop, az Ollama-binárissal, Vulkan iGPU + CPU, `test-results.md` „Többmodelles munka”): config nélkül a gyári indítósor; a config megjelenése a futó modellt nem érinti; erősorrend és naplósor; eszközazonosítás; `on_busy: next` (0 s, CPU-példány), `on_busy: wait` lejár → tovább / időben felszabadul → `waited`; sor és érkezési sorrend; KV az NVMe-re és vissza (a visszatöltés után a llama-server 1 tokent számol, kimenet azonos); csere oda-vissza a KV-val (1 token, kimenet azonos, csere 3–4,4 s a laptopon); RAM-hiánynál a CPU-tartalék, majd a parkoló kiesik, a `keep_alive: -1` marad; dinamikus KV Vulkanon (másolással, 4096 → 8192). A slot-visszatöltés eltérő kontextusba (nagyobb, kisebb, dinamikus KV) és eltérő KV-típusnál (elutasítva) külön llama-server teszttel. Egységtesztek: `multimodel` (config, erősorrend, eszközök, koordinátor, RAM, kirakási sorrend, KV-fájlok), `llm` (kapcsoló-felülírás, `zones_state`), `server` (no-op config nélkül, elhelyezés és sor, opció-erősorrend, áldozatválasztás); a CI-ben `-race`-szel zöld. Az image (`0.40.16-dualbuf-d805f68`) lefordult a Vulkan-backenddel.

**Nyitva maradt** (`OPEN-QUESTIONS.md`):
1. A csere folyamatszintű; a kért ~1 s / 10 GB-hoz a folyamaton belüli park kell (pinned GPU-formájú példány minden GPU-s súlyhoz, a pufferek elengedése/visszaépítése a llama.cpp-ben). A NAS-mérés (NAS-TESTS 6.) dönti el, kell-e.
2. Vulkan helyben növelés (sparse binding) nincs; a dinamikus KV Vulkanon másolással megy.
3. `zones: off` a dinamikus KV-t is kikapcsolja; a CPU-tartalék keep_alive-ja a modellé.
4. ROCm, X080, két GPU, a fájlzárolás és a valódi modellek: csak a NAS-on.

**Reggeli NAS-sorrend** (`NAS-TESTS.md`): 0. előkészítés (render node-ok a compose-ban, `inference compute` sorok, RADV) → 1. config nélkül azonos-e a 30069-cel → 2. config, naplósor, eszközök → 3. RX 550 (embedding, Vulkan) → 6. **csere-mérés** (a legfontosabb szám: `load_duration` visszacserénél, `keep_in_ram: pinned` és `none`) → 4. futási korlát/on_busy → 8. KV az NVMe-re → 7. RAM-keret → 5. → 9. (ha van második X080).

## Fázisok

| fázis | állapot | mivel ellenőrizve |
|---|---|---|
| 1. Felderítés | kész | forrás (Ollama v0.40.16, llama.cpp b12177, fork), git-történet (#20137), `vulkaninfo` a laptopon |
| 2. Terv | kész | – |
| 3a. Config, erősorrend, eszközök, naplósor | kész | egységtesztek, laptop (Ollama) |
| 3b. Futási korlát, on_busy, sor | kész | egységtesztek (`-race` a CI-ben), laptop |
| 3c. Példány eszközönként, CPU-tartalék | kész | laptop |
| 3d. RAM-keret, LRU | kész | egységtesztek (cgroup v1/v2), laptop (`ram_limit`) |
| 3e. KV-mentés (RAM, NVMe) | kész | llama-server teszt, laptop (Ollama) |
| 3f. Csere (park) | folyamatszinten kész; folyamaton belüli: terv | laptop (Ollama); NAS: mérés |
| 3g. Mért memória az ütemezőben | kész (RAM: cgroup; VRAM: `zones_state`) | egységteszt |
| 3h. Vulkan az image-ben | kész | CI image-build (`d805f68`) |
| 3i. Vulkan dinamikus KV | másolással kész; sparse: nem | laptop (Ollama) |
| Dokumentáció | kész | DECISIONS 50–63, NAS-TESTS, OPEN-QUESTIONS, test-results |

## Hol mi van
- Kód: `multimodel/` (config, feloldás, koordinátor, RAM, kirakás, KV-fájlok, zárolás), `server/sched_dualbuf.go` (+ horgok: `sched.go`, `routes.go`, `images.go`), `llm/llama_server_dualbuf.go` (+ horgok: `server.go`, `llama_server.go`).
- Példa-config: `docs/dualbuf.example.yaml`; laptopos config és szkriptek: `tools/dualbuf/multimodel/`.
- Image/compose: `Dockerfile` (`rocm-vulkan`, `EXTRA_APT_PACKAGES`, `LLAMA_SERVER_VULKAN_TARGETS`), `.github/workflows/dualbuf-image.yaml` (a `multimodel` ág, `[skip image]`), `.github/workflows/multimodel-ci.yaml`, `docker-compose.multimodel.yaml`.
