# Többmodelles működés: haladás

## Összefoglaló, 2. kör (folyamaton belüli csere, Vulkan sparse KV) — reggelre

**Elkészült** (csaki89/llama.cpp `multimodel` 6b19b5a28 és csaki89/ollama `multimodel`, mindkettő pusholva; a patch 003 a llama.cpp-forrás diffje a b12177 + 001 fára):
- **Folyamaton belüli GPU-csere (park/unpark)** a llama.cpp-ben: `--park` mellett minden GPU-s súlynak (az 1. zónáénak és a kimeneti rétegnek is) pinned, GPU-formájú példánya van; `POST /park` a folyamatot életben hagyva felszabadítja a súlyokat, a KV-t (a VMM/sparse arénákkal együtt) és a compute puffereket, a projektort a CPU-ra tölti; `POST /unpark` DMA-val visszatölt, a KV új növelhető szakaszokba kerül, a zónák a mért szabad memóriával újramérnek. Parkolt állapotban a modell a CPU-n kiszolgál; az első CPU-s kérés demote-ol (repack a pinned példányból, RAM-ellenőrzéssel), a repackelt példány a `POST /park/drop`-ig marad.
- **Az Ollama ezt használja** (`keep_in_ram: pinned`): áldozatnál `POST /park`, a parkolt példány kérésénél unpark (előbb helyet csinál), a CPU-ra helyezett kérés a parkolt példányon fut. Folyamatszintű csere csak akkor, ha a pinned példányok nem fértek a RAM-keretbe (naplósor mindkét oldalon). RAM-hiánynál elsőként a demote-olt példányok mennek, a parkolt példányok a RAM-könyvelésben vannak, tartalék max(1 GiB, 5%).
- **Vulkan dinamikus KV helyben növeléssel** (sparse binding, a VMM-interfész Vulkan-megvalósítása); tartalék csak hiányzó képességnél, a naplósor megnevezi.
- **`zones` és `dynamic_kv` két kapcsoló** (`--no-zones-flex` / `LLAMA_ARG_ZONES_FLEX`).

**Bizonyítottan működik** (laptop, Iris Xe; részletek: `test-results.md` „2. kör”):
- llama-server: park/unpark 3× azonos kimenettel, a KV megmarad (`prompt_n=1`); flex sávval is; demote és drop; park 70–162 ms, unpark 0,19–0,53 s / 935 MiB.
- Ollamán át: 8 csere, mind folyamaton belül (park 0,23–0,54 s, unpark 0,29–0,89 s), 2 llama-server folyamat összesen, azonos válaszok, a KV megmarad; a pinned példányok elutasításakor a folyamatszintű tartalék naplósorral; valódi RAM-hiánynál a parkolt modell LRU-kirakása; CPU-s kérés a parkolt példányon demote-tal.
- Vulkan sparse KV: 28/28 réteg helyben nő (4096 → 24576) és zsugorodik, a felszabadított memória a meghajtó szerint visszajön (+490/560 MiB); a kimenet azonos a másolásos úttal.
- CI: Go vet + `-race` tesztek zöldek; az image a park- és sparse-kóddal ROCm/HIP-re és Vulkanra fordul (`0.40.16-dualbuf-3ab2eb5`, majd a későbbi buildek).

**Nyitva** (`OPEN-QUESTIONS.md`): a parkoló folyamat VRAM-maradéka diszkrét GPU-n (az iGPU-n nem mérhető tisztán); a rekurrens állapot (hibrid modellek) nem parkol; a projektor-park laptopon nem tesztelt (nincs vision modell); a 10 GB-os csereidő csak a NAS-on mérhető.

**Reggeli NAS-sorrend** (`NAS-TESTS.md`): 0. előkészítés → 1. config nélkül azonos → 2. config/eszközök → **10. folyamaton belüli csere az X080-on** (a fő mérés: unpark-idő 10 GB-nál, cél ~1 s; VRAM-maradék sysfs-szel; összevetés `keep_in_ram: none`-nal és a hideg betöltéssel; parkolt modell a CPU-n; projektor) → **11. Vulkan sparse KV a RX 550-en** → 3. → 4. → 8. → 7. → 5. → 9.

## Összefoglaló, 1. kör

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

**Bizonyítottan működik** (laptop, az Ollama-binárissal, Vulkan iGPU + CPU, `test-results.md` „Többmodelles munka”): config nélkül a gyári indítósor; a config megjelenése a futó modellt nem érinti; erősorrend és naplósor; eszközazonosítás; `on_busy: next` (0 s, CPU-példány), `on_busy: wait` lejár → tovább / időben felszabadul → `waited`; sor és érkezési sorrend; KV az NVMe-re és vissza (a visszatöltés után a llama-server 1 tokent számol, kimenet azonos); csere oda-vissza a KV-val (1 token, kimenet azonos, csere 3–4,4 s a laptopon); RAM-hiánynál a CPU-tartalék, majd a parkoló kiesik, a `keep_alive: -1` marad; dinamikus KV Vulkanon (másolással, 4096 → 8192). A slot-visszatöltés eltérő kontextusba (nagyobb, kisebb, dinamikus KV) és eltérő KV-típusnál (elutasítva) külön llama-server teszttel. Egységtesztek: `multimodel` (config, erősorrend, eszközök, koordinátor, RAM, kirakási sorrend, KV-fájlok), `llm` (kapcsoló-felülírás, `zones_state`), `server` (no-op config nélkül, elhelyezés és sor, opció-erősorrend, áldozatválasztás); a CI-ben `-race`-szel zöld. Az image lefordult a Vulkan-backenddel és a RADV-vel, a füstteszt zöld (`ghcr.io/csaki89/ollama-dualbuf:0.40.16-dualbuf-986377d`, mozgó címke: `:multimodel`).

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
| 3h. Vulkan az image-ben | kész | CI image-build (`986377d`, füstteszt zöld) |
| 3i. Vulkan dinamikus KV | másolással kész; sparse: nem | laptop (Ollama) |
| Dokumentáció | kész | DECISIONS 50–63, NAS-TESTS, OPEN-QUESTIONS, test-results |

## Hol mi van
- Kód: `multimodel/` (config, feloldás, koordinátor, RAM, kirakás, KV-fájlok, zárolás), `server/sched_dualbuf.go` (+ horgok: `sched.go`, `routes.go`, `images.go`), `llm/llama_server_dualbuf.go` (+ horgok: `server.go`, `llama_server.go`).
- Példa-config: `docs/dualbuf.example.yaml`; laptopos config és szkriptek: `tools/dualbuf/multimodel/`.
- Image/compose: `Dockerfile` (`rocm-vulkan`, `EXTRA_APT_PACKAGES`, `LLAMA_SERVER_VULKAN_TARGETS`), `.github/workflows/dualbuf-image.yaml` (a `multimodel` ág, `[skip image]`), `.github/workflows/multimodel-ci.yaml`, `docker-compose.multimodel.yaml`.

