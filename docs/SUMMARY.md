# Összefoglaló: dual-buffer az Ollama 0.40.2-ben

## Mi készült el
- **Fork és munkakörnyezet:** `csaki89/ollama` (origin), `upstream` = `ollama/ollama` (push: `DISABLED`), `dual-buffer` branch a `v0.40.2` tagből, a `main` érintetlen. `CLAUDE.md` + `.claude/settings.json` (a szabályok és tiltások).
- **Felderítés** (`docs/DESIGN.md`): hogyan épül és indul a llama-server, milyen kapcsolókat/env-eket kap, melyik modell melyik motoron fut, hogyan állítható tartósan a `use_mmap: false`, van-e RAM-becslés, a b11351 és az e14a77971 különbsége, és a beépítési mód döntése.
- **A dual-buffer beépítése:** `llama/compat/003-dual-buffer.patch`, az Ollama saját patch-mechanizmusával (a `001`, `002` után alkalmazódik). 11 llama.cpp-fájl, a ../llama-dualbuf diffjéből a b11351-re átültetve (4 hunk kézzel, a scheduler-regisztráció a b11351 `sched_reserve()`-jéhez igazítva; DECISIONS.md). Az Ollama Go kódja változatlan: a `LLAMA_ARG_DUAL_BUFFER` és a `LLAMA_ARG_LOAD_MODE` a környezetből átmegy a llama-serverre.
- **Laptopos igazolás Vulkannal** (`docs/test-results.md`): a patch-elt llama-server lefordul (Ollama build-útvonal, MSVC/Ninja), a naplóban megjelenik a `dual-buffer: 87 weights also stored in Vulkan_Host (308.18 MiB)` sor és mindkét puffer; kapcsoló nélkül az eredeti viselkedés; a prompt 324,8 tok/s a host-puffer 330,3-jával szemben (98%) és 2,7x a repack 121,4-énél.
- **GitHub Actions** (`.github/workflows/dualbuf-image.yaml`): `dual-buffer` pushra felépíti a ROCm image-et (csak gfx1030, a Dockerfile `rocm` célja, `FLAVOR=rocm`) és feltölti a `ghcr.io/csaki89/ollama-dualbuf` címre `0.40.2-dualbuf-<commit>` és `latest` címkével. Dockerfile-változás: 2 új `ARG` + 1 `RUN` (gfx1030-ra szűkítés és a többi architektúra rocBLAS/hipBLASLt-fájljainak törlése).
- **`docker-compose.yaml`** a TrueNAS egyéni apphoz (teszt: 30069-es port).

## Ami nincs kész
- **Az `ollama` binárissal végzett laptopos próba.** A laptopon nincs Go, a telepítés tiltott, a Go 1.26.10 MSI letöltve és hash-ellenőrizve a scratchpadben (`...\scratchpad\go-dl\go1.26.10.windows-amd64.msi`). Telepítés után: `cmake -B build . -DOLLAMA_LLAMA_BACKENDS=vulkan` (Developer PowerShell, `VULKAN_SDK` beállítva) és `cmake --build build`, majd a Modelfile (`FROM` a GGUF, `PARAMETER num_gpu 14`, `PARAMETER use_mmap false`) importja; ugyanezt a naplósort kell látni.
- **A workflow nem futott**, az image nincs lebuildelve (push az origin-re csak kérésre; a workflow és a Dockerfile-változás megtekinthető a `git diff v0.40.2..dual-buffer -- Dockerfile .github docker-compose.yaml` kimenetében). Az első build ideje és az image mérete ismeretlen. A `docker-compose.yaml` image-címkéjében a commit helye (`REPLACE_WITH_COMMIT`) az első sikeres build után tölthető ki.
- **ROCm/HIP, a NAS, MoE (qwen3-coder:30b), a compat-blobok (qwen3-vl:8b)**: nem futottak, a mérés a NAS-on lesz.
- **Automatikus RAM-keret:** nincs. Az Ollama nem becsli a gazda-RAM-ot, a `--fit` sem számol a host-ikerrel.

## Nyitott kérdések
1. **A gfx1030-only image mérete és a rocBLAS-szűrés:** a törlés a fájlnévben szereplő `gfx…` alapján történik; ha a rocBLAS `TensileManifest`/`.dat` fájlok más névsémát használnak, a szűrés kevesebbet vagy többet törölhet. Az első buildnél a `Smoke test` lépés `du -sh /usr/lib/ollama` kimenete mutatja; a futási próbát (a modell betöltése a ROCm-on) a NAS-on el kell végezni.
2. **A `LLAMA_ARG_LOAD_MODE=none` globális hatása forrásból következtetett** (DESIGN.md 4.), futtatással nem ellenőrzött. Ha a NAS-on a naplóban `load_mode = mmap` látszik, a Modelfile-os `use_mmap false` a biztos út.
3. **MoE:** a `MUL_MAT_ID` súlyok iker-cseréje nem tesztelt (DESIGN: ha a gráf egy nézetet kap, nincs offload és a viselkedés a mai marad); a qwen3-coder:30b-nél lehet, hogy nincs prompt-nyereség. A mérés dönt.
4. **RAM-többlet:** a repackelt CPU-s rész másodszor is foglal (pinned host). qwen3-coder:30b-nél a CPU-s rész ~8-12 GB, a 128 GB-ból belefér; gpt-oss:120b-nél nem (`LLAMA_ARG_DUAL_BUFFER_PATTERN=ffn_` / `LLAMA_ARG_DUAL_BUFFER_LIMIT=<MiB>`).
5. **Az Ollama VRAM-számlálója:** a `*_Host` és `CPU*` pufferek nem számítanak GPU-nak (`isGPUBuffer`), így a dual-buffer nem rontja az Ollama VRAM-becslését (kedvező), a RAM-használatot viszont nem látja.

## Hogyan húzza le a NAS a ghcr.io image-et
Az Actionsből, `GITHUB_TOKEN`-nel feltöltött csomag **alapból privát**.
- **Nyilvánossá tétel (ajánlott):** GitHub → a fiók „Packages” → `ollama-dualbuf` → „Package settings” → „Change visibility” → Public. Utána a NAS bejelentkezés nélkül húzza: `docker pull ghcr.io/csaki89/ollama-dualbuf:<címke>`, a TrueNAS is tudja. Az image nem tartalmaz titkot (Ollama és llama.cpp, mindkettő MIT), a hátránya csak az, hogy bárki letöltheti.
- **Privát marad:** a NAS shellben (root) `docker login ghcr.io -u csaki89`, jelszónak egy személyes hozzáférési token (classic, `read:packages` jogosultság). A token a NAS-on a Docker konfigurációs fájljában (`config.json`) tárolódik, lejárhat (akkor a frissítés/újratelepítés elhasal), és a TrueNAS rendszerfrissítése a bejelentkezést elveszítheti. Előny: az image nem nyilvános.
A különbség tehát: nyilvánosnál nincs titok a NAS-on és nincs lejárat; privátnál van token-kezelés. A compose-ban a címke rögzített, ne `latest` legyen.

## Hogyan kell a NAS-on elindítani
1. Push az origin-re (kifejezett kérésre), az Actions lefut; a `Compute tags` lépés kiírja a címkét (`0.40.2-dualbuf-<commit>`).
2. Láthatóság beállítása vagy `docker login` (lásd fent).
3. A `docker-compose.yaml`-ban a `REPLACE_WITH_COMMIT` helyére a címke commit-része.
4. TrueNAS: Apps → Discover Apps → „Install via YAML” → a fájl tartalma, az app neve pl. `ollama-dualbuf`. A teszt a **30069**-es porton fut a mostani (30068) mellett, ugyanazt a `/mnt/nvme/ollama/models` mappát használja.
5. Ellenőrzés: `docker logs ollama-dualbuf` (az `OLLAMA_DEBUG=1`-gyel bővebb) → `using llama-server for model`, `load_tensors: dual-buffer: N weights also stored in ROCm_Host (X MiB)`, `ROCm0 model buffer`, `CPU_REPACK model buffer`, `ROCm_Host model buffer`, és a `load_mode = none`. Ha `dual-buffer ... disabling it` figyelmeztetést látsz, az ok a sorban van (mmap, `--no-repack` vagy nincs host-puffer).

**Közös modellmappa:**
- Mindkét Ollama ugyanazt a `manifests/` és `blobs/` könyvtárat olvassa. A teszt-instance `OLLAMA_NOPRUNE=1`-gyel indul, hogy induláskor ne töröljön blobokat; mindkét példány azonos Ollama-verzió (0.40.2), így a manifest-formátum megegyezik.
- Ne tölts le és ne törölj modellt a teszt-instance-szal, amíg a másik fut (félkész `-partial` blobok, versenyhelyzet). Az új modelleket a meglévőn húzd le.
- Modellenkénti `use_mmap: false`: `printf 'FROM qwen3-vl:8b\nPARAMETER num_gpu 30\nPARAMETER use_mmap false\n' > Modelfile && ollama create qwen3-vl-dualbuf-30 -f Modelfile` a teszt-instance-ban. Ez csak egy kis manifestet ír a közös mappába (a blobok közösek); a másik Ollama is látni fogja a nevet (ártalmatlan). Globálisan ezt a compose `LLAMA_ARG_LOAD_MODE=none` váltja ki, így Modelfile nélkül is működik (lásd a nyitott kérdés 2.).

## Hogyan kell mérni
**Mi fut mely motoron:** a 0.40.2-ben minden GGUF modell a llama-serveren fut (`llm.NewLlamaServer`), tehát a **qwen3-vl:8b, a Qwen2.5-1.5B és a qwen3-coder:30b is**; Ollama „saját motorja” GGUF-ra nincs. Egyedül a `safetensors` modell megy az `mlxrunner`-re, ami a NAS ROCm image-ében nem érhető el (DESIGN.md 3.). A qwen3-vl:8b blobját a compat réteg fordítja (és mmap nélkül tölti). A naplóban a `using llama-server for model` sor ezt minden modellnél megerősíti.

**Elrendezés:** két Ollama fut egyszerre a NAS-on (a mostani `ollama/ollama:0.40.2-rocm` a 30068-on és a teszt a 30069-en). A GPU-t egyszerre csak az egyik használhatja (10 GB), ezért mérés közben a másikat `keep_alive: 0`-val ürítsd ki, vagy állítsd le.

| # | modell | `num_gpu` | motor | cél |
|---|---|---|---|---|
| 1 | qwen3-vl:8b | 30, 18, 6 | llama-server | a ../llama-dualbuf NAS-mérésével azonos sorozat |
| 2 | Qwen2.5-1.5B (importált GGUF, vagy bármely kicsi) | 14 | llama-server | a laptopos próba ismétlése ROCm-on |
| 3 | qwen3-coder:30b | pl. 20 (a VRAM-hoz igazítva) | llama-server | MoE-próba (lásd nyitott kérdés 3.) |

**Konfigurációk mindhárom modellhez:**
- A) régi: `ollama/ollama:0.40.2-rocm` (30068), `use_mmap` be- és kikapcsolt állapotban is (a kikapcsolt a méltányos referencia a `--load-mode none` miatt).
- B) új, `LLAMA_ARG_DUAL_BUFFER` nélkül (a compose-ban a változót töröld vagy `0`): az új image alapviselkedése, a patch hatása nélkül.
- C) új, `LLAMA_ARG_DUAL_BUFFER=1`.
A B és C között a kapcsoló az egyetlen különbség (ugyanaz az image és ugyanaz a `LLAMA_ARG_LOAD_MODE=none`). A `LLAMA_ARG_NO_HOST` ne legyen beállítva (felülírja a viselkedést).

**Parancs (egy mérés, a kérés opciói felülírják a Modelfile-t):**
```
HOST=http://<nas>:30069   # 30068 a régi
for N in 30 18 6; do
  for i in 1 2 3; do
    curl -s $HOST/api/generate -d '{"model":"qwen3-vl:8b","prompt":"'"$PROMPT"'","stream":false,
      "options":{"num_gpu":'$N',"use_mmap":false,"num_predict":300,"temperature":0,"seed":1}}' \
    | jq -r '"num_gpu='$N' prompt \(.prompt_eval_count/(.prompt_eval_duration/1e9)|floor) t/s, gen \(.eval_count/(.eval_duration/1e9)*10|floor/10) t/s"'
  done
done
```
- A `PROMPT` ~100 tokenes legyen (a régi mérés: 98 tokenes prompt, 300 generált token); egy hosszabb (~1500 token) prompt is érdemes, mert ott a prompt-nyereség jobban látszik.
- Az első kérés a betöltést is méri: dobd el, és minden `num_gpu` váltásnál az új betöltés után mérj 3 ismétlést, átlaggal és szórással. A `cache_prompt` miatt a prompt-sebesség méréséhez minden kérésnek más (vagy `keep_alive:0`-val újratöltött) promptot kell használnia, különben a gyorsítótárból jön.
- Naplóból ellenőrizd mérésenként a puffereket (`CPU_REPACK`, `ROCm_Host`), a `load_mode`-ot és az `offloaded N/M layers` sort.
- Várt: C-ben a prompt-sebesség a host-puffer (`--no-repack` nélküli, dual-buffer nélküli alapmód) szintjén (a ../llama-dualbuf mérése szerint +23% / +59% / +64% 30/18/6 réteggel a repackhez képest), a generálás a repack szintjén (28,7-29,5 / 18,8-20,0 / 14,0-14,5 tok/s).
- A RAM-többlet mérése: `free -m` a betöltés előtt és után (B vs C), várt különbség ≈ a naplóbeli `ROCm_Host` iker mérete.
