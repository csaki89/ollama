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
- **Az image lebuildelve** (workflow run `42250551239`, sikeres, kb. 34 perc): `ghcr.io/csaki89/ollama-dualbuf:0.40.2-dualbuf-7ce1f36` (és `latest`), 1,76 GB, `ollama --version`: 0.40.2, a rocBLAS kernelek csak gfx1030-ra. A ROCm backend **futását** (modell betöltése a GPU-n) még senki nem próbálta. A `docker-compose.yaml` a fenti rögzített címkét használja.
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
- **Nyilvánossá tétel (ajánlott, a GitHub felületén kell megtenni):** GitHub → Packages → `ollama-dualbuf` → Package settings → Change visibility → Public. Utána a NAS bejelentkezés nélkül húzza: `docker pull ghcr.io/csaki89/ollama-dualbuf:<címke>`, a TrueNAS is tudja. Az image nem tartalmaz titkot (Ollama és llama.cpp, mindkettő MIT), a hátránya csak az, hogy bárki letöltheti.
- **Privát marad:** a NAS shellben (root) `docker login ghcr.io -u csaki89`, jelszónak egy személyes hozzáférési token (classic, `read:packages` jogosultság). A token a NAS-on a Docker konfigurációs fájljában (`config.json`) tárolódik, lejárhat (akkor a frissítés/újratelepítés elhasal), és a TrueNAS rendszerfrissítése a bejelentkezést elveszítheti. Előny: az image nem nyilvános.
A különbség tehát: nyilvánosnál nincs titok a NAS-on és nincs lejárat; privátnál van token-kezelés. A compose-ban a címke rögzített, ne `latest` legyen.

## Hogyan kell a NAS-on elindítani
1. Push az origin-re (kifejezett kérésre), az Actions lefut; a `Compute tags` lépés kiírja a címkét (`0.40.2-dualbuf-<commit>`).
2. Láthatóság beállítása vagy `docker login` (lásd fent).
3. A `docker-compose.yaml` már a rögzített címkét tartalmazza (`0.40.2-dualbuf-7ce1f36`); újabb build esetén írd át a `Compute tags` lépés által kiírt címkére.
4. TrueNAS: Apps → Discover Apps → „Install via YAML” → a fájl tartalma, az app neve pl. `ollama-dualbuf`. A teszt a **30069**-es porton fut a mostani (30068) mellett, ugyanazt a `/mnt/nvme/ollama/models` mappát használja.
5. Ellenőrzés: `docker logs ollama-dualbuf` (az `OLLAMA_DEBUG=1`-gyel bővebb) → `using llama-server for model`, `load_tensors: dual-buffer: N weights also stored in ROCm_Host (X MiB)`, `ROCm0 model buffer`, `CPU_REPACK model buffer`, `ROCm_Host model buffer`, és a `load_mode = none`. Ha `dual-buffer ... disabling it` figyelmeztetést látsz, az ok a sorban van (mmap, `--no-repack` vagy nincs host-puffer).

**Közös modellmappa:**
- Mindkét Ollama ugyanazt a `manifests/` és `blobs/` könyvtárat olvassa. A teszt-instance `OLLAMA_NOPRUNE=1`-gyel indul, hogy induláskor ne töröljön blobokat; mindkét példány azonos Ollama-verzió (0.40.2), így a manifest-formátum megegyezik.
- Ne tölts le és ne törölj modellt a teszt-instance-szal, amíg a másik fut (félkész `-partial` blobok, versenyhelyzet). Az új modelleket a meglévőn húzd le.
- Modellenkénti `use_mmap: false`: `printf 'FROM qwen3-vl:8b\nPARAMETER num_gpu 30\nPARAMETER use_mmap false\n' > Modelfile && ollama create qwen3-vl-dualbuf-30 -f Modelfile` a teszt-instance-ban. Ez csak egy kis manifestet ír a közös mappába (a blobok közösek); a másik Ollama is látni fogja a nevet (ártalmatlan). Globálisan ezt a compose `LLAMA_ARG_LOAD_MODE=none` váltja ki, így Modelfile nélkül is működik (lásd a nyitott kérdés 2.).
- **Mire figyelj, amíg a produkciós Ollama ugyanezt a mappát használja:**
  - A mappa írható (nincs `:ro`), mert az `ollama create` manifestet ír a `manifests/` alá. A blobok nem másolódnak (a `FROM <modell>` variáns ugyanazt a blobot használja), a variáns neve a produkciós Ollama listájában is megjelenik.
  - A teszt-instance-szal ne `pull`/`rm`/`create` olyan néven, ami a produkciósban használt név (felülírná a manifestet). A variánsnak mindig új nevet adj (`<modell>-dualbuf-<num_gpu>`), és a produkciós modelleket ne töröld.
  - Az `OLLAMA_NOPRUNE=1` marad; enélkül a teszt-instance induláskor a számára ismeretlen (félkész vagy másik instance által húzott) blobokat törölhetné. A produkciósat ez nem védi: az továbbra is takarít, ha neki nincs beállítva, ezért a variáns törlése (`ollama rm`) a blobot nem érinti, amíg más manifest hivatkozik rá.
  - Egyszerre ne futtass nagy modellt mindkét példányon: közös a RAM és a GPU (a `OLLAMA_MAX_LOADED_MODELS=1` példányonként értendő).
- **Erőforráskorlát és RAM-védelem:** a compose 60 CPU-t és 100 GB RAM-ot ad a konténernek (`deploy.resources.limits`). Ez cgroup-korlát, amit a dual-buffer automatikus RAM-védelme olvas (llama-dualbuf DECISIONS 11–13): a másolatokra jutó keret = korlát − használat (az inaktív page cache nélkül) − a CPU-n maradó modellrész − tartalék (max(1 GiB, a korlát 5%-a)). Ha a keret nem elég, kevesebb tenzor kap host-másolatot, ha 0, a dual-buffer figyelmeztetéssel kikapcsol. A naplósor jelzi a forrást: `free RAM N MiB (… cgroup limit …)`. A TrueNAS custom appban a `deploy.resources` érvényesülését ellenőrizd (`docker inspect` → `HostConfig.Memory`), különben a védelem a gazdagép RAM-ját látja.
- A compose healthchecket (`ollama list`) és naplókorlátot (10 MB × 3) is tartalmaz.

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

## Ellenőrzés a NAS-on: image lehúzása, ROCm-könyvtár és rocBLAS gfx-kernelek
A GitHub Actions smoke testje ugyanezt futtatja; a NAS-on így nézhető meg (a csomagnak publikusnak kell lennie, vagy előtte `docker login ghcr.io`):
```
IMG=ghcr.io/csaki89/ollama-dualbuf:0.40.2-dualbuf-7ce1f36
docker pull $IMG && docker run --rm --entrypoint /bin/sh $IMG -c 'ollama --version 2>&1 | tail -1; du -sh /usr/lib/ollama/*/; ls -la /usr/lib/ollama/rocm_v7_2; echo "rocBLAS gfx targets:"; ls /usr/lib/ollama/rocm_v7_2/rocblas/library | grep -o "gfx[0-9a-z]*" | sort | uniq -c'
```
Várt: `0.40.2`, a rocBLAS-nél csak `gfx1030`. A tényleges ROCm-futást a teszt-instance indítása után a naplóban kell nézni (`docker logs ollama-dualbuf`: `ROCm0` eszköz, `HSA_OVERRIDE_GFX_VERSION`).

## Rugalmas sáv (`--flex-layers`): mérés a NAS-on
A patch (`003`) tartalmazza a rugalmas sávot is (a GPU-s rétegek első N-je futás közben a CPU-ra vehető és visszatehető; DECISIONS.md 12–18). Az Ollama alapból nem kapcsolja be, és nincs zónaváltási szabály: a mozgatást csak a hibakereső `LLAMA_FLEX_TEST` váltja ki. A mérőszkript ezt használja, a produkciós compose-ba **nem** kell semmit tenni.

**Futtatás a NAS-on** (a repó klónjából, root vagy docker-csoport; a szkript Python 3, nem a zsh):
```
sudo python3 tools/dualbuf/nas-flex-bench.py --image ghcr.io/csaki89/ollama-dualbuf:<címke>
```
- Előtte a produkciós Ollama ne tartson modellt a GPU-n (a VRAM-értékeket torzítaná): `curl -s localhost:30068/api/generate -d '{"model":"<modell>","keep_alive":0}'`, vagy várd meg a lejáratot. A szkript figyelmeztet, ha a GPU 300 MiB-nál többet használ indulás előtt.
- Alapból: modellek `qwen3-vl:8b` (dense) és `qwen3-coder:30b` (MoE), 0/1/2 levett réteg, mindegyik HIP-gráfokkal és `GGML_CUDA_DISABLE_GRAPHS=1`-gyel (12 konténerindítás, modellenként néhány perc betöltéssel; a teljes futás kb. fél–egy óra). A tesztkonténer: 100 GB memória- és 60 CPU-korlát, `/dev/kfd`, `/dev/dri/renderD129`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`, a `/mnt/nvme/ollama/models` csak olvashatóan, port 30070.
- Részhalmaz: `--models qwen3-vl:8b --layers 0,1 --no-graphs-off`; a dense modellnél beállítható az `-ngl`: `--num-gpu qwen3-vl:8b=30`; a MoE-nél ne add meg (az Ollama/`--fit` dönt). `--dry-run` csak kiírja a docker parancsokat.
- Eredmény: `flex-bench-<időbélyeg>/results.md` (táblázat: generálás és hosszú prompt tok/s, VRAM, felszabadult MiB, levétel/visszatétel ideje, gráfhasítások, értelmes-e a kimenet, eltér-e a k=0-tól és a gráfok nélküli futástól), `results.json` és a konténernaplók a `logs/` alatt.
- Ha az Ollama a csak olvasható modellmappán nem indulna el, a napló mutatja; ilyenkor a mappát ideiglenesen írhatóra kell csatolni (a szkriptben a `:ro`).

**Mit mér és hogyan:** a bemelegítő kérés tölti be a modellt, ennek első tokenjei után a `LLAMA_FLEX_TEST="1:take,..."` leveszi a rétegeket (a levétel és a K/V mozgatása a naplóból: `flex_move: layer N moved to CPU (…ms…)`). A szabad VRAM a `mem_info_vram_used` (`0000:86:00.0`) különbsége a k=0 esethez képest. A visszatételt (`20000:put`) a szkript töltelék-promptokkal éri el, a visszatételi idő szintén a naplóból jön. A gráfhasítások a `sched_reserve: graph … splits = pp / tg` sorokból (a szkript `LLAMA_FLEX_TEST_RESERVE=1`-et állít). A szkriptet a laptopon nem tudtam futtatni (nincs Python), az első futás a NAS-on lesz; a naplóformátum-illesztést valódi llama-server naplóval ellenőriztem.

**MoE-nél mire számíts:** a sáv a 0..N-1. réteg; a `--fit` szerint ezek az elöl lévő, teljes rétegek, ha jutott rájuk VRAM. A hátsó rétegek szakértői CPU-n vannak, ezek nem mozognak. Részletek és korlát: DECISIONS.md 16.
