# Nyitott kérdések (többmodelles munka)

Amit nem tudtam eldönteni vagy megvalósítani; mindegyiknél a pontos akadály és a mostani állapot.

## Lezárva (2. kör)
- ~~A GPU-csere folyamatszintű~~ → **folyamaton belüli park/unpark kész** (llama.cpp `--park`, `POST /park`, `/unpark`; az Ollama ezt használja, a folyamatszintű csere csak akkor marad, ha a pinned példányok nem fértek a RAM-keretbe, naplósorral). Laptop: `test-results.md` „Folyamaton belüli park”.
- ~~`zones: off` a dinamikus KV-t is kikapcsolja~~ → két külön kapcsoló: `zones: off` = `LLAMA_ARG_ZONES_FLEX=0` (`--no-zones-flex`), a dinamikus KV marad.
- ~~Vulkan helyben növelés~~ → **kész** (sparse binding a ggml-vulkanban; tartalék csak hiányzó képességnél, a naplósor megnevezi).

## Nyitva
1. **A parkoló folyamat VRAM-maradéka diszkrét GPU-n nem mért.** A laptop iGPU-ján a heap közös, a pinned host-példányok (935 MiB) és a host-K/V is a „device” heap-et terhelik, ezért a park után a szabad memória (6025 MiB) nem adja ki tisztán a HIP/Vulkan-kontextus maradékát (a kezdeti 7384 MiB-hoz képest 1359 MiB, ebből ~1047 MiB a pinned példány és a K/V, a maradék ~310 MiB a Vulkan-kontextus és a staging pufferek becsülhetően). Mérés: `NAS-TESTS.md` 6. (a sysfs `mem_info_vram_used` az X080-on).
2. **A rekurrens állapot (hibrid modellek, `llama_memory_recurrent`) parkoláskor a helyén marad** (a napló: `the recurrent state stays where it is`); a `park_kv` csak a `llama_kv_cache` részeket helyezi át. A tervezett modellek (qwen3-coder, qwen3-vl, embedding) nem hibridek, a qwen3.5/qwen3-next-nél ez néhány tíz-száz MiB VRAM marad. Megoldás: a rekurrens puffer `relayout`-szerű áthelyezése (nincs megcsinálva). A **projektor (mmproj) megoldva**: park-kor a CPU-n töltődik újra (a parkolt modell képpel is kiszolgálhat), unpark-kor vissza az eszközre; laptopon nem tesztelt (nincs projektoros modell), NAS-TESTS 10.7.
9. **Egy betöltés kevés memóriánál csendben leállt** a laptopon (Vulkan, a sparse aréna létrehozása után, miközben egy másik Ollama-példány is foglalt): hibaüzenet nélkül; a memória felszabadítása után ugyanaz a parancs hibátlan. Nem reprodukáltam célzottan; ha a NAS-on előjön, a `GGML_VK_SPARSE=0`-val érdemes összevetni.
3. **Parkolt állapotban a CPU-s kimenet eltérhet a GPU-stól** (más numerikus út, `temperature 0` mellett is); unpark után a kimenet bájtra azonos a park előttivel (mérve). A parkolt modell első CPU-s kérése demote-ol (repack a pinned példányból, RAM-ellenőrzéssel), a repackelt példány a `POST /park/drop`-ig megmarad, RAM-hiánynál az Ollama ezt dobja el elsőként.
4. **Az Ollama-oldali park a laptopon csak kis modellekkel igazolható**; a 10 GB-os csereidő (cél ~1 s, PCIe ~11,6 GB/s) a NAS-on mérendő (`NAS-TESTS.md` 6.).
5. **`on_busy: wait` `wait_timeout` nélkül**: 60 s beépített alapérték.
6. **A futási hely a kérés teljes idejére szól**, a betöltéssel (és az unparkkal) együtt.
7. **A CPU-tartalék példány keep_alive-ja** a modellé; RAM-hiánynál elsőként megy.
8. **Egy kártya két backenden** (X080 ROCm-ként és Vulkanként is látszik): az Ollama a ROCm-ot tartja meg, egy `backend: vulkan` bejegyzés erre a kártyára `missing` lesz (szándékos).

## Vulkan–ROCm egyezés (1. lépés)
10. **A fit memóriabudget nélkül a heap teljes méretét látja szabadnak.** A `--fit` (és vele az auto zónák két fitje) a betöltés előtt a `ggml_backend_dev_memory` szabad értékét használja; `VK_EXT_memory_budget` nélkül ez a heap mérete (a laptopon a letiltással mérve: `everything fits with the max context`, nincs flex sáv). A zónakeret betöltés után már a könyvelésből (vagy amdgpu sysfs-ből) számol, de a fit felső becslése a betöltéskor túlfoglalhat. A fit a ggml-ből nem kap más mérést; megoldás egy, a fitnek adott könyvelés (a többi folyamat foglalását ez sem látja). Nem csináltam meg.
11. **Nagy visszaadási egység és kis K/V-szakasz.** A Vulkan-darab nem lehet kisebb a meghajtó mért visszaadási egységénél (a laptopon 8 MiB). Ha egy réteg K-ja vagy V-je a max kontextuson sem nagyobb egy darabnál (kis modell, kis kontextus), a szakasz egyetlen darab, a zsugorítás ilyenkor nem ad vissza memóriát (nincs mit leválasztani). Nagyobb szakaszokkal (`-c 65536`) mérve a visszaadás a másolásos úttal egyező (90,5% / 90,4%). A ROCm-os úton ugyanez a helyzet a 2 MiB-os darabbal sokkal ritkább.
12. **UMA (integrált) eszközön** a mért szabad memóriából a host-pufferek (pinned példányok, `Vulkan_Host`) is levonódnak, mert a heap közös; a fit ezt nem számolja, ezért a mért zónakeret szűkebb lehet a vártnál (laptop, `--fit-target 6200`: negatív keret, a teljes flex sáv lemegy). Ez mérés, nem hiba; diszkrét kártyán nem jelentkezik.
