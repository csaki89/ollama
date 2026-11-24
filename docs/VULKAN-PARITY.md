# Vulkan és ROCm: a dual-buffer, a zónák, a flex rétegek és a dinamikus KV egyezése

Felmérés a `multimodel` branchen (a45647ddb..6b19b5a28 után), módosítás előtt. Minta: a ROCm (HIP) út. Mérés: laptop, Intel Iris Xe (Vulkan, UMA/iGPU), qwen2.5-1.5B Q4_K_M, `--auto-zones --dual-buffer -fa on --load-mode none -c 16384`, `--fit-target` 5800 / 6000 / 6200 / 6500 MiB (a 2. és 3. zóna kikényszerítésére), egy ~11 000 tokenes, utána két rövid kérés. Szkript: `tools/dualbuf/vk-parity.sh`.

| elem | ROCm (minta) | Vulkan ma | eltérés |
|---|---|---|---|
| dual-buffer (CPU-s súlyok host-példánya, op offload) | `ROCm_Host` | `Vulkan_Host`, `dual-buffer: 46 weights also stored in Vulkan_Host (162.84 MiB)` | nincs |
| auto zónák (két fit, flex sáv) | `auto zones: n_ctx ... (zone 1); ... (zone 1 + 2)`, `flex band N whole layers` | ugyanezek a sorok (FIT=6000: 27 réteg az 1. zónában, 2 rétegű sáv) | nincs |
| flex rétegek mozgása a K/V-vel | `flex_move_part: layer L moved to CPU`, `flex_move: layer L K/V moved to ROCm_Host` | ugyanez `Vulkan_Host`-tal; a hosszú kérés leveszi, a rövid utáni zsugorítás visszahozza (`2 flex steps off` → `2 back`) | nincs |
| dinamikus KV helyben növelés/zsugorítás | VMM, `N/N layers in place (VMM)`, `grown/shrunk in place` | sparse binding (2. kör), `28/28 layers in place`, `grown/shrunk in place`; a flex rétegek K/V-je másolással (mint ROCm-on) | nincs |
| **a VMM-darab mérete** | futásidőben: granularitás, igazítás és a meghajtó **mért** foglalási egysége (egy minimális foglalás előtti és utáni szabad memória, kétszer); egy naplósor mutatja az összetevőket | **fix 2 MiB** (`GGML_VK_SPARSE_CHUNK_KIB`), csak a sparse blokkra kerekítve; nincs mért foglalási egység, és a darabszám nem figyeli az eszköz foglalásszám-korlátját (`maxMemoryAllocationCount`, Windowson tipikusan 4096) – nagy kontextusnál a darabok száma túllépheti | **eltér** |
| a zónakeret a mért szabad memóriából | `hipMemGetInfo` (mért) + amdgpu sysfs (`mem_info_vram_used`) összevetés | `ggml_backend_dev_memory`: `VK_EXT_memory_budget`-tel mért (heap budget − usage); **a bővítmény nélkül a heap teljes mérete**, vagyis nem mérés, és a zónakód ezt nem tudja (a keret a teljes VRAM-ból indulna); a sysfs-összevetés amdgpu-n backendtől függetlenül megy | **eltér** (tartalék és naplósor hiányzik) |
| a felszabadított memória visszajön | VMM-darab foglalásának felszabadítása, mért | `vkFreeMemory`, a budget szerint mérve (+490/560 MiB az iGPU-n) | nincs |
| `zones_state` és a többi naplósor | `zones_state ... device free: booked, driver, sysfs` | ugyanez (a `driver` a Vulkan budgetje) | költségvetés nélkül a `driver` érték félrevezető (a heap mérete) |
| tartalékág: nincs helyben növelés | `no VMM support` → másolás | `no growable (sparse) KV buffers: <hiányzó képesség>; the KV grows by copy` | nincs |
| tartalékág: nincs mért szabad memória | – (CUDA/HIP mindig mér) | nincs: a heap mérete mérésként megy | **eltér** |
| HIP-specifikus utak (KFD-korlát, „unmap keeps memory”, darabonkénti címfoglalás) | kell | nem kell (a `vkFreeMemory` azonnal visszaad) | szándékos, a backend képessége dönt |

**UMA (integrált) eszközön** a mért szabad memóriából a host-látható foglalások (pinned host-példányok, `Vulkan_Host` pufferek) is levonódnak, mert a heap közös; ezért a mért keret szűkebb lehet, mint amit a fit becsült (FIT=6200: `device memory budget -306 MiB`, a teljes sáv lemegy). Ez mérés, nem hiba; diszkrét kártyán (ROCm vagy Vulkan) a host-pufferek nem a VRAM-ból fogynak.

## Teendők (ebben a lépésben)
1. Vulkan VMM-darab futásidőben: a sparse blokk (igazítás), a meghajtó mért foglalási egysége (a budget `heapUsage` változása egy blokknyi foglalásra, kétszer) és az eszköz foglalásszám-korlátja; egy naplósor a ROCm-os mintájára; költségvetés nélkül a foglalási egység „nem mérhető”, a darab az igazításból és a korlátból jön.
2. A zónakeret mért memóriából: a backend megmondja, mér-e (`ggml_backend_dev_memory_measured` proc); ha nem, a keret a könyvelésből (az eszköz teljes memóriája − a modell eszközön lévő súlyai − a KV − a compute pufferek − a margó), és egy naplósor megnevezi a hiányzó képességet (`VK_EXT_memory_budget`). Tesztkapcsoló a letiltáshoz: `GGML_VK_DISABLE_MEMORY_BUDGET`.
3. A ROCm-os út nem változik (a proc ott nincs, a mért ágon marad).

## Eredmény
Lásd a végén (a módosítások után kitöltve).

## Eredmény (e795e5860 után; mérés: laptop, Iris Xe, `tools/dualbuf/vk-parity.sh`)
| elem | állapot |
|---|---|
| dual-buffer, auto zónák, flex-mozgás a K/V-vel, naplósorok | egyezik (a módosítás előtt is) |
| dinamikus KV helyben növelés és zsugorítás | egyezik |
| a VMM-darab futásidőben | **egyezik**: `sparse block 65536; driver booking unit 65536 (measured); release unit 8388608 (measured: 8 allocations of it freed bring the heap usage back); allocation limit 2091911 (...); chunk 8388608`. A ROCm-os mintához (granularitás + mért foglalási egység) képest egy összetevővel több: a **visszaadási egység** (a legkisebb méret, amelynek felszabadítása a heap-használatot visszaállítja). Ok: ezen a meghajtón a 64 KiB-os darabokat a meghajtó belső blokkokból szolgálta ki, a zsugorításkor felszabadított 196 MiB-ból csak 72 MiB jött vissza; a mért 8 MiB-os darabbal 896-ból 811 MiB (90,5%), a másolásos úttal 490-ből 443 MiB (90,4%) – a kettő egyezik. ROCm-on ez a mérés nincs (a HIP-út változatlan). |
| a zónakeret a mért szabad memóriából | **egyezik**: `VK_EXT_memory_budget` mellett mért; nélküle (`GGML_VK_DISABLE_MEMORY_BUDGET=1`) a backend jelzi (`no VK_EXT_memory_budget ... not a measurement`), a zónák a kernel sysfs-ét (amdgpu) vagy a könyvelést használják (`the budget follows the booking only`), a `zones_state`-ben `driver -` |
| a felszabadított memória visszajön (mérve) | egyezik (fent) |
| tartalékágak naplósorral | egyezik: `GGML_VK_SPARSE=0` → `the KV grows by copy`; költségvetés nélkül → könyvelés |
| ROCm-út | változatlan: a `ggml_backend_dev_memory_measured` proc a CUDA/HIP-backendben nincs, a zónák a mért ágon maradnak; a `zones_state` formája (`driver %.0f`) ugyanaz |

Nyitva (OPEN-QUESTIONS.md): a fit költségvetés nélkül a heap teljes méretét látja szabadnak (felső becslés); nagy visszaadási egységnél egy kis modell egy K/V-szakasza egyetlen darab, ilyenkor a zsugorítás nem ad vissza memóriát (a mechanizmus nagyobb szakaszokkal mérve működik).

# 2. lépés: a NAS-on talált hibák (terv, a kód előtt)

## Okok (kódból és a laptopon mérve)
1. **Másolásos növelés helyben növelés helyett.** A `relayout` eszközönként egy arénát kér a nem-flex rétegek K/V-jének max-kontextusú méretére, szakaszonként +64 MiB címtartalékkal (közös kód, a ROCm-on is így). A HIP-aréna (`per_chunk`, DECISIONS 41) a címtartományt darabonként foglalja, a leképezéssel együtt: a ténylegesen lefoglalt cím a kártyára leképezett memóriával arányos. A Vulkan-aréna egyetlen sparse `VkBuffer` a teljes kért méretre: a 64 MiB-os tartalékokkal együtt túllépte a `sparseAddressSpaceSize`-t (RADV-n ~4 GiB) → `no growable (sparse) KV buffers` → minden réteg másolással.
2. **A másolásos tartalék csúcsa.** A betöltéskori K/V rétegenként közös pufferben van (pufferenként több réteg); a `set_dynamic` utáni teljes relayout az új rétegpuffereket a régi közös puffer mellé foglalja, és a régit csak a végén engedi el: a csúcs a teljes régi + a teljes új K/V. A további növelések már rétegenként mennek (a `resize_peak` ezt könyveli).
3. **A Vulkan host-pufferei nem látszanak a cgroupban.** A `Vulkan_Host` (dual-buffer példányok, a nem repackelhető CPU-súlyok, flex példányok, host-K/V, parkolt példányok) a meghajtón át (`vkAllocateMemory`, diszkrét kártyán a kernel GTT-je) foglal, amit a konténer memória-cgroupja nem számol; a `hipHostMalloc` (ROCm) felhasználói lapokat rögzít, azokat számolja. A llama-server RAM-őre és az Ollama-tervezés (`PlanRAM`) a cgroup `memory.current`-jéből dolgozik.
4. **Split-generálás.** A laptopon (Iris Xe, qwen2.5-1.5B, minden réteg a GPU-n, az FFN a CPU-n: 58 split tokenenként) a Vulkan-részek tokenenként ~22 ms-mal többet adnak, mint a csak-GPU-s futás ugyanazon műveletei: váltásonként ~0,75 ms, a gráfra várásban (`GGML_SCHED_TIMING`). A GPU→CPU visszaolvasás összevonása a gráf beküldésével (egy várakozás kettő helyett) a laptopon nem változtatott. Hogy ez a NAS-on a teljes ok-e, és mekkora a ROCm-os megfelelője, csak ott mérhető.
5. **Teljesen GPU-n, hosszú kontextusnál lassabb.** A sparse darab a RADV-n a sparse blokk (64 KiB), a ROCm-on 2 MiB (a szakaszigazítás, `GGML_CUDA_VMM_MIN_ALIGN`). 64 KiB-os kötésekkel a K/V nem kap nagy (2 MiB-os) laptábla-töredékeket; a hosszú kontextus K/V-olvasása így több TLB-hibával jár, rövid kontextusnál nem számít. A „nagyobb méretben maradt” KV ezt erősíti (több sparse darab). Mérés a NAS-on: sparse vs. `GGML_VK_SPARSE=0` ugyanazon a hosszú kérésen.
6. Csak dokumentálva (OPEN-QUESTIONS).
7. **Könyvelt és mért szabad memória eltér.** A könyvelés a ggml-pufferek névleges méretét látja (súlyok, K/V `cell_bytes` x cella, compute pufferek). Nem látja: (a) a Vulkan-backend saját eszközfoglalásait (`prealloc_x/y/split_k`, `add_rms_partials`, `sync_staging`: a gráf futása közben nőnek, nem a scheduler foglalja); (b) a sparse szakaszok darabra kerekítését (szakaszonként legfeljebb egy darab, a laptopon 8 MiB x 56 szakasz). A CUDA/HIP-pool ugyanígy kívül esik, de ott a NAS-mérések egyeztek (DECISIONS 45-46).

## Tervezett eltérések a ROCm-os úttól (csak ezek)
| eltérés | a Vulkan API melyik része kényszeríti ki |
|---|---|
| a szakasz saját sparse `VkBuffer`-t kap (nem egy közös aréna-puffert), a címtartomány a szakasz max méretére, tartalék nélkül; ha az eszköz korlátja (`sparseAddressSpaceSize`, `maxBufferSize`) nem engedi, a szakasz kisebb tartományt kap, és ha kinövi, a meglévő memóriadarabjai egy nagyobb tartományba **kötődnek át** (adatmásolás nélkül), a régi tartomány felszabadul | a HIP-ben a címtartomány darabonként foglalható és a leképezés címhez kötött (`cuMemAddressReserve`/`cuMemMap`); Vulkanon a sparse címtartomány egy `VkBuffer`, a mérete létrehozáskor rögzül, és a sparse címtartomány összesen korlátos. Egy tenzor egy pufferben kell legyen, ezért a „több tartomány” szakaszonkénti puffert és átkötést jelent |
| a darab a sparse blokkból, a 2 MiB-os szakaszigazításból (mint a HIP), a mért könyvelési és visszaadási egységből és a foglalásszám-korlátból | a visszaadási egység és a foglalásszám-korlát Vulkan-specifikus (`vkAllocateMemory` darabonként, `maxMemoryAllocationCount`); a többi a HIP-pel azonos szabály |
| `ggml_backend_dev_internal_memory` proc: a backend saját eszközfoglalásai a ggml-pufferek névleges méretén felül (prealloc/staging pufferek, a sparse szakaszok darabkerekítése); a zónakönyvelés ezt a compute pufferrel együtt könyveli | a Vulkan-backend a köztes puffereit maga foglalja a gráf futásakor, nem a schedulerrel; a HIP-backend ilyen procot nem ad (változatlan) |
| `ggml_backend_dev_host_memory` proc: a backend host-foglalásainak összege; a llama-server megméri, hogy a cgroup látja-e őket (egy host-puffer foglalása előtt és után a `memory.current`), és ami nem látszik, azt a RAM-számításhoz adja, naplóban és az Ollamának | a host-látható Vulkan-memória a meghajtó foglalása (GTT), nem felhasználói lap; hogy a cgroup számolja-e, a meghajtótól függ, ezért mérve |
| a másolásos tartalék közös régi pufferből rétegenként halad: a használt sorok a hostra mennek, a közös puffer az új pufferek előtt felszabadul | nem API-kényszer: a tartalék csak ott fut, ahol nincs aréna (Vulkan hiányzó képességgel); a ROCm-on az arénás eszközök útja nem változik |
| `ggml_backend_dev_async_readback` proc (alapból ki, `GGML_VK_ASYNC_READBACK=1`) és `GGML_SCHED_TIMING` mérősor a schedulerben | a Vulkan szinkron olvasása külön parancspuffer + beküldés + várakozás; a NAS-mérés dönt róla |

## Eredmény (2. lépés; f8133efac, c639f4f84, bbd033c19)
| hiba | állapot | eltérés a ROCm-tól (az API miatt) |
|---|---|---|
| 1. másolásos növelés | javítva: szakaszonkénti sparse tartomány, kevés címtérnél átkötéses növelés (laptopon `GGML_VK_SPARSE_VA_MIB`-bel ellenőrizve) | a tartomány pufferenként rögzül, ezért átkötés (HIP: darabonkénti címfoglalás) |
| 2. tartalék csúcsa | javítva: a közös régi puffer az új pufferek előtt szabadul (használt sorok a hoston át) | nincs (csak a tartalékon fut) |
| 3. RAM | javítva: driver host-foglalás mérve, ami a cgroupnak nem látszik, hozzáadódik (llama RAM-őr, Ollama `PlanRAM`, park-válasz) | proc + mérés (a HIP pinned memóriáját a cgroup látja) |
| 4. split-generálás | **nyitva**: mérőeszköz (`GGML_SCHED_TIMING`) és kísérleti kapcsoló (`GGML_VK_ASYNC_READBACK`), NAS-mérés kell | – |
| 5. hosszú kontextus | javítva ok-feltevés alapján (2 MiB-os darabigazítás, mint a HIP-en), NAS-mérés igazolja | nincs (azonos szabály) |
| 6. betöltési margó | dokumentálva (OPEN-QUESTIONS 13) | – |
| 7. könyvelés | javítva: a backend saját eszközpufferei és a darabkerekítés könyvelve (`ggml_backend_dev_internal_memory`) | proc (a Vulkan-backend a köztes puffereit maga foglalja) |
