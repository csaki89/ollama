# Vulkan és ROCm: a dual-buffer, a zónák, a flex rétegek és a dinamikus KV egyezése

Felmérés a `multimodel` branchen (a45647ddb..6b19b5a28 után), módosítás előtt. Minta: a ROCm (HIP) út. Mérés: laptop, Intel Iris Xe (Vulkan, UMA/iGPU), qwen2.5-1.5B Q4_K_M, `--auto-zones --dual-buffer -fa on --load-mode none -c 16384`, `--fit-target` 5800 / 6000 / 6200 / 6500 MiB (a 2. és 3. zóna kikényszerítésére), egy ~11 000 tokenes, utána két rövid kérés. Szkript: `tools/dualbuf/vk-parity.sh`.

| elem | ROCm (minta) | Vulkan ma | eltérés |
|---|---|---|---|
| dual-buffer (CPU-s súlyok host-példánya, op offload) | `ROCm_Host` | `Vulkan_Host`, `dual-buffer: 46 weights also stored in Vulkan_Host (162.84 MiB)` | nincs |
| auto zónák (két fit, flex sáv) | `auto zones: n_ctx ... (zone 1); ... (zone 1 + 2)`, `flex band N whole layers` | ugyanezek a sorok (FIT=6000: 27 réteg az 1. zónában, 2 rétegű sáv) | nincs |
| flex rétegek mozgása a K/V-vel | `flex_move_part: layer L moved to CPU`, `flex_move: layer L K/V moved to ROCm_Host` | ugyanez `Vulkan_Host`-tal; a hosszú kérés leveszi, a rövid utáni zsugorítás visszahozza (`2 flex steps off` → `2 back`) | nincs |
| dinamikus KV helyben növelés/zsugorítás | VMM, `N/N layers in place (VMM)`, `grown/shrunk in place` | **darabos KV BDA-val (3. lépés, a sparse helyett)**: közönséges eszközpuffer-darabok, a shaderek darabtáblán át érik el őket; `28/28 layers in place`, `grown/shrunk in place`; a flex rétegek K/V-je másolással (mint ROCm-on) | nincs (eltérések lent) |
| **a VMM-darab mérete** | futásidőben: granularitás, igazítás és a meghajtó **mért** foglalási egysége (egy minimális foglalás előtti és utáni szabad memória, kétszer); egy naplósor mutatja az összetevőket | **fix 2 MiB** (`GGML_VK_SPARSE_CHUNK_KIB`), csak a sparse blokkra kerekítve; nincs mért foglalási egység, és a darabszám nem figyeli az eszköz foglalásszám-korlátját (`maxMemoryAllocationCount`, Windowson tipikusan 4096) – nagy kontextusnál a darabok száma túllépheti | **eltér** |
| a zónakeret a mért szabad memóriából | `hipMemGetInfo` (mért) + amdgpu sysfs (`mem_info_vram_used`) összevetés | `ggml_backend_dev_memory`: `VK_EXT_memory_budget`-tel mért (heap budget − usage); **a bővítmény nélkül a heap teljes mérete**, vagyis nem mérés, és a zónakód ezt nem tudja (a keret a teljes VRAM-ból indulna); a sysfs-összevetés amdgpu-n backendtől függetlenül megy | **eltér** (tartalék és naplósor hiányzik) |
| a felszabadított memória visszajön | VMM-darab foglalásának felszabadítása, mért | `vkFreeMemory`, a budget szerint mérve (+490/560 MiB az iGPU-n) | nincs |
| `zones_state` és a többi naplósor | `zones_state ... device free: booked, driver, sysfs` | ugyanez (a `driver` a Vulkan budgetje) | költségvetés nélkül a `driver` érték félrevezető (a heap mérete) |
| tartalékág: nincs helyben növelés | `no VMM support` → másolás | `no growable (chunked) KV buffers: the device has no bufferDeviceAddress / shaderInt64 ...; the KV grows by copy` | nincs |
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

# 3. lépés: a split-határ (GPU–CPU váltás) a két backendben

## Mi történik egy váltáskor (kódból)
A scheduler (`ggml_backend_sched_compute_splits`) splitenként: bemenetek másolása (`ggml_backend_sched_copy_input`), majd `graph_compute_async`. A CPU-split bemenete a GPU-split kimenete; a GPU-split bemenete a CPU-é.

**HIP (ggml-cuda):**
- *gráf:* a kernelek (vagy egy CUDA/HIP-gráf) a backend streamjére kerülnek, aszinkron, egy felhasználói módú parancssorba (doorbell), rendszerhívás nélkül;
- *GPU→CPU:* `ggml_backend_synchronize(GPU)` = `hipStreamSynchronize` (az alapértelmezett ütemezés pörög/enged, alvás nélkül), utána `ggml_backend_tensor_copy` → `buffer_get_tensor` = `hipMemcpyAsync` (DMA) + `hipStreamSynchronize`;
- *CPU→GPU:* `cpy_tensor_async` csak GPU↔GPU, ezért szinkron `buffer_set_tensor` = `hipMemcpyAsync` + `hipStreamSynchronize` – a host-forrás a CPU compute puffere (`ROCm_Host`, rögzített);
- *várakozás:* minden várakozás pörgő stream-szinkron, a parancspufferek kezelése a HIP runtime-ban (nincs alkalmazásoldali költség).

**Vulkan (ggml-vulkan, a javítás előtt):**
- *gráf:* parancspuffer-rögzítés, és a gráf 80%-ánál egy beküldés az `almost_ready` kerítéssel, a végén egy második beküldés kerítés nélkül (`ggml_vk_compute_forward`);
- *GPU→CPU:* `ggml_backend_vk_synchronize`: (1) egy **üres** `vkQueueSubmit` csak a kerítésért, (2) **alvó** `vkWaitForFences` az `almost_ready` kerítésre (a kernel felébresztésének késleltetése), (3) pörgés a fő kerítésen, (4) `ggml_vk_graph_cleanup` (parancspool-takarítás). Utána a szinkron `buffer_get_tensor` = `ggml_vk_buffer_read`: **új ideiglenes parancspuffer**, `vkQueueSubmit` az eszköz kerítésével, **alvó** `vkWaitForFences`, pool-takarítás;
- *CPU→GPU:* a scheduler előbb `ggml_backend_synchronize(Vulkan)`-t hív (üresjáratban is: átviteli sor beküldése és teljes `graph_cleanup`), majd `cpy_tensor_async` a rögzített host-forrásból a compute parancspufferbe; AMD RDNA-n (`async_use_transfer_queue`) az átviteli sorra, ami a következő gráfnál **külön beküldés** + idővonal-szemafor-várakozás.

**A különbség:** a HIP váltásonként egy aszinkron indítást és egy pörgő szinkront fizet (a másolás ugyanazon a streamen). A Vulkan GPU→CPU váltásonként 4 beküldést (2 a gráf, 1 üres a kerítésért, 1 a visszaolvasás) és 2 alvó várakozást, CPU→GPU váltásonként egy üresjárati szinkront takarítással, RDNA-n még egy átviteli beküldést és szemafort.

## Amit a Vulkan API másképp követel
Egy `vkQueueSubmit` rendszerhívás (a HIP felhasználói módú sorába írás nem az), és a gazda csak kerítésen vagy idővonal-szemaforon át tudja meg, hogy a GPU kész. Ezért a HIP-elv legközelebbi megfelelője: **váltásonként egy beküldés és egy pörgő kerítésvárás**, a másolás ugyanabban a parancspufferben, mint a gráf. Ennél kevesebb a Vulkan API-val nem lehetséges (a parancspuffer csak beküldéssel indul).

## A javítás (alapból be, kártyafüggetlen; `GGML_VK_LOW_LATENCY_SYNC=0` a régi viselkedés, tesztkapcsoló)
1. Kis gráfnál (az előző gráf 1 GFLOP alatt, `flops_per_submit == 0` – ugyanaz a feltétel, ami alapján a ggml-vulkan sem darabolja a beküldést) nincs `almost_ready` beküldés és alvó várakozás: a várakozás az elejétől pörög, mint a HIP stream-szinkronja.
2. A kis gráf utolsó parancspuffere nyitva marad (`defer_last_submit`): a scheduler a visszaolvasást mögé rögzíti (`ggml_backend_dev_async_readback`, alapból be), és a szinkronizálás az egészet **egy** beküldéssel küldi, a **kerítéssel együtt** (nincs üres beküldés).
3. Üresjárati szinkronizálás (semmi nincs rögzítve vagy beküldve az előző óta) azonnal visszatér, mint egy üres stream szinkronja.
4. Kis gráfok alatt a CPU→GPU bemásolás a compute parancspufferbe kerül (nem az átviteli sorra), mint a HIP másolása a compute streamen.
Tartalék: ha az eszköz nem támogatja az aszinkron végrehajtást (`GGML_VK_DISABLE_ASYNC` vagy az eszköz), nincs `get_tensor_async`, a visszaolvasás szinkron marad, napló: `ggml_backend_vk_init: ...: no async execution ...: a GPU/CPU split boundary reads its output synchronously`.

## Darabos KV BDA-val (3. lépés; a sparse-os út helyett)
A sparse-os K/V-út megszűnt (`GGML_VK_SPARSE*` nincs többé). Egy réteg K-ja és V-je (egy szakasz) **közönséges eszközpufferekből** (darabokból) áll; a KV-t olvasó és író shaderek a darabokat a **darabtáblán** át érik el: a tenzor kötésének helyén a szakasz táblája áll (fejléc: a kötés szakaszbeli kezdete, a darab bájtban és a fastdiv-értékei; utána minden darab BDA-címe), a címet **egyetlen** közös számítás adja (`kv_chunk.glsl`: `kvc_chunk` + `kvc_addr`, a `KVC_ADDR`/`KVC_AT` makrókon át), az elem típusának `buffer_reference`-én olvas/ír. Növeléskor új darab jön a végére (a meglévők helyben maradnak, nincs másolás, nincs dupla foglalás), a tábla új bejegyzést kap; zsugorításkor a felső darabok `vkFreeMemory`-val felszabadulnak. Nincs összesített címtér-korlát (nincs sparse címtér, nincs `maxBufferSize`-korlát a szakaszra).

**Az érintett shaderek** (mind egy `KV_CHUNKED` fordítással, `_kvc`/`_kvcd` néven; a gyári fordításuk SPIR-V-je bájtra azonos az eredetivel, laptopon glslc-vel összevetve):
- figyelés: `flash_attn.comp` (scalar, dot2, int8/MMQ), `flash_attn_cm1.comp` (a közvetlen `coopMatLoad` a darab címéről), a közös `flash_attn_dequant.glsl` és `flash_attn_mmq_funcs.glsl` (minden K/V-blokkolvasás `FA_BLK`-on, minden f16-vektorolvasás `FA_K4`/`FA_V4`-en át);
- KV-írás: `copy_to_quant.comp` (set_rows minden KV-típusra, és f32→kvantált másolás), a `rope_*.comp` (`rope_funcs.glsl`: a ROPE+VIEW+SET_ROWS fúzió írása és a K-shift helybeni rope-ja), `rms_norm.comp` (az RMS_NORM+VIEW+SET_ROWS és az RMS_NORM+MUL+ROPE(+SET_ROWS) fúzió írása);
- másolások és előmenetek: `copy.comp`, `contig_copy.comp` (az AMD-s f16 FA-előmenet és a K-shift másolásai), `copy_from_quant.comp` a `dequant_funcs.glsl`-lel (kvantált K olvasása a K-shifthez), `dequant_q8_0.comp` (a q8_0 FA-előmenet).
A shaderlogika nem változott: csak a K/V-hozzáférések kerültek makróba (gyári fordításban az eredeti indexelés).

**A host oldal:** a pufferműveletek (set/get/memset/clear/cpy, a backend aszinkron set/get-je) darabonként mennek; a KV-t érintő műveletek közül csak azok futnak Vulkanon darabos K/V-vel, amelyeknek van `KV_CHUNKED` fordítása (`supports_op`: set_rows, FA, helybeni rope, a K-shift másolásai); minden más a scheduleren át másik backendre kerül (a nézet a hoston át, darabonként másolódik).

**Eltérések a ROCm-os úttól, okkal:**

| eltérés | ok |
|---|---|
| a darab egy szakaszon belül a K/V-sor egész számú többszöröse (256 sor többszöröse), az eszköz darabja (granularitás, 2 MiB igazítás, mért könyvelési egység, visszaadási egység, foglalásszám-korlát) fölötti legkisebb ilyen, amelyet a foglalási egység 1/16-nál kisebb veszteséggel tölt ki; a tenzor elhelyezésekor dől el (`init_tensor`), addig az eszköz darabjával foglal | a HIP-ben a darabok egy összefüggő virtuális tartományban vannak, egy olvasás átnyúlhat a határukon; itt minden darab külön `VkBuffer`, egy olvasás (egy blokk, egy f16-vektor, egy coopmat-csempe) nem lóghat át a következőbe |
| a darabtábla a kötés helyén, kötés-eltolásonként egy tábla | a push constant-ok mérete (az FA-é már 128 bájt, a garantált minimum) nem bír több mezőt; a tábla egy tárolópuffer, BDA nélkül olvasható |
| a coopmat2 FA-nak nincs darabos fordítása: darabos K/V-vel a coopmat1 vagy a scalar FA fut | a coopmat2 shader a K/V-csempéket `coopMatLoadTensorNV`-vel tölti, tenzor-címzéssel egyetlen pufferből; darabonkénti betöltéshez a shader logikáját kellene átírni (nyitott kérdés) |
| a bf16 coopmat1 FA-nak nincs darabos fordítása (bf16 K/V-vel darabosan a scalar FA fut); az Intel Xe kétfázisú FA-ja (`fa_decode_ph1/2`) darabos K/V-vel nem választódik | ritka utak; a nyitott kérdések között |
| a szakasz legfeljebb 4 GiB (afölött az a réteg másolással nő) | a shaderek 32 bites szakaszon belüli eltolással címeznek (mint a gyári kötés is) |
| tartalék: `bufferDeviceAddress` vagy `shaderInt64` nélkül nincs darabos KV, rétegenként másolásos növelés, egy naplósor a hiányzó képességgel | a darabtábla 64 bites címeket tart |

**Laptop (Iris Xe, qwen2.5-1.5B, `-c 16384`, `--auto-zones --dual-buffer -fa on`, 4701 tokenes kérés + két rövid, `temperature 0`):** a kimenet bájtra azonos a dinamikus KV nélküli (fix KV) futással, a másolásos úttal (`LLAMA_KV_NO_VMM=1`), és darabos KV-val akkor is, ha egy szakasz 64 darabból áll (`GGML_VK_KV_CHUNK_KIB=128`, 256 soros darabok, a figyelés darabhatárokon át olvas); f16 és q8_0 K/V-vel is (q8_0: 272 bájtos sor, 512 soros = 139 264 bájtos darab). Növelés helyben (`KV 4096 -> 4864 cells ... 14 grown in place`), zsugorítás helyben (`--no-zones-flex`: `14 shrunk in place`, a szakaszok 2 490 368 → 2 097 152 bájt), flex-mozgás (`flex_move: layer 8 K/V moved to Vulkan_Host`), park/unpark 3 körben (a válaszok azonosak, a K/V a darabos szakaszokból a hostra és vissza). A felszabaduló memória mérése és a sebesség a gyárihoz képest a diszkrét kártyán mérendő (NAS-TESTS 15); UMA-n a mért szabad memória a rendszer zaja miatt erre alkalmatlan.

# 4. lépés: MoE split-generálás (qwen3-coder:30b) Vulkanon a CPU sebességén

**Hiba (NAS, RX 6700):** qwen3-coder:30b, auto zónák, dual-buffer: Vulkanon ~20 tok/s, csak CPU-n ~21, ROCm-on ugyanazon a kártyán ~35. A betöltés (CPU_REPACK, host-iker, GPU-s rétegdarabok) a két backenden azonos; a generálás Vulkanon tokenenként ~16–20 ms-mal lassabb.

## Összevetés generáláskor (1 tokenes köteg)
ROCm-oldal: a kódból (ggml-cuda és a scheduler) levezetve; Vulkan-oldal: a kódból és a laptopon mérve (`GGML_SCHED_DEBUG=2`, `GGML_SCHED_TIMING=32`, granite-3.1-1b-a400m Q4_K_M, a szakértők a CPU-n: `-ngl 99 -ot exps=CPU --dual-buffer -fa on`, ugyanaz a rétegszerkezet, mint a coder szakértő-override-os rétegeinél).

| kérdés | ROCm (HIP) | Vulkan | eltérés |
|---|---|---|---|
| op offload 1 tokenes kötegnél | `get_op_batch_size`: MUL_MAT `ne[1]`, MUL_MAT_ID/ROPE `ne[2]`, GET_ROWS 0, egyébként `nrows`; küszöb `GGML_OP_OFFLOAD_MIN_BATCH`, alap 32 | `ggml_vk_get_op_batch_size`: betűre ugyanez, ugyanaz a környezeti változó és alapérték | nincs: generáláskor egyik sem küldi a CPU-s súlyokat a GPU-ra (a MUL_MAT_ID sem) |
| hol fut a réteg (szakértők a CPU-n) | figyelem, router (gate_inp, softmax, argsort/top-k, get_rows, normalizálás) a GPU-n; a 3 MUL_MAT_ID + SWIGLU a CPU-n; a súlyozás és az összeadás a GPU-n | ugyanez (a `supports_op` mindkét backendben lefedi ezeket a műveleteket) | nincs |
| váltások tokenenként | rétegenként 2 (GPU→CPU, CPU→GPU), azaz a szakértő-override-os rétegek számának kétszerese (+ a bemenet és a kimenet) | ugyanannyi (laptop: 24 rétegből 50 split gráfonként) | nincs |
| a szakértő-súlyok forrása a CPU-n | a tenzor saját puffere: `CPU_REPACK` (repackelt példány); a nem repackelhető típus (pl. q6_K `ffn_down_exps`) a `ROCm_Host`-ban | `CPU_REPACK`, a q6_K a `Vulkan_Host`-ban (a napló: `buffer type overridden to CPU_REPACK` / `Vulkan_Host`); a host-ikert csak az op offload olvassa (köteg ≥ 32) | nincs; a `Vulkan_Host` a CPU számára gyorsítótárazott memóriát kér (`eHostCached`, a RADV-n van ilyen) |
| az aktivációk útja GPU→CPU | a CPU-split bemenetei: `ffn_norm` (reshape-nézet) és a top-k azonosítók (nézet); a scheduler nézet-bemenetet szinkron olvas: `synchronize` + bemenetenként `buffer_get_tensor` = `hipMemcpy` D2H a rögzített CPU-compute-pufferbe (pörgő szinkron) | ugyanez a hívássor; a `buffer_get_tensor` = `ggml_vk_buffer_read` | **költségben eltér** (lent) |
| CPU→GPU | `ffn_moe_down` (nem nézet): szinkron `hipMemcpy` H2D | a compute parancspufferbe rögzített másolás (2. lépés) | Vulkanon olcsóbb |

**Egy váltás költsége a javítás előtt.** GPU→CPU: (1) a gráf utolsó parancspufferének beküldése a kerítéssel, pörgő várás (2. lépés), (2) **bemenetenként** (2 db) egy új ideiglenes parancspuffer az **átviteli soron** (AMD-n külön DMA-motor), `vkQueueSubmit`, **alvó** `vkWaitForFences` (kernel-ébresztés), pool-takarítás. Laptopon (Iris Xe, Windows) egy ilyen olvasás átlag **~190–225 µs**, váltásonként kettő, a granite 24 rétegén tokenenként ~10 ms. A HIP ugyanezt két ~10–20 µs-os `hipMemcpy`-vel fizeti (felhasználói módú sor, pörgő szinkron). A coder 48 rétegén ~96 visszaolvasás tokenenként: ez a NAS-on mért 16–20 ms nagyságrendje. A végrehajtás helye tehát egyezik, **a különbség a váltásonkénti költség**.

## A javítás (csak ggml-vulkan; alapból be, `GGML_VK_LOW_LATENCY_SYNC=0` a régi út, tesztkapcsoló)
1. **Előreolvasás a split-határon** (`vk_readahead`): egy kis gráf (a 2. lépés `defer_last_submit` feltétele) szinkronja után az eszköz megjegyzi, mely tartományokat olvasták vissza szinkron (a gráf azonosítója: kontextus, gráfcím, csomópontszám, első/utolsó csomópont és címe). Amikor ugyanez a gráf legközelebb szinkronizál, a megjegyzett tartományok másolása a gráf nyitott parancspufferébe kerül (barrier, `copyBuffer` egy gazda-oldali (host-cached) pufferbe, HOST-barrier), **ugyanazzal az egy beküldéssel és kerítéssel**. A szinkron olvasás, ha a tartomány benne van és azóta semmilyen beküldés (és a gazda semmit nem írt az eszköz memóriájába) nem történt, csak `memcpy`. Érvényesség: minden `vkQueueSubmit` (a queue-handle-ben, egy helyen) és minden gazda-oldali írás az eszköz memóriájába növel egy eszközszintű számlálót; az előreolvasás csak a saját beküldésének számlálóértékénél érvényes. Ha a minta változik, az olvasás a rendes úton megy (és a mintát újratanulja). GPU→CPU váltásonként így **1 beküldés + 1 pörgő várás**, mint a HIP stream-szinkronja, a két külön kör nélkül.
2. **Kis szinkron másolások** (≤ 1 MiB; `buffer_get/set_tensor`, pl. a felhasználói bemenetek gráfonként): a compute soron, pörgő kerítésvárással (nem az átviteli soron alvó várással), mint egy HIP-másolás a compute streamen.
3. **Sűrűbb kerítés-lekérdezés kis gráf után:** a pörgő várás 1000 helyett 10 `pause`-onként kérdezi a kerítést (a 1000 `pause` a mai x86 magokon több tíz µs).
4. **Mérés:** `GGML_SCHED_TIMING=N` mellett 2 másodpercenként egy Vulkan-sor: `ggml_vulkan: sched timing: per second: G graph recordings g us, S synchronizes s us, R sync reads r us (H from the read-ahead), W sync writes w us`.

## Eredmény (laptop, Iris Xe; a kimenet bájtra azonos)
| futás | tg tok/s | CPU `input copies` / gráf | sync read |
|---|---|---|---|
| granite MoE, szakértők a CPU-n, régi út (`GGML_VK_LOW_LATENCY_SYNC=0`) | 23,9–25,3 | 24,5–29,3 ms | 193–225 µs, 0 előreolvasásból |
| ugyanez, új út | **31,8** | 15,9 ms | 0,6 µs, mind előreolvasásból |
| qwen2.5-1.5B sűrű, FFN a CPU-n (a bemenet nem nézet: aszinkron visszaolvasás, 2. lépés) | 13,3–13,7 régi / 12,6–13,8 új | – | nincs szinkron olvasás |
A laptopon a split lassabb a csak-CPU-snál (granite 45,6 tok/s), mert az Iris Xe-n a GPU-s figyelés rétegenként ~0,7 ms; ez a diszkrét kártyán nem áll fenn. Kimenet: granite split új = régi (389 karakter), qwen2.5 FFN-split új = régi (751 karakter), bájtra.

# 5. lépés: a maradék MoE-többlet és a host-memória (Vulkan)

**Hiba (NAS, RX 6700, az előreolvasásos build):** qwen3-coder:30b, auto zónák, dual-buffer: Vulkanon ~26 tok/s, ROCm-on ~35, csak CPU-n ~21,5 (tokenenként ~10 ms többlet); gpt-oss:20b Vulkanon ~30, ROCm-on ~40–44. Az OS szerint mindkét backend ~30 GB RAM-ot foglal; Vulkanon ebből ~15 GB (`Vulkan_Host`) a GPU-meghajtón át, GTT-ként, ezt a konténer cgroupja és az RSS nem látja.

## A host-pufferek memóriatípusa (kódból; a laptopon a naplósorral mérve)
Minden puffer, amelyet a CPU olvas, a `Vulkan_Host` puffertípusból (`ggml_vk_host_malloc`) vagy egy staging pufferből jön:

| puffer | ki olvassa | kért típus (sorrendben) | RADV, diszkrét kártya (NAS) | laptop (Iris Xe, Windows, naplóból) |
|---|---|---|---|---|
| dual-buffer eredeti formájú példány, a nem repackelhető súlyok (pl. q6_K szakértők), flex host-példány, park-példány, a CPU compute és kimeneti puffere | CPU (generálás), GPU (op offload másolás) | HOST_VISIBLE\|HOST_COHERENT\|HOST_CACHED, utána HOST_VISIBLE\|HOST_COHERENT | GTT, HOST_CACHED típus (snoopolt, gyorsítótárazott); a meghajtó foglalja, a cgroup nem látja | 2-es típus: DEVICE_LOCAL\|HOST_VISIBLE\|HOST_COHERENT\|HOST_CACHED (UMA) |
| `sync_staging` (visszaolvasás), előreolvasás staging | CPU | ugyanaz | ugyanaz | ugyanaz |

A típus tehát **gyorsítótárazott és koherens** mindkét eszközön; ROCm-on a `hipHostMalloc` folyamat-RAM-ot (névtelen leképezés, KFD userptr) rögzít, ez is gyorsítótárazott. A különbség a ROCm-tól nem a gyorsítótárazás, hanem (a) a lapok a GPU-meghajtó (TTM) leképezésén át jönnek, nem a folyamat névtelen memóriájából, ezért a cgroup és az RSS nem látja őket; (b) a TTM CPU-leképezése 4 KiB-os lapokkal megy (nem mérve, OPEN-QUESTIONS 27).

**Új naplósor** (betöltéskor, felhasználásonként és típusonként egyszer): `ggml_vulkan: <eszköz>: Vulkan_Host buffers (dual-buffer and flex copies, CPU compute and output buffers): memory type N (<jelzők>), heap M (<méret> MiB, system memory|device-local): cached for the CPU`; ugyanígy a `staging buffer (readbacks)` és a `read-ahead staging buffer`. Gyorsítótár nélküli típusnál WARN-szint: `NOT cached for the CPU (write-combined or uncached: slow CPU reads)`.

**Változás csak gyorsítótárazott típus hiányában:** akkor a `Vulkan_Host` közönséges folyamat-RAM (gyorsítótárazott a CPU-nak), a GPU a staging pufferen át kapja (nem rögzített host-mutatóként), naplósor: `Vulkan_Host buffers: plain process RAM (cached for the CPU); the GPU reads them through the staging buffer`. RADV-n, ANV-n, NVIDIA-n van gyorsítótárazott típus, ott semmi nem változik.

**Eltérés a ROCm-tól, indokkal: nem importált folyamat-RAM.** A ROCm-megfelelő a folyamat névtelen memóriája lenne `VK_EXT_external_memory_host`-tal importálva (gyorsítótárazott, a cgroup látja). A RADV-n ez amdgpu userptr-BO: a kernel CS-útja minden beküldésnél újra lekéri a userptr-BO-k lapjait (`amdgpu_ttm_tt_get_user_pages` a beküldés BO-listájának minden userptr-elemére), a RADV pedig a puffereszköz-cím (BDA) bekapcsolásakor minden BO-t minden beküldés listájába tesz (globális BO-lista). 15 GB-nál ez tokenenként ~100 beküldés mindegyikében ~4 millió lap bejárása volna. (Kódból levezetve, nem mérve.) A meghajtós GTT marad, a könyvelést a RAM-oldal kezeli (lent).

## A maradék többlet részei (laptop, granite-3.1-1b-a400m MoE, szakértők a CPU-n, 200 token, `GGML_SCHED_TIMING=32`)
A Vulkan-sor most részletesebb: `per second: R graph recordings r us, P replays p us, S synchronizes s us (B submissions b us, W GPU waits w us), ... sync reads ..., ... sync writes ...; ms per second: recording .., replay .., submission .., GPU wait .., sync reads .., sync writes ..`. Tokenenként = ms/s ÷ tok/s.

| rész (tokenenként, 25 GPU-split gráf, 50 váltás) | újrajátszás nélkül | újrajátszással | ROCm-megfelelő |
|---|---|---|---|
| gráf-rögzítés (CPU) | 25 × 76–79 µs = **~1,9 ms** | 25 × 23–25 µs = **~0,6 ms** (aláírás + beillesztés) | kernelindítások, a GPU már fut közben |
| beküldés (`vkQueueSubmit`) | 25 × 22–26 µs = ~0,6 ms | ugyanannyi | ajtócsengő (felhasználói módú sor) |
| GPU-rész (beküldéstől a kerítésig) | 25 × ~440 µs = ~11 ms (Iris Xe; a diszkrét kártyán sokkal kevesebb) | ugyanannyi | a stream szinkronja |
| visszaolvasás | ~49, mind előreolvasásból, 0,5 µs | ugyanaz | `hipMemcpy` D2H |
| szinkron bemenetírás | ~33 × 1,2–1,4 µs (UMA: közvetlen memcpy) | ugyanaz | `hipMemcpy` H2D |
| CPU-rész (szakértők) | ~8,4 ms gráfonként | ugyanaz | ugyanaz |

A laptopon a Vulkan-útból jövő, a ROCm-on nem létező rész a **gráf-rögzítés**: a HIP a kerneleket rögtön indítja és a GPU közben dolgozik, a Vulkan előbb a teljes parancspuffert rögzíti (a GPU addig áll). A beküldés és a GPU-várás a HIP-ben is megvan (más formában). A tok/s a laptopon alig mozdul (41,8–44,6 újrajátszással, 42,3–43,6 nélküle), mert itt a GPU-rész (Iris Xe, ~11 ms) és a közös memória-sávszélesség dominál; a diszkrét kártyán a rögzítés közvetlenül a tokenidőhöz adódik.

## A javítás: a változatlan split-gráfok parancspufferének újrajátszása (`vk_replay`, csak ggml-vulkan)
1. **Mikor:** kis gráf (a 2. lépés `defer_last_submit` feltétele: < 1 GFLOP, egy beküldés), alacsony késleltetésű mód, nincs perf logger és debug marker.
2. **Aláírás:** csomópontonként a tenzor címe, op, típus, jelzők, használatszám, op-paraméterek, `ne`/`nb`, puffer, adatcím, nézetforrás és -eltolás, ugyanez a forrásokra. Ebből következik minden, amit a rögzítés kiad (fúziók, barrierek, pipeline-ok, push constantok, kötések).
3. **Rögzítés:** ha egy gráf második alkalommal jön ugyanazzal az aláírással, a parancsai egy saját parancspufferbe kerülnek (saját pool, nem one-time-submit), saját leíró-készlet-tartománnyal (a többi gráf a `replay_desc_reserved` fölött indul). Az előtte rögzített másolások (CPU→GPU bemenet) a saját parancspufferükben maradnak, utána új parancspuffer jön (barrierrel) a visszaolvasásnak és az előreolvasásnak. Ha a rögzítés közben beküldés, pufferfelszabadítás vagy host-másolás történt, a bejegyzés nem marad meg (a parancsok egyszer lefutnak); két ilyen után a gráf a rendes úton marad.
4. **Újrajátszás:** azonos aláírásnál a bejegyzés parancspuffere a beküldésbe kerül (a sorrend: bemásolás → gráf → visszaolvasás, barrierekkel), rögzítés nincs. Egy parancspuffer szinkron-ablakonként egyszer kerül be.
5. **Érvénytelenítés:** bármely eszközpuffer felszabadítása (a kezelő egy új pufferé lehet) minden bejegyzést eldob (a parancspufferek a következő gráf-takarításkor szabadulnak); az aláírás minden más változást (KV-növekedés, új gráf) lefed.
6. **Kapcsoló:** `GGML_VK_GRAPH_REPLAY=0` (tesztkapcsoló); a `GGML_VK_LOW_LATENCY_SYNC=0` ezt is kikapcsolja.

**Eredmény (laptop):** granite split: rögzítés 82–87 ms/s → újrajátszás 26–28 ms/s; 10 futás újrajátszással, 6 nélküle és 4 kényszerített rögzítési hibával: futásonként (200, ill. 120 token) a kimenet mind bájtra azonos. qwen2.5-1.5B sűrű, FFN a CPU-n: rögzítés 46 ms/s → 10 ms/s, kimenet azonos (751 karakter); teljes GPU és auto zónák (nagy gráfok, nincs újrajátszás): azonos.

## Ami nem változott, és miért
- **Szinkron bemenetírások (OPEN-QUESTIONS 22):** a laptopon UMA miatt az eszközpuffer host-látható, az írás közvetlen memcpy (1,2 µs); a diszkrét kártya útja (staging + beküldés + várás) itt nem kényszeríthető ki (az UMA-ág a `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` előtt dönt). A RX 6700-on ReBAR-ral ugyanez a közvetlen út megy, nélküle bemenetenként egy beküldés. A halasztott írás (a következő beküldés elé) minden olvasási és beküldési utat érint; diszkrét kártya nélkül a helyessége nem ellenőrizhető, ezért nincs benne. A NAS-sor (`sync writes N us`) megmutatja, kell-e (OPEN-QUESTIONS 26).
- **A beküldés és a kerítés-várás váltásonként:** a Vulkan API része (egy `vkQueueSubmit` és egy kerítés); a HIP felhasználói módú sora és memóriajel-alapú szinkronja ennél olcsóbb. A NAS-mérés (`submissions`, `GPU waits` átlaga a ROCm `final wait`-jéhez képest) dönti el, mennyi marad (OPEN-QUESTIONS 25).

## RAM-könyvelés
- A `ggml_backend_dev_host_memory` (Vulkan: a nem eszközlokális meghajtós foglalások összege) és a cgroup-mérés (DECISIONS 75) megmaradt. **Új betöltési naplósor** (közös kód, csak ha a backend ad ilyen memóriát; ROCm-on nincs sor): `load_tensors: RAM: driver host memory X MiB (backend host buffers), not seen by the memory cgroup: where a cgroup limit bounds the RAM, the RAM guards and the RAM plan add +X MiB to the cgroup's usage`, illetve `..., in the memory cgroup's usage already: +0 MiB` / `no memory cgroup: in the system's used memory already: +0 MiB`.
- Ollama: a `host memory:` sorból kiolvasott értékről naplósor (`dualbuf: runner driver host memory not seen by the memory cgroup uncharged_mib=...`), és javítva a dupla számolás: ha a `ram_limit` egy rendszer-olvasást ír felül (nincs cgroup-korlát), a használat a rendszeré, abban a GTT már benne van; eddig ilyenkor is hozzáadódott. Most csak cgroup-ból olvasott használathoz adódik (`RAM.UsedFrom`).

# 6. lépés: a GPU órajele (compute-teljesítményállapot)
**Mérés (NAS, RX 6700, image 679b2aa):** azonos elrendezés mellett ROCm 38,4–39,4, Vulkan 25,5, csak CPU ~24,6 tok/s (coder); generálás közben a GPU órajele ROCm alatt 2670 MHz, Vulkan alatt 500–630 MHz. Egy mellette futó ROCm-folyamattal a Vulkan is 2670 MHz-en fut: 38,99 tok/s. A különbség tehát nem a Vulkan-útban volt (az 2–3 ms tokenenként), hanem a meghajtó energiaprofiljában: az amdgpu aktív KFD-sor mellett vált compute profilra.

| | ROCm | Vulkan előtte | Vulkan most |
|---|---|---|---|
| compute profil | a HIP-runtime KFD-sorai váltják ki | nincs (a Vulkan-munka amdgpu CS) | egy üres KFD compute-sor, amíg a backend él (DECISIONS 95) |
| feltétel | `/dev/kfd` | – | `/dev/kfd` a konténerben, AMD-eszköz KFD-csomóponttal |
| KFD nélkül | nem fut | – | a régi viselkedés + figyelmeztető sor (DECISIONS 96) |

Eltérés a ROCm-tól: a ROCm a sorait munkára használja, itt a sor üres, csak a compute-állapotot tartja; a sor pufferei rendszermemóriában vannak (néhány MiB, a context save area méretétől függően).
