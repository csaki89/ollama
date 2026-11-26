# Többmodelles működés: felmérés a kódról

Alap: `MULTIMODEL-DESIGN.md` és a DECISIONS 50–69. pontja; a kód a `multimodel` branchen. Ez a felmérés kódmódosítás nélkül készült: minden sor azt írja le, mi van a kódban és mi nincs.

Jelölések:
- **kész:** a megállapodás szerint működik;
- **eltér:** van kód, de valamiben eltér a megállapodástól;
- **hiányzik:** nincs hozzá kód.

A „tesztelve” oszlopban az egységteszt a `go test` (a CI-ben `-race`-szel), a laptop az Ollamán át futó végponttól-végpontig próba (`test-results.md`, Vulkan iGPU + CPU). A NAS-os többmodelles futások (NAS-TESTS 1–11) eredménye a repóban nincs rögzítve.

## Összesítő

| rész | kód (hol) | megfelel? | kártya-, gép- vagy telepítésspecifikus rész a kódban | tesztelve | munka | állapot |
|---|---|---|---|---|---|---|
| dualbuf.yaml beolvasás, ellenőrzés | `multimodel/config.go` (`Parse`, `validate`), `multimodel/store.go` (mtime/méret szerinti újraolvasás, utolsó érvényes) | igen: szigorú olvasás (`KnownFields`), hibás fájlnál az utolsó érvényes + `ERROR` sor, a futó modell a betöltéskori pillanatképpel fut (`dualbufConfigFor`) | nincs (út: `OLLAMA_DUALBUF_CONFIG`, alapból `<OLLAMA_MODELS>/dualbuf.yaml`) | egységteszt (`TestParse*`, `TestStoreLastValid`, `TestExampleFileParses`), laptop | nincs | kész |
| `devices` | `multimodel/devices.go` (`MatchDevices`) | igen: PCI-cím vagy névrész, a backendnek egyeznie kell, két találat vagy egy kártya két bejegyzésre hiba, a hiányzó eszközt a modellek átugorják | nincs (a backendnevek – rocm, vulkan, cuda, metal – általánosak) | egységteszt (`TestMatchDevices`: PCI, név, két kártya, ütközés); laptop csak név szerint, egy eszközzel | nincs; a PCI-s és a két GPU-s ág eszközön nincs kipróbálva | kész |
| modellenkénti helyek (`models.<név>.devices`), minták | `multimodel/resolve.go` (`SectionFor`, `Resolve`) | igen: pontos `név:tag`, különben a leghosszabb illeszkedő minta; tag nélkül `:latest` | nincs | egységteszt (`TestSectionFor`), laptop | nincs | kész |
| precedencia | `multimodel/resolve.go`, `server/routes.go` (két `FromMap`), `server/sched_dualbuf.go` (`dualbufPrepare`) | igen: env < `defaults` < Modelfile < a modell szakasza < kérés, mezőnként a forrással; naplósor `dualbuf effective settings` | nincs | egységteszt (`TestResolvePrecedence`, `TestDualbufModelOptionsPrecedence`, `TestResolveNoConfig`), laptop | nincs | kész |
| `on_busy`, `wait_timeout` | `multimodel/coord.go` (`Acquire`) | igen: `wait` csak az első eszközre vár `wait_timeout`-ig, `next` azonnal továbblép (naplósor, hogy a `wait_timeout`-ot figyelmen kívül hagyja); beépített alap 60 s (DECISIONS 53) | nincs | egységteszt (`TestCoordWaitThenGet`, `TestCoordWaitTimeoutNext`, `TestCoordNextNoWait`), laptop | nincs | kész |
| `max_running` | `multimodel/coord.go`, `config.go` (`MaxRunning`), `sched_dualbuf.go` (`dualbufStarted`) | igen: kérésenkénti hely a kérés végéig, a betöltéssel együtt; a GPU+CPU-n futó példány a CPU-n is foglal (mindig GPU → CPU sorrendben, így körkörös várakozás nincs) | nincs | egységteszt (`TestCoordFreeAndNoLimit`, `TestCoordAlsoCPU`, `TestCoordReleaseOnDone`), laptop | nincs | kész |
| `reserve_mib` | **nincs** sem config-mezőként, sem a tervben; a RAM-tartalék beégetett: `max(1 GiB, a keret 5%-a)` (`multimodel/ram.go` `RAMReserve`, a llama-server RAM-védelmével azonos); VRAM-oldalon a fit margója (Ollama `--fit-target`) érvényes | – | – (a beégetett tartalék nem kártyafüggő, de nem állítható) | – | kicsi, de előbb el kell dönteni, mit jelent: RAM-tartalék a kerethez, eszközönkénti VRAM-tartalék a fitnek, vagy mindkettő. Utána egy mező a `Device`-ban és/vagy a gyökérben, átadás a `PlanRAM`-nak, illetve a llama-server `--fit-target`-jének, egy validálás és egy egységteszt | hiányzik |
| ütemező: sorba állítás | `multimodel/coord.go` (`addLocked`, `dispatchLocked`, `takeLocked`) | igen: ha nincs szabad hely, a kérés sorba áll, és a listája elsőként felszabaduló eszközén indul, érkezési sorrendben. Egy korábbi várakozó által kért eszközt egy későbbi nem vehet el. Hibát nem dob, csak a kérés context-je szakítja meg | nincs | egységteszt (`TestCoordQueueArrivalOrder`, `TestCoordWaiterKeepsItsTurn`, `TestCoordCancel`, `-race`), laptop (`busy.sh`) | nincs | kész |
| ütemező: eszközhöz rendelés | `server/sched_dualbuf.go` (`dualbufPlacementGPUs`, `schedulerModelKey`), `server/sched.go` (`load`) | igen: csak a kijelölt eszköz kerül a llama-server elé (`main_gpu 0`), a fit a GPU és a CPU között oszt; eszközönként külön példány (`<út>\x00@<eszköz>`). Ha a kijelölt eszközt a felderítés nem látja, az Ollama helyez (naplósorral). Egy modell több GPU közötti osztása nincs (nem is volt megállapodás) | nincs | laptop (egy iGPU + CPU); diszkrét GPU-n és két GPU-val nincs rögzített eredmény | NAS-os igazolás | kész |
| ütemező: demote a lista következő szintjére | `multimodel/coord.go` (`Devices[1:]`), `sched_dualbuf.go` (`dualbufPrepare`: CPU-ra `num_gpu 0`; ha a modell GPU-s példánya parkol, a CPU-s kérés azon fut, újratöltés nélkül) | igen: az első eszköz foglalt → a lista következő **szabad** eszköze (várakozás csak az elsőre); a CPU-s példány repackel; RAM-hiánynál elsőként megy (`KindCPUFallback`) | nincs | egységteszt (`TestCoordNextNoWait`, `TestDualbufPrepareAndQueue`), laptop | nincs | kész |
| ütemező: promote a lista első szabad eszközére | nincs külön kód: minden kérés a lista elejéről indul (`Acquire`), így a következő kérés a felszabadult első eszközre kerül | igen, a „következő kérés újra a lista elejéről” megállapodás szerint. A futó (tartalékeszközön lévő) kérés nem költözik át; a CPU-s példány a keep_alive-jáig betöltve marad | nincs | egységteszt (a sor- és szabad-hely tesztek), laptop | nincs | kész |
| park/unpark | llama-server: `--park`, `POST /park`, `/unpark`, `/park/drop` (patch 003); Ollama: `sched_dualbuf.go` (`dualbufVictim`, `dualbufTryPark`, `dualbufUnparkFor`, `dualbufBeforeUnload`/`AfterLoad`), `llm/llama_server_dualbuf.go` | nagyrészt. Folyamaton belüli csere `keep_in_ram: pinned`-del; folyamatszintű tartalék (KV a tmpfs-be, a modellfájl `mlock`-kal), ha a pinned példányok nem fértek a RAM-keretbe. A parkolt példány nem számít az `OLLAMA_MAX_LOADED_MODELS`-be. **Eltérések:** (1) a `dualbufVictim` tétlen példány híján egy foglaltat is választ (az Ollama megvárja, amíg felszabadul), a megállapodás tétlen áldozatot mond; (2) a `keep_alive: -1`-es modellt ez az út nem szűri: `swap.enabled: false` mellett az eszköz-helyhiány teljesen kirakhatja | nincs | egységteszt csak az áldozatválasztásra (`TestDualbufVictimSameDevice`); laptop: 8 csere, mind folyamaton belül, azonos kimenet, a KV megmarad. Diszkrét GPU-n (VRAM-maradék, 10 GB-os csereidő) nincs rögzített eredmény (OPEN-QUESTIONS 1, 4) | kicsi: a tétlenségi és a `keep_alive: -1` szűrés az eszközáldozatnál. Közepes: egységtesztek a park/unpark ütemezési útjára (ma csak a végponttól-végpontig próba fedi) | eltér |
| LRU-kiürítés a RAM-kereten belül, `keep_alive: -1` soha | `multimodel/evict.go` (`EvictionOrder`, `PlanEvictions`), `multimodel/ram.go` (cgroup v1/v2, `ram_limit`, a driver cgroup által nem látott host-memóriája), `sched_dualbuf.go` (`dualbufRoom`) | nagyrészt. Sorrend: a CPU-tartalék példányok és a demote-olt másolatok, a parkolók, végül a tétlenek, mindegyiken belül LRU. A `keep_alive: -1` kimarad (csak a CPU-tartalék példánya mehet). **Eltérések:** (1) csak új betöltés előtt fut; ha a betöltött modellek RAM-ja futás közben nő (KV, demote), nem ürít; (2) az első betöltés igénybecslése durva: a fájlméret mínusz az átadott **összes** GPU szabad memóriája (nem csak a kijelölt eszközé), plusz 1 GiB, dual-bufferrel kétszerese. Később a példány mért értéke számít; (3) a tartalék nem állítható (lásd `reserve_mib`) | nincs (a cgroup-utak általánosak; az `OLLAMA_DUALBUF_RAM_FREE_MIB` csak tesztsegéd) | egységteszt (`TestEvictionOrderAndPlan`, `TestCgroupRAMv2`, `TestCgroupRAMv1AndNone`), laptop (`ram_limit`, valódi RAM-hiánynál a parkolt modell LRU-kirakása); a `dualbufRoom` maga nincs egységtesztelve | kicsi–közepes: a becslés a kijelölt eszközre, egy futás közbeni ellenőrzés (pl. a `zones_state`/host-memória sorainál), egységteszt a `dualbufRoom`-ra | eltér |
| NVMe-s KV-mentés | `multimodel/kvfiles.go` (kísérő JSON, `CopyKV`, `Compatible`), `sched_dualbuf.go` (`dualbufBeforeUnload`: lejárat, LRU-kirakás és leállítás előtt; `dualbufAfterLoad`: visszatöltés), `llm` `SaveSlots`/`RestoreSlots` | igen: `kv_save` mellett teljes kirakás előtt minden nem üres slot a RAM-könyvtárba, onnan a `swap.kv_dir`-be (alap `<models>/dualbuf-kv`). Visszatöltés csak azonos digestnél és KV-típusnál; a fájl a visszatöltés után törlődik (egyszeri). Megjegyzés: a fájlnév az eszközt is tartalmazza, ezért egy GPU-n mentett KV nem töltődik vissza a CPU-s példányba; leállításkor a mentés legfeljebb 2 percig tarthat, a compose-ban nincs ehhez igazított `stop_grace_period` (a Docker alapból 10 s után leállítja) | nincs | egységteszt (`TestKVFiles`), laptop (`slot-restore.sh`, `slot-restore-edge.sh`, Ollamán át); a NAS-on nincs rögzített eredmény | kicsi: `stop_grace_period` a compose-ban (config), és döntés arról, hogy a KV átvihető legyen-e eszközök között | kész |

## Kártya-, gép- és telepítésspecifikus részek

- **A Go-kódban** (`multimodel/`, `server/sched_dualbuf.go`, `llm/llama_server_dualbuf.go`) nincs ilyen. A beégetett értékek általánosak:
  - a RAM-tartalék;
  - az időkorlátok (1, 2 és 5 perc);
  - a `/dev/shm` mint RAM-könyvtár, a temp könyvtár tartalékkal.
- **A konfigurációban és a buildben** vannak telepítésfüggő értékek. Ezek a megállapodás szerint ott a helyükön vannak, csak a teljesség kedvéért:
  - a `dualbuf-image` workflow környezeti változói: `GPU_ARCH: gfx1030`, `CPU_VARIANTS: haswell`;
  - a compose-ban a `HSA_OVERRIDE_GFX_VERSION`, a render node-ok, a port, a modellmappa és az erőforráskorlát. Az utóbbiak most `${...}` változók.

## Javaslat

Rendbe tenni, nem újraírni.
- A részek megvannak, a megállapodás szerint működnek, és egységtesztekkel, illetve a laptopos próbával fedettek; a hiány és az eltérések kicsik és jól körülhatárolhatók.
- Egy újraírás a meglévő, `-race`-szel tesztelt koordinátort és a laptopon igazolt park-utat dobná el.

Sorrend:
1. `reserve_mib`: a jelentés eldöntése, majd a mező bevezetése.
2. Az eszközáldozat tétlenségi és `keep_alive: -1` szűrése.
3. A RAM-igény becslése a kijelölt eszközzel.
4. Egységtesztek a `dualbufRoom`-ra és a park/unpark ütemezési útjára.
5. A futás közbeni RAM-ellenőrzés.
6. A `stop_grace_period` a compose-ban.

A NAS-os igazolás (diszkrét GPU, két eszköz, PCI-s párosítás, csereidő, NVMe-s KV) a NAS-tesztek dolga.
