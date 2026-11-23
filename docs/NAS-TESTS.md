# NAS-tesztek: többmodelles működés (multimodel branch)

Sorrendben futtasd; minden pontnál: mit, hogyan, mit várunk. Az image: `ghcr.io/csaki89/ollama-dualbuf:multimodel` (a fix címkét a `dualbuf-image` workflow „Compute tags” lépése írja ki). A compose: `docker-compose.multimodel.yaml` (port **30071**). Mérés közben a többi Ollama-példány ne tartson modellt a GPU-n (`keep_alive: 0`), mert közös a VRAM és a modellmappa.

Rövidítések: `H=http://<nas>:30071`, `L='docker logs ollama-multimodel 2>&1'`.

## 0. Előkészítés (5 perc)
1. A render node-ok: `ls -l /dev/dri/by-path/` → melyik `renderD12x` a RX 550 és melyik az X080/RX 6700; a compose `devices:` sorait ehhez igazítsd (ha egy nem létező node marad benne, a konténer nem indul).
2. Pull és indítás (TrueNAS → Install via YAML a `docker-compose.multimodel.yaml` tartalmával), majd `docker exec ollama-multimodel ollama --version` → `0.40.16`.
3. Eszközök: `eval $L | grep "inference compute"` → várt: egy `library=ROCm` sor (X080/RX 6700, `pci_id=0000:..`) és egy `library=Vulkan` sor a RX 550-re (`description` benne „RX 550”/„POLARIS12”). A RX 550 **nem** jelenhet meg ROCm-ként (a ROCm 7 nem támogatja a gfx803-at; az Ollama ellenőrzése kiszűri). Ha a nagy kártya Vulkanon is megjelenne, a duplikátum-szűrés a ROCm-ot tartja meg (várt).
4. Sparse binding a RX 550-en (a Vulkanos helyben növelés jövőbeli feltétele, ma csak információ): `docker exec ollama-multimodel sh -c 'apt list --installed 2>/dev/null | grep mesa-vulkan; ls /usr/share/vulkan/icd.d/'` → `radeon_icd.x86_64.json`. A `vulkaninfo` nincs az image-ben; a gazdán vagy egy `ubuntu:24.04` konténerben `apt-get install -y vulkan-tools mesa-vulkan-drivers` után: `vulkaninfo 2>/dev/null | grep -E "deviceName|sparseBinding|sparseResidencyBuffer|SPARSE_BINDING"` → várt (Mesa-forrásból nem igazolt, `MULTIMODEL-RESEARCH.md` 9.): `sparseBinding = true`, `sparseResidencyBuffer = true` a POLARIS12-n. Ha `false`: a Vulkanos dinamikus KV marad a másolásos úton (ez ma is így van).
5. Nincs rendszerszintű llama.cpp config: `docker exec ollama-multimodel sh -c 'ls /etc/llama.cpp ~/.config/llama.cpp 2>&1'` → mindkettő „No such file” (különben minden llama-server betöltené).

## 1. Config nélkül változatlan (15 perc)
A `/mnt/nvme/ollama/models/dualbuf.yaml` még **ne** létezzen.
- `curl -s $H/api/generate -d '{"model":"qwen3-coder:30b","prompt":"Write a haiku.","stream":false,"options":{"temperature":0,"seed":1,"num_predict":32}}' | jq -r .response`
- `eval $L | grep 'msg="starting llama-server"' | tail -1` → a kapcsolók (a `--port` kivételével) **azonosak** a `dual-buffer-0.40.16` példány (30069) ugyanerre a kérésre adott sorával; a naplóban nincs `dualbuf` sor; a pufferek (`ROCm0 model buffer`, `CPU_REPACK`, `ROCm_Host`, `zones:`) ugyanazok; a kimenet azonos.
- Ugyanez `qwen3-vl:8b`-vel és `qwen3-embedding:0.6b`-vel (`/api/embed`, 1024 dim).

## 2. A config beolvasása, erősorrend, eszközazonosítás (10 perc)
1. Másold a `docs/dualbuf.example.yaml`-t a `/mnt/nvme/ollama/models/dualbuf.yaml`-ba; a `devices` `match` mezőibe a 0.3 pont PCI-címét (X080/RX 6700) és a RX 550 nevét vagy PCI-címét írd.
2. Rakd ki a modelleket (`keep_alive: 0`), majd egy kérés a `qwen3-coder:30b`-re.
3. Várt naplósorok: `dualbuf config loaded ... devices=3 models=3 swap=true`; `dualbuf devices matched="cpu=CPU, rx550=Vulkan Vulkan0 ... 0000:..., x080=ROCm ROCm0 ... 0000:..."` (nincs `missing`, nincs `dualbuf device match` hiba); `dualbuf effective settings model=qwen3-coder:30b section=qwen3-coder:30b placement=x080 settings="devices=[x080,cpu](model:qwen3-coder:30b) on_busy=wait(model:...) wait_timeout=2m0s(model:...) zones=auto(model:...) dual_buffer=true(defaults) dynamic_kv=true(defaults) kv_type=f16(defaults) max_context=131072(model:...) keep_alive=30m0s(defaults) ..."`; a llama-server sorában `-c 131072 ... --split-mode none --main-gpu 0`.
4. Hibás config: írj a fájlba egy elírt kulcsot (`dual_bufer: true`), rakd ki és töltsd be újra a modellt → `level=ERROR msg="DUALBUF CONFIG INVALID, the model loads with the last valid config" ... error="... field dual_bufer not found ..."`, a modell az előző beállítással tölt. Javítsd vissza.

## 3. Elhelyezés: X080 (ROCm) és RX 550 (Vulkan), a Vulkanos dinamikus KV (10 perc)
- `curl -s $H/api/embed -d '{"model":"qwen3-embedding:0.6b","input":"hello world"}' | jq '.embeddings[0]|length'` → `1024`.
- Napló: `dualbuf: model placed on its device model=qwen3-embedding:0.6b device=rx550 ... library=Vulkan`; a llama-server sorában `--split-mode none --main-gpu 0`, a környezetben `GGML_VK_VISIBLE_DEVICES` (debug szinten látszik); `Vulkan0 model buffer size`.
- A Vulkanos dinamikus KV: a 4096-os `max_context` mellett a KV-t egy ~3000 tokenes inputtal érdemes megnövelni → `set_dynamic: ... by copy`, `relayout: KV ... -> ... cells (... new by copy)`; hiba nélkül. Ha a Vulkan-backend a RX 550-en nem működne (meghajtó), a napló `ggml_vulkan` hibát ad: ekkor a `devices: [rx550, cpu]` lista következő eleme, a CPU veszi át.
- `docker exec ollama-multimodel ollama ps` → a coder `100% GPU`-hoz közeli osztással, az embedding a RX 550-en.

## 4. Futási korlát, on_busy, sor (10 perc)
A laptopos `tools/dualbuf/multimodel/busy.sh` mintájára (a `localhost:11500` helyett `$H`, modellek: A = `qwen3-coder:30b`, 300 token; B = `qwen3-vl:8b` (`on_busy: next`), 150 token; C = `qwen3-coder:30b` 20 token; D = embedding):
- Várt: A → `device=x080 reason=free`; B → `device=cpu reason=next waited=0s`, és **új** llama-server `-ngl 0`-val (a qwen3-vl CPU-s példánya, „demote”, a betöltés a repack miatt hosszabb); C → `wait_timeout` (120 s) leteltéig vár az x080-ra; ha közben az A befejezi: `reason=waited`, ha nem: CPU, vagy sor (`no device free, the request is queued`, majd `reason=queued`).
- A megosztott modell CPU-helye: a coder MoE-szakértői a CPU-n vannak, ezért generálás közben a CPU-helyet is foglalja (`max_running: 1`): amíg az A fut, a B CPU-s kérése **vár** (`reason=queued`), és az A vége után indul. Ha ez nem kívánt, a `cpu: { max_running: 2 }` oldja.
- A következő kérés újra a lista elejéről indul: B után egy újabb `qwen3-vl:8b` kérés, amikor az x080 szabad → `device=x080`.

## 5. A CPU-s példány RAM-ban marad, és elsőként szabadul (5 perc)
- A 4. után: `ollama ps` → a `qwen3-vl:8b` kétszer (x080 és CPU, `100% CPU`).
- `docker stats --no-stream ollama-multimodel` → a memória a CPU-s példány méretével nőtt.
- RAM-hiánynál (7. pont) az első kirakott: `dualbuf: unloading for RAM ... kind=cpu-fallback`.

## 6. Csere (park) az X080-on, 10 GB körüli GPU-résszel — a fő mérés (20 perc)
Config: `swap: { enabled: true, keep_in_ram: pinned, save_kv: true }`. A két modell ugyanarra a kártyára: `qwen3-coder:30b` (a GPU-része ~9–10 GB) és `qwen3-vl:8b`.
1. Coder, egy ~2000 tokenes prompttal (tartsd meg a promptot), 32 token. Jegyezd fel: `load_duration`.
2. `qwen3-vl:8b` kérés. Várt: `dualbuf: making room on the device device=x080 ... victim=<coder> idle=true park=true`, `KV saved to RAM ... tokens=[~2000] bytes=~200MB duration=<1s`, `model parked`, majd `model file pinned in RAM ... bytes=~18.5e9` (háttérben). A VRAM felszabadul: `rocm-smi --showmeminfo vram` (vagy a sysfs `mem_info_vram_used`) a coder kirakása után a coder GPU-részével csökken, mielőtt a vl betölt.
3. Ugyanaz a coder-prompt újra. Várt: `making room ... victim=<qwen3-vl> park=true`, `KV restored ... from=ram tokens=[~2000] duration=<0.5s`, a llama-server `prompt eval time = ... / 1 tokens`, a válasz bájtra azonos az 1. lépésével.
4. **Mérendő a csereidő:** a 3. lépés `load_duration`-je. A folyamatszintű csere becslése: 6–12 s (llama-server indítás ~1–2 s, a két fit, a page cache-ből olvasás a zárolt fájlból ~2–5 GB/s, a ~8 GB-os CPU-rész repackje, a GPU-rész feltöltése PCIe 3.0 x16-on ~1 s / 10 GB, warmup). A kért cél ~1 s / 10 GB, ez folyamatszintű cserével **nem várható** (`OPEN-QUESTIONS.md` 1.). Ugyanezt mérd `keep_in_ram: none`-nal is (a fájl a page cache-ben marad, de kiszorulhat) és a 30069-es példány hideg betöltésével: így látszik, mennyit ad a zárolás és a KV-mentés.
5. A zárolás: `cat /sys/fs/cgroup/memory.stat | grep unevictable` a konténer cgroupjában (vagy `docker exec ... cat /sys/fs/cgroup/memory.stat`) → a parkoló modell(ek) fájlméretével nő; a park megszűnésekor (visszatöltés) csökken.

## 7. RAM-keret, LRU, keep_alive -1 (15 perc)
- `ram_limit: 40GiB` a configba (a 100 GB-os konténerkorlát alatt), rakd ki a modelleket, majd töltsd be sorban: coder (park), vl, embedding (keep_alive -1), coder újra, egy negyedik modell (pl. qwen2.5 1.5B).
- Várt: `dualbuf: RAM is short for the load, freeing ... limit="40.0 GiB" used=... source=ram_limit`, a sorrend: `kind=cpu-fallback` → `parked model dropped from RAM` (a legrégebben használt parkoló) → `unloading for RAM ... kind=idle`; az embedding (`keep_alive: -1`) soha nincs a kirakottak között (`ollama ps`: `Forever`).
- `ram_limit` nélkül a forrás `source=cgroup`, `limit="100.0 GiB"`.

## 8. KV-mentés az NVMe-re (10 perc)
Config: a coder szakaszába `kv_save: true`.
- Egy ~30 000 tokenes beszélgetés a coderrel, majd `keep_alive: 0`. Várt: `KV saved to RAM ... tokens=[~30000] bytes=~3e9` (a qwen3-coder:30b KV-ja f16-tal ~96 KiB/token: 48 réteg × 4 KV-fej × 128 × 2 × 2 bájt), `KV saved to the disk dir=/models/dualbuf-kv duration=` (NVMe-n ~1–3 s). `ls -la /mnt/nvme/ollama/models/dualbuf-kv` → `kv-<digest>-x080-f16-slot0.bin` és `.json`.
- Újratöltés ugyanazzal a beszélgetéssel: `KV restored from=disk tokens=[~30000]`, a llama-server ~1 tokent számol; a dinamikus KV a visszatöltéskor nő (`relayout: KV ... cells`), a flex rétegek a zónaszabály szerint mozognak.
- Eltérő KV-típus (`kv_type: q8_0` a szakaszban): a visszatöltés elmarad (`the saved KV is not restored ... kv_type`), a prompt újraszámolódik.

## 9. Két ROCm-kártya (ha megjön a második X080)
- A `devices` két bejegyzése PCI-címmel (`x080a`, `x080b`); név szerinti `match` esetén a napló `matches 2 devices ... use the PCI address` hibát ad, és a bejegyzés kimarad.
- Egy modell `devices: [x080a, x080b, cpu]`: az első foglalt → a második (`reason=next`), mindkettőn külön példány (`ollama ps` két sor).

## Mit küldj vissza
Minden pontnál a `dualbuf` naplósorok (`eval $L | grep dualbuf`), a 6. pont `load_duration`-jei (három futás), a 7. pont kirakási sorrendje és a 8. pont mentési/visszatöltési ideje.
