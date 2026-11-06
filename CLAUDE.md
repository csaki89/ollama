# CLAUDE.md

See `AGENTS.md` for the shared agent instructions for this repository.

# CLAUDE.md (dual-buffer fork)

## Szabályok
- Soha ne nyiss pull requestet, issue-t vagy kommentet, és ne tölts fel semmit az ollama/ollama repóba. Az `upstream` remote csak frissítések lehúzására való (a push URL `DISABLED`).
- A gh alapértelmezett repója a saját fork (csaki89/ollama).
- A munka a `dual-buffer` branchben folyik (a `v0.40.2` tagből indult), a `main` tiszta marad.
- A dual-buffer branchen a commitok előre jóvá vannak hagyva, nem kell egyenként rákérdezni. A commit üzenetet te írod: rövid, és tartalmazza: `Assisted-by: Claude`.
- Push az originre (csaki89/ollama) csak kifejezett kérésre.
- A `../llama-dualbuf` repó (csaki89/llama.cpp, dual-buffer branch) csak olvasható forrás: onnan olvasni, másolni szabad, módosítani, commitolni vagy pusholni nem.
- Semmit ne telepíts kérés nélkül (sem rendszerszinten, sem csomagkezelővel). A hiányzó eszközöket sorold fel, és kérdezz.

## Viszony az AGENTS.md-hez és a CONTRIBUTING.md-hez
Ez egy privát fork, nem az ollama/ollama-ba szánt hozzájárulás.
- Az `AGENTS.md` csak build-útmutató (`cmake -B build .`, `cmake --build build --parallel 8`, gyors Go-iterációhoz `go build .`; részletek: `docs/development.md`). Érvényes, ütközés nincs.
- A `CONTRIBUTING.md` az upstream PR/issue folyamatot írja le (előzetes issue nem triviális változáshoz, `<package>: <leírás>` commit cím, tesztek, kevés új függőség). Ez itt nem kötelező, mert nem nyitunk PR-t. A commit üzenetre a fenti rövid forma + `Assisted-by: Claude` érvényes. A "tesztet is írj, viselkedést tesztelj" elv jó gyakorlat marad.

## Cél
Saját Ollama ROCm image, amelynek llama-server komponense a llama.cpp dual-buffer módosítását tartalmazza. A TrueNAS-on egy compose fájllal, egyéni appként fut.

A dual-buffer a csaki89/llama.cpp repó `dual-buffer` branchében kész és tesztelt. Forrásként a `../llama-dualbuf` mappa szolgál (olvasható): `CLAUDE.md`, `docs/DESIGN.md`, `docs/DECISIONS.md`, `docs/SUMMARY.md`, `docs/test-results.md`. A cél: a CPU-n maradó súlyokat repackelt (generálás) és eredeti formájú host-példányban (op offload, prompt) is tartani.

## Környezet (NAS)
- Jelenleg: `ollama/ollama:0.40.2-rocm`, AMD RX 6700 (gfx1031), `HSA_OVERRIDE_GFX_VERSION=10.3.0`, ezért a GPU-kódot gfx1030-ra kell fordítani.
- Az Ollama 0.40.2 a modelleket beépített llama-server folyamattal futtatja, és a `LLAMA_ARG_` környezeti változókat átadja neki.
- Igazolt: a `use_mmap: false` opció (kérésben vagy Modelfile-ban) split módban is `--load-mode none`-nal indítja a llama-servert.

## Hol van a llama.cpp ebben a repóban
- A llama-server külön épül: `llama/server/CMakeLists.txt`, FetchContent-tel a rögzített llama.cpp forrásból (`LLAMA_CPP_VERSION`: b11351 a v0.40.2-ben).
- Az Ollama-specifikus patchek: `llama/compat/` (pl. `001-llama-cpp-hooks.patch`, `002-clef.patch`).
- A Linux/ROCm image a `Dockerfile`-ban épül (`llama-server-rocm_v7_2` stage, `cmake -S llama/server --preset rocm_v7_2_linux`, ROCm 7.2.1 az `rocm/dev-almalinux-8` alapon).
- Az ROCm image fordítása Linuxon/Dockerben történik, nem ezen a Windows gépen.

## Fejlesztői környezet (laptop, Windows)
- Natív eszközök: VS 2022 Community (MSVC), a VS-be épített CMake és Ninja, Vulkan SDK `C:\VulkanSDK\1.4.363.1`. Az msys2 cmake/git nem a natív toolchain része.
- Fordítás vcvars64 környezetben (Developer PowerShell), `VULKAN_SDK` beállítva.
