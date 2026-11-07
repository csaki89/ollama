# Haladás

| Fázis | Állapot | Megjegyzés |
|---|---|---|
| 1. Felderítés (DESIGN.md) | kész | Döntés: `llama/compat/003-dual-buffer.patch` |
| 2. Beépítés | kész | `llama/compat/003-dual-buffer.patch`, a patch lefordul (CPU build), DECISIONS.md |
| 3. Laptopos próba (Vulkan) | részben kész | llama-server közvetlenül mérve (dual: host-szint, 2,7x a repack); az ollama bináris nem: Go megvan, de nincs C fordító a cgo-hoz (test-results.md) |
| 4. GitHub Actions | kész (nem futtatva) | `.github/workflows/dualbuf-image.yaml`, Dockerfile: 2 ARG + 1 RUN; futtatás csak push után |
| 5. docker-compose.yaml | kész (nem futtatva) | a címkét az első sikeres build után kell beírni |
| 6. SUMMARY.md | kész | |
| 7. Automatikus zónák (`--auto-zones`, llama-dualbuf e86f222a5) | kész, CI-ben fordítva | 003 patch háromutas összefésüléssel (DECISIONS 19-23); laptopon a magkönyvtárak fordultak, az eszközök fordítása memóriahiány miatt megszakadt |
