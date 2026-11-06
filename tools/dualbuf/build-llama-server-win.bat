@echo off
rem Builds the patched llama-server (CPU + Vulkan) on Windows with MSVC/Ninja.
rem Run from the repository root. Needs: Visual Studio 2022 (C++ tools), Vulkan SDK (VULKAN_SDK set).
rem Usage: build-llama-server-win.bat [vulkan]   ("vulkan" skips the CPU part)
rem Output: dist\lib\ollama (llama-server.exe + ggml dlls)
rem The Vulkan build dir is short on purpose: the shader generator sub-build exceeds MAX_PATH otherwise.
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "VSCMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake"
set "PATH=%VSCMAKE%\CMake\bin;%VSCMAKE%\Ninja;%PATH%"
if "%VULKAN_SDK%"=="" set "VULKAN_SDK=C:\VulkanSDK\1.4.363.1"
set CMAKE_GENERATOR=Ninja
set "ROOT=%CD%"

if /i "%1"=="vulkan" goto vulkan
cmake -S llama/server --preset cpu_windows -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON || exit /b 1
cmake --build build/llama-server-cpu || exit /b 1
cmake --install build/llama-server-cpu --component llama-server || exit /b 1

:vulkan
rem reuse the patched llama.cpp source of the CPU build (the compat/dual-buffer patches are applied idempotently)
cmake -S llama/server --preset vulkan -B C:/b/vk -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP="%ROOT%\build\llama-server-cpu\_deps\llama_cpp-src" || exit /b 1
cmake --build C:/b/vk --target ggml-vulkan || exit /b 1
cmake --install C:/b/vk --component llama-server || exit /b 1
echo BUILD-OK
