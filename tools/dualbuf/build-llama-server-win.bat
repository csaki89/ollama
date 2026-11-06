@echo off
rem Builds the patched llama-server (CPU + Vulkan) on Windows with MSVC/Ninja.
rem Run from the repository root. Needs: Visual Studio 2022 (C++ tools), Vulkan SDK (VULKAN_SDK set).
rem Output: dist\lib\ollama (llama-server.exe + ggml dlls), build\llama-server-*\
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "VSCMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake"
set "PATH=%VSCMAKE%\CMake\bin;%VSCMAKE%\Ninja;%PATH%"
if "%VULKAN_SDK%"=="" set "VULKAN_SDK=C:\VulkanSDK\1.4.363.1"
set CMAKE_GENERATOR=Ninja

cmake -S llama/server --preset cpu_windows -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON || exit /b 1
cmake --build build/llama-server-cpu || exit /b 1
cmake --install build/llama-server-cpu --component llama-server || exit /b 1

cmake -S llama/server --preset vulkan || exit /b 1
cmake --build build/llama-server-vulkan --target ggml-vulkan || exit /b 1
cmake --install build/llama-server-vulkan --component llama-server || exit /b 1
echo BUILD-OK
