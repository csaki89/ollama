# Starts the patched llama-server (dist\lib\ollama) the way Ollama does and compares prompt/generation speed
# for: host (default), repack (LLAMA_ARG_NO_HOST=1), dual (LLAMA_ARG_DUAL_BUFFER=1).
# Usage (repo root): powershell -File tools\dualbuf\bench-server.ps1 -Model <gguf> [-Ngl 14] [-Reps 3] [-OutDir <dir>]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$Ngl = 14,
    [int]$Reps = 3,
    [int]$PromptRepeat = 90,
    [string]$Bin = "dist\lib\ollama",
    [string]$OutDir = "build\bench-server",
    [int]$Port = 18090
)
$ErrorActionPreference = "Stop"
$exe = Join-Path $Bin "llama-server.exe"
$backend = Join-Path $Bin "vulkan\ggml-vulkan.dll"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$modes = [ordered]@{
    "host"   = @{}
    "repack" = @{ LLAMA_ARG_NO_HOST = "1" }
    "dual"   = @{ LLAMA_ARG_DUAL_BUFFER = "1" }
}
$sentence = "The quick brown fox jumps over the lazy dog while the engineers measure the prompt processing speed of a hybrid CPU and GPU setup. "
$prompt = $sentence * $PromptRepeat
$rows = @()
foreach ($name in $modes.Keys) {
    $log = Join-Path $OutDir "$name.log"
    $argsList = @("--model", $Model, "--port", $Port, "--host", "127.0.0.1", "--no-webui", "--offline",
        "-c", 4096, "-np", 1, "--log-verbosity", 4, "--no-log-prefix", "--no-log-timestamps",
        "--load-mode", "none", "--flash-attn", "auto", "-b", 512, "-ub", 512, "-ngl", $Ngl)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = (Resolve-Path $exe).Path
    $psi.Arguments = ($argsList -join " ")
    $psi.UseShellExecute = $false
    $psi.RedirectStandardError = $true
    $psi.RedirectStandardOutput = $true
    $psi.EnvironmentVariables["GGML_BACKEND_PATH"] = (Resolve-Path $backend).Path
    foreach ($k in $modes[$name].Keys) { $psi.EnvironmentVariables[$k] = $modes[$name][$k] }
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    $sb = New-Object System.Text.StringBuilder
    $h = { if ($EventArgs.Data) { [void]$Event.MessageData.AppendLine($EventArgs.Data) } }
    $e1 = Register-ObjectEvent $p ErrorDataReceived -Action $h -MessageData $sb
    $e2 = Register-ObjectEvent $p OutputDataReceived -Action $h -MessageData $sb
    [void]$p.Start(); $p.BeginErrorReadLine(); $p.BeginOutputReadLine()
    try {
        $ok = $false
        for ($i = 0; $i -lt 180; $i++) {
            Start-Sleep -Seconds 1
            if ($p.HasExited) { break }
            try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq "ok") { $ok = $true; break } } catch {}
        }
        if (-not $ok) { throw "llama-server ($name) did not become ready" }
        $res = @()
        for ($r = 0; $r -lt $Reps + 1; $r++) {
            $body = @{ prompt = $prompt; n_predict = 48; temperature = 0; cache_prompt = $false } | ConvertTo-Json
            $o = Invoke-RestMethod "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
            if ($r -gt 0) { $res += $o }   # first request is a warm-up
        }
        $pp = $res | ForEach-Object { $_.timings.prompt_per_second }
        $tg = $res | ForEach-Object { $_.timings.predicted_per_second }
        $rows += [pscustomobject]@{
            mode = $name; prompt_tokens = $res[0].timings.prompt_n
            pp_mean = [math]::Round(($pp | Measure-Object -Average).Average, 1); pp_min = [math]::Round(($pp | Measure-Object -Minimum).Minimum, 1); pp_max = [math]::Round(($pp | Measure-Object -Maximum).Maximum, 1)
            tg_mean = [math]::Round(($tg | Measure-Object -Average).Average, 1)
            out_hash = (Get-FileHash -InputStream ([IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes($res[0].content))) -Algorithm MD5).Hash.Substring(0, 8)
        }
    }
    finally {
        if (-not $p.HasExited) { $p.Kill() }
        Start-Sleep -Milliseconds 500
        Unregister-Event -SourceIdentifier $e1.Name; Unregister-Event -SourceIdentifier $e2.Name
        $sb.ToString() | Set-Content $log
    }
}
$rows | Format-Table -AutoSize | Out-String | Write-Host
$rows | ConvertTo-Csv -NoTypeInformation | Set-Content (Join-Path $OutDir "results.csv")
foreach ($name in $modes.Keys) {
    Write-Host "== $name"
    Select-String -Path (Join-Path $OutDir "$name.log") -Pattern "dual-buffer|model buffer size|load_mode|system_info|using device" | ForEach-Object { $_.Line }
}
