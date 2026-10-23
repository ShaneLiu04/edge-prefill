# GPU dense prefill baseline via official llama.cpp b11748 CUDA 12.4 build
# (same commit as our pinned source). Outputs CSV for fig_cpu_gpu.
# Usage: powershell -File run_gpu_baseline.ps1
$ErrorActionPreference = "Stop"
$dir = "D:\edge-llm\llama-gpu"
$model = "D:\edge-llm\models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$outDir = "D:\prefill\benchmarks\results\exp-gpu-dense"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

if (-not (Test-Path "$dir\llama-bench.exe")) {
    Expand-Archive -Path "$dir\llama-bin.zip" -DestinationPath "$dir\bin" -Force
    Expand-Archive -Path "$dir\cudart.zip" -DestinationPath "$dir\cudart" -Force
    # cudart DLLs must sit next to the exes
    Copy-Item "$dir\cudart\*" "$dir\bin\" -Force
}

$bench = "$dir\bin\llama-bench.exe"
$stamp = [int][double]::Parse((Get-Date -UFormat %s))
$csv = "$outDir\gpu_dense_${stamp}.csv"

# pp (prefill) throughput at our sweep lengths; full layer offload;
# FA off matches our sparse graph family, FA on shows the GPU ceiling.
& $bench -m $model -ngl 99 -t 4 -p 2048,4096,8192,16384 -n 0 -fa off,on -r 3 -o csv | Out-File -FilePath $csv -Encoding utf8

Get-Content $csv
"---"
"saved: $csv"
"env: Quadro RTX 5000 16GB, driver 556.18, CUDA 12.4, llama.cpp b11748 win-cuda x64"
