# run_ollama_gpu.ps1 -- start Ollama so it uses the Radeon 780M (Ryzen 7 8845HS iGPU).
#
# The 780M (gfx1103) isn't on AMD's ROCm support list, so Ollama has to use its
# Vulkan backend. Run this from PowerShell:
#
#     powershell -ExecutionPolicy Bypass -File chatbot\run_ollama_gpu.ps1
#
# It closes any running Ollama, saves the settings for your Windows user (so the
# tray app picks them up next time too), and starts `ollama serve` in this window.

Get-Process -Name "ollama*", "ollama app" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

$settings = @{
    "OLLAMA_VULKAN"          = "1"      # must be exactly 1 ("true" is ignored)
    "OLLAMA_IGPU_ENABLE"     = "1"      # allow integrated GPUs
    "OLLAMA_NUM_PARALLEL"    = "1"      # one slot: less memory, better prompt caching
    "OLLAMA_MAX_LOADED_MODELS" = "1"
    "OLLAMA_KEEP_ALIVE"      = "30m"
    "OLLAMA_FLASH_ATTENTION" = "1"      # faster attention, smaller KV cache
}
foreach ($name in $settings.Keys) {
    [Environment]::SetEnvironmentVariable($name, $settings[$name], "User")  # persistent
    Set-Item -Path "Env:$name" -Value $settings[$name]                      # this window
}

Write-Host "Starting Ollama with Vulkan enabled..." -ForegroundColor Cyan
Write-Host "In a second terminal: python -m chatbot.check_model" -ForegroundColor Cyan
ollama serve
