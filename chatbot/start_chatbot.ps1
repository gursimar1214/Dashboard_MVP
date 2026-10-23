# Starts the chatbot API. Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File chatbot\start_chatbot.ps1
#
# Langfuse keys are read from chatbot\langfuse.env (copy langfuse.env.example).
# Only the supported Langfuse settings are accepted from this file.

$envFile = Join-Path $PSScriptRoot "langfuse.env"
foreach ($name in @("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL")) {
    Remove-Item -Path ("Env:" + $name) -ErrorAction SilentlyContinue
}
if (Test-Path -LiteralPath $envFile -PathType Leaf) {
    foreach ($raw in Get-Content $envFile) {
        $line = $raw.Trim().TrimStart([char]0xFEFF)
        if ($line -eq "" -or $line.StartsWith("#") -or -not $line.Contains("=")) { continue }
        $name, $value = $line.Split("=", 2)
        $name = $name.Trim()
        $value = $value.Trim().Trim('"').Trim("'")
        if ($name -notin @("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL", "CHATBOT_LANGFUSE_ROWS")) {
            Write-Warning "Ignoring unsupported setting '$name' in langfuse.env."
            continue
        }
        if ($name -in @("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") -and [string]::IsNullOrWhiteSpace($value)) {
            Remove-Item -Path ("Env:" + $name) -ErrorAction SilentlyContinue
            continue
        }
        Set-Item -Path ("Env:" + $name) -Value $value
    }
    if ($env:LANGFUSE_PUBLIC_KEY -and $env:LANGFUSE_SECRET_KEY) {
        Write-Host "Loaded Langfuse settings from langfuse.env"
    } else {
        Write-Host "Langfuse keys are missing from langfuse.env -> tracing stays OFF."
    }
} else {
    Write-Host "No chatbot\langfuse.env found -> Langfuse tracing stays OFF (the chatbot works normally)."
}

$env:OLLAMA_HOST = "http://10.10.3.100:11434"
$env:CHATBOT_LLM_MODEL = "qwen3:8b"

Write-Host "Ollama server: $env:OLLAMA_HOST"
Write-Host "Model: $env:CHATBOT_LLM_MODEL"

# uvicorn must start from the project root (the folder that CONTAINS chatbot\)
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m uvicorn chatbot.main:app --port 8000
