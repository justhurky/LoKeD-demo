param(
    [string]$Model = "gemma4:12b"
)

$ErrorActionPreference = "Stop"

$ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
$ollamaExe = $null
if ($ollamaCommand) {
    $ollamaExe = $ollamaCommand.Source
} else {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama.exe"
    )
    $ollamaExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $ollamaExe) {
    Write-Host "Ollama nincs telepítve vagy nem található a szokásos helyeken." -ForegroundColor Yellow
    Write-Host "Telepítés után futtasd újra ezt a scriptet, vagy használj LM Studio / llama.cpp OpenAI-kompatibilis endpointot."
    exit 1
}

Write-Host "Lokális HÖK AI modell letöltése: $Model"
& $ollamaExe pull $Model

Write-Host ""
Write-Host "HÖK AI beállítás:"
Write-Host "- LLM szolgáltató: local"
Write-Host "- Modellnév: $Model"
Write-Host "- Host URL: http://127.0.0.1:11434/v1"
Write-Host "- Felhő API: kikapcsolva"
