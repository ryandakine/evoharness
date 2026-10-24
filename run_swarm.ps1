param(
    [string]$Prompt,
    [double]$PriceIn = 0,
    [double]$PriceOut = 0
)
$ErrorActionPreference = 'Stop'
$swarmPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $swarmPython)) {
    throw 'Local environment is missing. Install the harness in .venv first.'
}
if (-not $Prompt) {
    $Prompt = Read-Host 'What should the swarm work on?'
}
if (-not $Prompt.Trim()) {
    throw 'Please provide a task for the swarm.'
}
& $swarmPython (Join-Path $PSScriptRoot 'examples\run_swarm.py') --prompt $Prompt --temperature omit --max-tokens 8192 --price-in $PriceIn --price-out $PriceOut
exit $LASTEXITCODE
