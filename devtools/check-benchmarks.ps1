# .\devtools\check-benchmarks.ps1
#
# Run ASV benchmark discovery and optional smoke benchmark runs through the
# repository's existing virtual environment.
#
# Examples:
#   .\devtools\check-benchmarks.ps1
#   .\devtools\check-benchmarks.ps1 -RunSmoke
#   .\devtools\check-benchmarks.ps1 -RunRepresentative
#   .\devtools\check-benchmarks.ps1 -RunSmoke -Bench BenchmarkTimeIVPSmokeRepeatSolve

[CmdletBinding()]
param(
    [switch]$RunSmoke,
    [switch]$RunRepresentative,
    [switch]$RunFull,
    [string]$Bench = "",
    [string]$Python = "..\.venv\Scripts\python.exe",
    [string]$Machine = "stark-local"
)

$ErrorActionPreference = "Stop"

$gitRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -eq 0 -and $gitRoot) {
    Set-Location $gitRoot
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

$runSwitches = @($RunSmoke, $RunRepresentative, $RunFull) | Where-Object { $_ }
if ($runSwitches.Count -gt 1) {
    throw "Choose only one of -RunSmoke, -RunRepresentative, or -RunFull."
}

$selectedBench = $Bench
if ([string]::IsNullOrWhiteSpace($selectedBench)) {
    if ($RunSmoke) {
        $selectedBench = "BenchmarkTimeIVPSmoke"
    }
    elseif ($RunRepresentative) {
        $selectedBench = "BenchmarkTimeIVPRepresentative"
    }
    elseif ($RunFull) {
        $selectedBench = "BenchmarkTimeIVPFull"
    }
}

$asvHome = ".\devtools\tmp\asv-home"
New-Item -ItemType Directory -Force -Path $asvHome | Out-Null
$asvHomePath = (Resolve-Path -LiteralPath $asvHome).Path

$previousUserProfile = $env:USERPROFILE
$previousHome = $env:HOME

try {
    $env:USERPROFILE = $asvHomePath
    $env:HOME = $asvHomePath

    Write-Host "Running ASV discovery check"
    Write-Host "Python: $Python"
    Write-Host "ASV home: $asvHome"
    Write-Host "Machine: $Machine"
    Write-Host ""

    & $Python -m asv machine --yes --machine $Machine
    if ($LASTEXITCODE -ne 0) {
        throw "ASV machine initialisation failed."
    }

    & $Python -m asv check -E "existing:$Python"
    if ($LASTEXITCODE -ne 0) {
        throw "ASV discovery check failed."
    }

    if (-not [string]::IsNullOrWhiteSpace($selectedBench)) {
        Write-Host ""
        Write-Host "Running ASV benchmarks"
        Write-Host "Bench:  $selectedBench"
        Write-Host ""

        & $Python -m asv run --quick -E "existing:$Python" --bench $selectedBench --machine $Machine
        if ($LASTEXITCODE -ne 0) {
            throw "ASV benchmark run failed."
        }
    }
}
finally {
    $env:USERPROFILE = $previousUserProfile
    $env:HOME = $previousHome
}
