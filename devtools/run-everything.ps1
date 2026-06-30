# run-everything.ps1
#
# Run the broad local verification sweep: tests, docs guards, example runners,
# competition reports, and runnable benchmark scripts.
#
# Example:
#   .\devtools\run-everything.ps1
#
# Quicker slices:
#   .\devtools\run-everything.ps1 -SkipBenchmarks
#   .\devtools\run-everything.ps1 -SkipCompetition -SkipBenchmarks

param(
    [string]$Python = "",

    [double]$CompetitionTimeout = 1000,

    [switch]$SkipTests,
    [switch]$SkipSlowTests,
    [switch]$SkipDocs,
    [switch]$FailOnDocsWarning,
    [switch]$SkipExamples,
    [switch]$SkipCompetition,
    [switch]$SkipBenchmarks
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Python)) {
    $venvPython = ".\.venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $Python = $venvPython
    }
    else {
        $Python = "python"
    }
}

function Write-Section {
    param([string]$Title)

    Write-Host ""
    Write-Host ("=" * 88)
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Title)
    Write-Host ("=" * 88)
}

function Invoke-PythonModule {
    param(
        [string]$Label,
        [string[]]$Arguments
    )

    Write-Section $Label
    Write-Host "Command: $Python $($Arguments -join ' ')"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: $Label"
    }
}

$started = Get-Date
Write-Host "STARK full local sweep"
Write-Host "Python: $Python"
Write-Host "Started: $($started.ToString('yyyy-MM-dd HH:mm:ss'))"

if (-not $SkipTests) {
    Invoke-PythonModule "tests: default pytest suite" @("-m", "pytest")
}

if (-not $SkipTests -and -not $SkipSlowTests) {
    Invoke-PythonModule "tests: slow pytest suite" @("-m", "pytest", "-m", "slow", "-o", "addopts=-ra")
}

if (-not $SkipDocs) {
    Write-Section "docs: static consistency guard"
    if ($FailOnDocsWarning) {
        & .\devtools\check-docs-consistency.ps1 -FailOnWarning
    }
    else {
        & .\devtools\check-docs-consistency.ps1
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: docs consistency guard"
    }
}

if (-not $SkipExamples) {
    Invoke-PythonModule "examples: getting started" @("-m", "examples.getting_started")
    Invoke-PythonModule "examples: problem" @("-m", "examples.problem")
    Invoke-PythonModule "examples: methods" @("-m", "examples.methods")
    Invoke-PythonModule "examples: diagnostics" @("-m", "examples.diagnostics")
    Invoke-PythonModule "examples: engines" @("-m", "examples.engines")
    Invoke-PythonModule "examples: inverters" @("-m", "examples.inverters")
    Invoke-PythonModule "examples: core" @("-m", "examples.core")
}

if (-not $SkipCompetition) {
    Invoke-PythonModule "competition: report guard" @(
        "-m",
        "competition.check_reports",
        "--timeout",
        "$CompetitionTimeout"
    )
}

if (-not $SkipBenchmarks) {
    # Current runnable benchmark modules. Older transitional benchmark files
    # that target removed Executor/resolvent APIs are intentionally left out
    # until they are redesigned around the current architecture.
    $benchmarkModules = @(
        "benchmarks.algebraist.timing_explicit",
        "benchmarks.algebraist.timing_implicit_fixed",
        "benchmarks.algebraist.timing_imex_adaptive",
        "benchmarks.schemes.bench_scheme_refactor",
        "benchmarks.inverters.bench_defect",
        "benchmarks.inverters.bench_jacobi",
        "benchmarks.inverters.bench_richardson"
    )

    foreach ($module in $benchmarkModules) {
        Invoke-PythonModule "benchmark: $module" @("-m", $module)
    }
}

$finished = Get-Date
$elapsed = $finished - $started

Write-Host ""
Write-Host ("=" * 88)
Write-Host "Full local sweep completed."
Write-Host "Finished: $($finished.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host ("Elapsed: {0:hh\:mm\:ss}" -f $elapsed)
Write-Host ("=" * 88)
