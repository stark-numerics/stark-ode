# .\devtools\build-docs.ps1
#
# Regenerate the generated API reference and build the Sphinx HTML docs.
#
# The narrative documentation is hand-written. The API reference under
# docs/reference/api is generated from package docstrings so it can act as a
# public-surface smell detector.

param(
    [string]$Python = "",
    [switch]$CleanHtml
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repo = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $repo

if ([string]::IsNullOrWhiteSpace($Python)) {
    $candidatePythons = @(
        (Join-Path $repo ".venv\Scripts\python.exe"),
        (Join-Path (Split-Path $repo -Parent) ".venv\Scripts\python.exe")
    )

    $Python = "python"
    foreach ($candidate in $candidatePythons) {
        if (Test-Path -LiteralPath $candidate) {
            $Python = $candidate
            break
        }
    }
}

function Invoke-DocsStep {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    Write-Host ""
    Write-Host ("-" * 88)
    Write-Host $Label
    Write-Host ("-" * 88)

    try {
        & $Action
    }
    catch {
        Write-Host ""
        Write-Host "FAILED: $Label"
        Write-Host ""
        Write-Host "PowerShell exception:"
        Write-Host $_.Exception.Message
        Write-Host ""
        Write-Host "Recommended recovery:"
        Write-Host "  1. Check whether the docs extra is installed:"
        Write-Host "       python -m pip install -e `".[docs]`""
        Write-Host "  2. Re-run this script from the repository root."
        Write-Host "  3. If generated API files are half-written, inspect:"
        Write-Host "       git diff -- docs/reference/api"
        throw
    }
}

function Remove-PackageContentsAutomodule {
    param([Parameter(Mandatory = $true)][string]$Path)

    $lines = [System.IO.File]::ReadAllLines($Path)
    $keep = @()
    $changed = $false

    foreach ($line in $lines) {
        if ($line -eq "Module contents" -or $line -eq "Package contents") {
            $changed = $true
            break
        }
        $keep += $line
    }

    if ($changed) {
        $updated = ($keep -join "`r`n").TrimEnd() + "`r`n"
        [System.IO.File]::WriteAllText($Path, $updated, [System.Text.UTF8Encoding]::new($false))
    }
}

Write-Host ""
Write-Host ("=" * 88)
Write-Host "Build STARK-ODE documentation"
Write-Host ("=" * 88)
Write-Host "Repo:   $repo"
Write-Host "Python: $Python"

Invoke-DocsStep "Regenerate API reference" {
    $apiRoot = Join-Path $repo "docs\reference\api"
    if (Test-Path -LiteralPath $apiRoot) {
        Remove-Item -LiteralPath $apiRoot -Recurse -Force
    }

    & $Python -m sphinx.ext.apidoc -o "docs\reference\api" "stark"
    if ($LASTEXITCODE -ne 0) {
        throw "sphinx-apidoc failed with exit code $LASTEXITCODE."
    }

    Get-ChildItem -LiteralPath $apiRoot -Filter "*.rst" -File | ForEach-Object {
        Remove-PackageContentsAutomodule -Path $_.FullName
    }
}

Invoke-DocsStep "Build HTML documentation" {
    if ($CleanHtml) {
        $htmlRoot = Join-Path $repo "docs\_build\html"
        if (Test-Path -LiteralPath $htmlRoot) {
            Remove-Item -LiteralPath $htmlRoot -Recurse -Force
        }
    }

    & $Python -m sphinx -b html "docs" "docs\_build\html"
    if ($LASTEXITCODE -ne 0) {
        throw "Sphinx build failed with exit code $LASTEXITCODE."
    }
}

Write-Host ""
Write-Host "Docs build complete."
Write-Host ""
Write-Host "Open:"
Write-Host "  docs\_build\html\index.html"
Write-Host ""
Write-Host "Useful smell-detector pages:"
Write-Host "  docs\_build\html\reference\index.html"
Write-Host "  docs\_build\html\reference\api\modules.html"
Write-Host ""
