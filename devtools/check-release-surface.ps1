# .\devtools\check-release-surface.ps1
#
# Audit the public surface before release. By default this script is
# non-destructive: it reports generated caches, forbidden imports, and
# top-level `stark` exports that should be reviewed.
#
# Examples:
#   .\devtools\check-release-surface.ps1
#   .\devtools\check-release-surface.ps1 -FailOnUnexpectedTopLevel
#   .\devtools\check-release-surface.ps1 -CleanCaches

[CmdletBinding()]
param(
    [switch]$CleanCaches,
    [switch]$VerboseCaches,
    [switch]$FailOnUnexpectedTopLevel,
    [int]$ContextLines = 0
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_common.ps1"

$gitRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -eq 0 -and $gitRoot) {
    Set-Location $gitRoot
}

function Write-Section {
    param([Parameter(Mandatory = $true)][string]$Title)

    Write-Host ""
    Write-Host ("=" * 88)
    Write-Host $Title
    Write-Host ("=" * 88)
}

function Get-TopLevelExports {
    param([Parameter(Mandatory = $true)][string]$Path)

    $exports = @()
    $insideAll = $false

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*__all__\s*=\s*\[') {
            $insideAll = $true
            continue
        }

        if (-not $insideAll) {
            continue
        }

        if ($line -match '^\s*\]') {
            break
        }

        if ($line -match '^\s*["'']([^"'']+)["'']\s*,?\s*$') {
            $exports += $Matches[1]
        }
    }

    return $exports
}

function Get-GeneratedCacheItems {
    $cacheDirectoryNames = @(
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "htmlcov"
    )

    $files = @(
        Get-ChildItem -Force -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -match '\.(pyc|pyo|nbc|nbi)$' -or
                $_.Name -eq ".coverage"
            }
    )

    $dirs = @(
        Get-ChildItem -Force -Recurse -Directory -ErrorAction SilentlyContinue |
            Where-Object { $cacheDirectoryNames -contains $_.Name }
    )

    return [pscustomobject]@{
        Files = $files
        Dirs  = $dirs
    }
}

function Show-ForbiddenImportAudit {
    param([string[]]$Rules)

    $violations = @()

    foreach ($rule in $Rules) {
        $matcher = New-DevtoolsSearchRegex -Pattern $rule -DottedToken
        $matches = @(
            Find-DevtoolsRegexMatches `
                -Matcher $matcher `
                -Roots $script:DevtoolsDefaultRoots `
                -Extensions $script:DevtoolsDefaultTextExtensions `
                -AllMatches
        )

        foreach ($fileMatch in $matches) {
            $violations += [pscustomobject]@{
                Rule  = $rule
                Match = $fileMatch
            }
        }
    }

    if ($violations.Count -eq 0) {
        Write-Host "No forbidden imports found."
        return 0
    }

    Write-Host "Forbidden imports remain:"
    Write-Host ""

    foreach ($violation in $violations) {
        Write-Host "Rule: $($violation.Rule)"
        Write-DevtoolsMatchReport `
            -MatchesByFile @($violation.Match) `
            -ContextLines $ContextLines `
            -Header ""
    }

    return $violations.Count
}

$recommendedTopLevel = @(
    "Configuration",
    "Derivative",
    "DerivativeSignature",
    "DerivativeStyle",
    "Frame",
    "FrameField",
    "Interval",
    "Linearizer",
    "LinearizerSignature",
    "LinearizerStyle",
    "Method",
    "System",
    "Tolerance"
)

$forbiddenImports = @(
    "stark.accelerators",
    "stark.carriers",
    "stark.contracts",
    "stark.interface",
    "stark.algebraist",
    "stark.engines.backends",
    "stark.engines.accelerators",
    "stark.engines.algebraist",
    "stark.engines.carriers",
    "stark.executor"
)

$failures = 0

Write-Host "STARK release-surface audit"
Write-Host "Repo: $((Resolve-Path -LiteralPath '.').Path)"

Write-Section "Generated cache cleanup"
if ($CleanCaches) {
    Write-Host "Cleaning generated caches."
    & "$PSScriptRoot\clean-caches.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "Cache cleanup failed."
    }
}
else {
    Write-Host "Dry run only. Pass -CleanCaches to remove generated cache files."

    if ($VerboseCaches) {
        & "$PSScriptRoot\clean-caches.ps1" -DryRun
        if ($LASTEXITCODE -ne 0) {
            throw "Cache dry run failed."
        }
    }
    else {
        $cacheItems = Get-GeneratedCacheItems
        Write-Host ("Generated cache files: {0}" -f @($cacheItems.Files).Count)
        Write-Host ("Generated cache directories: {0}" -f @($cacheItems.Dirs).Count)
        if (@($cacheItems.Files).Count -gt 0 -or @($cacheItems.Dirs).Count -gt 0) {
            Write-Host "Pass -VerboseCaches to list every generated cache path."
        }
    }
}

Write-Section "Forbidden import audit"
$forbiddenCount = Show-ForbiddenImportAudit -Rules $forbiddenImports
if ($forbiddenCount -gt 0) {
    $failures += 1
}

Write-Section "Top-level stark exports"
$topLevelPath = ".\stark\__init__.py"
$exports = @(Get-TopLevelExports -Path $topLevelPath)
$recommendedSet = @{}
foreach ($name in $recommendedTopLevel) {
    $recommendedSet[$name] = $true
}

$exportSet = @{}
foreach ($name in $exports) {
    $exportSet[$name] = $true
}

$unexpected = @($exports | Where-Object { -not $recommendedSet.ContainsKey($_) })
$missing = @($recommendedTopLevel | Where-Object { -not $exportSet.ContainsKey($_) })

Write-Host "Current exports:"
foreach ($name in $exports) {
    Write-Host "  $name"
}

if ($unexpected.Count -gt 0) {
    Write-Host ""
    Write-Host "Review these exports before release:"
    foreach ($name in $unexpected) {
        Write-Host "  $name"
    }

    if ($FailOnUnexpectedTopLevel) {
        $failures += 1
    }
}
else {
    Write-Host ""
    Write-Host "Top-level exports match the current recommended lean surface."
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Recommended exports missing from top level:"
    foreach ($name in $missing) {
        Write-Host "  $name"
    }
    $failures += 1
}

Write-Section "Next checks"
Write-Host "Recommended after any cleanup:"
Write-Host "  python -m pytest -q"
Write-Host "  .\devtools\check-release-surface.ps1"
Write-Host ""
Write-Host "Suggested commit if the cleanup is intentional:"
Write-Host "  git add ."
Write-Host "  git commit -m `"Tighten release surface hygiene`""

if ($failures -gt 0) {
    Write-Host ""
    Write-Host "Release-surface audit completed with $failures review item(s)."
    exit 1
}

Write-Host ""
Write-Host "Release-surface audit passed."
