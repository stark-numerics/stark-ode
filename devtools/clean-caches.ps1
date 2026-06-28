#.\devtools\clean-caches.ps1 -DryRun
#.\devtools\clean-caches.ps1

[CmdletBinding()]
param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$gitRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -eq 0 -and $gitRoot) {
    Set-Location $gitRoot
}

$directories = @(
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "htmlcov"
)

$files = Get-ChildItem -Force -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match '\.(pyc|pyo|nbc|nbi)$' -or
        $_.Name -eq ".coverage"
    }

$dirs = Get-ChildItem -Force -Recurse -Directory -ErrorAction SilentlyContinue |
    Where-Object { $directories -contains $_.Name }

foreach ($item in @($files) + @($dirs)) {
    if ($DryRun) {
        Write-Host "Would remove $($item.FullName)"
    }
    else {
        Remove-Item -Force -Recurse $item.FullName
    }
}

if (-not $DryRun) {
    Write-Host "Cache cleanup complete."
}
