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
$root = (Resolve-Path -LiteralPath ".").Path

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
    Where-Object { $directories -contains $_.Name } |
    Sort-Object { $_.FullName.Length } -Descending

function Clear-CacheReadOnlyAttribute {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.FileSystemInfo]$Item
    )

    try {
        if (($Item.Attributes -band [System.IO.FileAttributes]::ReadOnly) -ne 0) {
            $Item.Attributes = $Item.Attributes -band (-bnot [System.IO.FileAttributes]::ReadOnly)
        }
    }
    catch {
        Write-Verbose "Could not clear read-only attribute on $($Item.FullName): $($_.Exception.Message)"
    }
}

function Remove-CacheItem {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.FileSystemInfo]$Item
    )

    $path = $Item.FullName
    if (-not $path.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove cache outside repository root: $path"
    }

    if ($DryRun) {
        Write-Host "Would remove $path"
        return
    }

    if ($Item.PSIsContainer) {
        Get-ChildItem -LiteralPath $path -Force -Recurse -ErrorAction SilentlyContinue |
            ForEach-Object {
                Clear-CacheReadOnlyAttribute -Item $_
            }
    }
    Clear-CacheReadOnlyAttribute -Item $Item

    try {
        Remove-Item -LiteralPath $path -Force -Recurse -ErrorAction Stop
    }
    catch {
        throw "Could not remove cache item '$path': $($_.Exception.Message)"
    }
}

foreach ($item in @($files) + @($dirs)) {
    Remove-CacheItem -Item $item
}

if (-not $DryRun) {
    Write-Host "Cache cleanup complete."
}
