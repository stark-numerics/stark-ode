# .\devtools\list-stark-classes.ps1
# Run from repo root:
#   .\devtools\list-stark-classes.ps1
# Optional:
#   .\devtools\list-stark-classes.ps1 -Csv > .\devtools\stark-classes.csv

param(
    [string]$PackagePath = ".\stark",
    [switch]$Csv,
    [switch]$IncludePrivate
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PackagePath)) {
    throw "Package path not found: $PackagePath. Run this from the repo root, or pass -PackagePath."
}

$PackagePath = (Resolve-Path $PackagePath).Path
$PackageRoot = Split-Path $PackagePath -Parent
$PackageRootWithSlash = $PackageRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar

$Rows = @()

Get-ChildItem -Path $PackagePath -Recurse -Filter "*.py" | ForEach-Object {
    $File = $_.FullName

    if ($File.StartsWith($PackageRootWithSlash, [System.StringComparison]::OrdinalIgnoreCase)) {
        $RelativeModulePath = $File.Substring($PackageRootWithSlash.Length)
    }
    else {
        $RelativeModulePath = $_.Name
    }

    $RelativeFile = ".\" + $RelativeModulePath

    $Module = $RelativeModulePath `
        -replace '\\', '.' `
        -replace '/', '.' `
        -replace '\.py$', '' `
        -replace '\.__init__$', ''

    $Lines = Get-Content -Path $File

    for ($i = 0; $i -lt $Lines.Count; $i++) {
        $Line = $Lines[$i]

        if ($Line -match '^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s*:') {
            $ClassName = $Matches[1]
            $Bases = ""

            if ($Matches.Count -gt 2) {
                $Bases = $Matches[2].Trim()
            }

            if (-not $IncludePrivate -and $ClassName.StartsWith("_")) {
                continue
            }

            $Rows += [pscustomobject]@{
                Class  = $ClassName
                Module = $Module
                Bases  = $Bases
                Line   = $i + 1
                File   = $RelativeFile
            }
        }
    }
}

$Rows = $Rows | Sort-Object @{ Expression = { $_.Class.ToLowerInvariant() } }, Module, Line

if ($Csv) {
    $Rows | ConvertTo-Csv -NoTypeInformation
}
else {
    foreach ($Row in $Rows) {
        if ($Row.Bases) {
            "$($Row.Class) : $($Row.Bases)`t$($Row.Module):$($Row.Line)`t$($Row.File)"
        }
        else {
            "$($Row.Class)`t$($Row.Module):$($Row.Line)`t$($Row.File)"
        }
    }
}