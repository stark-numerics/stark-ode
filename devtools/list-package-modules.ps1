# .\devtools\list-package-modules.ps1
#
# Examples:
#   .\devtools\list-package-modules.ps1
#   .\devtools\list-package-modules.ps1 -PackagePath .\stark
#   .\devtools\list-package-modules.ps1 -PackagePath .\stark -WithFiles
#   .\devtools\list-package-modules.ps1 -PackagePath .\stark -OnlyPackages

param(
    [string]$PackagePath = ".\stark",

    [switch]$WithFiles,

    [switch]$OnlyPackages,

    [switch]$OnlyModules
)

$ErrorActionPreference = "Stop"

if ($OnlyPackages -and $OnlyModules) {
    throw "Use only one of -OnlyPackages or -OnlyModules."
}

if (-not (Test-Path $PackagePath)) {
    throw "Package path not found: $PackagePath. Run from repo root, or pass -PackagePath."
}

$PackagePath = (Resolve-Path $PackagePath).Path
$PackageRoot = Split-Path $PackagePath -Parent
$PackageRootWithSlash = $PackageRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar

$Rows = @()

Get-ChildItem -Path $PackagePath -Recurse -File -Filter "*.py" | ForEach-Object {
    $file = $_.FullName

    if ($file.StartsWith($PackageRootWithSlash, [System.StringComparison]::OrdinalIgnoreCase)) {
        $relativeModulePath = $file.Substring($PackageRootWithSlash.Length)
    }
    else {
        $relativeModulePath = $_.Name
    }

    $module = $relativeModulePath `
        -replace '\\', '.' `
        -replace '/', '.' `
        -replace '\.py$', ''

    $isPackage = $module.EndsWith(".__init__")

    if ($isPackage) {
        $module = $module -replace '\.__init__$', ''
    }

    if ($OnlyPackages -and -not $isPackage) {
        return
    }

    if ($OnlyModules -and $isPackage) {
        return
    }

    $Rows += [pscustomobject]@{
        Module = $module
        Kind   = if ($isPackage) { "package" } else { "module" }
        File   = ".\" + $relativeModulePath
    }
}

$Rows = $Rows | Sort-Object Module

foreach ($row in $Rows) {
    if ($WithFiles) {
        "$($row.Module)`t$($row.Kind)`t$($row.File)"
    }
    else {
        $row.Module
    }
}