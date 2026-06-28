# .\devtools\rename-python-package.ps1
#
# Move/rename a Python package directory and update dotted import references.
# This is intended for large package architecture moves such as:
#   stark\schemes -> stark\methods\schemes
#   stark\carriers -> stark\engines\carriers
#
# Defaults include stark/, tests/, docs/, examples/, competition/, benchmarks/,
# README.md, and pyproject.toml.
#
# Examples:
#   .\devtools\rename-python-package.ps1 `
#       -Before .\stark\schemes `
#       -After .\stark\methods\schemes
#
#   .\devtools\rename-python-package.ps1 `
#       -Before .\stark\carriers `
#       -After .\stark\engines\carriers `
#       -Yes

param(
    [Parameter(Mandatory = $true)]
    [string]$Before,

    [Parameter(Mandatory = $true)]
    [string]$After,

    [string]$PackageRoot = ".\stark",

    [string[]]$Roots = @(),

    [string[]]$Extensions = @(),

    [int]$ContextLines = 1,

    [switch]$NoGitMove,

    [switch]$Yes,

    [switch]$DryRun,

    [switch]$NoVerify
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_common.ps1"

function Resolve-DevtoolsMaybeMissingPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (Test-Path -LiteralPath $Path) {
        return (Resolve-Path -LiteralPath $Path).Path
    }

    $parent = Split-Path $Path -Parent
    $leaf = Split-Path $Path -Leaf
    if ([string]::IsNullOrWhiteSpace($parent)) {
        $parent = "."
    }

    return (Join-Path (Resolve-Path -LiteralPath $parent).Path $leaf)
}

$beforePath = Resolve-DevtoolsMaybeMissingPath -Path $Before
$afterPath = Resolve-DevtoolsMaybeMissingPath -Path $After

if (-not (Test-Path -LiteralPath $beforePath)) {
    throw "Before package directory does not exist: $Before"
}

$beforeItem = Get-Item -LiteralPath $beforePath
if (-not $beforeItem.PSIsContainer) {
    throw "Before must be a directory. Use rename-python-module.ps1 for .py files."
}

if (-not (Test-Path -LiteralPath (Join-Path $beforePath "__init__.py"))) {
    throw "Before directory is not a Python package: missing __init__.py in $Before"
}

if (Test-Path -LiteralPath $afterPath) {
    throw "Destination already exists: $After"
}

$oldPackage = Get-DevtoolsModuleNameFromPath -Path $beforePath -PackageRoot $PackageRoot
$newPackage = Get-DevtoolsModuleNameFromPath -Path $afterPath -PackageRoot $PackageRoot

if ($oldPackage -eq $newPackage) {
    throw "Old and new package names are identical."
}

# Prefix-safe module replacement. This matches both the package itself and its
# submodules, e.g. stark.schemes and stark.schemes.explicit.
$oldPackagePattern = "(?<![A-Za-z0-9_\.])$([regex]::Escape($oldPackage))(?![A-Za-z0-9_])"
$matcher = [System.Text.RegularExpressions.Regex]::new($oldPackagePattern)

$matchesByFile = @(
    Find-DevtoolsRegexMatches `
    -Matcher $matcher `
    -Roots $Roots `
    -Extensions $Extensions `
    -AllMatches
)

Write-Host ""
Write-Host "Python package move:"
Write-Host "  Directory:"
Write-Host "    $beforePath"
Write-Host "    -> $afterPath"
Write-Host ""
Write-Host "  Package:"
Write-Host "    $oldPackage"
Write-Host "    -> $newPackage"
Write-Host ""

if ($matchesByFile.Count -eq 0) {
    Write-Host "No import/reference matches found."
}
else {
    Write-DevtoolsMatchReport `
        -MatchesByFile $matchesByFile `
        -ContextLines $ContextLines `
        -Header "Found references:"
}

if ($DryRun) {
    Write-Host "Dry run only. No changes made."
    exit 0
}

if (-not $Yes) {
    $response = Read-Host "Move package and update references? Type y to continue, anything else to exit"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "No changes made."
        exit 0
    }
}

# Update text before moving so references inside the moved package are handled safely.
foreach ($fileMatch in $matchesByFile) {
    $path = $fileMatch.Path
    $text = [System.IO.File]::ReadAllText($path)
    $updated = $matcher.Replace($text, $newPackage)

    if ($updated -ne $text) {
        Write-DevtoolsUtf8NoBom -Path $path -Text $updated
        Write-Host "updated $(Format-DevtoolsRelativePath -Path $path)"
    }
}

$afterParent = Split-Path $afterPath -Parent
if (-not (Test-Path -LiteralPath $afterParent)) {
    New-Item -ItemType Directory -Force -Path $afterParent | Out-Null
}

if (-not $NoGitMove) {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        & git mv $beforePath $afterPath
        if ($LASTEXITCODE -ne 0) {
            throw "git mv failed."
        }
    }
    else {
        Move-Item -LiteralPath $beforePath -Destination $afterPath
    }
}
else {
    Move-Item -LiteralPath $beforePath -Destination $afterPath
}

if (-not $NoVerify) {
    if (Test-Path -LiteralPath $beforePath) {
        throw "Post-check failed: source package still exists: $beforePath"
    }

    if (-not (Test-Path -LiteralPath $afterPath)) {
        throw "Post-check failed: destination package was not created: $afterPath"
    }

    if (-not (Test-Path -LiteralPath (Join-Path $afterPath "__init__.py"))) {
        throw "Post-check failed: destination package has no __init__.py: $afterPath"
    }

    $remaining = @(
        Find-DevtoolsRegexMatches `
        -Matcher $matcher `
        -Roots $Roots `
        -Extensions $Extensions `
        -AllMatches
    )

    if ($remaining.Count -gt 0) {
        Write-DevtoolsMatchReport `
            -MatchesByFile $remaining `
            -ContextLines $ContextLines `
            -Header "Post-check failed: old package path '$oldPackage' still appears here:"

        throw "Package move did not eliminate all references to '$oldPackage'."
    }

    Write-Host "Post-check passed: package moved and no references remain for '$oldPackage'."
}

Write-Host ""
Write-Host "Done. Review with:"
Write-Host "  git diff"
Write-Host "  python -m compileall .\stark .\tests .\examples .\competition"
