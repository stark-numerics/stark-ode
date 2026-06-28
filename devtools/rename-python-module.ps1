# .\devtools\rename-python-module.ps1
#
# Move/rename a Python module file and update import references.
# Defaults include stark/, tests/, docs/, examples/, competition/, benchmarks/,
# README.md, and pyproject.toml.
#
# Examples:
#   .\devtools\rename-python-module.ps1 `
#       -Before .\stark\algebraist\delta.py `
#       -After .\stark\algebraist\delta_specialist.py
#
#   .\devtools\rename-python-module.ps1 `
#       -Before .\stark\algebraist\core.py `
#       -After .\stark\engines\algebraist\core.py `
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

function Get-SameDirectoryRelativeImportPattern {
    param(
        [Parameter(Mandatory = $true)][string]$ImportingFile,
        [Parameter(Mandatory = $true)][string]$OldFile
    )

    $importingDir = Split-Path $ImportingFile -Parent
    $oldDir = Split-Path $OldFile -Parent

    if ($importingDir -eq $oldDir) {
        $oldLeaf = [System.IO.Path]::GetFileNameWithoutExtension($OldFile)
        return "(from\s+\.)$([regex]::Escape($oldLeaf))(\s+import\s+)"
    }

    return $null
}

function Get-SameDirectoryRelativeReplacement {
    param(
        [Parameter(Mandatory = $true)][string]$ImportingFile,
        [Parameter(Mandatory = $true)][string]$OldFile,
        [Parameter(Mandatory = $true)][string]$NewFile,
        [Parameter(Mandatory = $true)][string]$PackageRoot
    )

    $importingDir = Split-Path $ImportingFile -Parent
    $oldDir = Split-Path $OldFile -Parent
    $newDir = Split-Path $NewFile -Parent

    if ($importingDir -eq $oldDir -and $importingDir -eq $newDir) {
        $newLeaf = [System.IO.Path]::GetFileNameWithoutExtension($NewFile)
        return "`$1$newLeaf`$2"
    }

    $newModule = Get-DevtoolsModuleNameFromPath -Path $NewFile -PackageRoot $PackageRoot
    return "from $newModule import "
}

$beforePath = Resolve-DevtoolsMaybeMissingPath -Path $Before
$afterPath = Resolve-DevtoolsMaybeMissingPath -Path $After

if (-not (Test-Path -LiteralPath $beforePath)) {
    throw "Before file does not exist: $Before"
}

if (-not $beforePath.EndsWith(".py", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Before must be a .py file."
}

if (-not $afterPath.EndsWith(".py", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "After must be a .py file."
}

if (Test-Path -LiteralPath $afterPath) {
    throw "Destination already exists: $After"
}

if ([System.IO.Path]::GetFileName($beforePath) -eq "__init__.py" -or [System.IO.Path]::GetFileName($afterPath) -eq "__init__.py") {
    throw "Use rename-python-package.ps1 for package directory moves; this script does not move __init__.py directly."
}

$oldModule = Get-DevtoolsModuleNameFromPath -Path $beforePath -PackageRoot $PackageRoot
$newModule = Get-DevtoolsModuleNameFromPath -Path $afterPath -PackageRoot $PackageRoot
$absoluteMatcher = New-DevtoolsSearchRegex -Pattern $oldModule -AllowPartialMatches

$matchesByFile = @()
$files = Get-DevtoolsTextFiles -Roots $Roots -Extensions $Extensions

foreach ($file in $files) {
    $path = $file.FullName
    $text = [System.IO.File]::ReadAllText($path)
    $hasAbsolute = $absoluteMatcher.IsMatch($text)

    $relativePattern = $null
    $hasRelative = $false

    if ($file.Extension -eq ".py") {
        $relativePattern = Get-SameDirectoryRelativeImportPattern `
            -ImportingFile $path `
            -OldFile $beforePath

        if ($null -ne $relativePattern) {
            $hasRelative = [regex]::IsMatch($text, $relativePattern)
        }
    }

    if (-not ($hasAbsolute -or $hasRelative)) {
        continue
    }

    $lines = $text -split "`r?`n"
    $lineHits = @()

    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        if ($absoluteMatcher.IsMatch($line) -or ($null -ne $relativePattern -and [regex]::IsMatch($line, $relativePattern))) {
            $lineHits += ($i + 1)
        }
    }

    if ($lineHits.Count -gt 0) {
        $matchesByFile += [pscustomobject]@{
            Path  = $path
            Lines = $lines
            Hits  = $lineHits
        }
    }
}

Write-Host ""
Write-Host "Python module move:"
Write-Host "  File:"
Write-Host "    $beforePath"
Write-Host "    -> $afterPath"
Write-Host ""
Write-Host "  Module:"
Write-Host "    $oldModule"
Write-Host "    -> $newModule"
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
    $response = Read-Host "Move file and update references? Type y to continue, anything else to exit"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "No changes made."
        exit 0
    }
}

# Update text before moving so references inside the moved file are handled safely.
foreach ($fileMatch in $matchesByFile) {
    $path = $fileMatch.Path
    $text = [System.IO.File]::ReadAllText($path)
    $updated = $absoluteMatcher.Replace($text, $newModule)

    if ([System.IO.Path]::GetExtension($path) -eq ".py") {
        $relativePattern = Get-SameDirectoryRelativeImportPattern `
            -ImportingFile $path `
            -OldFile $beforePath

        if ($null -ne $relativePattern) {
            $replacement = Get-SameDirectoryRelativeReplacement `
                -ImportingFile $path `
                -OldFile $beforePath `
                -NewFile $afterPath `
                -PackageRoot $PackageRoot

            $updated = [regex]::Replace($updated, $relativePattern, $replacement)
        }
    }

    if ($updated -ne $text) {
        Write-DevtoolsUtf8NoBom -Path $path -Text $updated
        Write-Host "updated $(Format-DevtoolsRelativePath -Path $path)"
    }
}

$afterDir = Split-Path $afterPath -Parent
if (-not (Test-Path -LiteralPath $afterDir)) {
    New-Item -ItemType Directory -Force -Path $afterDir | Out-Null
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
        throw "Post-check failed: source file still exists: $beforePath"
    }

    if (-not (Test-Path -LiteralPath $afterPath)) {
        throw "Post-check failed: destination file was not created: $afterPath"
    }

    $remaining = @(
        Find-DevtoolsRegexMatches `
        -Matcher $absoluteMatcher `
        -Roots $Roots `
        -Extensions $Extensions `
        -AllMatches
    )

    if ($remaining.Count -gt 0) {
        Write-DevtoolsMatchReport `
            -MatchesByFile $remaining `
            -ContextLines $ContextLines `
            -Header "Post-check failed: old module path '$oldModule' still appears here:"

        throw "Module move did not eliminate all references to '$oldModule'."
    }

    Write-Host "Post-check passed: file moved and no references remain for '$oldModule'."
}

Write-Host ""
Write-Host "Done. Review with:"
Write-Host "  git diff"
Write-Host "  python -m compileall .\stark .\tests .\examples .\competition"
