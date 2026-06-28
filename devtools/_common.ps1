# devtools/_common.ps1
#
# Shared helpers for STARK maintenance scripts.
# Dot-source this file from scripts in devtools/:
#   . "$PSScriptRoot\_common.ps1"

Set-StrictMode -Version Latest

$script:DevtoolsDefaultRoots = @(
    ".\stark",
    ".\tests",
    ".\docs",
    ".\examples",
    ".\competition",
    ".\benchmarks",
    ".\README.md",
    ".\pyproject.toml"
)

$script:DevtoolsDefaultTextExtensions = @(
    ".py",
    ".pyi",
    ".md",
    ".rst",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
    ".ps1"
)

$script:DevtoolsIgnoredPathFragments = @(
    "/.git/",
    "/.hg/",
    "/.svn/",
    "/.venv/",
    "/venv/",
    "/.pytest_cache/",
    "/.mypy_cache/",
    "/.ruff_cache/",
    "/.asv/",
    "/.jupyter-runtime/",
    "/.pip-tmp/",
    "/__pycache__/"
)

function Test-DevtoolsIgnoredPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $normal = ($Path -replace '\\', '/').ToLowerInvariant()
    foreach ($fragment in $script:DevtoolsIgnoredPathFragments) {
        if ($normal.Contains($fragment)) {
            return $true
        }
    }

    return $false
}

function Get-DevtoolsTextFiles {
    param(
        [string[]]$Roots = @(),
        [string[]]$Extensions = @()
    )

    if ($null -eq $Roots -or $Roots.Count -eq 0) {
        $Roots = $script:DevtoolsDefaultRoots
    }

    if ($null -eq $Extensions -or $Extensions.Count -eq 0) {
        $Extensions = $script:DevtoolsDefaultTextExtensions
    }

    $extensionSet = @{}
    foreach ($extension in $Extensions) {
        $normalized = $extension
        if (-not $normalized.StartsWith(".")) {
            $normalized = ".$normalized"
        }
        $extensionSet[$normalized.ToLowerInvariant()] = $true
    }

    $seen = @{}
    $files = @()

    foreach ($root in $Roots) {
        if (-not (Test-Path -LiteralPath $root)) {
            continue
        }

        $item = Get-Item -LiteralPath $root

        if ($item.PSIsContainer) {
            Get-ChildItem -LiteralPath $item.FullName -Recurse -File | ForEach-Object {
                if (Test-DevtoolsIgnoredPath -Path $_.FullName) {
                    return
                }

                $extension = $_.Extension.ToLowerInvariant()
                if (-not $extensionSet.ContainsKey($extension)) {
                    return
                }

                if (-not $seen.ContainsKey($_.FullName)) {
                    $seen[$_.FullName] = $true
                    $files += $_
                }
            }
        }
        else {
            if (Test-DevtoolsIgnoredPath -Path $item.FullName) {
                continue
            }

            $extension = $item.Extension.ToLowerInvariant()
            if (-not $extensionSet.ContainsKey($extension)) {
                continue
            }

            if (-not $seen.ContainsKey($item.FullName)) {
                $seen[$item.FullName] = $true
                $files += $item
            }
        }
    }

    return ($files | Sort-Object FullName)
}

function New-DevtoolsSearchRegex {
    param(
        [Parameter(Mandatory = $true)][string]$Pattern,
        [switch]$AllowPartialMatches,
        [switch]$Regex,
        [switch]$DottedToken,
        [switch]$IgnoreCase
    )

    if ($Regex) {
        $regexPattern = $Pattern
    }
    elseif ($AllowPartialMatches) {
        $regexPattern = [regex]::Escape($Pattern)
    }
    elseif ($DottedToken) {
        $regexPattern = "(?<![A-Za-z0-9_\.])$([regex]::Escape($Pattern))(?![A-Za-z0-9_\.])"
    }
    else {
        $regexPattern = "(?<![A-Za-z0-9_])$([regex]::Escape($Pattern))(?![A-Za-z0-9_])"
    }

    $options = [System.Text.RegularExpressions.RegexOptions]::None
    if ($IgnoreCase) {
        $options = $options -bor [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    }

    return [System.Text.RegularExpressions.Regex]::new($regexPattern, $options)
}

function Test-DevtoolsImportLikeLine {
    param([Parameter(Mandatory = $true)][string]$Line)

    return (
        $Line -match '^\s*(from|import)\s+' -or
        $Line -match '^\s*__all__\s*=' -or
        $Line -match '^\s*".*"\s*,?\s*$' -or
        $Line -match "^\s*'.*'\s*,?\s*$" -or
        $Line -match '^\s*(python|py)\s+-m\s+'
    )
}

function Find-DevtoolsRegexMatches {
    param(
        [Parameter(Mandatory = $true)][System.Text.RegularExpressions.Regex]$Matcher,
        [string[]]$Roots = @(),
        [string[]]$Extensions = @(),
        [switch]$AllMatches
    )

    $matchesByFile = @()
    $files = @(Get-DevtoolsTextFiles -Roots $Roots -Extensions $Extensions)

    foreach ($file in $files) {
        $path = $file.FullName
        $text = [System.IO.File]::ReadAllText($path)

        if (-not $Matcher.IsMatch($text)) {
            continue
        }

        $lines = $text -split "`r?`n"
        $lineHits = @()

        for ($i = 0; $i -lt $lines.Count; $i++) {
            $line = $lines[$i]
            if (-not $Matcher.IsMatch($line)) {
                continue
            }

            if ($AllMatches -or (Test-DevtoolsImportLikeLine -Line $line)) {
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

    return $matchesByFile
}

function Format-DevtoolsRelativePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    try {
        return (Resolve-Path -LiteralPath $Path -Relative)
    }
    catch {
        return $Path
    }
}

function Write-DevtoolsMatchReport {
    param(
        [Parameter(Mandatory = $true)]$MatchesByFile,
        [int]$ContextLines = 0,
        [string]$Header = "Matches found:"
    )

    $MatchesByFile = @($MatchesByFile)

    if ($MatchesByFile.Count -eq 0) {
        return
    }

    if (-not [string]::IsNullOrWhiteSpace($Header)) {
        Write-Host ""
        Write-Host $Header
        Write-Host ""
    }

    foreach ($fileMatch in $MatchesByFile) {
        $relativePath = Format-DevtoolsRelativePath -Path $fileMatch.Path
        Write-Host $relativePath

        foreach ($lineNumber in $fileMatch.Hits) {
            $start = [Math]::Max(1, $lineNumber - $ContextLines)
            $end = [Math]::Min($fileMatch.Lines.Count, $lineNumber + $ContextLines)

            for ($n = $start; $n -le $end; $n++) {
                $marker = if ($n -eq $lineNumber) { ">" } else { " " }
                $lineText = $fileMatch.Lines[$n - 1]
                Write-Host ("  {0} {1,5}: {2}" -f $marker, $n, $lineText)
            }

            Write-Host ""
        }
    }
}

function Write-DevtoolsUtf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )

    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function Get-DevtoolsModuleNameFromPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$PackageRoot = ".\stark"
    )

    $resolvedPath = if (Test-Path -LiteralPath $Path) {
        (Resolve-Path -LiteralPath $Path).Path
    }
    else {
        $parent = Split-Path $Path -Parent
        $leaf = Split-Path $Path -Leaf
        if ([string]::IsNullOrWhiteSpace($parent)) {
            $parent = "."
        }
        Join-Path (Resolve-Path -LiteralPath $parent).Path $leaf
    }

    $resolvedRoot = (Resolve-Path -LiteralPath $PackageRoot).Path
    if (-not $resolvedPath.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path '$Path' is not under package root '$PackageRoot'."
    }

    $relative = $resolvedPath.Substring($resolvedRoot.Length).TrimStart("\", "/")
    $rootName = Split-Path $resolvedRoot -Leaf

    if ($relative.EndsWith(".py", [System.StringComparison]::OrdinalIgnoreCase)) {
        if ($relative -eq "__init__.py" -or $relative.EndsWith("\__init__.py") -or $relative.EndsWith("/__init__.py")) {
            $packageRelative = (Split-Path $relative -Parent) -replace "[\\/]", "."
            if ([string]::IsNullOrWhiteSpace($packageRelative)) {
                return $rootName
            }
            return "$rootName.$packageRelative"
        }

        $withoutExtension = $relative.Substring(0, $relative.Length - 3)
        $moduleSuffix = $withoutExtension -replace "[\\/]", "."
        return "$rootName.$moduleSuffix"
    }

    $packageSuffix = $relative.TrimEnd("\", "/") -replace "[\\/]", "."
    if ([string]::IsNullOrWhiteSpace($packageSuffix)) {
        return $rootName
    }

    return "$rootName.$packageSuffix"
}

function Test-DevtoolsPythonModuleExists {
    param([Parameter(Mandatory = $true)][string]$Module)

    $relative = $Module -replace '\.', [System.IO.Path]::DirectorySeparatorChar
    return (
        (Test-Path -LiteralPath "$relative.py") -or
        (Test-Path -LiteralPath (Join-Path $relative "__init__.py"))
    )
}

function Get-DevtoolsPythonModuleFile {
    param([Parameter(Mandatory = $true)][string]$Module)

    $relative = $Module -replace '\.', [System.IO.Path]::DirectorySeparatorChar
    $moduleFile = "$relative.py"
    $packageFile = Join-Path $relative "__init__.py"

    if (Test-Path -LiteralPath $moduleFile) {
        return (Resolve-Path -LiteralPath $moduleFile).Path
    }

    if (Test-Path -LiteralPath $packageFile) {
        return (Resolve-Path -LiteralPath $packageFile).Path
    }

    return $null
}

function Test-DevtoolsNameInModule {
    param(
        [Parameter(Mandatory = $true)][string]$Module,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if (Test-DevtoolsPythonModuleExists -Module "$Module.$Name") {
        return $true
    }

    $moduleFile = Get-DevtoolsPythonModuleFile -Module $Module
    if ($null -eq $moduleFile) {
        return $false
    }

    $text = [System.IO.File]::ReadAllText($moduleFile)
    $escapedName = [regex]::Escape($Name)

    $patterns = @(
        "(?m)^\s*(class|def)\s+$escapedName\b",
        "(?m)^\s*$escapedName\s*=",
        "(?m)^\s*from\s+[^\r\n]+\s+import\s+[^\r\n]*\b$escapedName\b",
        "(?m)^\s*import\s+[^\r\n]+\s+as\s+$escapedName\b",
        "['`"]$escapedName['`"]"
    )

    foreach ($pattern in $patterns) {
        if ([regex]::IsMatch($text, $pattern)) {
            return $true
        }
    }

    return $false
}
