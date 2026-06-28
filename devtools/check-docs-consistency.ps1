# .\devtools\check-docs-consistency.ps1
#
# Static smoke-checks for README/docs drift.
# It checks:
#   - advertised `python -m ...` commands point to importable repo modules;
#   - simple `import stark...` and `from stark... import ...` snippets point to
#     modules/names that exist in the source tree;
#   - local markdown links point to existing files;
#   - optional forbidden terms are absent.
#
# This is deliberately a warning-oriented docs guard, not a full doctest runner.
# Use -FailOnWarning in CI once the docs are clean.
#
# Examples:
#   .\devtools\check-docs-consistency.ps1
#   .\devtools\check-docs-consistency.ps1 -FailOnWarning
#   .\devtools\check-docs-consistency.ps1 -ForbiddenTerm StarkIVP,Marcher

param(
    [string[]]$Roots = @(".\README.md", ".\docs"),

    [string[]]$Extensions = @(".md", ".rst"),

    [string[]]$ForbiddenTerm = @(),

    [switch]$FailOnWarning
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_common.ps1"

$warnings = @()

function Add-DocsWarning {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int]$Line,
        [Parameter(Mandatory = $true)][string]$Message,
        [string]$Text = ""
    )

    $script:warnings += [pscustomobject]@{
        Path    = $Path
        Line    = $Line
        Message = $Message
        Text    = $Text
    }
}

function Split-ImportNames {
    param([Parameter(Mandatory = $true)][string]$Text)

    $cleaned = $Text.Trim()
    $cleaned = $cleaned.TrimStart("(").TrimEnd(")")
    $parts = $cleaned -split ","
    $names = @()

    foreach ($part in $parts) {
        $name = ($part -replace "\s+as\s+[A-Za-z_][A-Za-z0-9_]*", "").Trim()
        if ($name -eq "" -or $name -eq "*") {
            continue
        }
        if ($name -match "^([A-Za-z_][A-Za-z0-9_]*)$") {
            $names += $Matches[1]
        }
    }

    return $names
}

function Test-DocsLocalLinkExists {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$LinkTarget
    )

    if ($LinkTarget -match "^[a-zA-Z][a-zA-Z0-9+.-]*:") {
        return $true
    }

    if ($LinkTarget.StartsWith("#")) {
        return $true
    }

    $withoutAnchor = ($LinkTarget -split "#", 2)[0]
    if ([string]::IsNullOrWhiteSpace($withoutAnchor)) {
        return $true
    }

    if ($withoutAnchor -match "^mailto:") {
        return $true
    }

    $sourceDir = Split-Path $SourcePath -Parent
    $candidate = Join-Path $sourceDir $withoutAnchor
    return (Test-Path -LiteralPath $candidate)
}

$files = @(Get-DevtoolsTextFiles -Roots $Roots -Extensions $Extensions)

foreach ($file in $files) {
    $path = $file.FullName
    $relativePath = Format-DevtoolsRelativePath -Path $path
    $lines = [System.IO.File]::ReadAllLines($path)

    for ($i = 0; $i -lt $lines.Count; $i++) {
        $lineNumber = $i + 1
        $line = $lines[$i]

        foreach ($term in $ForbiddenTerm) {
            if ([string]::IsNullOrWhiteSpace($term)) {
                continue
            }
            if ($line.Contains($term)) {
                Add-DocsWarning `
                    -Path $relativePath `
                    -Line $lineNumber `
                    -Message "Forbidden/stale term appears: $term" `
                    -Text $line
            }
        }

        if ($line -match "(?:^|\s)(?:python|py)\s+-m\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)") {
            $module = $Matches[1]
            if (-not (Test-DevtoolsPythonModuleExists -Module $module)) {
                Add-DocsWarning `
                    -Path $relativePath `
                    -Line $lineNumber `
                    -Message "Advertised python -m module does not exist: $module" `
                    -Text $line
            }
        }

        if ($line -match "^\s*import\s+(stark(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\b") {
            $module = $Matches[1]
            if (-not (Test-DevtoolsPythonModuleExists -Module $module)) {
                Add-DocsWarning `
                    -Path $relativePath `
                    -Line $lineNumber `
                    -Message "Imported module does not exist: $module" `
                    -Text $line
            }
        }

        if ($line -match "^\s*from\s+(stark(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import\s+(.+)$") {
            $module = $Matches[1]
            $nameText = $Matches[2]

            if (-not (Test-DevtoolsPythonModuleExists -Module $module)) {
                Add-DocsWarning `
                    -Path $relativePath `
                    -Line $lineNumber `
                    -Message "Imported-from module does not exist: $module" `
                    -Text $line
            }
            else {
                $names = Split-ImportNames -Text $nameText
                foreach ($name in $names) {
                    if (-not (Test-DevtoolsNameInModule -Module $module -Name $name)) {
                        Add-DocsWarning `
                            -Path $relativePath `
                            -Line $lineNumber `
                            -Message "Name '$name' is not visibly exported or defined by $module" `
                            -Text $line
                    }
                }
            }
        }

        # Markdown links: [text](target). This intentionally ignores reference-style links.
        $linkMatches = [regex]::Matches($line, "\[[^\]]+\]\(([^)]+)\)")
        foreach ($linkMatch in $linkMatches) {
            $target = $linkMatch.Groups[1].Value.Trim()
            if ($target -eq "") {
                continue
            }
            if (-not (Test-DocsLocalLinkExists -SourcePath $path -LinkTarget $target)) {
                Add-DocsWarning `
                    -Path $relativePath `
                    -Line $lineNumber `
                    -Message "Local documentation link target does not exist: $target" `
                    -Text $line
            }
        }
    }
}

if ($warnings.Count -eq 0) {
    Write-Host "Docs consistency check passed."
    exit 0
}

Write-Host ""
Write-Host "Docs consistency warnings:"
Write-Host ""

foreach ($warning in $warnings) {
    Write-Host "$($warning.Path):$($warning.Line)"
    Write-Host "  $($warning.Message)"
    if (-not [string]::IsNullOrWhiteSpace($warning.Text)) {
        Write-Host "  $($warning.Text.Trim())"
    }
    Write-Host ""
}

Write-Host "Found $($warnings.Count) docs consistency warning(s)."

if ($FailOnWarning) {
    exit 1
}

exit 0
