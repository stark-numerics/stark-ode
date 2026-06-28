# .\devtools\rename-symbol.ps1
#
# Rename a symbol/text token across the repo and verify that the old token is gone.
# Defaults include stark/, tests/, docs/, examples/, competition/, benchmarks/,
# README.md, and pyproject.toml.
#
# Examples:
#   .\devtools\rename-symbol.ps1 -Before BoundDerivative -After DerivativeBinding
#   .\devtools\rename-symbol.ps1 -Before StarkIVP -After SystemIVP -Yes
#   .\devtools\rename-symbol.ps1 -Before "stark.schemes" -After "stark.methods.schemes" -DottedToken
#   .\devtools\rename-symbol.ps1 -Before "old text" -After "new text" -AllowPartialMatches

param(
    [Parameter(Mandatory = $true)]
    [string]$Before,

    [Parameter(Mandatory = $true)]
    [string]$After,

    [string[]]$Roots = @(),

    [string[]]$Extensions = @(),

    [int]$ContextLines = 1,

    [switch]$AllowPartialMatches,

    [switch]$DottedToken,

    [switch]$Yes,

    [switch]$DryRun,

    [switch]$NoVerify
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_common.ps1"

if ($Before -eq $After) {
    throw "Before and After are identical."
}

$looksDotted = $DottedToken -or $Before.Contains(".")
$matcher = New-DevtoolsSearchRegex `
    -Pattern $Before `
    -AllowPartialMatches:$AllowPartialMatches `
    -DottedToken:$looksDotted

$matchesByFile = @(
    Find-DevtoolsRegexMatches `
    -Matcher $matcher `
    -Roots $Roots `
    -Extensions $Extensions `
    -AllMatches
)

if ($matchesByFile.Count -eq 0) {
    Write-Host "No matches found for '$Before'."
    exit 0
}

Write-DevtoolsMatchReport `
    -MatchesByFile $matchesByFile `
    -ContextLines $ContextLines `
    -Header "Found matches for '$Before':"

Write-Host "Replace '$Before' with '$After' in $($matchesByFile.Count) file(s)?"

if ($DryRun) {
    Write-Host "Dry run only. No changes made."
    exit 0
}

if (-not $Yes) {
    $response = Read-Host "Type y to replace, anything else to exit"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "No changes made."
        exit 0
    }
}

foreach ($fileMatch in $matchesByFile) {
    $path = $fileMatch.Path
    $text = [System.IO.File]::ReadAllText($path)
    $updated = $matcher.Replace($text, $After)

    if ($updated -ne $text) {
        Write-DevtoolsUtf8NoBom -Path $path -Text $updated
        Write-Host "updated $(Format-DevtoolsRelativePath -Path $path)"
    }
}

if (-not $NoVerify) {
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
            -Header "Post-check failed: '$Before' still appears here:"

        throw "Rename did not eliminate all matches for '$Before'. Review manually or rerun with -NoVerify if the remaining text is intentional."
    }

    Write-Host "Post-check passed: no matches remain for '$Before'."
}

Write-Host ""
Write-Host "Done. Review with:"
Write-Host "  git diff"
