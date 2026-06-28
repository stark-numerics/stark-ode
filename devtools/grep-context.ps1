# .\devtools\grep-context.ps1
#
# Search text across the repo and print line context.
# Defaults include stark/, tests/, docs/, examples/, competition/, benchmarks/,
# README.md, and pyproject.toml.
#
# Examples:
#   .\devtools\grep-context.ps1 -Pattern "CarrierBound"
#   .\devtools\grep-context.ps1 -Pattern "CarrierBound" -ContextLines 2
#   .\devtools\grep-context.ps1 -Pattern "stark.schemes" -AllowPartialMatches
#   .\devtools\grep-context.ps1 -Pattern "class .*Bound" -Regex

param(
    [Parameter(Mandatory = $true)]
    [string]$Pattern,

    [string[]]$Roots = @(),

    [string[]]$Extensions = @(),

    [int]$ContextLines = 1,

    [switch]$AllowPartialMatches,

    [switch]$Regex,

    [switch]$IgnoreCase,

    [switch]$DottedToken
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_common.ps1"

$matcher = New-DevtoolsSearchRegex `
    -Pattern $Pattern `
    -AllowPartialMatches:$AllowPartialMatches `
    -Regex:$Regex `
    -DottedToken:$DottedToken `
    -IgnoreCase:$IgnoreCase

$matchesByFile = @(
    Find-DevtoolsRegexMatches `
    -Matcher $matcher `
    -Roots $Roots `
    -Extensions $Extensions `
    -AllMatches
)

if ($matchesByFile.Count -eq 0) {
    Write-Host "No matches found for: $Pattern"
    exit 0
}

Write-DevtoolsMatchReport `
    -MatchesByFile $matchesByFile `
    -ContextLines $ContextLines `
    -Header "Matches for '$Pattern':"

$total = 0
foreach ($fileMatch in $matchesByFile) {
    $total += $fileMatch.Hits.Count
}
Write-Host "Found $total match(es) in $($matchesByFile.Count) file(s)."
