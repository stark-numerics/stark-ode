# .\devtools\find-imports.ps1
#
# Find import-like references to a module or symbol across the repo.
# Defaults include stark/, tests/, docs/, examples/, competition/, benchmarks/,
# README.md, and pyproject.toml.
#
# Examples:
#   .\devtools\find-imports.ps1 -Name "stark.schemes"
#   .\devtools\find-imports.ps1 -Name "SchemeRK4"
#   .\devtools\find-imports.ps1 -Name "stark.comparison.runner" -ContextLines 1
#   .\devtools\find-imports.ps1 -Name "stark.schemes" -AllMatches

param(
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [string[]]$Roots = @(),

    [string[]]$Extensions = @(),

    [int]$ContextLines = 0,

    [switch]$AllMatches,

    [switch]$AllowPartialMatches,

    [switch]$IgnoreCase
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_common.ps1"

$looksDotted = $Name.Contains(".")
$matcher = New-DevtoolsSearchRegex `
    -Pattern $Name `
    -AllowPartialMatches:$AllowPartialMatches `
    -DottedToken:$looksDotted `
    -IgnoreCase:$IgnoreCase

$matchesByFile = @(
    Find-DevtoolsRegexMatches `
    -Matcher $matcher `
    -Roots $Roots `
    -Extensions $Extensions `
    -AllMatches:$AllMatches
)

if ($matchesByFile.Count -eq 0) {
    Write-Host "No import-like matches found for '$Name'."
    Write-Host "Try again with -AllMatches if you want every textual occurrence."
    exit 0
}

Write-DevtoolsMatchReport `
    -MatchesByFile $matchesByFile `
    -ContextLines $ContextLines `
    -Header "Import-like matches for '$Name':"

Write-Host "Found $($matchesByFile.Count) file(s)."
