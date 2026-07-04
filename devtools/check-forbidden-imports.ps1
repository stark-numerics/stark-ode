# .\devtools\check-forbidden-imports.ps1
#
# Fail if forbidden import/text patterns are found.
# Defaults include stark/, tests/, docs/, examples/, competition/, benchmarks/,
# README.md, and pyproject.toml.
#
# Examples:
#   .\devtools\check-forbidden-imports.ps1
#   .\devtools\check-forbidden-imports.ps1 -Rule "stark.schemes"
#   .\devtools\check-forbidden-imports.ps1 -Rule "BoundDynamics","BoundLinearizer"
#   .\devtools\check-forbidden-imports.ps1 -RulesFile .\devtools\forbidden-imports.txt
#   .\devtools\check-forbidden-imports.ps1 -AllMatches

param(
    [string[]]$Rule = @(),

    [string]$RulesFile = "",

    [string[]]$Roots = @(),

    [string[]]$Extensions = @(),

    [int]$ContextLines = 0,

    [switch]$AllMatches,

    [switch]$AllowPartialMatches
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_common.ps1"

$rules = @()

if ($Rule.Count -gt 0) {
    $rules += $Rule
}

if ($RulesFile -ne "") {
    if (-not (Test-Path -LiteralPath $RulesFile)) {
        throw "Rules file not found: $RulesFile"
    }

    Get-Content -LiteralPath $RulesFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) {
            return
        }
        $rules += $line
    }
}

if ($rules.Count -eq 0) {
    $rules = @(
        "stark.machinery",
        "stark.comparison.comparator"
    )
}

$violations = @()

foreach ($ruleText in $rules) {
    $looksDotted = $ruleText.Contains(".")
    $matcher = New-DevtoolsSearchRegex `
        -Pattern $ruleText `
        -AllowPartialMatches:$AllowPartialMatches `
        -DottedToken:$looksDotted

    $matchesByFile = @(
    Find-DevtoolsRegexMatches `
        -Matcher $matcher `
        -Roots $Roots `
        -Extensions $Extensions `
        -AllMatches:$AllMatches
)

    foreach ($fileMatch in $matchesByFile) {
        foreach ($lineNumber in $fileMatch.Hits) {
            $violations += [pscustomobject]@{
                Rule  = $ruleText
                Path  = $fileMatch.Path
                Lines = $fileMatch.Lines
                Hits  = @($lineNumber)
            }
        }
    }
}

if ($violations.Count -eq 0) {
    Write-Host "No forbidden imports found."
    exit 0
}

Write-Host ""
Write-Host "Forbidden import check failed:"
Write-Host ""

foreach ($violation in $violations) {
    $relativePath = Format-DevtoolsRelativePath -Path $violation.Path
    Write-Host $relativePath
    Write-Host "  rule: $($violation.Rule)"

    $lineNumber = $violation.Hits[0]
    $start = [Math]::Max(1, $lineNumber - $ContextLines)
    $end = [Math]::Min($violation.Lines.Count, $lineNumber + $ContextLines)

    for ($n = $start; $n -le $end; $n++) {
        $marker = if ($n -eq $lineNumber) { ">" } else { " " }
        $lineText = $violation.Lines[$n - 1]
        Write-Host ("  {0} {1,5}: {2}" -f $marker, $n, $lineText)
    }

    Write-Host ""
}

Write-Host "Found $($violations.Count) forbidden occurrence(s)."
exit 1
