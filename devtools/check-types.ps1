# .\devtools\check-types.ps1
#
# Run Pyright/Pylance-style type analysis and write readable reports into
# devtools/tmp, which is local-only and ignored by git.
#
# The default scope covers package code, shipped examples, competition scripts,
# and tests. Use -Full to include benchmarks as well.
#
# Examples:
#   .\devtools\check-types.ps1
#   .\devtools\check-types.ps1 -Full
#   .\devtools\check-types.ps1 -Json
#   .\devtools\check-types.ps1 -Output .\devtools\tmp\pyright-public.txt

[CmdletBinding()]
param(
    [string]$Output = ".\devtools\tmp\pyright-report.txt",
    [switch]$Full,
    [switch]$Json
)

$ErrorActionPreference = "Stop"

$gitRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -eq 0 -and $gitRoot) {
    Set-Location $gitRoot
}

$repoRoot = (Resolve-Path -LiteralPath ".").Path
$pyright = Get-Command pyright -ErrorAction SilentlyContinue
if ($null -eq $pyright) {
    $venvPyright = ".\.venv\Scripts\pyright.exe"
    if (Test-Path -LiteralPath $venvPyright) {
        $pyrightCommand = (Resolve-Path -LiteralPath $venvPyright).Path
    }
    else {
        throw "Pyright was not found. Install it with: python -m pip install pyright"
    }
}
else {
    $pyrightCommand = $pyright.Source
}

$outputPath = $Output
if ($Json -and $outputPath.EndsWith(".txt", [System.StringComparison]::OrdinalIgnoreCase)) {
    $outputPath = $outputPath.Substring(0, $outputPath.Length - 4) + ".json"
}

$outputDirectory = Split-Path -Path $outputPath -Parent
if (-not [string]::IsNullOrWhiteSpace($outputDirectory)) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}

$jsonPath = if ($outputPath.EndsWith(".json", [System.StringComparison]::OrdinalIgnoreCase)) {
    $outputPath
}
else {
    $outputPath.Substring(0, $outputPath.Length - [System.IO.Path]::GetExtension($outputPath).Length) + ".json"
}

$scopeName = if ($Full) { "Full" } else { "Default" }
$include = if ($Full) {
    @("stark", "examples", "competition", "tests", "benchmarks")
}
else {
    @("stark", "examples", "competition", "tests")
}

function Format-LocalPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = try {
        (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    }
    catch {
        $Path
    }

    if ($resolved.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $resolved.Substring($repoRoot.Length).TrimStart("\", "/")
    }

    return $resolved
}

function Format-DiagnosticMessage {
    param([Parameter(Mandatory = $true)][string]$Message)

    $clean = $Message
    $clean = $clean.Replace([string][char]0x00a0, " ")
    $clean = $clean.Replace(([string][char]0x00c2) + ([string][char]0x00a0), " ")
    $clean = $clean.Replace(([string][char]0x252c) + ([string][char]0x00e1), " ")
    $clean = $clean.Replace("`r", "")
    return ($clean -split "`n")
}

Write-Host "Running Pyright"
Write-Host "Scope:  $scopeName"
Write-Host "Roots:  $($include -join ', ')"
Write-Host "Text:   $outputPath"
Write-Host "JSON:   $jsonPath"
Write-Host ""

$arguments = @("--project", ".\pyrightconfig.json", "--outputjson") + $include
$raw = & $pyrightCommand @arguments 2>&1
$exitCode = $LASTEXITCODE
$rawText = ($raw -join "`n")

Set-Content -LiteralPath $jsonPath -Value $rawText -Encoding utf8

try {
    $report = $rawText | ConvertFrom-Json
}
catch {
    Set-Content -LiteralPath $outputPath -Value $rawText -Encoding utf8
    Write-Host "Pyright did not return parseable JSON. Raw output written to:"
    Write-Host "  $outputPath"
    exit $exitCode
}

if ($Json) {
    Write-Host "JSON report written to:"
    Write-Host "  $jsonPath"
    exit $exitCode
}

$diagnostics = @($report.generalDiagnostics)
$errors = @($diagnostics | Where-Object { $_.severity -eq "error" })
$warnings = @($diagnostics | Where-Object { $_.severity -eq "warning" })
$information = @($diagnostics | Where-Object { $_.severity -eq "information" })
$files = @($diagnostics | Group-Object file)

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("Pyright type report")
$lines.Add("===================")
$lines.Add("")
$lines.Add("Scope: $scopeName")
$lines.Add("Roots: $($include -join ', ')")
$lines.Add("Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("Summary")
$lines.Add("-------")
$lines.Add("errors:       $($errors.Count)")
$lines.Add("warnings:     $($warnings.Count)")
$lines.Add("information:  $($information.Count)")
$lines.Add("files:        $($files.Count)")
$lines.Add("")

if ($diagnostics.Count -eq 0) {
    $lines.Add("No Pyright diagnostics found.")
}
else {
    $lines.Add("Diagnostics")
    $lines.Add("-----------")

    foreach ($fileGroup in ($files | Sort-Object Name)) {
        $relativePath = Format-LocalPath -Path $fileGroup.Name
        $lines.Add("")
        $lines.Add($relativePath)

        $orderedDiagnostics = @($fileGroup.Group | Sort-Object `
            @{ Expression = { $_.range.start.line } }, `
            @{ Expression = { $_.range.start.character } }, `
            @{ Expression = { $_.severity } })

        foreach ($diagnostic in $orderedDiagnostics) {
            $line = [int]$diagnostic.range.start.line + 1
            $column = [int]$diagnostic.range.start.character + 1
            $rule = if ([string]::IsNullOrWhiteSpace($diagnostic.rule)) {
                ""
            }
            else {
                " [$($diagnostic.rule)]"
            }

            $messageLines = @(Format-DiagnosticMessage -Message $diagnostic.message)
            $firstMessage = if ($messageLines.Count -gt 0) { $messageLines[0] } else { "" }
            $lines.Add(("  {0}:{1} {2}{3}: {4}" -f $line, $column, $diagnostic.severity, $rule, $firstMessage))

            foreach ($messageLine in @($messageLines | Select-Object -Skip 1)) {
                if ([string]::IsNullOrWhiteSpace($messageLine)) {
                    continue
                }
                $lines.Add("      $messageLine")
            }
        }
    }
}

Set-Content -LiteralPath $outputPath -Value $lines -Encoding utf8

Write-Host "Readable report written to:"
Write-Host "  $outputPath"
Write-Host ""
Write-Host ("Summary: {0} errors, {1} warnings, {2} information across {3} files." -f `
    $errors.Count, $warnings.Count, $information.Count, $files.Count)

exit $exitCode
