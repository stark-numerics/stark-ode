param(
    [string[]]$Roots = @(".\stark", ".\benchmarks", ".\tests")
)

. "$PSScriptRoot\_common.ps1"

$designFiles = @(
    Get-DevtoolsTextFiles -Roots $Roots -Extensions @(".md") |
        Where-Object { $_.Name -eq "DESIGN.md" }
)

$sections = @()

foreach ($file in $designFiles) {
    $lines = [System.IO.File]::ReadAllLines($file.FullName)

    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -notmatch '^##\s+To Do\b') {
            continue
        }

        $start = $i
        $end = $lines.Count - 1

        for ($j = $i + 1; $j -lt $lines.Count; $j++) {
            if ($lines[$j] -match '^##\s+\S') {
                $end = $j - 1
                break
            }
        }

        $body = @()
        for ($lineIndex = $start + 1; $lineIndex -le $end; $lineIndex++) {
            if ([string]::IsNullOrWhiteSpace($lines[$lineIndex])) {
                continue
            }
            $body += $lines[$lineIndex]
        }

        $sections += [pscustomobject]@{
            Path      = $file.FullName
            Line      = $start + 1
            BodyLines = $body
        }
    }
}

if ($sections.Count -eq 0) {
    Write-Host "No DESIGN.md To Do sections found."
    exit 0
}

foreach ($section in $sections) {
    $relativePath = Format-DevtoolsRelativePath -Path $section.Path
    Write-Host "${relativePath}:$($section.Line)"

    if ($section.BodyLines.Count -eq 0) {
        Write-Host "  (empty)"
    }
    else {
        foreach ($line in $section.BodyLines) {
            Write-Host "  $line"
        }
    }

    Write-Host ""
}
