# STARK devtools

These scripts are intended for broad repository maintenance, especially package
renames and public import-path cleanup. Run them from the repository root.

## Shared scanning rules

Most scripts dot-source `devtools/_common.ps1`. By default they scan:

- `stark/`
- `tests/`
- `docs/`
- `examples/`
- `competition/`
- `benchmarks/`
- `README.md`
- `pyproject.toml`

Generated/cache directories such as `.git/`, `.venv/`, `.pytest_cache/`,
`__pycache__/`, `.asv/`, and `.pip-tmp/` are ignored.

## Rename workflow

Use dry runs first:

```powershell
.\devtools\rename-python-package.ps1 `
    -Before .\stark\schemes `
    -After .\stark\methods\schemes `
    -DryRun
```

Then apply:

```powershell
.\devtools\rename-python-package.ps1 `
    -Before .\stark\schemes `
    -After .\stark\methods\schemes `
    -Yes
```

The rename scripts perform a post-check. They fail if the source path still
exists, the destination path was not created, or references to the old dotted
module/package path remain in the scanned roots.

For simple class/function/name renames:

```powershell
.\devtools\rename-symbol.ps1 -Before OldName -After NewName -Yes
```

For dotted textual moves without moving files:

```powershell
.\devtools\rename-symbol.ps1 `
    -Before stark.schemes `
    -After stark.methods.schemes `
    -DottedToken `
    -Yes
```

## Docs drift guard

```powershell
.\devtools\check-docs-consistency.ps1
```

This warns about stale README/docs references, including missing `python -m`
modules, missing simple `stark` imports, missing local markdown links, and
optional forbidden terms.

To turn warnings into a failing check:

```powershell
.\devtools\check-docs-consistency.ps1 -FailOnWarning
```

To look for known stale public API names:

```powershell
.\devtools\check-docs-consistency.ps1 -ForbiddenTerm StarkIVP,Marcher
```

## Type warning reports

Pylance uses Pyright, so the command-line Pyright report is the closest
repeatable way to inspect IDE-style warnings. The default check covers package
code, shipped examples, competition scripts, and tests: `stark`, `examples`,
`competition`, and `tests`.

```powershell
.\devtools\check-types.ps1
```

The report is written to `devtools/tmp/pyright-report.txt`. For a deeper
internal report that also includes benchmarks:

```powershell
.\devtools\check-types.ps1 -Full
```

For a machine-readable report:

```powershell
.\devtools\check-types.ps1 -Json
```

## Broad local sweep

```powershell
.\devtools\run-everything.ps1
```

This runs the default tests, docs consistency guard, current example groups,
competition report smoke, and maintained runnable benchmark scripts. Use the
skip switches when you only need a slice:

```powershell
.\devtools\run-everything.ps1 -SkipBenchmarks
.\devtools\run-everything.ps1 -SkipCompetition -SkipBenchmarks
```
