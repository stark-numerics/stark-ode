# Benchmarks

This directory contains two different kinds of benchmark material:

- manual smoke benchmarks for quick local before/after checks;
- ASV benchmarks for repeatable performance history across commits.

Neither kind of benchmark is a CI gate by default. Timings are local,
machine-specific evidence for contributors.

## Which Benchmark Should I Run?

Use the manual smoke benchmarks while you are actively editing code and want a
fast answer from the current dirty working tree:

- did this refactor obviously slow a hot path?
- should I pause before committing?
- did a small support-layer change affect scheme, resolvent, or inverter calls?

Use ASV when you want durable history across commits:

- did this branch get faster or slower than `main`?
- did setup improve but steady solves regress?
- did an optimization still help after several related commits?
- which commit introduced a performance change?

The short version:

- dirty tree, immediate feedback: run a manual smoke benchmark;
- committed code, historical comparison: run ASV.

## Manual Smoke Benchmarks

Run these from the `stark-ode` directory. Use the virtual environment explicitly
when working in this repository:

```powershell
& ..\.venv\Scripts\python.exe -m benchmarks.schemes.bench_scheme_refactor
& ..\.venv\Scripts\python.exe -m benchmarks.inverters.bench_defect
& ..\.venv\Scripts\python.exe -m benchmarks.inverters.bench_jacobi
& ..\.venv\Scripts\python.exe -m benchmarks.inverters.bench_richardson
```

More stable local runs:

```powershell
& ..\.venv\Scripts\python.exe -m benchmarks.schemes.bench_scheme_refactor --warmup 3 --repeat 10
& ..\.venv\Scripts\python.exe -m benchmarks.inverters.bench_defect
& ..\.venv\Scripts\python.exe -m benchmarks.inverters.bench_jacobi
& ..\.venv\Scripts\python.exe -m benchmarks.inverters.bench_richardson
```

Save a local baseline before risky work:

```powershell
& ..\.venv\Scripts\python.exe -m benchmarks.schemes.bench_scheme_refactor --warmup 3 --repeat 10 --save-baseline pre-change
```

Compare against that baseline later:

```powershell
& ..\.venv\Scripts\python.exe -m benchmarks.schemes.bench_scheme_refactor --warmup 3 --repeat 10 --compare-baseline pre-change
```

Manual baseline files are written under:

- `benchmarks/schemes/results/`

These directories are git-ignored. Commit only deliberately curated reference
baselines.

## ASV Benchmarks

ASV is the long-term benchmark layer. Use it after committing the code you want
to compare.

Install ASV support:

```powershell
& ..\.venv\Scripts\python.exe -m pip install -e ".[asv]"
```

Check that ASV can discover the suite against the current virtual environment:

```powershell
& ..\.venv\Scripts\python.exe -m asv check -E existing:..\.venv\Scripts\python.exe
```

Run a quick ASV smoke benchmark against the current environment:

```powershell
& ..\.venv\Scripts\python.exe -m asv run --quick -E existing:..\.venv\Scripts\python.exe --bench TimeAlgebraist
```

After committing on a feature branch, compare the current branch against
`main`:

```powershell
& ..\.venv\Scripts\python.exe -m asv run main..HEAD
```

This range only selects commits when `HEAD` is ahead of `main`. Check that with:

```powershell
git log --oneline main..HEAD
```

If that prints nothing, ASV will report `No commit hashes selected`.

When working directly on `main`, compare explicit commits instead. For example,
after making one new commit:

```powershell
& ..\.venv\Scripts\python.exe -m asv run HEAD~1..HEAD
```

For a named baseline commit:

```powershell
& ..\.venv\Scripts\python.exe -m asv run OLD_COMMIT..HEAD
```

Run only a targeted ASV benchmark family:

```powershell
& ..\.venv\Scripts\python.exe -m asv run main..HEAD --bench TimeAlgebraist
& ..\.venv\Scripts\python.exe -m asv run main..HEAD --bench TimeFPUTExplicit
& ..\.venv\Scripts\python.exe -m asv run main..HEAD --bench TimeInterface
```

Use the same explicit range rule for targeted runs:

```powershell
& ..\.venv\Scripts\python.exe -m asv run HEAD~1..HEAD --bench TimeAlgebraist
& ..\.venv\Scripts\python.exe -m asv run OLD_COMMIT..HEAD --bench TimeAlgebraist
```

Publish and preview local HTML results:

```powershell
& ..\.venv\Scripts\python.exe -m asv publish
& ..\.venv\Scripts\python.exe -m asv preview
```

ASV stores generated environments, raw results, and HTML output under `.asv/`.

## Current ASV Coverage

The current ASV suite covers:

- FPUT explicit solves;
- interface-layer costs;
- Algebraist combination and setup costs.

The manual scheme, resolvent, and inverter smoke benchmarks are not yet ASV
classes. They are useful immediate checks, but they do not yet create durable
ASV history.

## Useful Habit

Before a performance-sensitive refactor:

1. commit the current clean state;
2. save manual baselines for the hot paths you expect to touch;
3. optionally run targeted ASV on the current commit;
4. make the refactor;
5. rerun the manual comparisons while still in the dirty tree;
6. commit once tests and smoke benchmarks are acceptable;
7. run ASV over `main..HEAD` or over the relevant commit range.
