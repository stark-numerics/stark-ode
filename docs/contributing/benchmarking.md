# Benchmarking and comparison reports

This page is for maintainers and users reading STARK timing output.

STARK has two kinds of performance material:

- comparison reports under `competition/`, intended to be readable solver comparisons;
- ASV benchmarks under `benchmarks/`, intended for contributor regression tracking.

## Comparison reports

Comparison reports show named solver configurations on named problems. They may compare STARK with SciPy, Diffrax, or other libraries when optional dependencies are available.

Reports distinguish three timing views.

### Preparation time

Preparation time includes setup and warmup. This is where compilation, tracing, sparse setup, or other one-time work may appear.

### Warm run time

Warm run time measures repeated solves after setup and warmup. This is useful for throughput when the same solver shape is reused many times.

### Total time

Total time combines preparation and one measured solve. This is the most honest headline for one-off solves.

JIT-based and GPU-backed libraries may move substantial work into preparation. Total timing makes that cost visible.

## ASV benchmarks

Install benchmark tooling with:

```powershell
python -m pip install -e ".[asv]"
```

Check benchmark discovery:

```powershell
python -m asv check
```

Run a quick smoke benchmark in the current environment:

```powershell
python -m asv run --quick -E existing:.venv\Scripts\python.exe
```

Full ASV runs are useful after committing the code you want to benchmark:

```powershell
python -m asv run
```

Use comparison reports to explain solver behaviour. Use ASV to catch performance regressions.
