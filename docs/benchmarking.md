# Benchmarking For Contributors

This page is for maintainers and contributors working on STARK performance.
Ordinary users do not need ASV to solve ODEs with STARK.

STARK has two different kinds of timing material:

- **Comparison reports** show how STARK, SciPy, and Diffrax solve the same
  example problems.
- **ASV benchmarks** track whether internal changes make STARK faster or
  slower over time.

The comparison reports live in `examples/comparison/`. They are written to be
read by users and to show each solver in its natural style.

The ASV suite lives in `benchmarks/`. It is written for repeatable timing and
performance-regression tracking during development.

Use the comparison reports when you want an illustrative example. Use ASV when
you are changing STARK internals, reviewing a performance-sensitive pull
request, or deciding whether an optimization helped.

## Installing

Install the benchmark tooling with:

```powershell
python -m pip install -e ".[asv]"
```

This installs Airspeed Velocity, NumPy, and the virtualenv support ASV uses for
isolated benchmark environments.

## Development Checks

Check that ASV can discover and import the benchmark suite:

```powershell
python -m asv check
```

During active development, before the current changes are committed, use the
current virtual environment:

```powershell
python -m asv check -E existing:.venv\Scripts\python.exe
python -m asv run --quick -E existing:.venv\Scripts\python.exe --bench TimeFPUTExplicit
```

The `existing` environment tells ASV to use the package already installed in
the named Python environment. This is the right mode for checking uncommitted
benchmark code against the current working tree.

Run a quick smoke benchmark for the explicit FPUT suite:

```powershell
python -m asv run --quick --bench TimeFPUTExplicit
```

Quick runs are useful for checking that benchmark code works. They are not good
baseline measurements because each benchmark is run only once.

## Baseline Runs

Run the full local benchmark suite with:

```powershell
python -m asv run
```

Full ASV runs build the project from Git commits. Use them after committing the
code you want to benchmark.

If benchmark files in the working tree require package code that is not
committed yet, a full ASV run may import a mismatch: new benchmarks from the
working tree against old installed package code.

Publish local HTML output with:

```powershell
python -m asv publish
```

ASV stores generated environments, result files, and HTML output under `.asv/`.

## Machine Identity

The first ASV run asks for machine information such as machine name, operating
system, CPU, CPU count, and RAM. These values let ASV keep timing histories from
different machines separate.

Use stable, ordinary values. For example:

- machine: `JMF_Laptop`
- os: `Windows 11`
- arch: `AMD64`
- ram: `16GB`

If you change the machine name later, use ASV's `--machine` option to select
the recorded machine explicitly.

## Benchmark Layout

The ASV configuration is `asv.conf.json`.

The benchmark modules live in `benchmarks/` and follow ASV's usual naming
convention:

- `time_fput_explicit.py`
- `time_interface.py`
- `time_algebraist.py`

Benchmark class and method names should read clearly in ASV output. Prefer
names such as:

- `TimeFPUTExplicit.time_cash_karp`
- `TimeFPUTExplicit.time_cash_karp_algebraist`

## Default Problem

The first standard benchmark problem is the
Fermi-Pasta-Ulam-Tsingou beta lattice.

FPUT is a useful default because it is:

- deterministic;
- non-stiff;
- scalable by changing the chain size;
- structured enough to exercise custom STARK state objects;
- large enough to show whether Algebraist-generated kernels repay their setup
  cost.

The initial ASV suite uses small, medium, and large FPUT chains. Small cases
show overhead. Larger cases show whether generated translation kernels scale.

## What The First Suite Measures

The first benchmark campaign measures:

- explicit schemes without `algebraist=`;
- the same explicit schemes with `algebraist=`;
- construction/setup cost;
- steady repeated solve cost;
- interface-layer solves versus direct core-object solves.

This distinction is important for optimization work. `Algebraist` can make
repeated stage updates faster, but it may add setup or compilation cost. For a
tiny one-off solve, generic code may be faster. For a larger state or repeated
solves, generated kernels may win.

## NumPy Version

The initial ASV matrix installs the latest NumPy compatible with the selected
Python. This keeps local development smooth on new Python versions.

Historical NumPy pins can be added later when the project needs cross-version
performance history.

## Interpreting Results

Before optimizing `stark.algebraist.codegen`, record an ASV baseline from the
current implementation. Future optimization work should be judged against that
baseline.

Local benchmark results are machine-specific. Treat them as a performance
history for that machine, not as universal speed claims and not as user-facing
solver recommendations.
