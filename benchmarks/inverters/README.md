# Inverter smoke benchmarks

This directory contains manual, non-gating timing scripts for current
request-shaped inverter support.

Run from the repository root:

```powershell
python -m benchmarks.inverters.bench_defect
python -m benchmarks.inverters.bench_jacobi
python -m benchmarks.inverters.bench_richardson
```

These scripts are local before/after checks only. They are not portable solver
performance claims and are not CI gates.

The old legacy Krylov refactor benchmark was removed because it targeted the
unfinished bind-then-solve inverter surface. Add replacement Krylov benchmark
coverage once the request-shaped Krylov API lands.
