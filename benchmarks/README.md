# Benchmarks

This directory is contributor-facing. Users do not need it to solve ODEs with
STARK; contributors use it to check whether package changes alter performance.

The benchmark suite has three independent axes:

- representative problems;
- method stacks from `stark.methods.METHOD_CATALOGUE`;
- engines.

This matters because a useful benchmark suite should answer questions such as:

- did a scheme/resolvent/inverter stack regress on a stiff problem?
- did an engine change help large array states but hurt small scalar states?
- did an optimisation help Newton+dense while damaging IMEX or explicit paths?

## Structure

`benchmarks.catalogue` contains the benchmark-facing metadata. It does not run
benchmarks. `benchmarks.problems` contains the reusable problem definitions that
can build a `System`, fresh initial values, an interval, and optionally a
final-state reference. `benchmarks.builders` turns catalogue axes into runnable
IVPs.

The layers are:

- `stark.methods.catalogue`: package-level catalogue of method components and
  method-stack recipes;
- `benchmarks.catalogue`: benchmark-only catalogue of representative problems
  and engines;
- `benchmarks.problems`: reusable problem definitions used by benchmark
  runners;
- `benchmarks.builders`: construction helpers for problem/method/engine axes;
- ASV benchmark classes: small timing harnesses that combine problem, method,
  and engine entries from the catalogues.

See `DESIGN.md` for the benchmark architecture notes.

## What ASV Does

ASV, or Airspeed Velocity, is a benchmark runner for tracking performance across
commits. Unit tests answer "is the result correct?". ASV answers "did this
change make a selected workload faster or slower?".

The important difference from an ordinary timing script is history. ASV runs the
same benchmark classes against selected commits, records the timings, and can
publish an HTML report showing regressions and improvements over time. That is
why ASV is most useful after code is committed: it compares named revisions
rather than just the current dirty working tree.

For STARK, ASV should track combinations of representative problems, method
stacks, and engines. That lets us catch regressions where, for example, an
engine optimisation helps large array states but hurts a small stiff Newton
solve.

## Current ASV Layer

`time_ivp.py` contains the first catalogue-driven ASV classes:

- `BenchmarkTimeIVPSmoke...`;
- `BenchmarkTimeIVPRepresentative...`;
- `BenchmarkTimeIVPFull...`.

Each tier has setup, first-solve, repeat-solve, and error tracking classes.

## Quick Commands

Check ASV discovery against the current virtual environment:

```powershell
.\devtools\check-benchmarks.ps1
```

Run the IVP smoke benchmarks quickly:

```powershell
.\devtools\check-benchmarks.ps1 -RunSmoke
```

Run the representative benchmark tier:

```powershell
.\devtools\check-benchmarks.ps1 -RunRepresentative
```

Run the full currently compatible IVP matrix:

```powershell
.\devtools\check-benchmarks.ps1 -RunFull
```
