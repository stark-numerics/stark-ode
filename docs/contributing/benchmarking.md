# Benchmarking and competition reports

Last reviewed: 2026-06-17.

This is contributor documentation. User-facing timing interpretation lives in [Diagnostics](../diagnostics.md).

## Purpose

STARK uses competition reports and benchmarks for different jobs.

```text
competition reports   compare solver choices on named problems
benchmarks            track performance regressions over time
examples              teach users, not prove speed
```

Do not use one artifact for all three jobs.

## Timing categories

Competition reports should keep these categories separate:

```text
setup        prepare the solver/callable
warmup       first solve after setup, often including compilation/JIT/GPU setup
warm run     repeated solve after setup and warmup
total        setup + warmup + one measured solve
```

For JIT and GPU backends, total time and warm time may tell very different stories.

## Headline rules

Reports should state which headline they are using.

Useful summaries:

```text
lowest preparation time
fastest warm median time
fastest total median time
best accuracy
```

Do not report only warm median when a backend moves large work into preparation.

## Monitoring

Benchmarks should normally run without monitors. Monitors are diagnostics and change the hot path.

If a benchmark needs counts, run a separate diagnostic pass and label it as such.

## Optional backends

Optional backend rows should skip/report cleanly when a dependency is missing.

Do not let absence of JAX, CuPy, SciPy, or Numba fail unrelated reports unless the report explicitly requires that dependency.

## Backend case studies

Backend examples should distinguish:

```text
plain NumPy
NumPy + Numba
JAX arrays / JAX generated kernels
CuPy arrays / CuPy generated kernels
```

A backend case study should show setup syntax in the lesson files. Do not hide all setup in a shared helper if the purpose is to teach backend use.

## When a result looks surprising

Before drawing conclusions, ask:

```text
Did setup include compilation?
Did warmup include first-use kernel generation?
Was GPU work synchronized before timing stopped?
Was the IVP reused after already finishing?
Is the row labelled accurately?
Are we measuring backend compatibility or backend acceleration?
```
