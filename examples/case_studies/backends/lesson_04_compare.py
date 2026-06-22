"""Lesson 4: compare backend timing and accuracy without hiding setup cost."""

from __future__ import annotations

# The first three backend lessons show how to set up NumPy, JAX, and CuPy.  This
# lesson asks when those choices pay off.  It deliberately reports two figures:
#
# - one-shot latency: setup plus the first solve,
# - repeated throughput: prepared same-shaped solves after setup/compilation.
#
# Your machine matters.  JAX compilation, Numba compilation, GPU launch cost,
# CPU vector performance, and GPU model can all change the crossover points.
# Treat the table as a diagnostic, not as a universal benchmark.
#
# The accuracy table is just as important as the timing table.  It checks that
# the backends are solving the same problem before we interpret speedups.
#
# In a source checkout, run from the ``stark-ode`` directory with:
#
#     python -m examples.case_studies.backends.lesson_04_compare



import os
from dataclasses import dataclass, field
from statistics import median
from time import perf_counter
from typing import Any, Callable

import numpy as np

from examples.case_studies.backends.lesson_01_numpy import build_numpy_ivp
from examples.case_studies.backends.lesson_02_jax import (
    _kernel_returning_available,
    build_jax_ivp,
    jnp,
    synchronize_jax,
)
from examples.case_studies.backends.lesson_03_cupy import build_cupy_ivp, cp, synchronize_cupy


@dataclass(slots=True)
class TimingRow:
    size: int
    library: str
    setup: float | None
    first_solve: float | None
    repeat_median: float | None
    one_shot: float | None
    note: str = ""
    values: np.ndarray | None = field(default=None, repr=False)


@dataclass(slots=True)
class AccuracyRow:
    size: int
    library: str
    rel_l2: float | None
    max_abs: float | None
    note: str = ""


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "skipped"
    return f"{value:.6f}s"


def _format_factor(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or value == 0.0:
        return ""
    return f"{baseline / value:.2f}x"


def _format_error(value: float | None) -> str:
    if value is None:
        return "skipped"
    return f"{value:.3e}"


def _state_values(result: object, library: str) -> np.ndarray:
    values = result.state.u
    if library == "CuPy" and cp is not None:
        return np.asarray(cp.asnumpy(values), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _fresh_timing(
    *,
    size: int,
    library: str,
    build: Callable[..., Any],
    synchronize: Callable[[object], None] | None = None,
    repeats: int = 3,
    note: str = "",
) -> TimingRow:
    print(f"  running {library} size={size} setup/first solve...", flush=True)
    started = perf_counter()
    try:
        ivp = build(size=size)
    except Exception as exc:
        return TimingRow(size, library, None, None, None, None, f"skipped: {exc}")
    setup = perf_counter() - started

    started = perf_counter()
    result = ivp.final_result()
    if synchronize is not None:
        synchronize(result)
    first_solve = perf_counter() - started
    values = _state_values(result, library)

    durations = []
    for repeat in range(repeats):
        # ``final_result`` advances an IVP. Build a fresh IVP for each
        # measured solve, but start the timer after construction so this
        # column still reports repeat solve time rather than setup time.
        print(
            f"  running {library} size={size} repeat {repeat + 1}/{repeats}...",
            flush=True,
        )
        ivp = build(size=size)
        started = perf_counter()
        result = ivp.final_result()
        if synchronize is not None:
            synchronize(result)
        durations.append(perf_counter() - started)

    return TimingRow(
        size=size,
        library=library,
        setup=setup,
        first_solve=first_solve,
        repeat_median=median(durations),
        one_shot=setup + first_solve,
        note=note,
        values=values,
    )


def _sync_jax_result(result) -> None:
    synchronize_jax(result.state.u)


def _sync_cupy_result(result) -> None:
    del result
    synchronize_cupy()


def _sizes() -> tuple[int, ...]:
    raw = os.environ.get("STARK_BACKEND_SIZES")
    if not raw:
        return (256, 1024, 4096, 16384, 65536)
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _rows_for_size(size: int) -> list[TimingRow]:
    rows: list[TimingRow] = []
    print(f"Collecting backend timings for size={size}", flush=True)
    rows.append(
        _fresh_timing(
            size=size,
            library="NumPy",
            build=lambda size: build_numpy_ivp(size=size, accelerated=False),
            note="plain NumPy arrays; no Numba accelerator",
        )
    )
    rows.append(
        _fresh_timing(
            size=size,
            library="NumPy+Numba",
            build=lambda size: build_numpy_ivp(size=size, accelerated=True),
            note="generated Algebraist kernels; first solve may compile with Numba",
        )
    )
    if jnp is None:
        rows.append(TimingRow(size, "JAX", None, None, None, None, "skipped: JAX not installed"))
    elif not _kernel_returning_available():
        rows.append(TimingRow(size, "JAX", None, None, None, None, "skipped: DerivativeStyle.kernel_accepts_instant_returns missing"))
    else:
        rows.append(
            _fresh_timing(
                size=size,
                library="JAX",
                build=build_jax_ivp,
                synchronize=_sync_jax_result,
                note="return-style derivative; generated Algebraist kernels; Python solver control",
            )
        )
    if cp is None:
        rows.append(TimingRow(size, "CuPy", None, None, None, None, "skipped: CuPy not installed"))
    else:
        rows.append(
            _fresh_timing(
                size=size,
                library="CuPy",
                build=build_cupy_ivp,
                synchronize=_sync_cupy_result,
                note="GPU arrays; synchronize before timing stops",
            )
        )
    return rows


def _accuracy_rows(rows: list[TimingRow]) -> list[AccuracyRow]:
    baseline: dict[int, np.ndarray] = {
        row.size: row.values
        for row in rows
        if row.library == "NumPy" and row.values is not None
    }
    result: list[AccuracyRow] = []
    for row in rows:
        base = baseline.get(row.size)
        if row.values is None or base is None:
            result.append(AccuracyRow(row.size, row.library, None, None, row.note))
            continue
        error = row.values - base
        denominator = max(float(np.linalg.norm(base)), np.finfo(float).tiny)
        result.append(
            AccuracyRow(
                size=row.size,
                library=row.library,
                rel_l2=float(np.linalg.norm(error)) / denominator,
                max_abs=float(np.max(np.abs(error))),
                note=row.note,
            )
        )
    return result


def _print_timing_table(rows: list[TimingRow]) -> None:
    print("Timing table")
    print("------------")
    print("size | library     | setup     | first    | repeat   | one-shot x | repeat x")
    print("-----+-------------+-----------+----------+----------+------------+---------")
    repeat_baseline: dict[int, float | None] = {}
    one_shot_baseline: dict[int, float | None] = {}
    for row in rows:
        if row.library == "NumPy":
            repeat_baseline[row.size] = row.repeat_median
            one_shot_baseline[row.size] = row.one_shot
    for row in rows:
        repeat_base = repeat_baseline.get(row.size)
        one_shot_base = one_shot_baseline.get(row.size)
        print(
            f"{row.size:>4} | "
            f"{row.library:<11} | "
            f"{_format_seconds(row.setup):>9} | "
            f"{_format_seconds(row.first_solve):>8} | "
            f"{_format_seconds(row.repeat_median):>8} | "
            f"{_format_factor(row.one_shot, one_shot_base):>10} | "
            f"{_format_factor(row.repeat_median, repeat_base):>7}"
        )


def _print_accuracy_table(rows: list[TimingRow]) -> None:
    print("Accuracy check against plain NumPy")
    print("----------------------------------")
    print("size | library     | rel L2    | max abs")
    print("-----+-------------+-----------+----------")
    for row in _accuracy_rows(rows):
        print(
            f"{row.size:>4} | "
            f"{row.library:<11} | "
            f"{_format_error(row.rel_l2):>9} | "
            f"{_format_error(row.max_abs):>8}"
        )


def _print_notes(rows: list[TimingRow]) -> None:
    print("Backend notes")
    print("-------------")
    seen: set[str] = set()
    for row in rows:
        if row.library in seen:
            continue
        seen.add(row.library)
        if row.note:
            print(f"- {row.library}: {row.note}")
    print()
    print("Column guide")
    print("------------")
    print("- setup: construct the IVP and prepared backend objects.")
    print("- first: first solve after setup; may include JIT, GPU, or Numba compilation.")
    print("- repeat: median of repeated same-shaped solves, excluding fresh IVP construction.")
    print("- one-shot x: speed factor for setup + first solve, relative to plain NumPy.")
    print("- repeat x: speed factor for repeated prepared solves, relative to plain NumPy.")
    print("- Higher speed factors are faster: 2.00x is twice as fast, 0.50x is half as fast.")


def main() -> None:
    print("Lesson 4: backend comparison")
    print("============================")
    print("This lesson separates one-shot latency from repeated-solve throughput.")
    print("Use one-shot x when solving once or rarely. Use repeat x when solving")
    print("the same-shaped problem many times after setup/compilation.")
    print()

    rows: list[TimingRow] = []
    for size in _sizes():
        rows.extend(_rows_for_size(size))
    _print_timing_table(rows)
    print()
    _print_accuracy_table(rows)
    print()
    _print_notes(rows)
    print("Use STARK_BACKEND_SIZES=... to choose a different size sweep.")


if __name__ == "__main__":
    main()
