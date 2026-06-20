"""Compare CuPy algebra-provider strategies on the backend case-study problem.

This is an experiment, not a public example. It asks one narrow question:
which specialist strategy does CuPy actually want for the large array backend
case study?

The rows intentionally keep the problem, derivative, scheme, interval, and
timing convention aligned with ``examples.case_studies.backends``. Only the
CuPy engine's algebra providers change.

Run from the ``stark-ode`` directory with:

    python -m benchmarks.experiments.timing_cupy_specialists
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from statistics import median
from time import perf_counter
from typing import Any, Callable

import numpy as np

from examples.case_studies.backends.lesson_01_numpy import build_numpy_ivp
from examples.case_studies.backends.lesson_02_jax import (
    build_jax_ivp,
    jnp,
    synchronize_jax,
)
from examples.case_studies.backends.lesson_03_cupy import (
    cp,
    initial_cupy,
    rhs_cupy,
    synchronize_cupy,
)
from stark import Configuration, Frame, Interval, Method, System
from stark.core.contracts.accelerator import Accelerator
from stark.engines.shared.accelerators import AcceleratorNone
from stark.engines.shared.algebraist.arity import AlgebraistArity
from stark.engines.shared.algebraist.frame import (
    AlgebraistFrame,
    AlgebraistFrameLooped,
    AlgebraistFrameNormMax,
    AlgebraistFrameNormRMS,
)
from stark.engines.shared.algebraist.generator import (
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
    AlgebraistGeneratorTargetMutableVectorized,
)
from stark.engines.shared.algebraist.runtime import (
    AlgebraistRuntimeInnerProduct,
    AlgebraistRuntimeLinearCombine,
    AlgebraistRuntimeNorm,
    AlgebraistRuntimeSpecialist,
)
from stark.engines.cupy import EngineCupy
from stark.engines.cupy.allocator import EngineAllocatorCupy
from stark.engines.cupy.target import AlgebraistGeneratorTargetCupy
from stark.engines.cupy.carriers import CarrierCupy, CarrierNormCupyMax, CarrierNormCupyRMS
from stark.methods.schemes import SchemeCashKarp
from stark.problem.frame.frame import Frame as ProblemFrame


SIZE = 65536
PRECONDITION_SIZES = (256, 1024, 4096, 16384)


@dataclass(slots=True)
class TimingRow:
    library: str
    setup: float | None
    first: float | None
    repeat: float | None
    one_shot: float | None
    values: np.ndarray | None = field(default=None, repr=False)
    note: str = ""


@dataclass(frozen=True, slots=True)
class EngineCupyGeneratedVectorized:
    """CuPy engine variant using ordinary generated vectorized CuPy expressions."""

    frame: ProblemFrame
    dtype: Any = None
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    algebraist_frame: AlgebraistFrame = field(init=False)
    carriers: tuple[CarrierCupy, ...] = field(init=False, repr=False)
    allocator: EngineAllocatorCupy = field(init=False, repr=False)
    algebraist_inner_product: AlgebraistGeneratorInnerProduct = field(init=False, repr=False)
    algebraist_linear_combine: AlgebraistGeneratorLinearCombine = field(init=False, repr=False)
    algebraist_norm: AlgebraistGeneratorNorm = field(init=False, repr=False)
    algebraist_specialist: AlgebraistGeneratorSpecialist = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._prepare_generated(AlgebraistGeneratorTargetMutableVectorized())

    def _prepare_generated(self, target: object) -> None:
        if cp is None:
            raise RuntimeError("CuPy is not installed.")
        dtype = cp.float64 if self.dtype is None else self.dtype
        algebraist_frame = self.frame.to_algebraist_frame()
        carriers = self._carriers(algebraist_frame, dtype)
        allocator = EngineAllocatorCupy(algebraist_frame=algebraist_frame, carriers=carriers)

        object.__setattr__(self, "algebraist_frame", algebraist_frame)
        object.__setattr__(self, "carriers", carriers)
        object.__setattr__(self, "allocator", allocator)

        linear_combine = AlgebraistGeneratorLinearCombine(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            frame=algebraist_frame,
            accelerator=self.accelerator,
            target=target,
        )
        object.__setattr__(
            allocator,
            "linear_combine",
            tuple(linear_combine.provide(AlgebraistArity(arity)) for arity in range(1, 13)),
        )

        specialist = AlgebraistGeneratorSpecialist(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            frame=algebraist_frame,
            accelerator=self.accelerator,
            target=target,
        )
        object.__setattr__(allocator, "apply_translation", specialist.provide_unit_apply())

        norm = AlgebraistGeneratorNorm(
            translation=allocator.allocate_translation(),
            frame=algebraist_frame,
            accelerator=self.accelerator,
            target=target,
        )
        object.__setattr__(allocator, "norm", norm.provide())

        inner_product = AlgebraistGeneratorInnerProduct(
            translation=allocator.allocate_translation(),
            frame=algebraist_frame,
            accelerator=self.accelerator,
            target=target,
        )
        object.__setattr__(allocator, "inner_product", inner_product.provide())

        object.__setattr__(self, "algebraist_linear_combine", linear_combine)
        object.__setattr__(self, "algebraist_specialist", specialist)
        object.__setattr__(self, "algebraist_norm", norm)
        object.__setattr__(self, "algebraist_inner_product", inner_product)

    @staticmethod
    def _carriers(algebraist_frame: AlgebraistFrame, dtype: object) -> tuple[CarrierCupy, ...]:
        carriers: list[CarrierCupy] = []
        for field in algebraist_frame.fields:
            policy = field.policy
            if not isinstance(policy, AlgebraistFrameLooped) or policy.shape is None:
                raise ValueError("CuPy experiment requires shaped looped fields.")
            carriers.append(CarrierCupy(cp.zeros(policy.shape, dtype=dtype)))
        return tuple(carriers)


@dataclass(frozen=True, slots=True)
class EngineCupyGeneratedElementwise(EngineCupyGeneratedVectorized):
    """CuPy engine variant using the backend ElementwiseKernel target."""

    def __post_init__(self) -> None:
        self._prepare_generated(AlgebraistGeneratorTargetCupy())


@dataclass(frozen=True, slots=True)
class EngineCupyRuntime:
    """CuPy engine variant using runtime providers and CuPy carrier primitives."""

    frame: ProblemFrame
    dtype: Any = None
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    algebraist_frame: AlgebraistFrame = field(init=False)
    carriers: tuple[CarrierCupy, ...] = field(init=False, repr=False)
    allocator: EngineAllocatorCupy = field(init=False, repr=False)
    algebraist_inner_product: AlgebraistRuntimeInnerProduct = field(init=False, repr=False)
    algebraist_linear_combine: AlgebraistRuntimeLinearCombine = field(init=False, repr=False)
    algebraist_norm: AlgebraistRuntimeNorm = field(init=False, repr=False)
    algebraist_specialist: AlgebraistRuntimeSpecialist = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if cp is None:
            raise RuntimeError("CuPy is not installed.")
        dtype = cp.float64 if self.dtype is None else self.dtype
        algebraist_frame = self.frame.to_algebraist_frame()
        carriers = EngineCupyGeneratedVectorized._carriers(algebraist_frame, dtype)
        allocator = EngineAllocatorCupy(algebraist_frame=algebraist_frame, carriers=carriers)

        object.__setattr__(self, "algebraist_frame", algebraist_frame)
        object.__setattr__(self, "carriers", carriers)
        object.__setattr__(self, "allocator", allocator)

        field_norms = []
        for field, carrier in zip(algebraist_frame.fields, carriers, strict=True):
            if isinstance(field.norm, AlgebraistFrameNormRMS):
                field_norms.append(CarrierNormCupyRMS())
                continue
            if isinstance(field.norm, AlgebraistFrameNormMax):
                field_norms.append(CarrierNormCupyMax())
                continue
            if not field.norm.include:
                field_norms.append(carrier.norm)
                continue
            raise ValueError("CuPy runtime experiment requires RMS, max, or excluded norm fields.")

        norm = AlgebraistRuntimeNorm(frame=algebraist_frame, field_norms=tuple(field_norms))
        object.__setattr__(allocator, "norm", norm.provide())
        inner_product = AlgebraistRuntimeInnerProduct(frame=algebraist_frame)
        object.__setattr__(allocator, "inner_product", inner_product.provide())
        linear_combine = AlgebraistRuntimeLinearCombine(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            frame=algebraist_frame,
            accelerator=self.accelerator,
        )
        specialist = AlgebraistRuntimeSpecialist(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            frame=algebraist_frame,
            accelerator=self.accelerator,
        )

        object.__setattr__(self, "algebraist_linear_combine", linear_combine)
        object.__setattr__(self, "algebraist_specialist", specialist)
        object.__setattr__(self, "algebraist_norm", norm)
        object.__setattr__(self, "algebraist_inner_product", inner_product)


def build_cupy_ivp(*, size: int, engine: Callable[[Frame], object]) -> object:
    if cp is None or rhs_cupy is None:
        raise RuntimeError("CuPy is not installed.")
    frame = Frame({"u": {"translation": "du", "shape": (size,)}})
    system = System(derivative=rhs_cupy, frame=frame)
    return system.ivp(
        initial={"u": initial_cupy(size)},
        interval=Interval(present=0.0, step=0.005, stop=0.015),
        method=Method(scheme=SchemeCashKarp),
        engine=engine,
        configuration=Configuration(check_progress=False),
    )


def time_entry(
    *,
    library: str,
    build: Callable[[], object],
    values: Callable[[object], np.ndarray],
    synchronize: Callable[[object], None] | None = None,
    repeats: int,
    note: str = "",
) -> TimingRow:
    print(f"  {library}: setup and first solve...", flush=True)
    started = perf_counter()
    ivp = build()
    setup = perf_counter() - started

    started = perf_counter()
    result = ivp.final_result()
    if synchronize is not None:
        synchronize(result)
    first = perf_counter() - started
    final_values = values(result)

    durations = []
    for index in range(repeats):
        print(f"  {library}: repeat {index + 1}/{repeats}...", flush=True)
        ivp = build()
        started = perf_counter()
        result = ivp.final_result()
        if synchronize is not None:
            synchronize(result)
        durations.append(perf_counter() - started)

    return TimingRow(
        library=library,
        setup=setup,
        first=first,
        repeat=median(durations),
        one_shot=setup + first,
        values=final_values,
        note=note,
    )


def numpy_values(result: object) -> np.ndarray:
    return np.asarray(result.state.u, dtype=np.float64)


def cupy_values(result: object) -> np.ndarray:
    return np.asarray(cp.asnumpy(result.state.u), dtype=np.float64)


def sync_cupy_result(result: object) -> None:
    del result
    synchronize_cupy()


def precondition_with_jax(*, size: int, repeats: int) -> None:
    """Run the same-size JAX backend path before CuPy timing.

    The backend comparison lesson runs JAX before CuPy in one Python process.
    JAX and CuPy both use the GPU, but they own separate runtimes, memory pools,
    and compilation caches. This preconditioner lets the experiment distinguish
    CuPy specialist speed from mixed-framework GPU interference.
    """

    if jnp is None:
        print("JAX preconditioner skipped: JAX is not installed.", flush=True)
        return

    print("Preconditioning GPU with same-size JAX solves...", flush=True)
    ivp = build_jax_ivp(size=size)
    result = ivp.final_result()
    synchronize_jax(result.state.u)
    for index in range(repeats):
        print(f"  JAX precondition repeat {index + 1}/{repeats}...", flush=True)
        ivp = build_jax_ivp(size=size)
        result = ivp.final_result()
        synchronize_jax(result.state.u)


def precondition_like_backend_comparison(*, size: int, repeats: int) -> None:
    """Run the backend-comparison order before timing the CuPy strategy rows."""

    print("Preconditioning with the full backend comparison order...", flush=True)
    smaller_sizes = tuple(item for item in PRECONDITION_SIZES if item < size)
    for precondition_size in smaller_sizes:
        print(f"  precondition size={precondition_size}: NumPy...", flush=True)
        ivp = build_numpy_ivp(size=precondition_size, accelerated=False)
        ivp.final_result()

        print(f"  precondition size={precondition_size}: NumPy+Numba...", flush=True)
        ivp = build_numpy_ivp(size=precondition_size, accelerated=True)
        ivp.final_result()

        if jnp is not None:
            print(f"  precondition size={precondition_size}: JAX...", flush=True)
            ivp = build_jax_ivp(size=precondition_size)
            result = ivp.final_result()
            synchronize_jax(result.state.u)

        if cp is not None:
            print(f"  precondition size={precondition_size}: CuPy...", flush=True)
            ivp = build_cupy_ivp(size=precondition_size, engine=EngineCupy)
            result = ivp.final_result()
            sync_cupy_result(result)

    print(f"  precondition size={size}: NumPy+Numba...", flush=True)
    ivp = build_numpy_ivp(size=size, accelerated=True)
    ivp.final_result()

    if jnp is not None:
        print(f"  precondition size={size}: JAX setup/first...", flush=True)
        ivp = build_jax_ivp(size=size)
        result = ivp.final_result()
        synchronize_jax(result.state.u)
        for index in range(repeats):
            print(f"  precondition size={size}: JAX repeat {index + 1}/{repeats}...", flush=True)
            ivp = build_jax_ivp(size=size)
            result = ivp.final_result()
            synchronize_jax(result.state.u)


def format_seconds(value: float | None) -> str:
    if value is None:
        return "skipped"
    return f"{value:.6f}s"


def format_factor(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or value == 0.0:
        return ""
    return f"{baseline / value:.2f}x"


def print_timing(rows: list[TimingRow]) -> None:
    baseline = rows[0]
    print()
    print("CuPy specialist timing experiment")
    print("---------------------------------")
    print("library                  | setup     | first    | repeat   | one-shot x | repeat x")
    print("-------------------------+-----------+----------+----------+------------+---------")
    for row in rows:
        print(
            f"{row.library:<24} | "
            f"{format_seconds(row.setup):>9} | "
            f"{format_seconds(row.first):>8} | "
            f"{format_seconds(row.repeat):>8} | "
            f"{format_factor(row.one_shot, baseline.one_shot):>10} | "
            f"{format_factor(row.repeat, baseline.repeat):>7}"
        )


def print_accuracy(rows: list[TimingRow]) -> None:
    baseline = rows[0].values
    print()
    print("Accuracy against plain NumPy")
    print("----------------------------")
    print("library                  | rel L2    | max abs")
    print("-------------------------+-----------+----------")
    for row in rows:
        if baseline is None or row.values is None:
            print(f"{row.library:<24} | skipped   | skipped")
            continue
        error = row.values - baseline
        denominator = max(float(np.linalg.norm(baseline)), np.finfo(float).tiny)
        rel_l2 = float(np.linalg.norm(error)) / denominator
        max_abs = float(np.max(np.abs(error)))
        print(f"{row.library:<24} | {rel_l2:9.3e} | {max_abs:8.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CuPy algebra-provider strategies on the largest backend case-study run.",
    )
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--jax-before-cupy",
        action="store_true",
        help="run same-size JAX solves before CuPy rows to mimic the backend comparison ordering",
    )
    parser.add_argument(
        "--full-order-before-cupy",
        action="store_true",
        help="run smaller backend rows and same-size NumPy+Numba/JAX before CuPy rows",
    )
    args = parser.parse_args()

    if args.size < 1:
        raise ValueError("--size must be positive.")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")

    rows = [
        time_entry(
            library="NumPy",
            build=lambda: build_numpy_ivp(size=args.size, accelerated=False),
            values=numpy_values,
            repeats=args.repeats,
            note="plain NumPy baseline",
        )
    ]

    if cp is None:
        print("CuPy is not installed; only NumPy baseline was run.")
        print_timing(rows)
        return

    if args.full_order_before_cupy:
        precondition_like_backend_comparison(size=args.size, repeats=args.repeats)
    elif args.jax_before_cupy:
        precondition_with_jax(size=args.size, repeats=args.repeats)

    rows.extend(
        [
            time_entry(
                library="CuPy current",
                build=lambda: build_cupy_ivp(size=args.size, engine=EngineCupy),
                values=cupy_values,
                synchronize=sync_cupy_result,
                repeats=args.repeats,
                note="package EngineCupy",
            ),
            time_entry(
                library="CuPy generated vector",
                build=lambda: build_cupy_ivp(size=args.size, engine=EngineCupyGeneratedVectorized),
                values=cupy_values,
                synchronize=sync_cupy_result,
                repeats=args.repeats,
                note="whole-array generated expressions",
            ),
            time_entry(
                library="CuPy elementwise",
                build=lambda: build_cupy_ivp(size=args.size, engine=EngineCupyGeneratedElementwise),
                values=cupy_values,
                synchronize=sync_cupy_result,
                repeats=args.repeats,
                note="explicit ElementwiseKernel target",
            ),
            time_entry(
                library="CuPy runtime",
                build=lambda: build_cupy_ivp(size=args.size, engine=EngineCupyRuntime),
                values=cupy_values,
                synchronize=sync_cupy_result,
                repeats=args.repeats,
                note="runtime providers and carrier primitives",
            ),
        ]
    )

    print_timing(rows)
    print_accuracy(rows)
    print()
    print("Notes")
    print("-----")
    for row in rows:
        print(f"- {row.library}: {row.note}")


if __name__ == "__main__":
    main()
