from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from stark import Executor, Integrator, Interval, Marcher, Tolerance
from stark.algebraist import Algebraist, AlgebraistField, AlgebraistLooped
from stark.schemes.explicit_adaptive.cash_karp import SchemeCashKarp
from stark.schemes.explicit_fixed.rk4 import SchemeRK4


def make_algebraist() -> Algebraist:
    return Algebraist(
        fields=(AlgebraistField("dy", "y", policy=AlgebraistLooped(rank=1)),),
        generate_norm="l2",
    )


@dataclass(slots=True)
class ArrayState:
    y: np.ndarray

    def copy(self) -> ArrayState:
        return ArrayState(self.y.copy())


@dataclass(slots=True)
class ArrayTranslation:
    dy: np.ndarray

    def __call__(self, origin: ArrayState, result: ArrayState) -> None:
        result.y[...] = origin.y + self.dy

    def norm(self) -> float:
        return float(np.linalg.norm(self.dy))

    def __add__(self, other: ArrayTranslation) -> ArrayTranslation:
        return ArrayTranslation(self.dy + other.dy)

    def __rmul__(self, scalar: float) -> ArrayTranslation:
        return ArrayTranslation(scalar * self.dy)


class ArrayWorkbench:
    def __init__(self, size: int) -> None:
        self.size = size

    def allocate_state(self) -> ArrayState:
        return ArrayState(np.zeros(self.size))

    def copy_state(self, dst: ArrayState, src: ArrayState) -> None:
        dst.y[...] = src.y

    def allocate_translation(self) -> ArrayTranslation:
        return ArrayTranslation(np.zeros(self.size))


class ArrayDerivative:
    def __call__(
        self,
        interval: Interval,
        state: ArrayState,
        out: ArrayTranslation,
    ) -> None:
        out.dy[...] = np.array(
            [
                state.y[1] + 0.25 * interval.present,
                -state.y[0] + 0.1 * state.y[2],
                -0.5 * state.y[2] + 0.2,
            ]
        )


def build_state() -> ArrayState:
    return ArrayState(np.array([1.0, -0.5, 0.25]))


def test_fixed_explicit_algebraist_path_matches_generic_semantics() -> None:
    generic_state = build_state()
    algebraist_state = build_state()

    generic = SchemeRK4(ArrayDerivative(), ArrayWorkbench(3))
    generated = SchemeRK4(
        ArrayDerivative(),
        ArrayWorkbench(3),
        algebraist=make_algebraist(),
    )

    generic_dt = generic(
        Interval(present=0.0, step=0.05, stop=0.05),
        generic_state,
        Executor(),
    )
    generated_dt = generated(
        Interval(present=0.0, step=0.05, stop=0.05),
        algebraist_state,
        Executor(),
    )

    assert generated_dt == pytest.approx(generic_dt)
    np.testing.assert_allclose(
        algebraist_state.y,
        generic_state.y,
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_adaptive_explicit_algebraist_path_matches_generic_semantics() -> None:
    generic_scheme = SchemeCashKarp(ArrayDerivative(), ArrayWorkbench(3))
    generated_scheme = SchemeCashKarp(
        ArrayDerivative(),
        ArrayWorkbench(3),
        algebraist=make_algebraist(),
    )

    generic_state, generic_steps, generic_next_step = run_adaptive_solve(generic_scheme)
    generated_state, generated_steps, generated_next_step = run_adaptive_solve(
        generated_scheme
    )

    assert generated_steps == generic_steps
    assert generated_next_step == pytest.approx(
        generic_next_step,
        rel=1.0e-14,
        abs=1.0e-14,
    )
    np.testing.assert_allclose(
        generated_state.y,
        generic_state.y,
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    generic_report = generic_scheme.adaptive.report()
    generated_report = generated_scheme.adaptive.report()

    assert generated_report.accepted_dt == pytest.approx(
        generic_report.accepted_dt,
        rel=1.0e-14,
        abs=1.0e-14,
    )
    assert generated_report.proposed_dt == pytest.approx(
        generic_report.proposed_dt,
        rel=1.0e-14,
        abs=1.0e-14,
    )
    assert generated_report.next_dt == pytest.approx(
        generic_report.next_dt,
        rel=1.0e-14,
        abs=1.0e-14,
    )
    assert generated_report.error_ratio == pytest.approx(
        generic_report.error_ratio,
        rel=1.0e-14,
        abs=1.0e-14,
    )
    assert generated_report.rejection_count == generic_report.rejection_count


def run_adaptive_solve(
    scheme: SchemeCashKarp,
) -> tuple[ArrayState, int, float]:
    state = build_state()
    interval = Interval(present=0.0, step=0.05, stop=0.25)
    executor = Executor(tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-10))
    marcher = Marcher(scheme, executor)
    integrator = Integrator(executor=marcher.executor)

    steps = 0
    for _snapshot_interval, _snapshot_state in integrator.live(marcher, interval, state):
        steps += 1

    return state, steps, interval.step


def test_algebraist_generated_source_is_inspectable() -> None:
    algebraist = make_algebraist()

    assert algebraist.sources
    assert algebraist.kernel_sources
    assert algebraist.wrapper_sources

    assert "apply" in algebraist.sources
    assert "apply_kernel" in algebraist.sources
    assert "norm" in algebraist.sources
    assert "norm_kernel" in algebraist.sources

    algebraist.bind_explicit_scheme(SchemeRK4.tableau)
    assert "stage1_state" in algebraist.sources
    assert "stage1_state_kernel" in algebraist.sources
    assert "solution_state" in algebraist.sources
    assert "solution_state_kernel" in algebraist.sources
    assert "solution_combine" in algebraist.sources
    assert "solution_combine_kernel" in algebraist.sources

    for source in algebraist.sources.values():
        assert isinstance(source, str)
        assert source.startswith("def ")


def test_non_algebraist_scheme_does_not_carry_generated_source_state() -> None:
    scheme = SchemeRK4(ArrayDerivative(), ArrayWorkbench(3))

    assert not hasattr(scheme, "sources")
    assert not hasattr(scheme, "kernel_sources")
    assert not hasattr(scheme, "wrapper_sources")
    assert not hasattr(scheme, "generated_source")