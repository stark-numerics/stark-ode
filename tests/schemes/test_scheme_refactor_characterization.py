from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Executor, Integrator, Interval, Marcher, Tolerance
from stark.accelerators import Accelerator
from stark.resolvents import ResolventPicard
from stark.resolvents.policy import ResolventPolicy
from stark.schemes.explicit_adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.schemes.explicit_fixed.rk4 import SchemeRK4
from stark.schemes.implicit_fixed.backward_euler import SchemeBackwardEuler


@dataclass(slots=True)
class ScalarState:
    value: float = 0.0


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: ScalarState, result: ScalarState) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: ScalarTranslation) -> ScalarTranslation:
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> ScalarTranslation:
        return ScalarTranslation(scalar * self.value)


class ScalarWorkbench:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, dst: ScalarState, src: ScalarState) -> None:
        dst.value = src.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def exponential_growth(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval
    out.value = state.value


def constant_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 1.0


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def run_live(
    scheme,
    *,
    state: ScalarState,
    interval: Interval,
    executor: Executor | None = None,
) -> tuple[list[tuple[float, float, float]], ScalarState, Interval]:
    marcher = Marcher(scheme, executor or Executor())
    outputs = [
        (snapshot_interval.present, snapshot_interval.step, snapshot_state.value)
        for snapshot_interval, snapshot_state in Integrator().live(marcher, interval, state)
    ]
    return outputs, state, interval


def test_rk4_fixed_explicit_characterization() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.125, stop=0.25)
    state = ScalarState(1.0)

    outputs, final_state, final_interval = run_live(
        scheme,
        state=state,
        interval=interval,
    )

    assert len(outputs) == 2
    assert final_interval.present == pytest.approx(0.25)
    assert final_interval.step == pytest.approx(0.125)

    # Current RK4 behaviour for y' = y over two dt=1/8 steps:
    # (1 + z + z^2/2 + z^3/6 + z^4/24)^2, z = 1/8.
    assert final_state.value == pytest.approx(1.2840248281136155)


def test_bogacki_shampine_adaptive_explicit_characterization() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)

    outputs, final_state, final_interval = run_live(
        scheme,
        state=state,
        interval=interval,
        executor=Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9)),
    )

    assert len(outputs) == 2
    assert [present for present, _, _ in outputs] == pytest.approx([0.1, 0.3])
    assert final_interval.present == pytest.approx(0.3)
    assert final_interval.step == pytest.approx(0.0)
    assert final_state.value == pytest.approx(2.0)


def test_backward_euler_implicit_fixed_characterization() -> None:
    workbench = ScalarWorkbench()
    resolvent = ResolventPicard(
        constant_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=SchemeBackwardEuler.tableau,
    )
    scheme = SchemeBackwardEuler(
        constant_rhs,
        workbench,
        resolvent=resolvent,
    )
    interval = Interval(present=0.0, step=0.125, stop=0.25)
    state = ScalarState(0.0)

    outputs, final_state, final_interval = run_live(
        scheme,
        state=state,
        interval=interval,
    )

    assert len(outputs) == 2
    assert final_interval.present == pytest.approx(0.25)
    assert final_interval.step == pytest.approx(0.0)
    assert final_state.value == pytest.approx(0.25)