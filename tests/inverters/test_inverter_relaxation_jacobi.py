from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark.block import Block
from stark.block.operator import BlockOperatorDiagonal
from stark.inverters.relaxation import InverterRelaxationJacobi
from stark.inverters.support import InverterBudget, InverterTolerance
from stark.monitor import MonitorInverter
from stark.resolvents.requests.inverter import ResolventInverterRequest


@dataclass(slots=True)
class TranslationScalar:
    value: float = 0.0

    def __call__(self, origin, result) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "TranslationScalar") -> "TranslationScalar":
        return TranslationScalar(self.value + other.value)

    def __rmul__(self, scalar: float) -> "TranslationScalar":
        return TranslationScalar(scalar * self.value)


@dataclass(slots=True)
class ScaleEntryOperator:
    scale: float

    def __call__(self, source: TranslationScalar, target: TranslationScalar) -> None:
        target.value = self.scale * source.value

    def inverse(self, source: TranslationScalar, target: TranslationScalar) -> None:
        target.value = source.value / self.scale


def invert_entry(
    operator: ScaleEntryOperator,
    source: TranslationScalar,
    target: TranslationScalar,
) -> None:
    operator.inverse(source, target)


def test_jacobi_solves_diagonal_scaled_request_in_one_step() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([ScaleEntryOperator(2.0), ScaleEntryOperator(4.0)]),
        residual=Block([TranslationScalar(6.0), TranslationScalar(20.0)]),
    )
    output = Block([TranslationScalar(0.0), TranslationScalar(0.0)])
    inverter = InverterRelaxationJacobi[TranslationScalar](
        invert_entry,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=0.0),
        budget=InverterBudget(maximum_steps=2),
    )

    inverter(request, output)

    assert output[0].value == pytest.approx(3.0)
    assert output[1].value == pytest.approx(5.0)
    assert inverter.defect.block is not None
    assert inverter.defect.block.norm() == pytest.approx(0.0)


def test_jacobi_records_success_through_init_time_monitor() -> None:
    monitor = MonitorInverter()
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([ScaleEntryOperator(2.0), ScaleEntryOperator(4.0)]),
        residual=Block([TranslationScalar(6.0), TranslationScalar(20.0)]),
    )
    output = Block([TranslationScalar(0.0), TranslationScalar(0.0)])
    inverter = InverterRelaxationJacobi[TranslationScalar](
        invert_entry,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=0.0),
        budget=InverterBudget(maximum_steps=2),
        monitor=monitor,
    )

    inverter(request, output)

    assert len(monitor.solves) == 1
    solve = monitor.solves[0]
    assert solve.inverter == "Jacobi"
    assert solve.converged is True
    assert solve.iteration_count == 1
    assert solve.initial_residual == pytest.approx((6.0**2 + 20.0**2) ** 0.5)
    assert solve.final_residual == pytest.approx(0.0)
    assert solve.failure_reason is None


def test_jacobi_records_initial_acceptance() -> None:
    monitor = MonitorInverter()
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([ScaleEntryOperator(2.0)]),
        residual=Block([TranslationScalar(6.0)]),
    )
    output = Block([TranslationScalar(3.0)])
    inverter = InverterRelaxationJacobi[TranslationScalar](
        invert_entry,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=0.0),
        monitor=monitor,
    )

    inverter(request, output)

    solve = monitor.solves[0]
    assert solve.converged is True
    assert solve.iteration_count == 0
    assert solve.initial_residual == pytest.approx(0.0)
    assert solve.final_residual == pytest.approx(0.0)


def test_jacobi_records_failure_when_budget_is_exhausted() -> None:
    monitor = MonitorInverter()
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([ScaleEntryOperator(2.0)]),
        residual=Block([TranslationScalar(6.0)]),
    )
    output = Block([TranslationScalar(0.0)])
    inverter = InverterRelaxationJacobi[TranslationScalar](
        invert_entry,
        damping=0.5,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=0.0),
        budget=InverterBudget(maximum_steps=1),
        monitor=monitor,
    )

    inverter(request, output)

    assert output[0].value == pytest.approx(1.5)
    solve = monitor.solves[0]
    assert solve.converged is False
    assert solve.iteration_count == 1
    assert solve.initial_residual == pytest.approx(6.0)
    assert solve.final_residual == pytest.approx(3.0)
    assert solve.failure_reason == "maximum steps reached"


def test_jacobi_rejects_non_positive_damping() -> None:
    with pytest.raises(ValueError, match="damping"):
        InverterRelaxationJacobi(invert_entry, damping=0.0)


def test_jacobi_rejects_missing_diagonal_entry() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([None]),
        residual=Block([TranslationScalar(6.0)]),
    )
    output = Block([TranslationScalar(0.0)])
    inverter = InverterRelaxationJacobi[TranslationScalar](invert_entry)

    with pytest.raises(RuntimeError, match="entry 0"):
        inverter(request, output)
