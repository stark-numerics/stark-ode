from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark.core.block.operator import BlockOperatorDiagonal
from stark.core.block import Block
from stark.methods.inverters import InverterBiCGStab, InverterFGMRES, InverterGMRES
from stark.methods.inverters.legacy_support.policy import InverterPolicy
from stark import Configuration, Tolerance
from stark.diagnostics.monitor import MonitorInverter


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __rmul__(self, scalar: float) -> "ScalarTranslation":
        return ScalarTranslation(scalar * self.value)

    def __add__(self, other: "ScalarTranslation") -> "ScalarTranslation":
        return ScalarTranslation(self.value + other.value)


class ScalarAllocator:
    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def scalar_inner_product(left: ScalarTranslation, right: ScalarTranslation) -> float:
    return left.value * right.value


INVERTER_TYPES = (InverterGMRES, InverterFGMRES, InverterBiCGStab)


def scale_by_two(translation: ScalarTranslation, out: ScalarTranslation) -> None:
    out.value = 2.0 * translation.value


@pytest.mark.parametrize("inverter_type", INVERTER_TYPES)
def test_monitored_inverter_records_successful_solve(inverter_type) -> None:
    monitor = MonitorInverter()
    inverter = inverter_type(
        ScalarAllocator(),
        scalar_inner_product,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12)),
        policy=InverterPolicy(max_iterations=8, restart=4),
    )
    inverter.bind(BlockOperatorDiagonal([scale_by_two]))
    inverter.assign_monitor(monitor)

    solution = Block([ScalarTranslation(0.0)])
    rhs = Block([ScalarTranslation(4.0)])
    inverter(rhs, solution)

    assert len(monitor.solves) == 1
    solve = monitor.solves[0]
    assert solve.inverter == inverter.short_name
    assert solve.converged is True
    assert solve.failure_reason is None
    assert solve.iteration_count is not None
    assert solve.initial_residual == pytest.approx(4.0)
    assert solve.final_residual is not None
    assert solve.final_residual <= 1.0e-10


@pytest.mark.parametrize("inverter_type", INVERTER_TYPES)
def test_unmonitored_inverter_creates_no_records(inverter_type) -> None:
    monitor = MonitorInverter()
    inverter = inverter_type(ScalarAllocator(), scalar_inner_product)
    inverter.bind(BlockOperatorDiagonal([scale_by_two]))
    inverter.assign_monitor(monitor)
    inverter.unassign_monitor()

    solution = Block([ScalarTranslation(0.0)])
    rhs = Block([ScalarTranslation(4.0)])
    inverter(rhs, solution)

    assert monitor.solves == []


def test_monitored_inverter_records_failed_solve_before_reraising() -> None:
    class DiagonalOperator(BlockOperatorDiagonal):
        def __call__(self, block: Block, out: Block) -> None:
            out.items[0].value = 2.0 * block[0].value
            out.items[1].value = 3.0 * block[1].value

    monitor = MonitorInverter()
    inverter = InverterGMRES(
        ScalarAllocator(),
        scalar_inner_product,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-14, rtol=1.0e-14)),
        policy=InverterPolicy(max_iterations=1, restart=1),
    )
    inverter.bind(DiagonalOperator([]))
    inverter.assign_monitor(monitor)

    with pytest.raises(RuntimeError, match="failed to converge"):
        inverter(
            Block([ScalarTranslation(1.0), ScalarTranslation(1.0)]),
            Block([ScalarTranslation(0.0), ScalarTranslation(0.0)]),
        )

    assert len(monitor.solves) == 1
    solve = monitor.solves[0]
    assert solve.inverter == "GMRES"
    assert solve.converged is False
    assert solve.iteration_count == 1
    assert solve.initial_residual is not None
    assert solve.final_residual is not None
    assert solve.failure_reason is not None
