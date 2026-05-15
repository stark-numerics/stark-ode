from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark.accelerators import Accelerator
from stark.block.operator import BlockOperator
from stark.contracts import Block
from stark.inverters import InverterBiCGStab, InverterFGMRES, InverterGMRES
from stark.inverters.policy import InverterPolicy
from stark.inverters.tolerance import InverterTolerance


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: object, result: object) -> None:
        del origin, result

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "ScalarTranslation") -> "ScalarTranslation":
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ScalarTranslation":
        return ScalarTranslation(scalar * self.value)


class ScalarWorkbench:
    def allocate_state(self) -> object:
        return object()

    def copy_state(self, dst: object, src: object) -> None:
        del dst, src

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def scalar_inner_product(left: ScalarTranslation, right: ScalarTranslation) -> float:
    return left.value * right.value


INVERTER_TYPES = (InverterGMRES, InverterFGMRES, InverterBiCGStab)


class RecordingAccelerator:
    def __init__(self) -> None:
        self.name = "recording"
        self.strict = False
        self.decorate_calls = 0
        self.compile_examples_calls = 0

    def decorate(self, function=None, /, **kwargs):
        del kwargs
        self.decorate_calls += 1

        def decorate_function(target):
            return target

        if function is None:
            return decorate_function
        return decorate_function(function)

    def compile_examples(self, function, *signatures):
        del signatures
        self.compile_examples_calls += 1
        return function

    def resolve(self, target, request):
        del request
        return target

    def resolve_derivative(self, derivative):
        return derivative

    def resolve_linearizer(self, linearizer):
        return linearizer

    def resolve_support(self, worker, *, label=None, **values):
        del label, values
        return worker


class ExactScalarPreconditioner:
    def __init__(self) -> None:
        self.bound = False

    def bind(self, operator) -> None:
        del operator
        self.bound = True

    def __call__(self, rhs: Block, out: Block) -> None:
        out[0].value = 0.5 * rhs[0].value


def test_inverter_tolerance_matches_general_tolerance_contract() -> None:
    tolerance = InverterTolerance(atol=1.0e-6, rtol=1.0e-3)

    assert tolerance.bound(2.0) == 0.002001
    assert tolerance.ratio(0.001, 2.0) < 1.0
    assert tolerance.accepts(0.001, 2.0)


@pytest.mark.parametrize("inverter_type", INVERTER_TYPES)
def test_inverter_solves_scalar_linear_system(inverter_type) -> None:
    workbench = ScalarWorkbench()
    inverter = inverter_type(
        workbench,
        scalar_inner_product,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=InverterPolicy(max_iterations=8, restart=4),
    )

    def scale_by_two(translation: ScalarTranslation, out: ScalarTranslation) -> None:
        out.value = 2.0 * translation.value

    inverter.bind(BlockOperator([scale_by_two]))
    solution = Block([ScalarTranslation(0.0)])
    rhs = Block([ScalarTranslation(4.0)])

    inverter(rhs, solution)

    assert abs(solution[0].value - 2.0) < 1.0e-10


@pytest.mark.parametrize("inverter_type", INVERTER_TYPES)
def test_inverter_accepts_preconditioner(inverter_type) -> None:
    workbench = ScalarWorkbench()
    preconditioner = ExactScalarPreconditioner()
    inverter = inverter_type(
        workbench,
        scalar_inner_product,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=InverterPolicy(max_iterations=8, restart=4),
        preconditioner=preconditioner,
    )

    def scale_by_two(translation: ScalarTranslation, out: ScalarTranslation) -> None:
        out.value = 2.0 * translation.value

    inverter.bind(BlockOperator([scale_by_two]))
    solution = Block([ScalarTranslation(0.0)])
    rhs = Block([ScalarTranslation(4.0)])

    inverter(rhs, solution)

    assert preconditioner.bound is True
    assert abs(solution[0].value - 2.0) < 1.0e-10


@pytest.mark.parametrize("inverter_type", INVERTER_TYPES)
def test_inverter_solves_two_by_two_block_system(inverter_type) -> None:
    workbench = ScalarWorkbench()
    inverter = inverter_type(
        workbench,
        scalar_inner_product,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=InverterPolicy(max_iterations=16, restart=4),
    )

    def first_row(out: ScalarTranslation, block_item: ScalarTranslation) -> None:
        del block_item
        raise AssertionError("BlockOperator should route row operators componentwise only.")

    del first_row

    def apply_operator(block: Block, out: Block) -> None:
        out.items[0].value = 4.0 * block[0].value + block[1].value
        out.items[1].value = block[0].value + 3.0 * block[1].value

    class DenseBlockOperator(BlockOperator):
        def __call__(self, block: Block, out: Block) -> None:
            apply_operator(block, out)

    inverter.bind(DenseBlockOperator([]))
    solution = Block([ScalarTranslation(0.0), ScalarTranslation(0.0)])
    rhs = Block([ScalarTranslation(1.0), ScalarTranslation(2.0)])

    inverter(rhs, solution)

    assert abs(solution[0].value - (1.0 / 11.0)) < 1.0e-10
    assert abs(solution[1].value - (7.0 / 11.0)) < 1.0e-10


@pytest.mark.parametrize(
    ("inverter_type", "expected_message"),
    (
        (InverterGMRES, "bound to an operator"),
        (InverterFGMRES, "bound to an operator"),
        (InverterBiCGStab, "bound to an operator"),
    ),
)
def test_inverter_requires_bound_operator(inverter_type, expected_message) -> None:
    inverter = inverter_type(ScalarWorkbench(), scalar_inner_product)

    try:
        inverter(Block([ScalarTranslation(1.0)]), Block([ScalarTranslation()]))
    except RuntimeError as exc:
        assert expected_message in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected inverter to reject use before bind().")


def test_krylov_inverters_use_their_configured_accelerator() -> None:
    workbench = ScalarWorkbench()
    accelerator = RecordingAccelerator()

    gmres = InverterGMRES(workbench, scalar_inner_product, accelerator=accelerator)
    fgmres = InverterFGMRES(workbench, scalar_inner_product, accelerator=accelerator)

    assert gmres.accelerator is accelerator
    assert fgmres.accelerator is accelerator
    assert accelerator.decorate_calls >= 4
    assert accelerator.compile_examples_calls >= 4


@pytest.mark.parametrize("inverter_type", INVERTER_TYPES)
def test_built_in_inverters_hold_an_explicit_accelerator(inverter_type) -> None:
    accelerator = Accelerator.none()
    inverter = inverter_type(ScalarWorkbench(), scalar_inner_product, accelerator=accelerator)

    assert inverter.accelerator is accelerator










