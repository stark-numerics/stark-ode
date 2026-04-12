from __future__ import annotations

from dataclasses import dataclass

from stark import Block, BlockOperator, Inversion, InverterGMRES


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


def test_inversion_matches_resolution_style_contract() -> None:
    inversion = Inversion(atol=1.0e-6, rtol=1.0e-3, max_iterations=20, restart=5)

    assert inversion.bound(2.0) == 0.002001
    assert inversion.ratio(0.001, 2.0) < 1.0
    assert inversion.accepts(0.001, 2.0)


def test_gmres_solves_scalar_linear_system() -> None:
    workbench = ScalarWorkbench()
    inverter = InverterGMRES(
        workbench,
        scalar_inner_product,
        inversion=Inversion(atol=1.0e-12, rtol=1.0e-12, max_iterations=8, restart=4),
    )

    def scale_by_two(out: ScalarTranslation, translation: ScalarTranslation) -> None:
        out.value = 2.0 * translation.value

    inverter.bind(BlockOperator([scale_by_two]))
    solution = Block([ScalarTranslation(0.0)])
    rhs = Block([ScalarTranslation(4.0)])

    inverter(solution, rhs)

    assert abs(solution[0].value - 2.0) < 1.0e-10


def test_gmres_solves_two_by_two_block_system() -> None:
    workbench = ScalarWorkbench()
    inverter = InverterGMRES(
        workbench,
        scalar_inner_product,
        inversion=Inversion(atol=1.0e-12, rtol=1.0e-12, max_iterations=8, restart=4),
    )

    def first_row(out: ScalarTranslation, block_item: ScalarTranslation) -> None:
        del block_item
        raise AssertionError("BlockOperator should route row operators componentwise only.")

    del first_row

    def apply_operator(out: Block, block: Block) -> None:
        out.items[0].value = 4.0 * block[0].value + block[1].value
        out.items[1].value = block[0].value + 3.0 * block[1].value

    class DenseBlockOperator(BlockOperator):
        def __call__(self, out: Block, block: Block) -> None:
            apply_operator(out, block)

    inverter.bind(DenseBlockOperator([]))
    solution = Block([ScalarTranslation(0.0), ScalarTranslation(0.0)])
    rhs = Block([ScalarTranslation(1.0), ScalarTranslation(2.0)])

    inverter(solution, rhs)

    assert abs(solution[0].value - (1.0 / 11.0)) < 1.0e-10
    assert abs(solution[1].value - (7.0 / 11.0)) < 1.0e-10


def test_gmres_requires_bound_operator() -> None:
    inverter = InverterGMRES(ScalarWorkbench(), scalar_inner_product)

    try:
        inverter(Block([ScalarTranslation()]), Block([ScalarTranslation(1.0)]))
    except RuntimeError as exc:
        assert "bound to an operator" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected GMRES to reject use before bind().")
