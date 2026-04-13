from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import ResolverAnderson, ResolverBroyden, ResolverPicard, ResolverPolicy, ResolverTolerance
from stark.contracts import Block


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin, result) -> None:
        del origin, result

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "ScalarTranslation") -> "ScalarTranslation":
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ScalarTranslation":
        return ScalarTranslation(scalar * self.value)


class ScalarWorkbench:
    def allocate_state(self):
        return None

    def copy_state(self, dst, src) -> None:
        del dst, src

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def scalar_inner_product(left: ScalarTranslation, right: ScalarTranslation) -> float:
    return left.value * right.value


class CosineResidual:
    def __call__(self, out: Block, block: Block) -> None:
        x = block[0].value
        out.items[0] = ScalarTranslation(x - __import__("math").cos(x))


class QuadraticResidual:
    def __call__(self, out: Block, block: Block) -> None:
        x = block[0].value
        out.items[0] = ScalarTranslation(x * x - 2.0)


@pytest.mark.parametrize(
    ("resolver", "expected"),
    [
        (
            ResolverAnderson(
                ScalarWorkbench(),
                scalar_inner_product,
                tolerance=ResolverTolerance(atol=1.0e-10, rtol=1.0e-10),
                policy=ResolverPolicy(max_iterations=12),
                depth=3,
            ),
            0.7390851332151607,
        ),
        (
            ResolverBroyden(
                ScalarWorkbench(),
                scalar_inner_product,
                tolerance=ResolverTolerance(atol=1.0e-10, rtol=1.0e-10),
                policy=ResolverPolicy(max_iterations=12),
                depth=4,
            ),
            1.4142135623730951,
        ),
    ],
)
def test_new_resolvers_converge_on_scalar_problems(resolver, expected: float) -> None:
    if isinstance(resolver, ResolverAnderson):
        residual = CosineResidual()
        block = Block([ScalarTranslation(1.0)])
    else:
        residual = QuadraticResidual()
        block = Block([ScalarTranslation(1.0)])

    resolver(block, residual)

    assert abs(block[0].value - expected) < 1.0e-8


def test_anderson_improves_over_plain_picard_on_cosine_problem() -> None:
    workbench = ScalarWorkbench()
    picard = ResolverPicard(
        workbench,
        tolerance=ResolverTolerance(atol=1.0e-10, rtol=1.0e-10),
        policy=ResolverPolicy(max_iterations=6),
    )
    anderson = ResolverAnderson(
        workbench,
        scalar_inner_product,
        tolerance=ResolverTolerance(atol=1.0e-10, rtol=1.0e-10),
        policy=ResolverPolicy(max_iterations=6),
        depth=3,
    )

    picard_block = Block([ScalarTranslation(1.0)])
    anderson_block = Block([ScalarTranslation(1.0)])
    residual = CosineResidual()

    with pytest.raises(RuntimeError):
        picard(picard_block, residual)

    anderson(anderson_block, residual)
    assert abs(anderson_block[0].value - 0.7390851332151607) < 1.0e-8
