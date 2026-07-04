from __future__ import annotations

from dataclasses import dataclass

from stark.core.block import Block
from stark.core.contracts import BlockLike
from stark.methods.resolvents.equations.residual import ResolventResidual


@dataclass(slots=True)
class DummyTranslation:
    """Translation fixture used by residual protocol tests."""

    value: float

    def __call__(self, origin: DummyTranslation, result: DummyTranslation) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: DummyTranslation) -> DummyTranslation:
        return DummyTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> DummyTranslation:
        return DummyTranslation(scalar * self.value)


@dataclass(slots=True)
class DummyOperator:
    scale: float = 0.0

    def __call__(self, translation: DummyTranslation, out: DummyTranslation) -> None:
        out.value = self.scale * translation.value


class DummyResidual:
    def __call__(
        self,
        delta: BlockLike[DummyTranslation],
        out: BlockLike[DummyTranslation],
    ) -> None:
        out[0].value = delta[0].value - 1.0

    def differential(
        self,
        delta: BlockLike[DummyTranslation],
        out: DummyOperator,
    ) -> None:
        out.scale = 2.0 * delta[0].value


def residual_block(value: float) -> Block[DummyTranslation]:
    """Build the block shape expected by `ResolventResidual`."""

    return Block[DummyTranslation]([DummyTranslation(value)])


def accepts_residual(
    residual: ResolventResidual[DummyTranslation, DummyOperator],
) -> ResolventResidual[DummyTranslation, DummyOperator]:
    return residual


def test_resolvent_residual_uses_direct_call_convention() -> None:
    residual = accepts_residual(DummyResidual())
    out = residual_block(0.0)

    residual(residual_block(3.0), out)
    assert out[0].value == 2.0


def test_differential_residual_exposes_differential() -> None:
    residual = accepts_residual(DummyResidual())
    operator = DummyOperator()

    residual.differential(residual_block(3.0), operator)
    assert operator.scale == 6.0
