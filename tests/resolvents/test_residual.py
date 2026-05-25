from __future__ import annotations

from dataclasses import dataclass

from stark.resolvents.support.residual import ResolventResidual


@dataclass(slots=True)
class DummyBlock:
    value: float


@dataclass(slots=True)
class DummyOperator:
    scale: float = 0.0


class DummyResidual:
    def __call__(self, delta: DummyBlock, out: DummyBlock) -> DummyBlock:
        out.value = delta.value - 1.0
        return out

    def differential(self, delta: DummyBlock, out: DummyOperator) -> DummyOperator:
        out.scale = 2.0 * delta.value
        return out


def accepts_residual(residual: ResolventResidual[DummyBlock]) -> ResolventResidual[DummyBlock]:
    return residual


def test_resolvent_residual_uses_direct_call_convention() -> None:
    residual = accepts_residual(DummyResidual())
    out = DummyBlock(0.0)

    residual(DummyBlock(3.0), out)
    assert out.value == 2.0


def test_differential_residual_exposes_differential() -> None:
    residual = accepts_residual(DummyResidual())
    operator = DummyOperator()

    residual.differential(DummyBlock(3.0), operator)
    assert operator.scale == 6.0
