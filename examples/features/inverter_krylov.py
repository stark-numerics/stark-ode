from __future__ import annotations

"""Use the current matrix-free Krylov inverter and an optional preconditioner."""

from dataclasses import dataclass

from stark import Configuration, Tolerance
from stark.core.block import Block
from stark.methods.inverters.krylov import InverterKrylovArnoldi


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


class ScalarAllocator:
    def allocate_state(self) -> object:
        return object()

    def copy_state(self, source: object, out: object) -> None:
        del source, out

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def inner_product(left: ScalarTranslation, right: ScalarTranslation) -> float:
    return left.value * right.value


def scale_by_two(source: Block[ScalarTranslation], target: Block[ScalarTranslation]) -> None:
    target[0].value = 2.0 * source[0].value


class HalfPreconditioner:
    calls = 0

    def __call__(
        self,
        operator,
        source: Block[ScalarTranslation],
        target: Block[ScalarTranslation],
    ) -> None:
        del operator
        self.calls += 1
        target[0].value = 0.5 * source[0].value


@dataclass(slots=True)
class Request:
    operator: object
    residual: Block[ScalarTranslation]


configuration = Configuration(
    inverter_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
    inverter_maximum_steps=8,
)
preconditioner = HalfPreconditioner()
inverter = InverterKrylovArnoldi(
    ScalarAllocator(),
    inner_product,
    restart=4,
    configuration=configuration,
    preconditioner=preconditioner,
)
output = Block([ScalarTranslation(0.0)])
request = Request(scale_by_two, Block([ScalarTranslation(4.0)]))

inverter(request, output)  # improves output so operator(output) = residual

print("Krylov inverter")
print("operator: 2 * x")
print("residual: 4")
print(f"solution: {output[0].value:.6f}")
print(f"preconditioner calls: {preconditioner.calls}")
