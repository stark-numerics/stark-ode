"""Use the new inverter request shape with a small custom inverter.

A resolvent now hands an inverter a request describing the linear problem and a
separate output block to improve in place:

    inverter(request, output)

This example keeps the linear problem deliberately small. The custom inverter
performs damped Richardson updates so the request/output syntax and the defect
worker are the focus.
"""

from __future__ import annotations

from dataclasses import dataclass

from stark.block import Block
from stark.block.operator import BlockOperatorDiagonal
from stark.inverters.support import InverterDefect
from stark.resolvents.requests.inverter import ResolventInverterRequest


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin, result) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "ScalarTranslation") -> "ScalarTranslation":
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ScalarTranslation":
        return ScalarTranslation(scalar * self.value)


class DampedRichardson:
    """Tiny custom inverter using the new request/output call shape."""

    def __init__(self, damping: float, steps: int) -> None:
        self.damping = damping
        self.steps = steps
        self.defect = InverterDefect[ScalarTranslation]()

    def __call__(self, request, output: Block[ScalarTranslation]) -> None:
        for _ in range(self.steps):
            self.defect(request, output)
            assert self.defect.block is not None
            output += self.damping * self.defect.block


def scale_by_two(source: ScalarTranslation, target: ScalarTranslation) -> None:
    target.value = 2.0 * source.value


def main() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([ScalarTranslation(6.0)]),
    )
    output = Block([ScalarTranslation(0.0)])
    defect = InverterDefect[ScalarTranslation]()
    inverter = DampedRichardson(damping=0.5, steps=1)

    initial_defect = defect(request, output)
    inverter(request, output)
    final_defect = defect(request, output)

    print("Linear request: operator(output) = residual")
    print("operator:       output -> 2 * output")
    print("residual:       6.0")
    print("initial output: 0.0")
    print(f"initial defect: {initial_defect:.6g}")
    print(f"final output:   {output[0].value:.6g}")
    print(f"final defect:   {final_defect:.6g}")


if __name__ == "__main__":
    main()
