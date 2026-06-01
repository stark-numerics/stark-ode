"""Use the built-in Richardson relaxation inverter.

The new inverter call shape keeps the linear problem request separate from the
output block that is improved in place:

    inverter(request, output)

Richardson relaxation is intentionally simple. It is useful here as the first
built-in example of the new inverter surface before projection and recurrence
families are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

from stark.block import Block
from stark.block.operator import BlockOperatorDiagonal
from stark.inverters.relaxation import InverterRelaxationRichardson
from stark.inverters.support import InverterBudget, InverterDefect, InverterTolerance
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


def scale_by_two(source: ScalarTranslation, target: ScalarTranslation) -> None:
    target.value = 2.0 * source.value


def main() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([ScalarTranslation(6.0)]),
    )
    output = Block([ScalarTranslation(0.0)])
    defect = InverterDefect[ScalarTranslation]()
    inverter = InverterRelaxationRichardson[ScalarTranslation](
        damping=0.5,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=0.0),
        budget=InverterBudget(maximum_steps=4),
    )

    initial_defect = defect(request, output)
    inverter(request, output)
    final_defect = defect(request, output)

    print("Richardson relaxation inverter")
    print("================================")
    print("problem:        2 * output = 6")
    print("initial output: 0")
    print(f"initial defect: {initial_defect:.6g}")
    print(f"final output:   {output[0].value:.6g}")
    print(f"final defect:   {final_defect:.6g}")


if __name__ == "__main__":
    main()
