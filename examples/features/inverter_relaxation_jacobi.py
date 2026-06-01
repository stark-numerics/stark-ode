"""Use the built-in Jacobi relaxation inverter.

Jacobi relaxation uses the diagonal entries of the request operator. The caller
supplies the local inverse action for each diagonal entry, while the inverter
owns the block-level relaxation loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from stark.block import Block
from stark.block.operator import BlockOperatorDiagonal
from stark.inverters.relaxation import InverterRelaxationJacobi
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


@dataclass(slots=True)
class ScaleEntryOperator:
    scale: float

    def __call__(self, source: ScalarTranslation, target: ScalarTranslation) -> None:
        target.value = self.scale * source.value

    def inverse(self, source: ScalarTranslation, target: ScalarTranslation) -> None:
        target.value = source.value / self.scale


def invert_entry(
    operator: ScaleEntryOperator,
    source: ScalarTranslation,
    target: ScalarTranslation,
) -> None:
    operator.inverse(source, target)


def main() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([ScaleEntryOperator(2.0), ScaleEntryOperator(4.0)]),
        residual=Block([ScalarTranslation(6.0), ScalarTranslation(20.0)]),
    )
    output = Block([ScalarTranslation(0.0), ScalarTranslation(0.0)])
    defect = InverterDefect[ScalarTranslation]()
    inverter = InverterRelaxationJacobi[ScalarTranslation](
        invert_entry,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=0.0),
        budget=InverterBudget(maximum_steps=2),
    )

    initial_defect = defect(request, output)
    inverter(request, output)
    final_defect = defect(request, output)

    print("Jacobi relaxation inverter")
    print("==========================")
    print("problem:        [2, 4] * output = [6, 20]")
    print("initial output: [0, 0]")
    print(f"initial defect: {initial_defect:.6g}")
    print(f"final output:   [{output[0].value:.6g}, {output[1].value:.6g}]")
    print(f"final defect:   {final_defect:.6g}")


if __name__ == "__main__":
    main()
