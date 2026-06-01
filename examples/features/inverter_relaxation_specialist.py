from __future__ import annotations

from dataclasses import dataclass

from stark.block import Block, BlockSpecialist
from stark.block.operator import BlockOperatorDiagonal
from stark.inverters.relaxation import InverterRelaxationRichardson, InverterRelaxationStencilUpdate
from stark.inverters.support import InverterBudget, InverterTolerance
from stark.resolvents.requests.inverter import ResolventInverterRequest


@dataclass(slots=True)
class Scalar:
    value: float

    def __call__(self, origin: "Scalar", result: "Scalar") -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "Scalar") -> "Scalar":
        return Scalar(self.value + other.value)

    def __rmul__(self, scale: float) -> "Scalar":
        return Scalar(scale * self.value)


class ScalarRelaxationSpecialist:
    def provide(self, stencil: InverterRelaxationStencilUpdate):
        def kernel(step: float, *terms: Scalar) -> Scalar:
            sources = terms[:-1]
            result = terms[-1]
            result.value = step * stencil.scale * sum(
                coefficient * source.value
                for coefficient, source in zip(stencil.coefficients, sources, strict=True)
            )
            return result

        return kernel


def scale_by_two(source: Scalar, target: Scalar) -> None:
    target.value = 2.0 * source.value


def main() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([Scalar(6.0)]),
    )
    output = Block([Scalar(0.0)])
    specialist = BlockSpecialist(ScalarRelaxationSpecialist())
    inverter = InverterRelaxationRichardson[Scalar](
        damping=0.5,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=0.0),
        budget=InverterBudget(maximum_steps=4),
        specialist=specialist,
    )

    inverter(request, output)

    print(f"specialized Richardson output: {output[0].value:.6f}")


if __name__ == "__main__":
    main()
