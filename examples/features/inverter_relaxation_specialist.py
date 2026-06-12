from __future__ import annotations

"""Use an engine-generated specialist with a relaxation inverter."""

from stark import Configuration, Frame, Tolerance
from stark.block import Block, BlockSpecialist
from stark.block.operator import BlockOperatorDiagonal
from stark.engines import EngineNumpy
from stark.methods.inverters.relaxation import InverterRelaxationRichardson
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


def scale_by_two(source, target) -> None:
    target.dx[:] = 2.0 * source.dx


def main() -> None:
    engine = EngineNumpy(Frame({"x": {"translation": "dx", "shape": (1,)}}))

    residual = engine.allocator.allocate_translation()
    residual.dx[0] = 6.0
    output_delta = engine.allocator.allocate_translation()

    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([residual]),
    )
    output = Block([output_delta])
    inverter = InverterRelaxationRichardson(
        damping=0.5,
        configuration=Configuration(
            inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0),
            inverter_maximum_steps=4,
        ),
        specialist=BlockSpecialist(engine.algebraist_specialist),
    )

    inverter(request, output)

    print(f"specialized Richardson output: {output[0].dx[0]:.6f}")


if __name__ == "__main__":
    main()
