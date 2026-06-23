"""Use the built-in Richardson relaxation inverter.

The inverter acts on translation blocks. In ordinary user-facing code those
translations should come from an engine allocator, just like scheme workspaces
do, rather than from hand-written scalar wrapper classes.
"""

from __future__ import annotations

from typing import Any

from stark import Configuration, Frame, Tolerance
from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.engines import EngineNumpy
from stark.methods import InverterRelaxationRichardson
from stark.methods.inverters.support import InverterDefect
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


if __name__ == "__main__":
    engine = EngineNumpy(Frame.scalar("x", translation="dx"))
    basis = engine.translation_basis()
    coordinates = [0.0]
    image = [0.0]

    def scale_by_two(source: Any, target: Any) -> None:
        basis.coordinates(source, coordinates)
        image[0] = 2.0 * coordinates[0]
        basis.synthesize(image, target)

    residual = engine.allocator.allocate_translation()
    basis.synthesize([6.0], residual)
    output_delta = engine.allocator.allocate_translation()

    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([residual]),
    )
    output = Block([output_delta])
    defect = InverterDefect()
    inverter = InverterRelaxationRichardson(
        damping=0.5,
        configuration=Configuration(
            inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0),
            inverter_maximum_steps=4,
        ),
    )

    initial_defect = defect(request, output)
    inverter(request, output)
    final_defect = defect(request, output)

    print("Richardson relaxation inverter")
    print("================================")
    print("problem:        2 * output = 6")
    print("initial output: 0")
    print(f"initial defect: {initial_defect:.6g}")
    basis.coordinates(output[0], coordinates)
    print(f"final output:   {coordinates[0]:.6g}")
    print(f"final defect:   {final_defect:.6g}")
