"""Use the built-in Richardson relaxation inverter.

The inverter acts on translation blocks. In ordinary user-facing code those
translations should come from an engine allocator, just like scheme workspaces
do, rather than from hand-written scalar wrapper classes.
"""

from __future__ import annotations

from stark import Configuration, StarkLayout, Tolerance
from stark.block import Block
from stark.block.operator import BlockOperatorDiagonal
from stark.engines import StarkEngineNumpy
from stark.inverters.relaxation import InverterRelaxationRichardson
from stark.inverters.support import InverterDefect
from stark.resolvents.requests.inverter import ResolventInverterRequest


def scale_by_two(source, target) -> None:
    target.dx[:] = 2.0 * source.dx


def main() -> None:
    engine = StarkEngineNumpy(StarkLayout({"x": {"translation": "dx", "shape": (1,)}}))

    residual = engine.allocator.allocate_translation()
    residual.dx[0] = 6.0
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
    print(f"final output:   {output[0].dx[0]:.6g}")
    print(f"final defect:   {final_defect:.6g}")


if __name__ == "__main__":
    main()
