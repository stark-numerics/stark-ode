"""Use the inverter request shape with a small custom inverter.

A resolvent hands an inverter a request describing the linear problem and a
separate output block to improve in place:

    inverter(request, output)

The translations in this example come from an engine allocator. The custom
object is the inverter, not a hand-written scalar state model.
"""

from __future__ import annotations

from typing import Any

from stark import Frame
from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.engines import EngineNumpy
from stark.methods.inverters.support import InverterDefect
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


class DampedRichardson:
    """Tiny custom inverter using the request/output call shape."""

    def __init__(self, damping: float, steps: int) -> None:
        self.damping = damping
        self.steps = steps
        self.defect = InverterDefect()

    def __call__(self, request: Any, output: Block[Any]) -> None:
        for _ in range(self.steps):
            self.defect(request, output)
            assert self.defect.block is not None
            update = Block(
                [
                    self.damping * self.defect.block[index]
                    for index in range(len(self.defect.block))
                ]
            )
            output += update


def main() -> None:
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
    inverter = DampedRichardson(damping=0.5, steps=1)

    initial_defect = defect(request, output)
    inverter(request, output)
    final_defect = defect(request, output)

    print("Linear request: operator(output) = residual")
    print("operator:       output -> 2 * output")
    print("residual:       6.0")
    print("initial output: 0.0")
    print(f"initial defect: {initial_defect:.6g}")
    basis.coordinates(output[0], coordinates)
    print(f"final output:   {coordinates[0]:.6g}")
    print(f"final defect:   {final_defect:.6g}")


if __name__ == "__main__":
    main()
