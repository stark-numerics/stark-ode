"""Use the inverter request shape with a small custom inverter.

A resolvent hands an inverter a request describing the linear problem and a
separate output block to improve in place:

    inverter(request, output)

The translations in this example come from an engine allocator. The custom
object is the inverter, not a hand-written scalar state model.
"""

from __future__ import annotations

from stark import Frame
from stark.block import Block
from stark.block.operator import BlockOperatorDiagonal
from stark.engines import EngineNumpy
from stark.methods.inverters.support import InverterDefect
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


class DampedRichardson:
    """Tiny custom inverter using the request/output call shape."""

    def __init__(self, damping: float, steps: int) -> None:
        self.damping = damping
        self.steps = steps
        self.defect = InverterDefect()

    def __call__(self, request, output) -> None:
        for _ in range(self.steps):
            self.defect(request, output)
            assert self.defect.block is not None
            output += self.damping * self.defect.block


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
    print(f"final output:   {output[0].dx[0]:.6g}")
    print(f"final defect:   {final_defect:.6g}")


if __name__ == "__main__":
    main()
