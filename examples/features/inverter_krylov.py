"""Use a matrix-free Krylov inverter with engine-owned translations.

Krylov inverters are most useful when forming a dense matrix would be wasteful
or impossible. Dense inverters handle scalar and other small systems directly,
so this toy scalar problem is not a recommendation to use Krylov for one
unknown. It keeps the example focused on the matrix-free call shape:

* an operator writes ``target <- A source``
* a request pairs that operator with a residual block
* an optional preconditioner writes an approximate inverse action
"""

from __future__ import annotations

from typing import Any

from stark import Configuration, Frame, Tolerance
from stark.core.block import Block
from stark.core.contracts import BlockLike, BlockOperatorLike
from stark.engines import EngineNative
from stark.methods import InverterKrylovArnoldi
from stark.methods.resolvents.requests import ResolventInverterRequest


def main() -> None:
    engine = EngineNative(Frame.scalar("x", translation="dx"))
    allocator = engine.allocator
    basis = engine.translation_basis()
    coordinates = [0.0]
    image = [0.0]
    preconditioner_calls = 0

    def scale_by_two(
        source: BlockLike[Any],
        target: BlockLike[Any],
    ) -> BlockLike[Any]:
        basis.coordinates(source[0], coordinates)
        image[0] = 2.0 * coordinates[0]
        basis.synthesize(image, target[0])
        return target

    def precondition_by_half(
        operator: BlockOperatorLike[Any],
        source: BlockLike[Any],
        target: BlockLike[Any],
    ) -> None:
        nonlocal preconditioner_calls
        del operator
        preconditioner_calls += 1
        basis.coordinates(source[0], coordinates)
        image[0] = 0.5 * coordinates[0]
        basis.synthesize(image, target[0])

    configuration = Configuration(
        inverter_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        inverter_maximum_steps=8,
    )
    inverter = InverterKrylovArnoldi(
        allocator,
        allocator.inner_product,
        restart=4,
        configuration=configuration,
        preconditioner=precondition_by_half,
    )

    residual = allocator.allocate_translation()
    basis.synthesize([4.0], residual)
    output_delta = allocator.allocate_translation()
    output: Block[Any] = Block([output_delta])
    request = ResolventInverterRequest[Any](
        operator=scale_by_two,
        residual=Block([residual]),
    )

    inverter(request, output)  # improves output so operator(output) = residual
    basis.coordinates(output[0], coordinates)

    print("Krylov inverter")
    print("operator: 2 * x")
    print("residual: 4")
    print(f"solution: {coordinates[0]:.6f}")
    print(f"preconditioner calls: {preconditioner_calls}")


if __name__ == "__main__":
    main()
