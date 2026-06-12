from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, ClassVar

from stark.core.block import BlockBasis
from stark.core.block.materialize import BlockOperatorDiagonalMaterialize
from stark.core.contracts import (
    BlockLike,
    InverterOutputMode,
    InverterRequest,
    TranslationType,
)
from stark.methods.inverters.dense.native import InverterProviderDenseNative
from stark.methods.inverters.dense.provider import InverterProviderDense
from stark.methods.inverters.support import InverterDescriptor, MonitorInverterLike, with_inverter_monitoring


# Optional extension: records inverter monitor events.
# Provides: record_solve.
@with_inverter_monitoring
@dataclass(slots=True)
class InverterDense(Generic[TranslationType]):
    """
    Dense direct inverter.

    Problem:
        Solve request.operator(output) = request.residual.

    Algorithm:
        1. Prepare the block basis, materialiser, and dense provider.
        2. Materialise the request operator as a dense matrix.
        3. Analyse the request residual into dense coordinates.
        4. Invert the dense coordinate system with the provider.
        5. Synthesize the dense solution coordinates into output.
    """

    basis: BlockBasis[TranslationType]
    provider: InverterProviderDense = field(default_factory=InverterProviderDenseNative)
    monitor: MonitorInverterLike | None = None
    materializer: BlockOperatorDiagonalMaterialize[TranslationType] | None = field(init=False, default=None)
    image_coordinates: list[float] = field(init=False, default_factory=list)
    result_coordinates: list[float] = field(init=False, default_factory=list)
    call_body: Callable[[InverterRequest[TranslationType], BlockLike[TranslationType]], None] = field(init=False)

    descriptor: ClassVar[InverterDescriptor] = InverterDescriptor("Dense", "Dense direct")
    output_mode: ClassVar[InverterOutputMode] = InverterOutputMode.overwrite

    def __post_init__(self) -> None:
        self.call_body = self.call_prepare

    def __call__(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        return self.call_body(request, output)

    def call_prepare(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        # 1. Prepare the block basis, materialiser, and dense provider.
        source = 0.0 * output  # type: ignore[operator]
        image = 0.0 * output  # type: ignore[operator]
        self.materializer = BlockOperatorDiagonalMaterialize(
            operator=request.operator,  # type: ignore[arg-type]
            bases=self.basis.bases,
            source=source,
            image=image,
            refresh_initial=False,
        )

        if self.materializer.dimension != self.basis.dimension:
            raise ValueError("Dense materialiser dimension must match the block basis dimension.")

        self.provider.prepare(self.basis.dimension)
        self.image_coordinates = [0.0 for _ in range(self.basis.dimension)]
        self.result_coordinates = [0.0 for _ in range(self.basis.dimension)]
        self.call_body = (
            self.call_fast_one_block
            if len(self.basis.bases) == 1
            else self.call_fast
        )
        return self.call_body(request, output)

    def call_fast(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        materializer = self.materializer
        assert materializer is not None

        # 2. Materialise the request operator as a dense matrix.
        materializer.refresh(request.operator)  # type: ignore[arg-type]

        # 3. Analyse the request residual into dense coordinates.
        self.basis.coordinates(request.residual, self.image_coordinates)

        # 4. Invert the dense coordinate system with the provider.
        self.provider.invert(materializer.matrix, self.image_coordinates, self.result_coordinates)

        # 5. Synthesize the dense solution coordinates into output.
        self.basis.synthesize(self.result_coordinates, output)
        self.record_solve(
            converged=True,
            iteration_count=None,
            initial_residual=None,
            final_residual=None,
        )

    def call_fast_one_block(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        materializer = self.materializer
        assert materializer is not None

        # 2. Materialise the request operator as a dense matrix.
        materializer.refresh(request.operator)  # type: ignore[arg-type]

        # 3. Analyse the request residual into dense coordinates.
        basis = self.basis.bases[0]
        basis.coordinates(request.residual[0], self.image_coordinates)

        # 4. Invert the dense coordinate system with the provider.
        self.provider.invert(materializer.matrix, self.image_coordinates, self.result_coordinates)

        # 5. Synthesize the dense solution coordinates into output.
        output[0] = basis.synthesize(self.result_coordinates, output[0])
        self.record_solve(
            converged=True,
            iteration_count=None,
            initial_residual=None,
            final_residual=None,
        )


__all__ = ["InverterDense"]
